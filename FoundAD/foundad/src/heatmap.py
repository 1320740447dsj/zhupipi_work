"""
heatmap.py - 单张/批量图片异常热力图推理脚本

功能：
    加载 FoundAD 训练好的 predictor checkpoint，对输入图片（单张或文件夹）
    进行异常检测，输出与原图叠加的热力图。

用法示例：
    # 单张图片
    python foundad/src/heatmap.py \
        --ckpt logs/mvtec_1shot/dinov3_pretrained/pretrained.pth.tar \
        --input path/to/image.png \
        --output ./heatmaps

    # 整个文件夹（递归）
    python foundad/src/heatmap.py \
        --ckpt logs/mvtec_1shot/dinov3_pretrained/pretrained.pth.tar \
        --input path/to/image_folder \
        --output ./heatmaps

    # 自动从同目录 params.yaml 读取模型参数
    python foundad/src/heatmap.py \
        --ckpt logs/mvtec_1shot/dinov3_pretrained/pretrained.pth.tar \
        --input path/to/image_folder

    # 自定义模型参数（覆盖默认）
    python foundad/src/heatmap.py \
        --ckpt path/to/ckpt.pth.tar \
        --input path/to/image.png \
        --model dinov3 --crop_size 512 --n_layer 3 \
        --pred_depth 6 --pred_emb_dim 384
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# 让 src.* 导入生效（与项目其他入口保持一致）
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]  # FoundAD/foundad
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402
from src.foundad import VisionModule  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("heatmap")


# ----------------------------------------------------------------------------
# 配置加载
# ----------------------------------------------------------------------------
def load_meta_from_params_yaml(ckpt_path: Path) -> Optional[Dict[str, Any]]:
    """
    尝试在 checkpoint 同目录或上级目录寻找 params.yaml，
    返回其中的 meta 段（包含 model / crop_size / pred_depth 等）。
    """
    # 同目录: <log_dir>/pretrained.pth.tar 对应 <log_dir>/params.yaml
    candidates = [
        ckpt_path.parent / "params.yaml",
        ckpt_path.parent.parent / "params.yaml",
        ckpt_path.parent.parent.parent / "params.yaml",
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p, "r") as f:
                    params = yaml.safe_load(f)
                if isinstance(params, dict) and "meta" in params:
                    logger.info(f"Loaded meta from {p}")
                    return params["meta"]
            except Exception as e:
                logger.warning(f"Failed to parse {p}: {e}")
    return None


def build_meta(args: argparse.Namespace, ckpt_path: Path) -> Dict[str, Any]:
    """
    构建 _build_model 所需的 meta 字典：
    优先用命令行参数，其次用 params.yaml，最后用默认值。
    """
    meta = {
        "model": args.model,
        "crop_size": args.crop_size,
        "pred_depth": args.pred_depth,
        "pred_emb_dim": args.pred_emb_dim,
        "if_pred_pe": args.if_pe,
        "feat_normed": args.feat_normed,
        "n_layer": args.n_layer,
    }

    # 命令行未显式覆盖时，从 params.yaml 读取
    if args.from_yaml or (
        not args.no_auto_yaml and all(
            getattr(args, k, None) == default
            for k, default in {
                "model": "dinov3",
                "crop_size": 512,
                "pred_depth": 6,
                "pred_emb_dim": 384,
                "n_layer": 3,
            }.items()
        )
    ):
        yaml_meta = load_meta_from_params_yaml(ckpt_path)
        if yaml_meta:
            for k in ["model", "crop_size", "pred_depth", "pred_emb_dim",
                       "if_pred_pe", "feat_normed", "n_layer"]:
                if k in yaml_meta and getattr(args, k, None) in (None, _DEFAULTS.get(k)):
                    meta[k] = yaml_meta[k]

    return meta


# ----------------------------------------------------------------------------
# 模型构建
# ----------------------------------------------------------------------------
def build_model(meta: Dict[str, Any], ckpt_path: Path, device: torch.device) -> VisionModule:
    """构建 VisionModule 并加载 checkpoint 权重。"""
    model = VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        use_cuda=(device.type == "cuda"),
    )

    logger.info(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if "predictor" not in state:
        raise KeyError(
            f"Checkpoint {ckpt_path} does not contain 'predictor' key. "
            f"Available keys: {list(state.keys())}"
        )
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None and state.get("projector") is not None:
        model.projector.load_state_dict(state["projector"])

    model.to(device)
    model.eval()
    return model


# ----------------------------------------------------------------------------
# 图像 IO
# ----------------------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(input_path: Path) -> List[Path]:
    """收集输入路径下所有图片（支持单文件或递归文件夹）。"""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    img_paths = [
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]
    return sorted(img_paths)


def preprocess(pil_img: Image.Image, crop: int, device: torch.device) -> torch.Tensor:
    """Resize + ToTensor + Normalize，返回 [1,3,H,W] 张量。"""
    pil_resized = pil_img.resize((crop, crop), Image.BILINEAR)
    arr = np.array(pil_resized).astype(np.float32) / 255.0  # [H,W,3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (t - mean) / std


def to_numpy_uint8(img_tensor: torch.Tensor) -> np.ndarray:
    """反归一化回 [H,W,3] uint8。"""
    mean = IMAGENET_MEAN.to(img_tensor.device)
    std = IMAGENET_STD.to(img_tensor.device)
    x = (img_tensor * std + mean).clamp(0, 1)
    x = x[0].permute(1, 2, 0).detach().cpu().numpy()
    return (x * 255.0).astype(np.uint8)


# ----------------------------------------------------------------------------
# 热力图绘制与保存
# ----------------------------------------------------------------------------
def save_overlay_heatmap(
    rgb_uint8: np.ndarray,
    heat: np.ndarray,
    save_path: Path,
    alpha: float = 0.5,
) -> None:
    """
    将热力图叠加到原图上并保存。
    rgb_uint8: [H,W,3] 0~255 RGB
    heat:      [H,W]   0~1
    """
    import cv2

    H, W = heat.shape
    heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)  # BGR
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay_rgb).save(save_path)


def save_grid(
    rgb_uint8: np.ndarray,
    heat: np.ndarray,
    save_path: Path,
) -> None:
    """保存 [原图 | 热力图 | 叠加图] 三联图，便于对比。"""
    import cv2
    from matplotlib import cm, pyplot as plt

    heat_norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    heat_rgb = (cm.jet(heat_norm)[..., :3] * 255).astype(np.uint8)

    heat_255 = (heat_norm * 255.0).clip(0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)
    rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    overlay = cv2.cvtColor(cv2.addWeighted(heat_color, 0.5, rgb_bgr, 0.5, 0),
                           cv2.COLOR_BGR2RGB)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(rgb_uint8); axs[0].set_title("Original"); axs[0].axis("off")
    axs[1].imshow(heat_rgb);  axs[1].set_title("Heatmap");  axs[1].axis("off")
    axs[2].imshow(overlay);   axs[2].set_title("Overlay");  axs[2].axis("off")
    fig.tight_layout()
    fig.savefig(save_path); plt.close(fig)


# ----------------------------------------------------------------------------
# 推理主流程
# ----------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(
    model: VisionModule,
    img_paths: List[Path],
    crop: int,
    n_layer: int,
    device: torch.device,
    out_root: Path,
    input_root: Optional[Path],
    alpha: float,
    save_grid_flag: bool,
) -> None:
    """
    对一组图片逐张推理并保存热力图。
    input_root 为 None 时表示单张图片输入，保存到 out_root 下；
    否则保留相对目录结构。
    """
    logger.info(f"Processing {len(img_paths)} image(s) -> {out_root}")
    for i, path in enumerate(img_paths, 1):
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            logger.warning(f"[{i}/{len(img_paths)}] Failed to open {path}: {e}")
            continue

        img = preprocess(pil, crop, device)  # [1,3,crop,crop]

        # 模型前向：target_features 为冻结 encoder 的输出，
        # predict 为可训练 predictor 的输出，二者 MSE 作为异常分数
        enc = model.target_features(img, [str(path)], n_layer=n_layer)  # [1,P,D]
        pred = model.predict(enc)                                         # [1,P,D]
        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)          # [1,P]

        # 把 patch 级分数 reshape 回 spatial 维度，再双线性插值到原图大小
        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(
            l.view(1, 1, h, w),
            size=img.shape[2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)  # [crop,crop]

        # 归一化到 [0,1]
        pmin, pmax = pix.min(), pix.max()
        pix_norm = ((pix - pmin) / (pmax - pmin + 1e-8)).detach().cpu().numpy()

        img_uint8 = to_numpy_uint8(img)  # [crop,crop,3]

        # 输出路径：保留相对目录结构
        if input_root is not None:
            rel = path.relative_to(input_root)
            save_dir = out_root / rel.parent
        else:
            save_dir = out_root
        save_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = save_dir / f"{path.stem}_heatmap.png"
        save_overlay_heatmap(img_uint8, pix_norm, overlay_path, alpha=alpha)

        if save_grid_flag:
            grid_path = save_dir / f"{path.stem}_grid.png"
            save_grid(img_uint8, pix_norm, grid_path)

        logger.info(f"[{i}/{len(img_paths)}] Saved: {overlay_path}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
_DEFAULTS = {
    "model": "dinov3",
    "crop_size": 512,
    "pred_depth": 6,
    "pred_emb_dim": 384,
    "n_layer": 3,
    "if_pe": False,
    "feat_normed": False,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FoundAD heatmap inference: input image(s) -> output heatmap(s)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to FoundAD checkpoint (.pth.tar)")
    p.add_argument("--input", type=str, required=True,
                   help="Path to a single image or a folder of images")
    p.add_argument("--output", type=str, default="./heatmaps",
                   help="Output directory for heatmaps")

    # 模型参数（不指定则从 params.yaml 读取，再退回默认值）
    p.add_argument("--model", type=str, default=_DEFAULTS["model"],
                   choices=["dinov3", "dinov2", "dino", "siglip", "clip", "dinosiglip"])
    p.add_argument("--crop_size", type=int, default=_DEFAULTS["crop_size"])
    p.add_argument("--pred_depth", type=int, default=_DEFAULTS["pred_depth"])
    p.add_argument("--pred_emb_dim", type=int, default=_DEFAULTS["pred_emb_dim"])
    p.add_argument("--n_layer", type=int, default=_DEFAULTS["n_layer"])
    p.add_argument("--if_pe", action="store_true", default=_DEFAULTS["if_pe"],
                   help="Use positional encoding in predictor")
    p.add_argument("--feat_normed", action="store_true", default=_DEFAULTS["feat_normed"],
                   help="L2-normalize features before predictor")
    p.add_argument("--no_auto_yaml", action="store_true",
                   help="Disable auto-loading params.yaml next to ckpt")
    p.add_argument("--from_yaml", action="store_true",
                   help="Force load meta from params.yaml (overrides CLI defaults)")

    p.add_argument("--device", type=str, default="auto",
                   help="cuda:0 / cpu / auto")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Overlay weight of heatmap on original image (0~1)")
    p.add_argument("--save_grid", action="store_true",
                   help="Also save a 3-panel grid (original | heatmap | overlay)")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 模型
    meta = build_meta(args, ckpt_path)
    logger.info(f"Meta: {meta}")
    model = build_model(meta, ckpt_path, device)

    crop = meta["crop_size"]
    n_layer = meta.get("n_layer", 3)

    # 收集图片
    img_paths = collect_images(input_path)
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {input_path}")

    # 单张图片输入时，input_root 为 None
    input_root = input_path if input_path.is_dir() else None

    run_inference(
        model=model,
        img_paths=img_paths,
        crop=crop,
        n_layer=n_layer,
        device=device,
        out_root=out_root,
        input_root=input_root,
        alpha=args.alpha,
        save_grid_flag=args.save_grid,
    )

    logger.info(f"Done. Heatmaps saved to: {out_root}")


if __name__ == "__main__":
    main()
