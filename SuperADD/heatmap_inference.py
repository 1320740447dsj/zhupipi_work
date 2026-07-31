"""
单张图片异常检测热力图推理脚本。

功能:
    1. 加载已训练好的 SuperADD 模型(从 ./models/<category>.npz + .json)
    2. 输入一张测试图片
    3. 输出三联可视化: 原图 | 异常热力图 | 二值化掩码
    4. 同时保存单独的热力图叠加图与原始 anomaly map(.npy)

前置条件:
    1. 已按 README 完成 conda 环境安装
    2. 已下载 DINOv3 权重到 ./weights/
    3. 已运行 train-industrial 训练好模型(./models/<category>.npz + .json 存在)
       或从 https://owncloud.fraunhofer.de/index.php/s/THkX7W8AhRd2RCs 下载预训练模型

使用方法(在项目根目录运行):
    python heatmap_inference.py --image path/to/image.png --category custom
    python heatmap_inference.py --image path/to/image.png --category custom --output ./my_heatmap.png
    python heatmap_inference.py --image path/to/image.png --category custom --device cpu
    # 批量处理一个目录:
    python heatmap_inference.py --image-dir path/to/images --category custom
"""

import argparse
import sys
from pathlib import Path

# 把项目内部模块路径加入 sys.path, 以便直接 import industrial.*
# 这样在 Anaconda 环境下无需安装为 editable package
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "tracks" / "industrial" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "utils"))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from industrial.model import SuperADD
from industrial.paths import get_model_path, get_root_config


def load_model(category: str, device: str | None = None) -> SuperADD:
    """从磁盘加载 SuperADD 模型。

    Args:
        category (str): 类别名(对应 ./models/<category>.npz + .json)。
        device (str | None): 若指定, 覆盖模型配置中的 device (例如 'cpu' / 'cuda:0')。

    Returns:
        SuperADD: 加载好的模型实例。
    """
    model_path = get_model_path(category)
    if not Path(f"{model_path}.json").exists() or not Path(f"{model_path}.npz").exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}.json / {model_path}.npz。"
            f"请先运行 train-industrial 或下载预训练模型到 ./models/。"
        )

    model = SuperADD.from_disk(model_path)

    # 可选: 切换设备 (例如用户没有 GPU 时强制用 CPU)
    if device is not None and device != model.device:
        model.device = device
        model.backbone.dino = model.backbone.dino.to(device)
        model.augmented_preprocessing.device = device
        model.preprocessing.device = device
        # 重建 normalization 张量到新设备
        for pp in (model.augmented_preprocessing, model.preprocessing):
            pp.mean = pp.mean.to(device)
            pp.std = pp.std.to(device)
        # prototype embeddings 也搬到新设备
        if model.prototype_embeddings is not None:
            model.prototype_embeddings = {
                k: v.to(device) for k, v in model.prototype_embeddings.items()
            }
    return model


def predict_image(model: SuperADD, image_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """对单张图片进行异常检测推理。

    Args:
        model (SuperADD): 已加载的模型。
        image_path (Path): 输入图片路径。

    Returns:
        tuple: (original_rgb, anomaly_map, binary_map)
            - original_rgb: 原图 (H, W, 3) RGB uint8
            - anomaly_map: 异常分数图 (h, w) float32, 已归一化到 [0, 1]
            - binary_map: 二值化结果 (h, w) uint8, 值为 0 或 255
    """
    # 加载原图(RGB)
    image_pil = Image.open(image_path).convert("RGB")
    original_rgb = np.array(image_pil)

    # 转 tensor (C, H, W) float32 [0, 1]
    image_tensor = transforms.ToTensor()(image_pil)

    # 推理; predict 返回 (anomaly_map: np.ndarray, binary_result: np.ndarray)
    with torch.inference_mode():
        anomaly_map, binary_map = model.predict(image_tensor)

    # anomaly_map 范围归一化到 [0, 1] 便于可视化
    am = anomaly_map.astype(np.float32)
    lo, hi = float(am.min()), float(am.max())
    anomaly_map_norm = (am - lo) / (hi - lo + 1e-8)

    return original_rgb, anomaly_map_norm, binary_map


def make_visualization(
    original_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    binary_map: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """生成三联可视化图: 原图 | 热力图叠加 | 二值化掩码叠加。

    Args:
        original_rgb (np.ndarray): 原图 (H, W, 3) RGB。
        anomaly_map (np.ndarray): 归一化异常图 [0, 1]。
        binary_map (np.ndarray): 二值掩码 (0/255)。
        alpha (float): 热力图叠加透明度, 0~1。

    Returns:
        np.ndarray: 拼接后的可视化图 (H, 3W, 3) RGB。
    """
    h, w = original_rgb.shape[:2]

    # 1) 热力图(JET colormap), 先放大到原图尺寸
    am_uint8 = (anomaly_map * 255).astype(np.uint8)
    am_resized = cv2.resize(am_uint8, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap = cv2.applyColorMap(am_resized, cv2.COLORMAP_JET)  # BGR
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 叠加到原图
    overlay = cv2.addWeighted(original_rgb, 1 - alpha, heatmap, alpha, 0)

    # 2) 二值化掩码叠加(红色高亮)
    bm_resized = cv2.resize(binary_map, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_overlay = original_rgb.copy()
    red = np.zeros_like(original_rgb)
    red[..., 0] = 255  # 红色通道
    mask_bool = bm_resized > 0
    mask_overlay[mask_bool] = (original_rgb[mask_bool] * 0.4 + red[mask_bool] * 0.6).astype(np.uint8)

    # 横向拼接
    vis = np.concatenate([original_rgb, overlay, mask_overlay], axis=1)
    return vis


def process_single(
    model: SuperADD,
    image_path: Path,
    output_path: Path | None,
    save_npy: bool,
) -> None:
    """处理单张图片并保存可视化结果。

    Args:
        model (SuperADD): 模型实例。
        image_path (Path): 输入图片。
        output_path (Path | None): 输出图片路径, 若为 None 则默认 <image_stem>_heatmap.png。
        save_npy (bool): 是否同时保存原始 anomaly_map 为 .npy。
    """
    image_path = image_path.resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    original_rgb, anomaly_map, binary_map = predict_image(model, image_path)
    vis = make_visualization(original_rgb, anomaly_map, binary_map)

    # 决定输出路径
    if output_path is None:
        output_path = image_path.with_name(f"{image_path.stem}_heatmap.png")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存(用 cv2 写, 需要 BGR)
    cv2.imwrite(str(output_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[OK] {image_path.name} -> {output_path}")

    if save_npy:
        npy_path = output_path.with_suffix(".npy")
        np.save(npy_path, anomaly_map)
        print(f"[OK] anomaly map saved -> {npy_path}")


def process_dir(
    model: SuperADD,
    image_dir: Path,
    output_dir: Path,
    save_npy: bool,
) -> None:
    """批量处理目录下所有图片(*.png / *.jpg / *.jpeg / *.bmp)。

    Args:
        model (SuperADD): 模型实例。
        image_dir (Path): 输入图片目录。
        output_dir (Path): 输出目录。
        save_npy (bool): 是否同时保存 .npy。
    """
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])
    if not images:
        print(f"[WARN] 目录 {image_dir} 下没有图片")
        return

    print(f"Found {len(images)} images in {image_dir}")
    for img in images:
        out_path = output_dir / f"{img.stem}_heatmap.png"
        try:
            process_single(model, img, out_path, save_npy)
        except Exception as e:
            print(f"[FAIL] {img.name}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SuperADD 单图/批量异常热力图推理"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="单张输入图片路径(与 --image-dir 二选一)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="输入图片目录(批量处理, 与 --image 二选一)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="custom",
        help="类别名, 用于定位 ./models/<category>.npz (默认 custom)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="单图模式下的输出路径(默认 <image_stem>_heatmap.png)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./heatmap_outputs"),
        help="批量模式下的输出目录(默认 ./heatmap_outputs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备, 例如 'cuda' / 'cpu' / 'cuda:0' (默认沿用模型配置)",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="同时保存原始 anomaly_map 为 .npy 文件",
    )
    args = parser.parse_args()

    if args.image is None and args.image_dir is None:
        parser.error("必须指定 --image 或 --image-dir 之一")

    # 加载模型
    print(f"Loading model for category='{args.category}' ...")
    model = load_model(args.category, args.device)
    print(f"Model loaded (device={model.device}, threshold={model.threshold:.4f})")

    if args.image is not None:
        process_single(model, args.image, args.output, args.save_npy)
    else:
        process_dir(model, args.image_dir, args.output_dir, args.save_npy)


if __name__ == "__main__":
    main()
