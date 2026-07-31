import os
import logging
import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import cm, pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from src.datasets.dataset import build_dataloader
from src.utils.metrics import (
    calculate_pro,
    compute_imagewise_retrieval_metrics,
    compute_pixelwise_retrieval_metrics,
)
from src.helper import save_segmentation_grid
from src.utils.logging import CSVLogger
from src.foundad import VisionModule          

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("evaluator")

def _discover_classnames(test_root: str, configured: List[str]) -> List[str]:
    if configured:
        classnames = list(configured)
    else:
        root = Path(test_root)
        classnames = sorted(
            p.name for p in root.iterdir()
            if p.is_dir() and (p / "test").is_dir()
        )
    if not classnames:
        raise ValueError(f"No object classes with a 'test' directory found under {test_root}")
    return classnames



def _build_model(meta: Dict[str, Any]) -> VisionModule:
    return VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        encoder_path=meta.get("encoder_path"),
        crop_size=meta["crop_size"],
    )

@torch.inference_mode()
def _evaluate_single_ckpt(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading DINOv3 backbone...")
    model = _build_model(cfg["meta"])
    logger.info("Loading predictor checkpoint: %s", ckpt)
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()
    logger.info("Model ready on %s", device)

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)

    # error = cfg["meta"].get("loss_mode", "l2")

    dataset_name = cfg["data"].get("dataset", "mvtec")
    classnames = _discover_classnames(
        cfg["data"]["test_root"], cfg["data"].get("classnames", [])
    )
    if "K_top" in cfg["testing"]:
        K = cfg["testing"]["K_top"]
    elif dataset_name == "visa":
        K = cfg["testing"]["K_top_visa"]
    else:
        K = cfg["testing"]["K_top_mvtec"]

    
    logger.info(f"Evaluating {ckpt.name} on {dataset_name}")
    
    os.makedirs(Path(cfg["logging"]["folder"]), exist_ok=True)
    csv_path = Path(cfg["logging"]["folder"]) / f"{cfg['logging']['write_tag']}_eval.csv"
    csv_logger = CSVLogger(
        csv_path,
        ("%s", "checkpoint"), ("%s", "class"),
        ("%.8f", "inst_auroc"), ("%.8f", "inst_aupr"),
        ("%.8f", "inst_f1"), ("%.8f", "inst_f1_threshold"),
        ("%.8f", "pix_auroc"), ("%.8f", "pro_auc"),
        ("%.8f", "pix_f1"), ("%.8f", "pix_f1_threshold"),
    )

    inst_auc, inst_aupr, inst_f1 = [], [], []
    pix_auc, pro_auc, pix_f1 = [], [], []

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    for cls in tqdm(classnames, desc="Classes", unit="class", dynamic_ncols=True):
        _, loader, _ = build_dataloader(
            mode="test",
            root=cfg["data"]["test_root"],
            batch_size=1,
            classname=cls,
            resize=crop,
            datasetname=dataset_name,
            normal_dir=cfg["data"].get("normal_dir"),
        )

        tqdm.write(f"Evaluating {cls}...")

        patch_scores, labels = [], []
        pix_buf, img_buf, mask_buf, name_buf = [], [], [], []

        for batch in tqdm(
            loader, desc=f"Images [{cls}]", unit="image", leave=False, dynamic_ncols=True
        ):
            img = batch["image"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            paths = batch["image_path"]; labels.extend(batch["is_anomaly"]); name_buf.extend(batch["image_name"])

            enc = model.target_features(img, paths, n_layer=n_layer)
            pred = model.predict(enc)

            l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)

            topk = torch.topk(l, min(K, l.size(1)), dim=1).values.mean(dim=1)
            patch_scores.extend(topk.cpu())
            h = w = int(math.sqrt(l.size(1)))
            pix = F.interpolate(l.view(-1,1,h,w), size=img.shape[2:], mode="bilinear", align_corners=False)
            pix_buf.append(pix.squeeze(1).cpu()); img_buf.append(img.cpu()); mask_buf.append(mask.cpu())

        p_np = torch.tensor(patch_scores).numpy()
        p_np = (p_np - p_np.min()) / (p_np.max() - p_np.min() + 1e-8) # normed

        pix_all = torch.cat(pix_buf)
        gmin, gmax = pix_all.min(), pix_all.max()
        pix_norm = ((pix_all - gmin) / (gmax - gmin + 1e-8)).numpy()
        mask_np  = torch.cat(mask_buf).squeeze(1).numpy()

        inst = compute_imagewise_retrieval_metrics(p_np, np.array(labels))
        pix  = compute_pixelwise_retrieval_metrics(pix_norm, mask_np)
        pro = calculate_pro(
            mask_np, pix_norm, max_steps=cfg["testing"]["max_steps"],
            expect_fpr=cfg["testing"]["expect_fpr"], progress_desc=f"AUPRO [{cls}]",
        )

        logger.info("%s | AUROC_i %.4f | F1_i %.4f | AUROC_p %.4f | AUPRO %.4f | F1_p %.4f",
                    cls, inst["auroc"], inst["f1_max"], pix["auroc"], pro, pix["f1_max"])
        csv_logger.log(
            ckpt.name, cls, inst["auroc"], inst["aupr"],
            inst["f1_max"], inst["f1_max_threshold"], pix["auroc"], pro,
            pix["f1_max"], pix["f1_max_threshold"],
        )

        inst_auc.append(inst["auroc"]); inst_aupr.append(inst["aupr"])
        inst_f1.append(inst["f1_max"]); pix_auc.append(pix["auroc"])
        pro_auc.append(pro); pix_f1.append(pix["f1_max"])

        # Generate visualizations
        if cfg["testing"].get("segmentation_vis", False):
            std_cpu, mean_cpu = std.cpu(), mean.cpu()
            imgs_un = (torch.cat(img_buf) * std_cpu + mean_cpu).permute(0,2,3,1).numpy()
            out_dir = Path(cfg["logging"]["folder"]) / "segmentation" / cls
            save_segmentation_grid(out_dir, name_buf, imgs_un, mask_np, pix_norm)

    logger.info("Mean | AUROC_i %.4f | F1_i %.4f | AUROC_p %.4f | AUPRO %.4f | F1_p %.4f",
                np.mean(inst_auc), np.mean(inst_f1), np.mean(pix_auc), np.mean(pro_auc), np.mean(pix_f1))
    csv_logger.log(
        ckpt.name, "Mean", np.mean(inst_auc), np.mean(inst_aupr),
        np.mean(inst_f1), float("nan"), np.mean(pix_auc), np.mean(pro_auc),
        np.mean(pix_f1), float("nan"),
    )
    

@torch.inference_mode()
def _demo(ckpt: Path, cfg: Dict[str, Any]) -> None:
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = _build_model(cfg["meta"])
    state = torch.load(ckpt, map_location="cpu")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        model.projector.load_state_dict(state["projector"])
    model.to(device)
    model.eval()

    crop = cfg["meta"]["crop_size"]
    n_layer = cfg["meta"].get("n_layer", 3)
    out_root = Path(cfg["logging"]["folder"]) / "heatmaps"
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg["data"].get("dataset", "mvtec")
    
    test_root = Path(cfg["data"]["test_root"])
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIF", "*.TIFF", "*.WEBP")
    img_paths: List[Path] = []
    for ext in exts:
        img_paths += list(test_root.rglob(ext))
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(f"No images found under: {test_root}")
    print(f"[INFO] Found {len(img_paths)} images under {test_root}")
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def _load_and_preprocess(path: Path):
        pil = Image.open(path).convert("RGB")

        W0, H0 = pil.size

        pil_resized = pil.resize((crop, crop), Image.BILINEAR)

        img = torch.from_numpy(np.array(pil_resized)).float() / 255.0   # [H,W,3], 0~1
        img = img.permute(2, 0, 1).unsqueeze(0).to(device)              # [1,3,H,W]
        img = (img - mean) / std

        return pil, (W0, H0), img
    
    def _to_numpy_image(t_img: torch.Tensor):
        # t_img: [1,3,H,W]
        x = (t_img * std + mean).clamp(0, 1)
        x = x[0].permute(1, 2, 0).detach().cpu().numpy()  # [H,W,3]
        return (x * 255.0).astype(np.uint8)
    
    def _save_overlay_heatmap(rgb_uint8: np.ndarray, heat: np.ndarray, save_path: Path, alpha: float = 0.5):
        """
        rgb_uint8: [H,W,3] 0~255
        heat:      [H,W]   0~1
        """
        import cv2
        H, W = heat.shape

        heat_255 = (heat * 255.0).clip(0, 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_255, cv2.COLORMAP_JET)      # BGR
        rgb_bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)            # RGB->BGR
        overlay = cv2.addWeighted(heat_color, alpha, rgb_bgr, 1 - alpha, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        Image.fromarray(overlay_rgb).save(save_path)

    for i, path in enumerate(img_paths, 1):
        pil_orig, (W0, H0), img = _load_and_preprocess(path)

        enc = model.target_features(img, [str(path)], n_layer=n_layer)  # [1, P, D]
        pred = model.predict(enc)                                       # [1, P, D]

        l = F.mse_loss(enc, pred, reduction="none").mean(dim=2)         # [1, P]

        h = w = int(math.sqrt(l.size(1)))
        pix = F.interpolate(l.view(1, 1, h, w), size=img.shape[2:], mode="bilinear", align_corners=False)  # [1,1,H,W]
        pix = pix.squeeze(0).squeeze(0)  # [H,W]

        pmin, pmax = pix.min(), pix.max()
        pix_norm = (pix - pmin) / (pmax - pmin + 1e-8)                  # [H,W], 0~1

        img_uint8 = _to_numpy_image(img)                                 # [H,W,3] @ crop

        rel = path.relative_to(test_root)
        save_dir = (out_root / rel.parent)
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{path.stem}_heatmap.png"

        _save_overlay_heatmap(img_uint8, pix_norm.detach().cpu().numpy(), save_path)
        print(f"[{i}/{len(img_paths)}] Saved: {save_path}")


def main(args: Dict[str, Any]) -> None:
    ckpt = Path(args["ckpt_path"])
    print(f"loading {ckpt}...")
    _evaluate_single_ckpt(ckpt, args)
    logger.info("Finished. Metrics appended to CSV.")

if __name__ == "__main__":
    main()