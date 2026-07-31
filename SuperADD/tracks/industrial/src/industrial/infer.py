# Anaconda 环境下没有 uv workspace, 需要手动把 industrial 包的两个目录
# (tracks/industrial/src 与 utils) 加入 sys.path, Python 才能把两个目录下的
# industrial/ 子目录合并为同一个 namespace package
"""Batch folder inference for SuperADD industrial track.

加载训练好的 SuperADD 模型, 对文件夹中的所有图像进行异常推理,
输出热力图、二值分割、直方图与 CSV 汇总, 功能与 AdaptCLIP 文件夹推理脚本一致。
"""

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from torchvision import transforms
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent              # .../tracks/industrial/src/industrial
_PROJECT_ROOT = _HERE.parents[3]                      # 项目根目录
sys.path.insert(0, str(_HERE.parents[0]))             # tracks/industrial/src
sys.path.insert(0, str(_PROJECT_ROOT / "utils"))      # utils

from industrial.model import SuperADD
from industrial.paths import get_model_path, get_root_config

IMG_FORMATS = {"jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"}


def normalize(x: np.ndarray) -> np.ndarray:
    """Min-max 归一化到 [0, 1]。"""
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax - xmin < 1e-8:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def apply_ad_scoremap(image: np.ndarray, scoremap: np.ndarray) -> np.ndarray:
    """将异常热力图叠加到图像上, 返回 RGB。"""
    scoremap = normalize(scoremap)
    heatmap = cv2.applyColorMap((scoremap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return overlay


def collect_images(folder: str) -> List[Path]:
    """递归收集文件夹中的所有图像文件。"""
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    image_paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS
    ]
    image_paths.sort()
    return image_paths


def make_panel(image: np.ndarray, title: str, image_size: int) -> np.ndarray:
    """生成带标题的面板。"""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size + 40, image_size, 3), 255, dtype=np.uint8)
    canvas[40:, :, :] = panel
    cv2.putText(
        canvas,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def save_outputs(
    image_path: str,
    rel_path: Path,
    anomaly_map: np.ndarray,
    binary_mask: np.ndarray,
    image_score: float,
    output_dir: Path,
    image_size: int,
    threshold: float,
    mode: int,
) -> dict:
    """保存单张图像的推理输出(汇总图 + npy), 返回统计信息。"""
    src = cv2.imread(image_path)
    if src is None:
        return {
            "saved": False,
            "detected": False,
            "mask_area": 0,
            "map_max": 0.0,
            "map_mean": 0.0,
        }
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (image_size, image_size))

    # 将异常图和二值 mask resize 到与面板一致的尺寸, 避免 addWeighted 尺寸不匹配
    anomaly_map = cv2.resize(anomaly_map, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    binary_mask = cv2.resize(binary_mask.astype(np.uint8), (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    amap_norm = normalize(anomaly_map)
    amap_norm = np.nan_to_num(amap_norm, nan=0.0, posinf=0.0, neginf=0.0)
    overlay = apply_ad_scoremap(src, amap_norm)
    binary = (binary_mask.astype(np.uint8) * 255)
    detected = bool(binary.any())
    mask_area = int((binary > 0).sum())
    map_max = float(np.max(anomaly_map))
    map_mean = float(np.mean(anomaly_map))

    if mode == 1 and not detected:
        return {
            "saved": False,
            "detected": detected,
            "mask_area": mask_area,
            "map_max": map_max,
            "map_mean": map_mean,
        }
    if mode == 2 and detected:
        return {
            "saved": False,
            "detected": detected,
            "mask_area": mask_area,
            "map_max": map_max,
            "map_mean": map_mean,
        }

    hist_max = max(map_max, float(image_score), threshold, 1e-6)
    fig = plt.figure(figsize=(4, 3))
    plt.hist(anomaly_map.ravel(), bins=256, range=(0, hist_max), color="C0", alpha=0.8)
    plt.yscale("log")
    plt.axvline(x=threshold, color="r", linestyle="--", label=f"threshold: {threshold:.4f}")
    plt.axvline(x=image_score, color="g", linestyle="-.", label=f"image score: {image_score:.4f}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequency")
    plt.title(rel_path.name + " histogram")
    plt.legend(loc="upper right")
    plt.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    hist_img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    buf.close()
    plt.close(fig)

    raw_panel = make_panel(cv2.cvtColor(src, cv2.COLOR_RGB2BGR), "Raw", image_size)
    overlay_panel = make_panel(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), "Overlay", image_size)
    binary_panel = make_panel(
        cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        f"Segmentation | thr={threshold:.2f}",
        image_size,
    )
    if hist_img is None:
        hist_img = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        cv2.putText(
            hist_img,
            "Histogram unavailable",
            (20, image_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    hist_panel = make_panel(hist_img, f"Histogram | score={image_score:.4f}", image_size)

    top_row = np.hstack([raw_panel, overlay_panel])
    bottom_row = np.hstack([binary_panel, hist_panel])
    summary_img = np.vstack([top_row, bottom_row])

    out_summary = output_dir / "summary" / rel_path.with_suffix(".png")
    out_npy = output_dir / "npy" / rel_path.with_suffix(".npy")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_summary), summary_img)
    np.save(str(out_npy), anomaly_map)

    return {
        "saved": True,
        "detected": detected,
        "mask_area": mask_area,
        "map_max": map_max,
        "map_mean": map_mean,
    }


def run_inference(args):
    config = get_root_config()

    # 选择类别: 命令行优先, 否则用 config 中的第一个
    category = args.category if args.category else config['categories'][0]

    # 加载模型
    model_path = args.checkpoint_path if args.checkpoint_path else str(get_model_path(category))
    print(f"Loading model from {model_path}")
    model = SuperADD.from_disk(Path(model_path))

    # 覆盖阈值
    if args.threshold is not None:
        model.threshold = float(args.threshold)
    threshold = float(model.threshold)
    print(f"Using threshold: {threshold:.4f}")

    # 收集图像
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise ValueError(f"No valid images found in input folder: {args.input_dir}")
    print(f"Found {len(image_paths)} images")

    root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    saved_count = 0
    detected_count = 0

    to_tensor = transforms.ToTensor()

    for image_path in tqdm(image_paths, desc="infer", file=sys.stdout):
        img = Image.open(image_path).convert("RGB")
        img_tensor = to_tensor(img)

        with torch.no_grad():
            anomaly_map, binary_result = model.predict(img_tensor)

        # 可选高斯平滑
        if args.sigma > 0:
            anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)

        # 图像级分数 = 异常图最大值
        image_score = float(np.max(anomaly_map))

        rel_path = image_path.relative_to(root)
        save_result = save_outputs(
            image_path=str(image_path),
            rel_path=rel_path,
            anomaly_map=anomaly_map,
            binary_mask=binary_result,
            image_score=image_score,
            output_dir=output_dir,
            image_size=args.image_size,
            threshold=threshold,
            mode=args.mode,
        )
        saved_count += int(save_result["saved"])
        detected_count += int(save_result["detected"])
        csv_rows.append(
            {
                "image_path": str(image_path),
                "saved": int(save_result["saved"]),
                "detected": int(save_result["detected"]),
                "image_score": image_score,
                "map_max": save_result["map_max"],
                "map_mean": save_result["map_mean"],
                "mask_area": save_result["mask_area"],
            }
        )

    csv_path = output_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "saved", "detected", "image_score", "map_max", "map_mean", "mask_area"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    headers = ["item", "value"]
    rows = [
        ["images", len(image_paths)],
        ["saved", saved_count],
        ["detected", detected_count],
        ["mode", args.mode],
        ["output_dir", str(output_dir)],
        ["score_file", str(csv_path)],
        ["threshold", f"{threshold:.4f}"],
        ["category", category],
        ["sigma", args.sigma],
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"))


def build_parser():
    parser = argparse.ArgumentParser("SuperADD Folder Inference", add_help=True)
    parser.add_argument("--input_dir", type=str, default="./data/mvtec_ad_2/yw/test_public/bad", help="Folder containing images to infer")
    parser.add_argument("--output_dir", type=str, default="./outputs/infer/bad", help="Folder for inference outputs")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to trained model (empty: use config category model)")
    parser.add_argument("--category", type=str, default="", help="Category name (empty: use first in config)")
    parser.add_argument("--image_size", type=int, default=336, help="Panel image size for output visualization")
    parser.add_argument("--sigma", type=float, default=0.0, help="Gaussian smoothing sigma (0: disable)")
    parser.add_argument("--threshold", type=float, default=None, help="Override auto-detected threshold")
    parser.add_argument("--mode", type=int, choices=[0, 1, 2], default=2, help="0: save all, 1: save detected only, 2: save not-detected only")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_inference(args)
