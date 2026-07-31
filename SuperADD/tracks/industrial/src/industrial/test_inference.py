"""Run SuperADD inference for every image in a folder.

The output layout and four-panel summary image follow the supplied AdaptCLIP
inference script. The original ``test.py`` remains responsible for metrics.
"""

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parents[0]))
sys.path.insert(0, str(_PROJECT_ROOT / "utils"))

from industrial.model import SuperADD
from industrial.paths import (
    get_dataset_path,
    get_model_path,
    get_results_path,
    get_root_config,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def collect_images(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def normalize(anomaly_map: np.ndarray) -> np.ndarray:
    anomaly_map = np.nan_to_num(
        anomaly_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    minimum = float(anomaly_map.min())
    value_range = float(anomaly_map.max()) - minimum
    if value_range < 1e-8:
        return np.zeros_like(anomaly_map)
    return (anomaly_map - minimum) / value_range


def apply_ad_scoremap(image_rgb: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(
        (normalize(anomaly_map) * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.5, heatmap, 0.5, 0)


def make_panel(image: np.ndarray, title: str, image_size: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(
        image, (image_size, image_size), interpolation=cv2.INTER_LINEAR
    )
    canvas = np.full((image_size + 40, image_size, 3), 255, dtype=np.uint8)
    canvas[40:] = panel
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


def make_histogram(
    anomaly_map: np.ndarray,
    image_score: float,
    threshold: float,
    image_name: str,
) -> np.ndarray | None:
    hist_max = max(float(anomaly_map.max()), image_score, threshold, 1e-6)
    fig = plt.figure(figsize=(4, 3))
    plt.hist(
        anomaly_map.ravel(),
        bins=256,
        range=(0, hist_max),
        color="C0",
        alpha=0.8,
    )
    plt.yscale("log")
    plt.axvline(
        threshold,
        color="r",
        linestyle="--",
        label=f"threshold: {threshold:.4f}",
    )
    plt.axvline(
        image_score,
        color="g",
        linestyle="-.",
        label=f"image score: {image_score:.4f}",
    )
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequency")
    plt.title(f"{image_name} histogram")
    plt.legend(loc="upper right")
    plt.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    histogram = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    buffer.close()
    return histogram


def save_outputs(
    image_path: Path,
    relative_path: Path,
    anomaly_map: np.ndarray,
    image_score: float,
    output_dir: Path,
    image_size: int,
    threshold: float,
) -> None:
    with Image.open(image_path) as image:
        source_rgb = np.asarray(
            image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        )

    display_map = cv2.resize(
        anomaly_map,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    overlay_rgb = apply_ad_scoremap(source_rgb, display_map)
    binary = ((display_map > threshold).astype(np.uint8) * 255)
    histogram = make_histogram(
        anomaly_map, image_score, threshold, relative_path.name
    )

    raw_panel = make_panel(
        cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR), "Raw", image_size
    )
    overlay_panel = make_panel(
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), "Overlay", image_size
    )
    binary_panel = make_panel(
        binary, f"Segmentation | thr={threshold:.2f}", image_size
    )
    if histogram is None:
        histogram = np.full((image_size, image_size, 3), 255, dtype=np.uint8)
        cv2.putText(
            histogram,
            "Histogram unavailable",
            (20, image_size // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    histogram_panel = make_panel(
        histogram, f"Histogram | score={image_score:.4f}", image_size
    )

    summary = np.vstack(
        [
            np.hstack([raw_panel, overlay_panel]),
            np.hstack([binary_panel, histogram_panel]),
        ]
    )
    summary_path = output_dir / "summary" / relative_path.with_suffix(".png")
    map_path = output_dir / "npy" / relative_path.with_suffix(".npy")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(summary_path), summary):
        raise OSError(f"Failed to save summary image: {summary_path}")
    np.save(map_path, anomaly_map)


def run_inference(args: argparse.Namespace) -> None:
    config = get_root_config()
    category = args.category or config["categories"][0]
    model_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else get_model_path(category)
    )
    model = SuperADD.from_disk(model_path)
    threshold = float(model.threshold if args.threshold is None else args.threshold)

    input_dir = (
        Path(args.input_dir).expanduser().resolve()
        if args.input_dir
        else get_dataset_path("mvtec_ad_2")
        / category
        / args.split
        / args.condition
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else get_results_path()
        / "folder_inference"
        / category
        / args.split
        / args.condition
    )
    image_paths = collect_images(input_dir)
    if not image_paths:
        raise ValueError(f"No valid images found in input folder: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"model: {model_path}")
    print(f"category: {category}")
    print(f"threshold: {threshold:.4f}")
    print(f"images: {len(image_paths)}")

    csv_rows: list[dict[str, str | int | float]] = []
    saved_count = 0
    to_tensor = transforms.ToTensor()

    for image_path in tqdm(image_paths, desc="infer", file=sys.stdout):
        with Image.open(image_path) as image:
            image_tensor = to_tensor(image.convert("RGB"))

        with torch.inference_mode():
            anomaly_map, _ = model.predict(image_tensor)
        anomaly_map = np.nan_to_num(
            anomaly_map.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        if args.sigma > 0:
            anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)

        image_score = float(anomaly_map.max())
        detected = bool(np.any(anomaly_map > threshold))
        should_save = (
            args.save_mask_filter == 0
            or (args.save_mask_filter == 1 and not detected)
            or (args.save_mask_filter == 2 and detected)
        )
        relative_path = image_path.relative_to(input_dir)
        if should_save:
            save_outputs(
                image_path=image_path,
                relative_path=relative_path,
                anomaly_map=anomaly_map,
                image_score=image_score,
                output_dir=output_dir,
                image_size=args.image_size,
                threshold=threshold,
            )
            saved_count += 1

        csv_rows.append(
            {
                "image_path": str(image_path),
                "saved": int(should_save),
                "detected": int(detected),
                "image_score": image_score,
                "map_max": float(anomaly_map.max()),
                "map_mean": float(anomaly_map.mean()),
            }
        )

    csv_path = output_dir / "scores.csv"
    fieldnames = [
        "image_path",
        "saved",
        "detected",
        "image_score",
        "map_max",
        "map_mean",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"saved: {saved_count}/{len(image_paths)}")
    print(f"output_dir: {output_dir}")
    print(f"score_file: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SuperADD Folder Inference")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="D:/sclead/val_data/good",
        help=(
            "Folder containing images to infer; empty uses the configured "
            "dataset/category/split/condition path"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="D:/sclead/val_test/SuperADD/good",
        help="Folder for inference outputs; empty uses results_dir/folder_inference",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Trained SuperADD model path; empty uses the configured category model",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="yw",
        help="Model category; empty uses the first category in config.json",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test_public",
        help="Dataset split used when input_dir is empty",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="bad",
        help="Image condition used when input_dir is empty, such as bad or good",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=518,
        help="Width and height of each visualization panel",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=4.0,
        help="Gaussian smoothing sigma; use 0 to disable",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary mask threshold",
    )
    parser.add_argument(
        "--save_mask_filter",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help=(
            "Which images to save: 0=all, 1=without detected mask, "
            "2=with detected mask"
        ),
    )
    return parser


if __name__ == "__main__":
    run_inference(build_parser().parse_args())
