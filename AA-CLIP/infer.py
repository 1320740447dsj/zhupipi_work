"""Batch folder inference for AA-CLIP.

This script runs a trained AA-CLIP image adapter on every image below an input
folder. Unlike ``test.py``, it does not require labels, masks, or dataset
metadata. Outputs mirror the folder-inference format used by AdaptCLIP:
summary images, raw anomaly maps, and a CSV score file.
"""

import argparse
import csv
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from dataset.constants import CLASS_NAMES, DOMAINS
from forward_utils import (
    calculate_similarity_map,
    get_adapted_single_class_text_embedding,
)
from model.adapter import AdaptedCLIP
from model.clip import create_model


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class FolderImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int):
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "image_path": str(path)}


@dataclass
class InferenceState:
    model: AdaptedCLIP
    text_embedding: torch.Tensor
    device: torch.device
    checkpoint_epoch: int | None
    text_adapter_loaded: bool


def collect_images(folder: str) -> List[Path]:
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input folder not found: {root}")
    return sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available; use --device cpu")
    return torch.device(requested_device)


def load_checkpoint(path: str, key: str):
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict) or key not in checkpoint:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain '{key}'")
    return checkpoint


def validate_class(dataset: str, class_name: str):
    if class_name == "object":
        return
    if dataset not in CLASS_NAMES:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available datasets: {sorted(CLASS_NAMES)}"
        )
    if class_name not in CLASS_NAMES[dataset]:
        raise ValueError(
            f"Class '{class_name}' is not defined for {dataset}. "
            f"Available classes: {CLASS_NAMES[dataset]}"
        )


def load_inference_state(args) -> InferenceState:
    validate_class(args.dataset, args.class_name)
    device = resolve_device(args.device)

    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.image_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = AdaptedCLIP(
        clip_model=clip_model,
        text_adapt_weight=args.text_adapt_weight,
        image_adapt_weight=args.image_adapt_weight,
        text_adapt_until=args.text_adapt_until,
        image_adapt_until=args.image_adapt_until,
        relu=args.relu,
    ).to(device)

    image_checkpoint = load_checkpoint(args.checkpoint_path, "image_adapter")
    model.image_adapter.load_state_dict(image_checkpoint["image_adapter"])

    text_adapter_loaded = bool(args.text_checkpoint_path)
    if text_adapter_loaded:
        text_checkpoint = load_checkpoint(args.text_checkpoint_path, "text_adapter")
        model.text_adapter.load_state_dict(text_checkpoint["text_adapter"])

    model.eval()
    text_model = model if text_adapter_loaded else clip_model
    with torch.no_grad():
        text_embedding = get_adapted_single_class_text_embedding(
            text_model,
            args.dataset,
            args.class_name,
            device,
        )

    return InferenceState(
        model=model,
        text_embedding=text_embedding,
        device=device,
        checkpoint_epoch=image_checkpoint.get("epoch"),
        text_adapter_loaded=text_adapter_loaded,
    )


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())
    if maximum <= minimum:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - minimum) / (maximum - minimum)).astype(np.float32)


def make_panel(image: np.ndarray, title: str, image_size: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    panel = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((image_size + 40, image_size, 3), 255, dtype=np.uint8)
    canvas[40:] = panel
    cv2.putText(
        canvas,
        title,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def create_histogram(
    anomaly_map: np.ndarray,
    image_score: float,
    threshold: float,
    title: str,
) -> np.ndarray:
    values = anomaly_map[np.isfinite(anomaly_map)]
    if values.size == 0:
        return np.full((300, 400, 3), 255, dtype=np.uint8)

    value_min = min(float(values.min()), threshold, image_score, 0.0)
    value_max = max(float(values.max()), threshold, image_score, value_min + 1e-6)
    figure = plt.figure(figsize=(4, 3))
    plt.hist(values.ravel(), bins=256, range=(value_min, value_max), color="C0", alpha=0.8)
    plt.yscale("log")
    plt.axvline(threshold, color="r", linestyle="--", label=f"threshold: {threshold:.4f}")
    plt.axvline(image_score, color="g", linestyle="-.", label=f"image score: {image_score:.4f}")
    plt.xlabel("Anomaly score")
    plt.ylabel("Frequency")
    plt.title(f"{title} histogram")
    plt.legend(loc="upper right")
    plt.tight_layout()

    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(figure)
    histogram = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    buffer.close()
    if histogram is None:
        return np.full((300, 400, 3), 255, dtype=np.uint8)
    return histogram


def save_outputs(
    image_path: str,
    relative_path: Path,
    anomaly_map: np.ndarray,
    image_score: float,
    output_dir: Path,
    image_size: int,
    threshold: float,
):
    source = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    source = cv2.resize(source, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    normalized = normalize_map(anomaly_map)
    heatmap = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(source, 0.5, heatmap, 0.5, 0)
    binary = ((anomaly_map > threshold).astype(np.uint8) * 255)
    histogram = create_histogram(anomaly_map, image_score, threshold, relative_path.name)

    top_row = np.hstack(
        [
            make_panel(source, "Raw", image_size),
            make_panel(overlay, "Overlay", image_size),
        ]
    )
    bottom_row = np.hstack(
        [
            make_panel(binary, f"Segmentation | thr={threshold:.2f}", image_size),
            make_panel(histogram, f"Histogram | score={image_score:.4f}", image_size),
        ]
    )
    summary = np.vstack([top_row, bottom_row])

    summary_path = output_dir / "summary" / relative_path.with_suffix(".png")
    map_path = output_dir / "npy" / relative_path.with_suffix(".npy")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(summary_path), summary):
        raise OSError(f"Failed to write summary image: {summary_path}")
    np.save(str(map_path), anomaly_map)


def run_inference(args):
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {Path(args.input_dir).resolve()}")

    state = load_inference_state(args)
    input_root = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = FolderImageDataset(image_paths, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=state.device.type == "cuda",
    )
    domain = args.domain if args.domain != "auto" else DOMAINS.get(args.dataset, "Industrial")
    rows = []

    with torch.inference_mode():
        for items in tqdm(loader, desc="infer"):
            images = items["image"].to(state.device, non_blocking=True)
            patch_features, detection_features = state.model(images)

            layer_maps = [
                calculate_similarity_map(
                    feature,
                    state.text_embedding,
                    args.image_size,
                    test=True,
                    domain=domain,
                )
                for feature in patch_features
            ]
            anomaly_maps = torch.cat(layer_maps, dim=1).sum(dim=1)
            image_scores = (detection_features @ state.text_embedding)[:, 1]
            image_scores = (image_scores + 1.0) / 2.0

            anomaly_maps = torch.nan_to_num(anomaly_maps).cpu().numpy()
            image_scores = torch.nan_to_num(image_scores).cpu().numpy()

            for path, anomaly_map, image_score in zip(
                items["image_path"], anomaly_maps, image_scores
            ):
                relative_path = Path(path).relative_to(input_root)
                has_mask = bool(np.any(anomaly_map > args.threshold))
                if args.save_mask_filter == 1 and has_mask:
                    continue
                if args.save_mask_filter == 2 and not has_mask:
                    continue

                score = float(image_score)
                save_outputs(
                    image_path=path,
                    relative_path=relative_path,
                    anomaly_map=anomaly_map,
                    image_score=score,
                    output_dir=output_dir,
                    image_size=args.image_size,
                    threshold=args.threshold,
                )
                rows.append(
                    {
                        "image_path": path,
                        "image_score": score,
                        "map_max": float(anomaly_map.max()),
                        "map_mean": float(anomaly_map.mean()),
                    }
                )

    csv_path = output_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "image_score", "map_max", "map_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"images found: {len(image_paths)}")
    print(f"images saved: {len(rows)}")
    print(f"output directory: {output_dir}")
    print(f"score file: {csv_path}")
    print(f"device: {state.device}")
    print(f"checkpoint epoch: {state.checkpoint_epoch}")
    print(f"text adapter loaded: {state.text_adapter_loaded}")


def build_parser():
    parser = argparse.ArgumentParser("AA-CLIP Folder Inference")
    parser.add_argument("--input_dir", default="../val_data/good", help="Input image folder")
    parser.add_argument("--output_dir", default="../val_test/AA-CLIP/good", help="Output folder")
    parser.add_argument(
        "--checkpoint_path",
        default="./image_adapter_15.pth",
        help="AA-CLIP image adapter checkpoint",
    )
    parser.add_argument(
        "--text_checkpoint_path",
        default="./text_adapter.pth",
        help="Optional AA-CLIP text_adapter.pth checkpoint",
    )
    parser.add_argument("--model_name", default="ViT-L-14-336")
    parser.add_argument("--dataset", default="Private", help="Dataset prompt mapping")
    parser.add_argument(
        "--class_name",
        default="object",
        help="Object class used to build normal/anomalous text prompts",
    )
    parser.add_argument(
        "--domain",
        choices=["auto", "Industrial", "Medical"],
        default="auto",
        help="Controls AA-CLIP Gaussian smoothing; auto uses the dataset domain",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--relu", action="store_true")
    parser.add_argument("--text_adapt_weight", type=float, default=0.1)
    parser.add_argument("--image_adapt_weight", type=float, default=0.1)
    parser.add_argument("--text_adapt_until", type=int, default=3)
    parser.add_argument("--image_adapt_until", type=int, default=6)
    parser.add_argument(
        "--save_mask_filter",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="0=save all, 1=only without detected mask, 2=only with detected mask",
    )
    return parser


if __name__ == "__main__":
    run_inference(build_parser().parse_args())
