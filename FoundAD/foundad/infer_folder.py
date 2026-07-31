"""Run FoundAD inference on every image in a folder and save visual results.

The output layout mirrors the supplied AdaptCLIP inference script:

    output_dir/
      summary/<relative image path>.png
      npy/<relative image path>.npy
      scores.csv

Each summary image contains the resized input, anomaly overlay, binary mask,
and anomaly-score histogram in a 2x2 grid.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from src.foundad import VisionModule


IMAGE_EXTENSIONS = {
    ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def collect_images(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class FolderImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], image_size: int) -> None:
        self.image_paths = list(image_paths)
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.image_paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return {"image": tensor, "image_path": str(path)}


@dataclass
class InferenceResult:
    path: Path
    anomaly_map: np.ndarray
    image_score: float


def load_params(params_path: Path) -> Dict[str, Any]:
    if not params_path.is_file():
        raise FileNotFoundError(f"Params file not found: {params_path}")
    with params_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if not isinstance(params, dict) or "meta" not in params:
        raise ValueError(f"Invalid FoundAD params file (missing 'meta'): {params_path}")
    return params


def build_model(meta: Dict[str, Any], checkpoint_path: Path, device: torch.device) -> VisionModule:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = VisionModule(
        model_name=meta["model"],
        pred_depth=meta["pred_depth"],
        pred_emb_dim=meta["pred_emb_dim"],
        use_cuda=False,
        if_pe=meta.get("if_pred_pe", True),
        feat_normed=meta.get("feat_normed", False),
        encoder_path=meta.get("encoder_path"),
        crop_size=meta["crop_size"],
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if "predictor" not in state:
        raise KeyError(f"Checkpoint has no 'predictor' state: {checkpoint_path}")
    model.predictor.load_state_dict(state["predictor"])
    if model.projector is not None:
        if "projector" not in state:
            raise KeyError(f"Checkpoint has no 'projector' state: {checkpoint_path}")
        model.projector.load_state_dict(state["projector"])
    model.to(device).eval()
    return model


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    minimum = float(anomaly_map.min())
    maximum = float(anomaly_map.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - minimum) / (maximum - minimum)).astype(np.float32)


def apply_scoremap(rgb: np.ndarray, anomaly_map: np.ndarray, alpha: float) -> np.ndarray:
    heatmap = cv2.applyColorMap(
        np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    source_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay_bgr = cv2.addWeighted(heatmap, alpha, source_bgr, 1.0 - alpha, 0)
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


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
        0.7,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def render_histogram(
    anomaly_map: np.ndarray,
    image_score: float,
    threshold: float,
    image_name: str,
) -> np.ndarray:
    fig, axis = plt.subplots(figsize=(4, 3))
    axis.hist(anomaly_map.ravel(), bins=256, range=(0.0, 1.0), color="C0", alpha=0.8)
    axis.set_yscale("log")
    axis.axvline(threshold, color="r", linestyle="--", label=f"threshold: {threshold:.4f}")
    axis.set_xlabel("Normalized anomaly score")
    axis.set_ylabel("Frequency")
    axis.set_title(f"{image_name} | raw score={image_score:.4g}")
    axis.legend(loc="upper right")
    fig.tight_layout()

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    histogram = cv2.imdecode(
        np.frombuffer(buffer.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR
    )
    buffer.close()
    if histogram is None:
        raise RuntimeError(f"Failed to render histogram for {image_name}")
    return histogram


def save_outputs(
    result: InferenceResult,
    relative_path: Path,
    output_dir: Path,
    image_size: int,
    threshold: float,
    overlay_alpha: float,
) -> None:
    with Image.open(result.path) as image:
        source_rgb = np.asarray(
            image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        )

    anomaly_map = result.anomaly_map
    overlay_rgb = apply_scoremap(source_rgb, anomaly_map, overlay_alpha)
    binary = (anomaly_map > threshold).astype(np.uint8) * 255
    histogram = render_histogram(anomaly_map, result.image_score, threshold, relative_path.name)

    raw_panel = make_panel(cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR), "Raw", image_size)
    overlay_panel = make_panel(
        cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR), "Overlay", image_size
    )
    binary_panel = make_panel(binary, f"Segmentation | thr={threshold:.2f}", image_size)
    histogram_panel = make_panel(
        histogram, f"Histogram | raw score={result.image_score:.4g}", image_size
    )
    summary = np.vstack(
        [np.hstack([raw_panel, overlay_panel]), np.hstack([binary_panel, histogram_panel])]
    )

    summary_path = output_dir / "summary" / relative_path.with_suffix(".png")
    map_path = output_dir / "npy" / relative_path.with_suffix(".npy")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(summary_path), summary):
        raise OSError(f"Failed to save summary image: {summary_path}")
    np.save(map_path, anomaly_map)


@torch.inference_mode()
def infer_batches(
    model: VisionModule,
    loader: DataLoader,
    device: torch.device,
    image_size: int,
    n_layer: int,
    top_k: int,
    sigma: float,
) -> List[InferenceResult]:
    results: List[InferenceResult] = []
    for batch in tqdm(loader, desc="infer", unit="batch"):
        images = batch["image"].to(device, non_blocking=True)
        paths = list(batch["image_path"])
        encoded = model.target_features(images, paths, n_layer=n_layer)
        predicted = model.predict(encoded)
        patch_loss = F.mse_loss(encoded, predicted, reduction="none").mean(dim=2)

        k = min(top_k, patch_loss.shape[1])
        image_scores = torch.topk(patch_loss, k=k, dim=1).values.mean(dim=1)
        side = math.isqrt(patch_loss.shape[1])
        if side * side != patch_loss.shape[1]:
            raise ValueError(
                f"Patch count {patch_loss.shape[1]} is not square; cannot create an anomaly map"
            )
        maps = F.interpolate(
            patch_loss.view(-1, 1, side, side),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        for path, anomaly_map, image_score in zip(paths, maps.cpu().numpy(), image_scores):
            if sigma > 0:
                anomaly_map = gaussian_filter(anomaly_map, sigma=sigma)
            anomaly_map = np.nan_to_num(anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
            results.append(
                InferenceResult(
                    path=Path(path),
                    anomaly_map=normalize_map(anomaly_map),
                    image_score=float(image_score.cpu()),
                )
            )
    return results


def resolve_device(requested: str) -> torch.device:
    if requested:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    params_path = Path(args.params_path).resolve()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    image_paths = collect_images(input_dir)
    if not image_paths:
        raise ValueError(f"No supported images found in: {input_dir}")

    params = load_params(params_path)
    meta = params["meta"]
    image_size = int(meta["crop_size"])
    n_layer = args.n_layer if args.n_layer is not None else int(meta.get("n_layer", 3))
    device = resolve_device(args.device)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = build_model(meta, checkpoint_path, device)
    dataset = FolderImageDataset(image_paths, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    results = infer_batches(
        model=model,
        loader=loader,
        device=device,
        image_size=image_size,
        n_layer=n_layer,
        top_k=args.top_k,
        sigma=args.sigma,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = 0
    for result in tqdm(results, desc="save", unit="image"):
        has_mask = bool(np.any(result.anomaly_map > args.threshold))
        if args.save_mask_filter == 1 and has_mask:
            continue
        if args.save_mask_filter == 2 and not has_mask:
            continue

        relative_path = result.path.relative_to(input_dir)
        save_outputs(
            result=result,
            relative_path=relative_path,
            output_dir=output_dir,
            image_size=image_size,
            threshold=args.threshold,
            overlay_alpha=args.overlay_alpha,
        )
        rows.append(
            {
                "image_path": str(result.path),
                "image_score": result.image_score,
                "map_max": float(result.anomaly_map.max()),
                "map_mean": float(result.anomaly_map.mean()),
            }
        )
        saved += 1

    csv_path = output_dir / "scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["image_path", "image_score", "map_max", "map_mean"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Images found : {len(image_paths)}")
    print(f"Images saved : {saved}")
    print(f"Output folder: {output_dir}")
    print(f"Score file   : {csv_path}")
    print(f"Device       : {device}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FoundAD folder inference")
    parser.add_argument("--input_dir", default="D:\\sclead\\val_data\\good", help="Folder containing input images")
    parser.add_argument("--output_dir", default="D:\\sclead\\val_test\\FoundAD\\good", help="Folder for inference outputs")
    parser.add_argument("--params_path", default="D:\\sclead\\FoundAD\\logs\\private_4shot_s42\\dinov3_s42\\params.yaml", help="Training params.yaml path")
    parser.add_argument("--checkpoint_path", default="D:\\sclead\\FoundAD\\logs\\private_4shot_s42\\dinov3_s42\\train-step25000.pth.tar", help="FoundAD checkpoint path")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker count")
    parser.add_argument("--device", default="", help="Device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--n_layer", type=int, default=None, help="Override params meta.n_layer")
    parser.add_argument("--top_k", type=int, default=10, help="Top patch count for image score")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian smoothing sigma")
    parser.add_argument("--threshold", type=float, default=0.5, help="Mask threshold in [0, 1]")
    parser.add_argument("--overlay_alpha", type=float, default=0.5, help="Heatmap overlay opacity")
    parser.add_argument(
        "--save_mask_filter",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="0=save all, 1=without detected mask, 2=with detected mask",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num_workers cannot be negative")
    if args.top_k < 1:
        parser.error("--top_k must be at least 1")
    if args.sigma < 0:
        parser.error("--sigma cannot be negative")
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be in [0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        parser.error("--overlay_alpha must be in [0, 1]")


if __name__ == "__main__":
    argument_parser = build_parser()
    parsed_args = argument_parser.parse_args()
    validate_args(argument_parser, parsed_args)
    run_inference(parsed_args)
