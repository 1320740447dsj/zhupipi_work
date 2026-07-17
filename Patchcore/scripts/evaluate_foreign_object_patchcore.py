"""Evaluate a saved PatchCore model on one image or a directory."""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.utils
from patchcore.datasets.mvtec import IMAGENET_MEAN, IMAGENET_STD


LOGGER = logging.getLogger("evaluate_foreign_object_patchcore")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, samples, resize, imagesize):
        self.samples = samples
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform_mask = transforms.Compose(
            [transforms.Resize(resize), transforms.CenterCrop(imagesize), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = PIL.Image.open(sample["image_path"]).convert("RGB")
        image_tensor = self.transform_img(image)
        if sample["mask_path"] is not None:
            mask = PIL.Image.open(sample["mask_path"]).convert("L")
            mask_tensor = self.transform_mask(mask)
            mask_tensor = (mask_tensor > 0).float()
        else:
            mask_tensor = torch.zeros([1, *image_tensor.size()[1:]])
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "is_anomaly": int(sample["is_anomaly"]),
            "image_path": str(sample["image_path"]),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Load PatchCore and save metrics/heatmaps.")
    parser.add_argument("--model_path", default='C:/Users/Administrator/Desktop/udesing/pathccore2/my_model/PatchCore/foreign_object_wr50_l2-3/models/mvtec_bottle', help="Folder containing patchcore_params.pkl.")
    parser.add_argument("--input", default='../good_test/13.png', help="Input image or directory.")
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--data_path", default=None, help="Dataset root for mask inference.")
    parser.add_argument("--mask_path", default="../1_mask.png", help="Mask for a single image.")
    parser.add_argument("--resize", type=int, default=332)
    parser.add_argument("--imagesize", type=int, default=332)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--faiss_on_gpu", action="store_true")
    parser.add_argument("--faiss_num_workers", type=int, default=8)
    parser.add_argument(
        "--assume_anomaly",
        action="store_true",
        help="Treat inputs whose label cannot be inferred as anomalous.",
    )
    return parser.parse_args()


def collect_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    return sorted(
        [
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def infer_label(image_path, assume_anomaly):
    parts = [part.lower() for part in image_path.parts]
    if "good" in parts:
        return 0
    if "test" in parts:
        return 1
    return int(assume_anomaly)


def find_mask(image_path, data_path, explicit_mask):
    if explicit_mask is not None:
        return Path(explicit_mask)
    parts = list(image_path.parts)
    lower_parts = [part.lower() for part in parts]
    if "good" in lower_parts:
        return None
    if "test" in lower_parts:
        test_idx = lower_parts.index("test")
        if test_idx + 1 < len(parts):
            anomaly = parts[test_idx + 1]
            filename = image_path.name
            roots = []
            if data_path:
                roots.append(Path(data_path))
            roots.append(Path(*parts[:test_idx]))
            for root in roots:
                mask_dir = root / "ground_truth" / anomaly
                for candidate in [mask_dir / filename, mask_dir / f"{image_path.stem}_mask.png"]:
                    if candidate.exists():
                        return candidate
                for candidate in mask_dir.glob(f"{image_path.stem}*"):
                    if candidate.suffix.lower() in IMAGE_EXTENSIONS:
                        return candidate
    return None


def load_patchcores(model_path, device, faiss_on_gpu, faiss_num_workers):
    model_path = Path(model_path)
    n_patchcores = len(list(model_path.glob("*nnscorer_search_index.faiss")))
    if n_patchcores == 0:
        raise FileNotFoundError(f"No PatchCore FAISS index found in {model_path}.")
    patchcores = []
    for idx in range(n_patchcores):
        prepend = "" if n_patchcores == 1 else f"Ensemble-{idx + 1}-{n_patchcores}_"
        nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
        patchcore_instance = patchcore.patchcore.PatchCore(device)
        patchcore_instance.load_from_path(
            load_path=str(model_path), device=device, nn_method=nn_method, prepend=prepend
        )
        patchcores.append(patchcore_instance)
    return patchcores


def normalize(values):
    values = np.asarray(values)
    denom = values.max() - values.min()
    if denom == 0:
        return np.zeros_like(values, dtype=np.float32)
    return (values - values.min()) / denom


def normalize_per_model(values):
    values = np.asarray(values)
    normalized = np.zeros_like(values, dtype=np.float32)
    for idx, model_values in enumerate(values):
        denom = model_values.max() - model_values.min()
        if denom == 0:
            normalized[idx] = model_values
        else:
            normalized[idx] = (model_values - model_values.min()) / denom
    return normalized


def save_heatmaps(samples, scores, segmentations, output_path):
    heatmap_dir = output_path / "heatmaps"
    overlay_dir = output_path / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    for sample, score, segmentation in zip(samples, scores, segmentations):
        image_path = Path(sample["image_path"])
        stem = "_".join(image_path.with_suffix("").parts[-4:])
        heatmap_path = heatmap_dir / f"{stem}_score_{float(score):.4f}.png"
        overlay_path = overlay_dir / f"{stem}_score_{float(score):.4f}.png"

        segmentation = normalize(segmentation)
        plt.imsave(heatmap_path, segmentation, cmap="jet")

        image = PIL.Image.open(image_path).convert("RGB").resize(
            (segmentation.shape[1], segmentation.shape[0])
        )
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.imshow(segmentation, cmap="jet", alpha=0.45)
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(overlay_path, bbox_inches="tight", pad_inches=0)
        plt.close()


def safe_auroc(metric_fn, predictions, targets):
    targets_array = np.asarray(targets)
    if len(np.unique(targets_array)) < 2:
        return np.nan
    return metric_fn(predictions, targets)["auroc"]


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()
    device = patchcore.utils.set_torch_device([args.gpu])
    patchcore.utils.fix_seeds(args.seed, device)

    image_paths = collect_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input}.")

    samples = []
    for image_path in image_paths:
        samples.append(
            {
                "image_path": image_path,
                "mask_path": find_mask(image_path, args.data_path, args.mask_path),
                "is_anomaly": infer_label(image_path, args.assume_anomaly),
            }
        )

    dataset = ImageListDataset(samples, resize=args.resize, imagesize=args.imagesize)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    patchcores = load_patchcores(args.model_path, device, args.faiss_on_gpu, args.faiss_num_workers)
    aggregator = {"scores": [], "segmentations": []}
    labels_gt = None
    masks_gt = None
    for patchcore_instance in patchcores:
        scores, segmentations, labels_gt, masks_gt = patchcore_instance.predict(dataloader)
        aggregator["scores"].append(scores)
        aggregator["segmentations"].append(segmentations)

    scores = normalize_per_model(np.asarray(aggregator["scores"]))
    scores = np.mean(scores, axis=0)
    segmentations = np.asarray(aggregator["segmentations"])
    segmentations = np.stack([normalize(model_segmentations) for model_segmentations in segmentations])
    segmentations = np.mean(segmentations, axis=0)

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    save_heatmaps(samples, scores, segmentations, output_path)

    labels_gt = labels_gt or [sample["is_anomaly"] for sample in samples]
    masks_gt = masks_gt or [np.zeros((1, args.imagesize, args.imagesize), dtype=np.float32) for _ in samples]
    instance_auroc = safe_auroc(
        patchcore.metrics.compute_imagewise_retrieval_metrics, scores, labels_gt
    )
    full_pixel_auroc = safe_auroc(
        patchcore.metrics.compute_pixelwise_retrieval_metrics, segmentations, masks_gt
    )
    anomaly_indices = [idx for idx, mask in enumerate(masks_gt) if np.sum(mask) > 0]
    if anomaly_indices:
        anomaly_pixel_auroc = safe_auroc(
            patchcore.metrics.compute_pixelwise_retrieval_metrics,
            [segmentations[idx] for idx in anomaly_indices],
            [masks_gt[idx] for idx in anomaly_indices],
        )
    else:
        anomaly_pixel_auroc = np.nan

    results_csv = output_path / "results.csv"
    with open(results_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image_path", "score", "is_anomaly", "mask_path"])
        for sample, score in zip(samples, scores):
            writer.writerow(
                [
                    sample["image_path"],
                    float(score),
                    sample["is_anomaly"],
                    sample["mask_path"] or "",
                ]
            )
        writer.writerow([])
        writer.writerow(["metric", "value"])
        writer.writerow(["instance_auroc", instance_auroc])
        writer.writerow(["full_pixel_auroc", full_pixel_auroc])
        writer.writerow(["anomaly_pixel_auroc", anomaly_pixel_auroc])

    LOGGER.info("instance_auroc: %s", instance_auroc)
    LOGGER.info("full_pixel_auroc: %s", full_pixel_auroc)
    LOGGER.info("anomaly_pixel_auroc: %s", anomaly_pixel_auroc)
    LOGGER.info("Saved results to %s", results_csv)


if __name__ == "__main__":
    main()
