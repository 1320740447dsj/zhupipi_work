"""Batch folder inference for AdaptCLIP.

This script loads a trained AdaptCLIP checkpoint and runs inference for all images
in a folder. It supports zero-shot mode and optional one-shot prompts via
--support_dir for PQ adapter.
"""

import argparse
import csv
import os
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from tabulate import tabulate
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import adaptcliplib
from adaptcliplib import PQAdapter, TextualAdapter, VisualAdapter, fusion_fun
from adaptcliplib.adaptclip import tokenize
from tools import get_transform
from tools.utils import IMG_FORMATS, normalize
from tools.visualization import apply_ad_scoremap


def resolve_model_cfg(pretrained_model: str) -> Tuple[int, int, int]:
    if pretrained_model == "ViT-L/14@336px":
        return 20, 14, 768
    if pretrained_model == "VITB16_PLUS_240":
        return 10, 16, 640
    if pretrained_model == "ViT-B-32-512":
        return 10, 32, 512
    if pretrained_model in ["ViT-L-14-CLIPA-336", "ViT-L-14-448"]:
        return 20, 14, 768
    if pretrained_model == "Vit-H-14-CLIP-224":
        return 30, 14, 1024
    raise ValueError(f"Unsupported pretrained_model: {pretrained_model}")


class FolderImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], preprocess):
        self.image_paths = list(image_paths)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        img = Image.open(path).convert("RGB")
        tensor = self.preprocess(img)
        return {
            "img": tensor,
            "img_path": str(path),
        }


@dataclass
class InferState:
    model: torch.nn.Module
    textual_learner: TextualAdapter
    visual_learner: VisualAdapter
    pq_learner: PQAdapter
    static_text_features: torch.Tensor
    learned_text_features: torch.Tensor
    device: str
    DPAM_layer: int


def collect_images(folder: str) -> List[Path]:
    root = Path(folder)
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    image_paths = [
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in IMG_FORMATS
    ]
    image_paths.sort()
    return image_paths


def prepare_static_text_feature_device(textual_learner: TextualAdapter, model, device: str):
    normal_description, abnormal_description = textual_learner.prompt()
    normal_tokens = tokenize(normal_description).to(device)
    abnormal_tokens = tokenize(abnormal_description).to(device)
    with torch.no_grad():
        normal_text_features = model.encode_text(normal_tokens).float()
        abnormal_text_features = model.encode_text(abnormal_tokens).float()

    avg_normal_text_features = torch.mean(normal_text_features, dim=0, keepdim=True)
    avg_abnormal_text_features = torch.mean(abnormal_text_features, dim=0, keepdim=True)
    return torch.cat((avg_normal_text_features, avg_abnormal_text_features), dim=0)


def load_infer_state(args) -> InferState:
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    DPAM_layer, patch_size, input_dim = resolve_model_cfg(args.pretrained_model)

    model, _ = adaptcliplib.load(args.pretrained_model, device=device)
    model.visual.DAPM_replace(DPAM_layer=DPAM_layer)

    textual_learner = TextualAdapter(model.to("cpu"), args.image_size, args.n_ctx)
    visual_learner = VisualAdapter(
        args.image_size,
        patch_size,
        input_dim=input_dim,
        reduction=args.vl_reduction,
    )
    pq_learner = PQAdapter(
        args.image_size,
        patch_size,
        context=args.pq_context,
        input_dim=input_dim,
        mid_dim=args.pq_mid_dim,
        layers_num=len(args.features_list),
    )

    checkpoint_adapter = torch.load(args.checkpoint_path, map_location="cpu")
    textual_learner.load_state_dict(checkpoint_adapter["textual_learner"])
    visual_learner.load_state_dict(checkpoint_adapter["visual_learner"])
    pq_learner.load_state_dict(checkpoint_adapter["pq_learner"])

    model.to(device)
    textual_learner.to(device)
    visual_learner.to(device)
    pq_learner.to(device)

    model.eval()
    textual_learner.eval()
    visual_learner.eval()
    pq_learner.eval()

    static_text_features = prepare_static_text_feature_device(textual_learner, model, device)

    learned_prompts, tokenized_prompts = textual_learner()
    learned_text_features = model.encode_text_learn(learned_prompts, tokenized_prompts).float()

    return InferState(
        model=model,
        textual_learner=textual_learner,
        visual_learner=visual_learner,
        pq_learner=pq_learner,
        static_text_features=static_text_features,
        learned_text_features=learned_text_features,
        device=device,
        DPAM_layer=DPAM_layer,
    )


def encode_support_features(
    support_paths: Sequence[Path],
    preprocess,
    state: InferState,
    features_list: Sequence[int],
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    tensors = []
    for p in support_paths:
        img = Image.open(p).convert("RGB")
        tensors.append(preprocess(img))

    if not tensors:
        raise ValueError("No support images found in --support_dir")

    support_batch = torch.stack(tensors, dim=0).to(state.device)
    with torch.no_grad():
        prompt_feats, prompt_patch_feats = state.model.encode_image(
            support_batch,
            features_list,
            DPAM_layer=state.DPAM_layer,
        )
    return prompt_feats, prompt_patch_feats


def save_outputs(
    image_path: str,
    rel_path: Path,
    anomaly_map: np.ndarray,
    image_score: float,
    output_dir: Path,
    image_size: int,
    threshold: float,
):
    src = cv2.imread(image_path)
    if src is None:
        return
    src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    src = cv2.resize(src, (image_size, image_size))

    amap_norm = normalize(anomaly_map)
    overlay = apply_ad_scoremap(src, amap_norm)
    binary = ((anomaly_map > threshold).astype(np.uint8) * 255)

    hist_max = max(float(np.max(anomaly_map)), float(image_score), threshold, 1e-6)
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

    def make_panel(image: np.ndarray, title: str) -> np.ndarray:
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

    raw_panel = make_panel(cv2.cvtColor(src, cv2.COLOR_RGB2BGR), "Raw")
    overlay_panel = make_panel(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), "Overlay")
    binary_panel = make_panel(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), f"Segmentation | thr={threshold:.2f}")
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
    hist_panel = make_panel(hist_img, f"Histogram | score={image_score:.4f}")

    top_row = np.hstack([raw_panel, overlay_panel])
    bottom_row = np.hstack([binary_panel, hist_panel])
    summary_img = np.vstack([top_row, bottom_row])

    out_summary = output_dir / "summary" / rel_path.with_suffix(".png")
    out_npy = output_dir / "npy" / rel_path.with_suffix(".npy")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_summary), summary_img)
    np.save(str(out_npy), anomaly_map)


def run_inference(args):
    preprocess, _ = get_transform(image_size=args.image_size)
    image_paths = collect_images(args.input_dir)
    if not image_paths:
        raise ValueError(f"No valid images found in input folder: {args.input_dir}")

    state = load_infer_state(args)
    root = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    support_feats = None
    support_patch_feats = None
    pq_enabled = args.pq_learner
    if pq_enabled:
        if not args.support_dir:
            print("warning: --pq_learner is enabled but --support_dir is missing, pq branch will be disabled")
            pq_enabled = False
        else:
            support_paths = collect_images(args.support_dir)
            support_feats, support_patch_feats = encode_support_features(
                support_paths=support_paths,
                preprocess=preprocess,
                state=state,
                features_list=args.features_list,
            )

    dataset = FolderImageDataset(image_paths=image_paths, preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    csv_rows = []

    for items in tqdm(dataloader, desc="infer"):
        query_image = items["img"].to(state.device)
        query_paths = items["img_path"]
        current_batchsize = query_image.shape[0]

        with torch.no_grad():
            query_feats, query_patch_feats = state.model.encode_image(
                query_image,
                args.features_list,
                DPAM_layer=state.DPAM_layer,
            )

        local_maps = []
        global_scores = []

        if args.visual_learner:
            global_vl_logit, local_vl_map = state.visual_learner(
                query_feats,
                query_patch_feats,
                state.static_text_features,
            )
            local_vl_map = local_vl_map[:, 1].detach()
            global_vl_score = global_vl_logit.softmax(-1)[:, 1].detach()
            local_maps.append(local_vl_map)
            global_scores.append(global_vl_score)

        if args.textual_learner:
            global_tl_logit, local_tl_map = state.textual_learner.compute_global_local_score(
                query_feats,
                query_patch_feats,
                state.learned_text_features,
            )
            local_tl_map = local_tl_map[:, 1].detach()
            global_tl_score = global_tl_logit.softmax(-1)[:, 1].detach()
            local_maps.append(local_tl_map)
            global_scores.append(global_tl_score)

        if pq_enabled:
            prompt_feats = support_feats.unsqueeze(0).repeat(current_batchsize, 1, 1)
            prompt_patch_feats = [
                x.unsqueeze(0).repeat(current_batchsize, 1, 1, 1)
                for x in support_patch_feats
            ]

            global_pq_logit, local_pq_map_list, align_score_list = state.pq_learner(
                query_feats,
                query_patch_feats,
                prompt_feats,
                prompt_patch_feats,
            )
            local_pq_map_list = [x[:, 1].unsqueeze(1) for x in local_pq_map_list]
            local_pq_map = torch.concat(local_pq_map_list, dim=1).mean(dim=1).detach()
            align_score = fusion_fun(align_score_list, fusion_type="harmonic_mean")[:, 0]

            if isinstance(global_pq_logit, list):
                global_pq_score = [x.softmax(-1).unsqueeze(-1) for x in global_pq_logit]
                global_pq_score = torch.concat(global_pq_score, dim=-1).mean(dim=-1).detach()[:, 1]
            else:
                global_pq_score = global_pq_logit.softmax(-1)[:, 1].detach()

            local_maps.append(local_pq_map)
            global_scores.append(global_pq_score)

        if not local_maps:
            raise RuntimeError("At least one local branch must be enabled")

        pixel_anomaly_map = fusion_fun(local_maps, fusion_type=args.fusion_type)
        if pq_enabled:
            pixel_anomaly_map = fusion_fun([pixel_anomaly_map, align_score], fusion_type="harmonic_mean")

        pixel_anomaly_map = torch.stack(
            [
                torch.from_numpy(gaussian_filter(i, sigma=args.sigma))
                for i in pixel_anomaly_map.cpu()
            ],
            dim=0,
        ).to(state.device)

        anomaly_map_max, _ = torch.max(pixel_anomaly_map.view(current_batchsize, -1), dim=1)
        if global_scores:
            image_anomaly_pred = fusion_fun(global_scores + [anomaly_map_max], fusion_type=args.fusion_type)
        else:
            image_anomaly_pred = anomaly_map_max

        pixel_anomaly_map = torch.nan_to_num(pixel_anomaly_map, nan=0.0, posinf=0.0, neginf=0.0)
        image_anomaly_pred = torch.nan_to_num(image_anomaly_pred, nan=0.0, posinf=0.0, neginf=0.0)

        anomaly_maps_np = pixel_anomaly_map.detach().cpu().numpy()
        image_scores_np = image_anomaly_pred.detach().cpu().numpy()

        for path, amap, score in zip(query_paths, anomaly_maps_np, image_scores_np):
            rel_path = Path(path).relative_to(root)
            has_mask = bool(np.any(amap > args.threshold))
            if args.save_mask_filter == 1 and has_mask:
                continue
            if args.save_mask_filter == 2 and not has_mask:
                continue
            save_outputs(
                image_path=path,
                rel_path=rel_path,
                anomaly_map=amap,
                image_score=float(score),
                output_dir=output_dir,
                image_size=args.image_size,
                threshold=args.threshold,
            )
            csv_rows.append(
                {
                    "image_path": path,
                    "image_score": float(score),
                    "map_max": float(np.max(amap)),
                    "map_mean": float(np.mean(amap)),
                }
            )

    csv_path = output_dir / "scores.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "image_score", "map_max", "map_mean"])
        writer.writeheader()
        writer.writerows(csv_rows)

    headers = ["item", "value"]
    rows = [
        ["images", len(image_paths)],
        ["output_dir", str(output_dir)],
        ["score_file", str(csv_path)],
        ["device", state.device],
        ["pq_enabled", pq_enabled],
    ]
    print(tabulate(rows, headers=headers, tablefmt="pipe"))


def build_parser():
    parser = argparse.ArgumentParser("AdaptCLIP Folder Inference", add_help=True)
    parser.add_argument("--input_dir", type=str, default="../val_data/good", help="Folder containing images to infer")
    parser.add_argument("--output_dir", type=str,default="../val_test/AdapClip2/good", help="Folder for inference outputs")
    parser.add_argument("--support_dir", type=str, default="", help="Optional support image folder for one-shot PQ")
    parser.add_argument("--checkpoint_path", type=str, default="./epoch_20.pth", help="Path to trained adapters checkpoint")
    parser.add_argument("--pretrained_model", type=str, default="ViT-L/14@336px", help="Pre-trained model name")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="Feature layers")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--image_size", type=int, default=518, help="Input image size")
    parser.add_argument("--n_ctx", type=int, default=12, help="Prompt context length")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian smoothing sigma") # 2 126   1  160    5 74
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary mask threshold")
    parser.add_argument("--fusion_type", type=str, default="average_mean", help="Fusion type")
    parser.add_argument("--vl_reduction", type=int, default=2, help="Visual adapter reduction")
    parser.add_argument("--pq_mid_dim", type=int, default=128, help="PQ adapter mid dim")
    parser.add_argument("--device", type=str, default="cuda", choices=["", "cuda", "cpu"], help="Inference device")
    parser.add_argument("--visual_learner", action="store_true", help="Enable visual adapter branch")
    parser.add_argument("--textual_learner", action="store_true", help="Enable textual adapter branch")
    parser.add_argument("--pq_learner", action="store_true", help="Enable prompt-query adapter branch")
    parser.add_argument("--pq_context", action="store_true", help="Enable context feature in PQ")
    parser.add_argument(
        "--save_mask_filter",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Filter which images to save by mask detection: 0=save all, 1=save only images without mask, 2=save only images with mask",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if not args.visual_learner and not args.textual_learner and not args.pq_learner:
        args.visual_learner = True
        args.textual_learner = True
    run_inference(args)