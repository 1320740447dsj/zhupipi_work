import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter

import AnomalyCLIP_lib
from prompt_ensemble import AnomalyCLIP_PromptLearner
from utils import get_transform


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GOOD_DIR_NAMES = {"good", "normal", "ok"}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def min_max_normalize(scoremap, eps=1e-8):
    scoremap = scoremap.astype(np.float32)
    return (scoremap - scoremap.min()) / (scoremap.max() - scoremap.min() + eps)


def overlay_heatmap(image_rgb, anomaly_map, alpha=0.45, colormap=cv2.COLORMAP_JET):
    heatmap_uint8 = (min_max_normalize(anomaly_map) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, colored, alpha, 0)


def make_side_by_side(image_rgb, heatmap_rgb):
    height, width = image_rgb.shape[:2]
    title_height = max(36, height // 14)
    gap = max(8, width // 64)
    canvas = np.full((height + title_height, width * 2 + gap, 3), 255, dtype=np.uint8)
    canvas[title_height:, :width] = image_rgb
    canvas[title_height:, width + gap:] = heatmap_rgb

    font_scale = max(0.6, width / 700.0)
    thickness = max(1, width // 320)
    cv2.putText(
        canvas,
        "Original",
        (12, title_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (25, 25, 25),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Heatmap",
        (width + gap + 12, title_height - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (25, 25, 25),
        thickness,
        cv2.LINE_AA,
    )
    return canvas


def safe_name(name):
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def iter_mvtec_defect_images(mvtec_root):
    mvtec_root = Path(mvtec_root)
    for class_dir in sorted(p for p in mvtec_root.iterdir() if p.is_dir()):
        test_dir = class_dir / "test"
        if not test_dir.is_dir():
            continue

        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            if defect_dir.name.lower() in GOOD_DIR_NAMES:
                continue

            for image_path in sorted(defect_dir.rglob("*")):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    yield class_dir.name, defect_dir.name, image_path


def load_prompt_learner(checkpoint_path, prompt_learner, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["prompt_learner"] if isinstance(checkpoint, dict) and "prompt_learner" in checkpoint else checkpoint
    prompt_learner.load_state_dict(state_dict)
    return prompt_learner.to(device).eval()


@torch.no_grad()
def build_text_features(model, prompt_learner, device):
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    prompts = prompts.to(device)
    tokenized_prompts = tokenized_prompts.to(device)
    compound_prompts_text = [prompt.to(device) for prompt in compound_prompts_text]

    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    return text_features / text_features.norm(dim=-1, keepdim=True)


@torch.no_grad()
def predict_anomaly_map(model, image_tensor, text_features, args, device):
    image = image_tensor.unsqueeze(0).to(device)
    image_features, patch_features = model.encode_image(
        image,
        args.features_list,
        DPAM_layer=args.dpam_layer,
    )
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    anomaly_maps = []
    for idx, patch_feature in enumerate(patch_features):
        if idx < args.feature_map_layer:
            continue

        patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
        similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
        similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
        anomaly_map = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
        anomaly_maps.append(anomaly_map)

    if not anomaly_maps:
        raise ValueError("No anomaly map was produced. Check --features_list and --feature_map_layer.")

    anomaly_map = torch.stack(anomaly_maps).sum(dim=0)
    anomaly_map = anomaly_map.squeeze(0).detach().cpu().numpy()
    return gaussian_filter(anomaly_map, sigma=args.sigma)


def save_visualization(image_path, anomaly_map, class_name, defect_name, args):
    image_rgb = np.array(Image.open(image_path).convert("RGB"))
    image_rgb = cv2.resize(image_rgb, (args.image_size, args.image_size), interpolation=cv2.INTER_CUBIC)
    heatmap_rgb = overlay_heatmap(image_rgb, anomaly_map, alpha=args.alpha)
    comparison_rgb = make_side_by_side(image_rgb, heatmap_rgb)

    output_dir = Path(args.output_dir) / safe_name(class_name) / safe_name(defect_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_heatmap.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(comparison_rgb, cv2.COLOR_RGB2BGR))
    return output_path


def load_model_and_preprocess(args, device):
    anomalyclip_parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model_name_or_path = args.clip_model_path or args.clip_model
    model, _ = AnomalyCLIP_lib.load(
        model_name_or_path,
        device=device,
        design_details=anomalyclip_parameters,
        download_root=args.download_root,
    )
    model.eval()
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)

    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), anomalyclip_parameters)
    prompt_learner = load_prompt_learner(args.checkpoint_path, prompt_learner, device)
    model.to(device)

    preprocess, _ = get_transform(args)
    text_features = build_text_features(model, prompt_learner, device)
    return model, preprocess, text_features


def main(args):
    setup_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    samples = list(iter_mvtec_defect_images(args.mvtec_root))
    if not samples:
        raise FileNotFoundError(
            f"No defect images were found under {args.mvtec_root}. "
            "Expected structure: class_name/test/defect_type/*.png"
        )

    print(f"Device: {device}")
    print(f"Found {len(samples)} defect images.")
    model, preprocess, text_features = load_model_and_preprocess(args, device)

    records = []
    for index, (class_name, defect_name, image_path) in enumerate(samples, start=1):
        try:
            pil_image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(pil_image)
            anomaly_map = predict_anomaly_map(model, image_tensor, text_features, args, device)
            output_path = save_visualization(image_path, anomaly_map, class_name, defect_name, args)
            records.append(
                {
                    "class": class_name,
                    "defect": defect_name,
                    "image_path": str(image_path),
                    "output_path": str(output_path),
                    "anomaly_max": float(np.max(anomaly_map)),
                    "anomaly_mean": float(np.mean(anomaly_map)),
                }
            )
            print(f"[{index}/{len(samples)}] saved: {output_path}")
        except Exception as exc:
            records.append(
                {
                    "class": class_name,
                    "defect": defect_name,
                    "image_path": str(image_path),
                    "output_path": "",
                    "anomaly_max": "",
                    "anomaly_mean": "",
                    "error": repr(exc),
                }
            )
            print(f"[{index}/{len(samples)}] failed: {image_path} -> {exc}")

    record_path = Path(args.output_dir) / "inference_records.csv"
    fieldnames = ["class", "defect", "image_path", "output_path", "anomaly_max", "anomaly_mean", "error"]
    with record_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    success_count = sum(1 for row in records if row.get("output_path"))
    print(f"Done. Saved {success_count}/{len(samples)} heatmap figures.")
    print(f"Record file: {record_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch visualize AnomalyCLIP heatmaps for all MVTec AD defect samples")
    parser.add_argument("--mvtec_root", type=str, default="C:\\Users\\Administrator\\Desktop\\data\\MVTec-AD\\mvtec_anomaly_detection\\mvtec_anomaly_detection", help="path to mvtec_anomaly_detection")
    parser.add_argument("--output_dir", type=str, default="./output", help="directory to save heatmap figures")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/9_12_4_multiscale/epoch_15.pth",
        help="AnomalyCLIP prompt_learner checkpoint",
    )
    parser.add_argument("--clip_model_path", type=str, default="", help="optional local base CLIP .pt path")
    parser.add_argument("--clip_model", type=str, default="ViT-L-14.pt", help="CLIP model name/path when --clip_model_path is empty")
    parser.add_argument("--download_root", type=str, default="./pretrained", help="where to cache the base CLIP model")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="visual layers used for patch maps")
    parser.add_argument("--feature_map_layer", type=int, default=0, help="skip patch maps before this index")
    parser.add_argument("--image_size", type=int, default=518, help="input and heatmap size")
    parser.add_argument("--depth", type=int, default=9, help="deep prompt depth")
    parser.add_argument("--n_ctx", type=int, default=12, help="shallow prompt length")
    parser.add_argument("--t_n_ctx", type=int, default=4, help="deep prompt length")
    parser.add_argument("--dpam_layer", type=int, default=20, help="DPAM layer used by AnomalyCLIP visual encoder")
    parser.add_argument("--sigma", type=float, default=4.0, help="Gaussian smoothing sigma for anomaly map")
    parser.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay alpha")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--cpu", action="store_true", help="force CPU inference")
    main(parser.parse_args())
