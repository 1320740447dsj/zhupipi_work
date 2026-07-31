# Anaconda does not provide a uv workspace, so expose both namespace paths.
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parents[0]))
sys.path.insert(0, str(_PROJECT_ROOT / "utils"))

import os.path

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from sklearn.metrics import f1_score
from torchvision import transforms
from tqdm import tqdm

from industrial.dataset import MVTecAD2Dataset
from industrial.metrics import compute_aupro, compute_image_auroc, compute_pixel_auroc
from industrial.model import SuperADD
from industrial.paths import (
    get_dataset_path,
    get_model_path,
    get_result_anomaly_images_path,
    get_result_anomaly_images_thresholded_path,
    get_result_csv_path,
    get_root_config,
)


def main() -> None:
    config = get_root_config()
    dataset_path = get_dataset_path("mvtec_ad_2")

    for split in config["test_split"]:
        print(f"\nprocessing split {split}")

        if split == "test_public" and not config["evaluate_good_images"]:
            raise ValueError(
                "I-AUROC requires normal test images. "
                "Set evaluate_good_images to true in config.json."
            )

        category_results: dict[str, dict[str, float]] = {}

        for category in config["categories"]:
            model = SuperADD.from_disk(get_model_path(category))
            test_data = MVTecAD2Dataset(
                dataset_path,
                category,
                split,
                transform=transforms.ToTensor(),
            )

            anomaly_maps: list[np.ndarray] = []
            binary_maps: list[np.ndarray] = []
            ground_truths: list[np.ndarray] = []
            image_labels: list[int] = []

            for sample in tqdm(
                test_data,
                desc=f"processing {category} images",
                file=sys.stdout,
            ):
                basename = os.path.basename(sample.image_path).removesuffix(".png")

                if sample.label == 0 and not config["evaluate_good_images"]:
                    continue

                anomaly_map, binary_result = model.predict(sample.image)
                anomaly_map = anomaly_map.astype(np.float32)

                if config["save_predictions"]:
                    tiff.imwrite(
                        get_result_anomaly_images_path(category, split)
                        / f"{basename}.tiff",
                        anomaly_map.astype(np.float16),
                    )
                    cv2.imwrite(
                        get_result_anomaly_images_thresholded_path(category, split)
                        / f"{basename}.png",
                        binary_result,
                    )

                if sample.label == -1:
                    continue

                if sample.mask is None:
                    ground_truth = np.zeros_like(binary_result, dtype=np.uint8)
                else:
                    ground_truth = sample.mask.numpy().astype(np.uint8)[0]
                    ground_truth = cv2.resize(
                        ground_truth,
                        binary_result.shape[::-1],
                        interpolation=cv2.INTER_NEAREST,
                    )

                anomaly_maps.append(anomaly_map)
                binary_maps.append(binary_result > 0)
                ground_truths.append(ground_truth > 0)
                image_labels.append(int(sample.label))

            if not ground_truths:
                continue

            ground_truth_pixels = np.concatenate(
                [ground_truth.reshape(-1) for ground_truth in ground_truths]
            )
            binary_pixels = np.concatenate(
                [binary_map.reshape(-1) for binary_map in binary_maps]
            )
            scores = {
                "I-AUROC": compute_image_auroc(image_labels, anomaly_maps),
                "P-AUROC": compute_pixel_auroc(ground_truths, anomaly_maps),
                "AUPRO": compute_aupro(
                    ground_truths,
                    anomaly_maps,
                    max_fpr=float(config.get("aupro_max_fpr", 0.30)),
                    num_thresholds=int(config.get("aupro_num_thresholds", 200)),
                ),
                "F1": float(
                    f1_score(
                        ground_truth_pixels,
                        binary_pixels,
                        zero_division=0,
                    )
                ),
            }
            category_results[category] = scores
            print(
                f"category {category}: "
                f"{scores['I-AUROC'] * 100:.2f}% (I-AUROC), "
                f"{scores['P-AUROC'] * 100:.2f}% (P-AUROC), "
                f"{scores['AUPRO'] * 100:.2f}% (AUPRO), "
                f"{scores['F1'] * 100:.2f}% (F1)"
            )

        if category_results:
            metric_columns = ["I-AUROC", "P-AUROC", "AUPRO", "F1"]
            results_df = (
                pd.DataFrame.from_dict(category_results, orient="index")
                .rename_axis("Category")
                .reset_index()
            )
            mean_row = {"Category": "mean"}
            mean_row.update(
                {
                    metric: float(results_df[metric].mean())
                    for metric in metric_columns
                }
            )
            results_df.loc[len(results_df)] = mean_row
            output_path = get_result_csv_path(f"scores_{split}")
            results_df.to_csv(output_path, index=False)

            print("\n---------- Results -----------")
            print(split)
            print(
                results_df.to_string(
                    index=False,
                    formatters={
                        metric: lambda value: f"{value * 100:.2f}%"
                        for metric in metric_columns
                    },
                )
            )
            print(f"\nmetrics saved to {output_path}")


if __name__ == "__main__":
    main()
