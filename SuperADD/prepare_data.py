"""Prepare MVTec-style datasets for SuperADD without changing source files."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AnomalyPair:
    defect_type: str
    image: Path
    mask: Path


@dataclass(frozen=True)
class CategoryPlan:
    name: str
    train_good: list[Path]
    test_good: list[Path]
    anomalies: list[AnomalyPair]


def find_images(directory: Path) -> list[Path]:
    """Return supported image files immediately below a directory."""
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def unique_stems(paths: list[Path], description: str) -> None:
    stems = [path.stem for path in paths]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"Duplicate stems in {description}: {', '.join(duplicates[:5])}")


def index_masks(directory: Path) -> dict[str, Path]:
    """Index both pic.png and pic_mask.png under the key pic."""
    result: dict[str, Path] = {}
    for path in find_images(directory):
        key = path.stem[:-5] if path.stem.endswith("_mask") else path.stem
        if key in result:
            raise ValueError(f"Duplicate masks for '{key}' in {directory}")
        result[key] = path
    return result


def validate_size(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask size mismatch: {image_path} {image.size}, "
                f"{mask_path} {mask.size}"
            )


def build_category_plan(category_dir: Path) -> CategoryPlan:
    """Validate one source category before writing any output."""
    name = category_dir.name
    train_good = find_images(category_dir / "train" / "good")
    test_good = find_images(category_dir / "test" / "good")
    if not train_good:
        raise ValueError(f"No training images: {category_dir / 'train' / 'good'}")
    if not test_good:
        raise ValueError(f"No normal test images: {category_dir / 'test' / 'good'}")
    unique_stems(train_good, f"{name}/train/good")
    unique_stems(test_good, f"{name}/test/good")

    test_root = category_dir / "test"
    defect_types = sorted(
        path.name for path in test_root.iterdir()
        if path.is_dir() and path.name != "good"
    )
    if not defect_types:
        raise ValueError(f"No anomaly directories below {test_root}")

    anomalies: list[AnomalyPair] = []
    errors: list[str] = []
    for defect_type in defect_types:
        images = find_images(test_root / defect_type)
        masks = index_masks(category_dir / "ground_truth" / defect_type)
        unique_stems(images, f"{name}/test/{defect_type}")
        image_stems = {path.stem for path in images}

        for extra in sorted(set(masks) - image_stems):
            errors.append(f"{defect_type}: mask without image: {extra}")
        for image_path in images:
            mask_path = masks.get(image_path.stem)
            if mask_path is None:
                errors.append(f"{defect_type}: missing mask: {image_path.name}")
                continue
            validate_size(image_path, mask_path)
            anomalies.append(AnomalyPair(defect_type, image_path, mask_path))

    if errors:
        details = "\n  - ".join(errors[:20])
        raise ValueError(f"Category '{name}' failed validation:\n  - {details}")
    if not anomalies:
        raise ValueError(f"No valid anomaly image/mask pairs in {category_dir}")
    return CategoryPlan(name, train_good, test_good, anomalies)


def save_png(source: Path, destination: Path, mask: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        if mask:
            output = image.convert("L").point(lambda value: 255 if value > 0 else 0)
        else:
            output = image.convert("RGB")
        output.save(destination, format="PNG")


def write_category(plan: CategoryPlan, dataset_root: Path) -> None:
    target = dataset_root / plan.name
    for path in plan.train_good:
        save_png(path, target / "train" / "good" / f"{path.stem}.png")
    for path in plan.test_good:
        save_png(path, target / "test_public" / "good" / f"{path.stem}.png")

    defect_types = {item.defect_type for item in plan.anomalies}
    add_prefix = len(defect_types) > 1 or defect_types != {"bad"}
    output_names: set[str] = set()
    for item in plan.anomalies:
        prefix = f"{item.defect_type}__" if add_prefix else ""
        output_stem = prefix + item.image.stem
        if output_stem in output_names:
            raise ValueError(f"Duplicate output name in {plan.name}: {output_stem}")
        output_names.add(output_stem)
        save_png(
            item.image,
            target / "test_public" / "bad" / f"{output_stem}.png",
        )
        save_png(
            item.mask,
            target
            / "test_public"
            / "ground_truth"
            / "bad"
            / f"{output_stem}_mask.png",
            mask=True,
        )


def update_config(config_path: Path, experiment_dir: Path, categories: list[str]) -> None:
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    config["datasets_dir"] = str((experiment_dir / "data").resolve())
    config["models_dir"] = str((experiment_dir / "models").resolve())
    config["results_dir"] = str((experiment_dir / "outputs").resolve())
    config["categories"] = categories
    config["test_split"] = ["test_public"]
    config["evaluate_good_images"] = True
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)
        file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert all categories under --src into an isolated SuperADD "
            "experiment. The source dataset is never modified."
        )
    )
    parser.add_argument("--src", type=Path, help="Raw dataset root")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="New experiment directory to create",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Optional category subset; default: auto-discover all categories",
    )
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update config.json to use the prepared experiment",
    )
    parser.add_argument(
        "--activate-only",
        action="store_true",
        help="Only activate an existing prepared experiment in config.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
        help="Config file used by --update-config or --activate-only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    experiment_dir = args.experiment_dir.resolve()
    dataset_root = experiment_dir / "data" / "mvtec_ad_2"
    if args.activate_only:
        manifest_path = experiment_dir / "dataset_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        update_config(config_path, experiment_dir, manifest["categories"])
        print(f"Activated experiment: {experiment_dir}")
        print(f"Updated config:       {config_path}")
        return
    if args.src is None:
        raise ValueError("--src is required unless --activate-only is used")
    source = args.src.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    if dataset_root.is_relative_to(source):
        raise ValueError("--experiment-dir must be outside the source dataset")
    if dataset_root.exists() and any(dataset_root.iterdir()):
        raise FileExistsError(
            f"Destination is not empty: {dataset_root}\n"
            "Use a new --experiment-dir to prevent stale or mixed data."
        )

    category_names = args.categories
    if category_names is None:
        category_names = sorted(
            path.name
            for path in source.iterdir()
            if path.is_dir() and (path / "train" / "good").is_dir()
        )
    if not category_names:
        raise ValueError(f"No category directories containing train/good in {source}")

    print(f"Source:      {source}")
    print(f"Destination: {dataset_root}")
    print(f"Categories:  {', '.join(category_names)}")
    print("\nValidating source data ...")
    plans = [build_category_plan(source / name) for name in category_names]

    print("Writing prepared data ...")
    statistics = []
    for plan in plans:
        write_category(plan, dataset_root)
        stats = {
            "category": plan.name,
            "train_good": len(plan.train_good),
            "test_good": len(plan.test_good),
            "test_bad": len(plan.anomalies),
            "masks": len(plan.anomalies),
        }
        statistics.append(stats)
        print(
            f"  {plan.name}: train_good={stats['train_good']}, "
            f"test_good={stats['test_good']}, test_bad={stats['test_bad']}, "
            f"masks={stats['masks']}"
        )

    manifest = {
        "source": str(source),
        "dataset_root": str(dataset_root),
        "categories": category_names,
        "statistics": statistics,
    }
    experiment_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = experiment_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=4)
        file.write("\n")

    if args.update_config:
        update_config(config_path, experiment_dir, category_names)
        print(f"Updated config: {config_path}")
    print(f"Manifest:      {manifest_path}")
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
