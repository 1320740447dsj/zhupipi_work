"""Generate MVTec-compatible metadata for a private multi-class dataset.

Expected layout:
    root/object1/{train,test,ground_truth}/...
    root/object2/{train,test,ground_truth}/...

A legacy single-class layout with train/ and test/ directly below root is
also supported.
"""

import argparse
import json
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_images(directory):
    if not directory.is_dir():
        return []
    return sorted(
        (path for path in directory.iterdir()
         if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )


def relative_path(path, root):
    return path.relative_to(root).as_posix()


def discover_classes(root, single_class_name):
    if (root / "train").is_dir() or (root / "test").is_dir():
        if not (root / "train").is_dir() or not (root / "test").is_dir():
            raise ValueError(
                f"Single-class root must contain both train/ and test/: {root}"
            )
        return [(single_class_name or root.name, root)]

    classes = []
    incomplete = []
    for class_dir in sorted(
            (path for path in root.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold()):
        has_train = (class_dir / "train").is_dir()
        has_test = (class_dir / "test").is_dir()
        if has_train and has_test:
            classes.append((class_dir.name, class_dir))
        elif has_train or has_test:
            incomplete.append(class_dir.name)

    if incomplete:
        raise ValueError(
            "These class directories do not contain both train/ and test/: "
            + ", ".join(incomplete)
        )
    if not classes:
        raise ValueError(
            f"No classes found under {root}. Expected object/train and object/test."
        )
    return classes


def find_mask(test_image, mask_dir):
    """Find image.png, image_mask.png, or image-mask.png with any image suffix."""
    masks = list_images(mask_dir)
    if not masks:
        return None, f"mask directory is missing or empty: {mask_dir}"

    exact = [mask for mask in masks
             if mask.name.casefold() == test_image.name.casefold()]
    if len(exact) == 1:
        return exact[0], None

    image_stem = test_image.stem.casefold()
    accepted_stems = {image_stem, f"{image_stem}_mask", f"{image_stem}-mask"}
    matches = [mask for mask in masks
               if mask.stem.casefold() in accepted_stems]
    if len(matches) == 1:
        return matches[0], None
    if not matches:
        return None, f"no mask found in {mask_dir}"
    return None, "ambiguous masks: " + ", ".join(mask.name for mask in matches)


def make_record(root, image, class_name, defect_name, anomaly, mask=None):
    return {
        "img_path": relative_path(image, root),
        "mask_path": relative_path(mask, root) if mask is not None else "",
        "cls_name": class_name,
        "specie_name": defect_name,
        "anomaly": int(anomaly),
    }


def build_meta(root, single_class_name=None, allow_missing_masks=False):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset root does not exist: {root}")

    classes = discover_classes(root, single_class_name)
    meta = {"train": {}, "test": {}}
    errors = []
    stats = {}

    for class_name, class_dir in classes:
        train_good_dir = class_dir / "train" / "good"
        test_dir = class_dir / "test"
        test_good_dir = test_dir / "good"
        train_good = list_images(train_good_dir)
        test_good = list_images(test_good_dir)

        if not train_good:
            errors.append(f"[{class_name}] no images found in {train_good_dir}")
        if not test_good:
            errors.append(f"[{class_name}] no images found in {test_good_dir}")

        train_records = [
            make_record(root, image, class_name, "good", anomaly=False)
            for image in train_good
        ]
        test_records = [
            make_record(root, image, class_name, "good", anomaly=False)
            for image in test_good
        ]

        defect_dirs = sorted(
            (path for path in test_dir.iterdir()
             if path.is_dir() and path.name.casefold() != "good"),
            key=lambda path: path.name.casefold(),
        ) if test_dir.is_dir() else []
        anomaly_count = 0
        missing_count = 0
        if not defect_dirs:
            errors.append(f"[{class_name}] no anomaly directories found in {test_dir}")

        for defect_dir in defect_dirs:
            defect_name = defect_dir.name
            anomaly_images = list_images(defect_dir)
            if not anomaly_images:
                errors.append(f"[{class_name}/{defect_name}] no anomaly images found")
                continue
            mask_dir = class_dir / "ground_truth" / defect_name
            for image in anomaly_images:
                mask, problem = find_mask(image, mask_dir)
                if problem:
                    missing_count += 1
                    message = f"[{class_name}/{defect_name}/{image.name}] {problem}"
                    if allow_missing_masks:
                        print(f"WARNING: {message}")
                    else:
                        errors.append(message)
                test_records.append(
                    make_record(root, image, class_name, defect_name, True, mask)
                )
                anomaly_count += 1

        meta["train"][class_name] = train_records
        meta["test"][class_name] = test_records
        stats[class_name] = {
            "train_good": len(train_good),
            "test_good": len(test_good),
            "test_anomaly": anomaly_count,
            "missing_mask": missing_count,
        }

    if errors:
        details = "\n  - ".join(errors)
        raise ValueError(
            f"Dataset validation failed with {len(errors)} error(s):\n  - {details}"
        )
    return root, meta, stats


def run(root, single_class_name=None, allow_missing_masks=False):
    root, meta, stats = build_meta(
        root, single_class_name, allow_missing_masks
    )
    meta_path = root / "meta.json"
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=4, ensure_ascii=False)
        file.write("\n")

    print("=" * 78)
    print(f"Generated: {meta_path}")
    print(f"Classes:   {len(stats)}")
    print("-" * 78)
    print(f"{'class':<24} {'train/good':>12} {'test/good':>12} "
          f"{'test/anomaly':>14} {'missing mask':>13}")
    for class_name, counts in stats.items():
        print(
            f"{class_name:<24} {counts['train_good']:>12} "
            f"{counts['test_good']:>12} {counts['test_anomaly']:>14} "
            f"{counts['missing_mask']:>13}"
        )
    print("=" * 78)
    return meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate meta.json for a multi-class MVTec-style dataset"
    )
    parser.add_argument(
        "--root", required=True,
        help="Dataset root containing object1/, object2/, ...",
    )
    parser.add_argument(
        "--single-class-name", "--cls_name", dest="single_class_name",
        help="Class name for the legacy single-class layout",
    )
    parser.add_argument(
        "--allow-missing-masks", action="store_true",
        help="Write records without masks (not suitable for pixel metrics)",
    )
    arguments = parser.parse_args()
    try:
        run(
            arguments.root,
            arguments.single_class_name,
            arguments.allow_missing_masks,
        )
    except ValueError as error:
        parser.error(str(error))
