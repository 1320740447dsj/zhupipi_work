"""
为私有数据集（MVTec-AD 同构格式）生成 full-shot.jsonl。

私有数据集目录结构（与 MVTec-AD 一致）：
    data/
      object1/
        train/good/001.png, 002.png ...
        test/good/001.png ...
        test/bad/001.png ...
        ground_truth/bad/001.png           # mask 与图片同名（默认）
                  或 001_mask.png          # MVTec 标准 _mask 后缀（自动探测）
      object2/
        ...

输出：dataset/metadata/Private/full-shot.jsonl
每行格式：
    {"image_path": "object1/test/bad/001.png",
     "label": 1,
     "class_name": "object1",
     "mask_path": "object1/ground_truth/bad/001.png"}

用法：
    python tools/gen_private_meta.py
    python tools/gen_private_meta.py --data_root ./data --out ./dataset/metadata/Private/full-shot.jsonl
    python tools/gen_private_meta.py --scan-classes        # 仅打印扫描到的 object 列表，不写文件
"""
import os
import json
import argparse
import glob
import runpy


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def find_mask(gt_dir, img_name, mask_suffix="_mask"):
    """在 gt_dir 下寻找与 img_name 对应的 mask 文件。
    优先尝试同名，其次尝试 _mask 后缀，最后退化为目录内任意同名前缀文件。
    """
    stem, ext = os.path.splitext(img_name)
    # 1. 完全同名
    cand = os.path.join(gt_dir, img_name)
    if os.path.exists(cand):
        return cand
    # 2. _mask 后缀（MVTec 标准：001.png -> 001_mask.png）
    cand = os.path.join(gt_dir, f"{stem}{mask_suffix}{ext}")
    if os.path.exists(cand):
        return cand
    # 3. 不带后缀的其它扩展名
    for e in IMG_EXTS:
        cand = os.path.join(gt_dir, f"{stem}{e}")
        if os.path.exists(cand):
            return cand
        cand = os.path.join(gt_dir, f"{stem}{mask_suffix}{e}")
        if os.path.exists(cand):
            return cand
    return None


def scan_classes(data_root):
    """扫描 data_root 下所有符合结构的 object 文件夹。"""
    classes = []
    if not os.path.isdir(data_root):
        return classes
    for name in sorted(os.listdir(data_root)):
        obj_dir = os.path.join(data_root, name)
        if not os.path.isdir(obj_dir):
            continue
        # 必须至少有 train/good 或 test 目录才算合法 object
        if os.path.isdir(os.path.join(obj_dir, "train", "good")) or os.path.isdir(
            os.path.join(obj_dir, "test")
        ):
            classes.append(name)
    return classes


def build_meta(data_root, obj_name):
    """为一个 object 生成所有 meta 行。"""
    rows = []
    obj_dir = os.path.join(data_root, obj_name)

    # ---- train/good：仅训练用，label=0，无 mask ----
    train_good_dir = os.path.join(obj_dir, "train", "good")
    if os.path.isdir(train_good_dir):
        for img_name in sorted(os.listdir(train_good_dir)):
            if not is_image(img_name):
                continue
            rows.append(
                {
                    "image_path": f"{obj_name}/train/good/{img_name}",
                    "label": 0,
                    "class_name": obj_name,
                    "mask_path": "",
                }
            )

    # ---- test/* ----
    test_dir = os.path.join(obj_dir, "test")
    if not os.path.isdir(test_dir):
        return rows
    for sub in sorted(os.listdir(test_dir)):
        sub_dir = os.path.join(test_dir, sub)
        if not os.path.isdir(sub_dir):
            continue
        is_bad = sub.lower() != "good"
        gt_sub_dir = os.path.join(obj_dir, "ground_truth", sub)
        for img_name in sorted(os.listdir(sub_dir)):
            if not is_image(img_name):
                continue
            mask_rel = ""
            if is_bad:
                mask_path = find_mask(gt_sub_dir, img_name)
                if mask_path is None:
                    print(
                        f"[WARN] mask not found for {obj_name}/test/{sub}/{img_name} "
                        f"in {gt_sub_dir}"
                    )
                else:
                    mask_rel = os.path.relpath(mask_path, data_root).replace("\\", "/")
            rows.append(
                {
                    "image_path": f"{obj_name}/test/{sub}/{img_name}",
                    "label": 1 if is_bad else 0,
                    "class_name": obj_name,
                    "mask_path": mask_rel,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=None,
        help="私有数据集根目录（默认使用 dataset/constants.py 中 DATA_PATH['Private']）",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出 jsonl 路径（默认 ./dataset/metadata/Private/full-shot.jsonl）",
    )
    parser.add_argument(
        "--scan-classes",
        action="store_true",
        help="仅打印扫描到的 object 列表，不生成 jsonl",
    )
    parser.add_argument(
        "--mask_suffix", default="_mask", help="MVTec 风格的 mask 后缀（默认 _mask）"
    )
    args = parser.parse_args()

    # 默认 data_root：与 constants.py 中 DATA_PATH['Private'] 保持一致
    if args.data_root is None:
        constants = runpy.run_path(
            os.path.join(PROJECT_ROOT, "dataset", "constants.py")
        )
        args.data_root = constants["DATA_PATH"]["Private"]
    print(f"[INFO] data_root = {args.data_root}")

    classes = scan_classes(args.data_root)
    print(f"[INFO] 扫描到 {len(classes)} 个 object: {classes}")

    if args.scan_classes:
        print("请将以上 object 名按顺序填入 dataset/constants.py 中 CLASS_NAMES['Private']。")
        return

    if not classes:
        raise RuntimeError(f"未在 {args.data_root} 下扫描到任何 object 文件夹，请检查目录结构。")

    # 默认输出路径
    if args.out is None:
        args.out = os.path.join(
            PROJECT_ROOT,
            "dataset",
            "metadata",
            "Private",
            "full-shot.jsonl",
        )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    all_rows = []
    for c in classes:
        rows = build_meta(args.data_root, c)
        n_bad = sum(1 for r in rows if r["label"] == 1)
        n_good = sum(1 for r in rows if r["label"] == 0)
        print(f"[INFO] {c}: 共 {len(rows)} 条 (good={n_good}, bad={n_bad})")
        all_rows.extend(rows)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 已写入 {args.out}，共 {len(all_rows)} 条记录。")


if __name__ == "__main__":
    main()
