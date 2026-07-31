"""
将用户原始数据结构重组为 SuperADD 项目期望的 MVTec AD 2 格式。

用户原始结构:
    data/
        ground_truth/bad/{pic1, pic2, ...}
        train/good/{pic1, pic2, ...}
        test/good/{pic1, pic2, ...}
        test/bad/{pic1, pic2, ...}

目标结构(SuperADD dataset.py 期望):
    data/mvtec_ad_2/
        <category>/
            train/good/*.png
            test_public/
                good/*.png
                bad/*.png
                ground_truth/bad/{stem}_mask.png

使用方法:
    python restructure_data.py --category custom
    # 若 mask 文件名本身已经带 _mask 后缀, 加 --mask-already-suffixed
"""

import argparse
import shutil
from pathlib import Path


def restructure(src_root: Path, dst_root: Path, category: str, mask_already_suffixed: bool) -> None:
    """把用户原始数据复制(不删除原文件)到 MVTec AD 2 目录结构。

    Args:
        src_root (Path): 用户原始 data 目录(包含 train/test/ground_truth)。
        dst_root (Path): 目标根目录(例如 ./data/mvtec_ad_2)。
        category (str): 类别名, 会作为一级子目录。
        mask_already_suffixed (bool): 若 mask 文件已经是 {stem}_mask.png, 设为 True 避免重复后缀。
    """
    cat_dir = dst_root / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    # 1) train/good -> <category>/train/good
    src_train = src_root / "train" / "good"
    dst_train = cat_dir / "train" / "good"
    dst_train.mkdir(parents=True, exist_ok=True)
    n_train = 0
    if src_train.exists():
        for p in src_train.glob("*.png"):
            shutil.copy2(p, dst_train / p.name)
            n_train += 1
    print(f"[train/good] copied {n_train} files -> {dst_train}")

    # 2) test/good -> <category>/test_public/good
    # 3) test/bad  -> <category>/test_public/bad
    for sub in ("good", "bad"):
        src_test = src_root / "test" / sub
        dst_test = cat_dir / "test_public" / sub
        dst_test.mkdir(parents=True, exist_ok=True)
        n = 0
        if src_test.exists():
            for p in src_test.glob("*.png"):
                shutil.copy2(p, dst_test / p.name)
                n += 1
        print(f"[test_public/{sub}] copied {n} files -> {dst_test}")

    # 4) ground_truth/bad -> <category>/test_public/ground_truth/bad/{stem}_mask.png
    src_gt = src_root / "ground_truth" / "bad"
    dst_gt = cat_dir / "test_public" / "ground_truth" / "bad"
    dst_gt.mkdir(parents=True, exist_ok=True)
    n_gt = 0
    if src_gt.exists():
        for p in src_gt.glob("*.png"):
            # 根据 mask_already_suffixed 决定是否追加 _mask 后缀
            if mask_already_suffixed or p.stem.endswith("_mask"):
                new_name = p.name
            else:
                new_name = f"{p.stem}_mask.png"
            shutil.copy2(p, dst_gt / new_name)
            n_gt += 1
    print(f"[test_public/ground_truth/bad] copied {n_gt} files -> {dst_gt}")

    print(f"\nDone. Category '{category}' prepared at: {cat_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Restructure user data into MVTec AD 2 layout.")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("./data"),
        help="原始 data 目录(默认 ./data)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("./data/mvtec_ad_2"),
        help="目标目录(默认 ./data/mvtec_ad_2)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="yw",
        help="类别名(自定义, 默认 custom)",
    )
    parser.add_argument(
        "--mask-already-suffixed",
        action="store_true",
        help="若 mask 文件已经是 {stem}_mask.png, 加此参数避免重复后缀",
    )
    args = parser.parse_args()
    restructure(args.src.resolve(), args.dst.resolve(), args.category, args.mask_already_suffixed)


if __name__ == "__main__":
    main()
