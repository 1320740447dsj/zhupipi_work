"""
私有数据集 metadata 生成脚本
==============================================
用途:
    扫描符合以下 MVTec-AD 风格目录结构的私有数据集,生成
    ./dataset/metadata/<dataset_name>/full-shot.jsonl

支持的目录结构 (与用户的描述一致):
    <DATA_PATH>/
      object1/
        train/good/           pic1.png, pic2.png, ...        (label=0, 无 mask)
        test/good/            pic1.png, ...                  (label=0, 无 mask)
        test/bad/             pic1.png, ...                  (label=1, 需要 mask)
        ground_truth/bad/     pic1_mask.png, ...             (与 test/bad 同名,_mask 后缀)
      object2/
        ...

用法:
    python tools/gen_private_metadata.py \
        --data_path /data/Custom \
        --dataset_name Custom \
        --classes object1 object2 object3 \
        --mask_suffix _mask \
        --img_exts .png .jpg .bmp

说明:
    1) --data_path       : 私有数据集根目录 (constants.py 中 DATA_PATH 对应的路径)
    2) --dataset_name    : 数据集名称,会创建 ./dataset/metadata/<dataset_name>/full-shot.jsonl
    3) --classes         : 数据集下所有 object 文件夹名,空格分隔 (与 constants.py 中 CLASS_NAMES 一致)
    4) --mask_suffix     : mask 文件相对图像文件名的后缀,默认 _mask
                           例如 图像 pic1.png -> mask pic1_mask.png
    5) --img_exts        : 视为图像的扩展名,默认 .png .jpg .jpeg .bmp
    6) --defect_subdir   : test 目录下异常子目录名,默认 bad (MVTec 中是 broken_large 等多种,
                           私有数据集中按用户描述使用 bad)
    7) --good_subdir     : 正常子目录名,默认 good
"""
import os
import json
import argparse
from glob import glob


def list_images(folder: str, img_exts):
    """返回 folder 下所有图像文件名 (不含路径),按名称排序。"""
    if not os.path.isdir(folder):
        return []
    files = []
    for f in os.listdir(folder):
        if os.path.splitext(f)[1].lower() in img_exts:
            files.append(f)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description="Generate full-shot.jsonl for private MVTec-style dataset")
    parser.add_argument("--data_path", type=str, required=True,
                        help="私有数据集根目录,例如 /data/Custom")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="数据集名称,例如 Custom (需与 constants.py 中一致)")
    parser.add_argument("--classes", type=str, nargs="+", required=True,
                        help="object 子目录列表,例如 object1 object2 object3")
    parser.add_argument("--mask_suffix", type=str, default="_mask",
                        help="mask 文件相对图像文件名的后缀,默认 _mask")
    parser.add_argument("--img_exts", type=str, nargs="+",
                        default=[".png", ".jpg", ".jpeg", ".bmp"],
                        help="图像扩展名列表")
    parser.add_argument("--defect_subdir", type=str, default="bad",
                        help="test 目录下异常子目录名,默认 bad")
    parser.add_argument("--good_subdir", type=str, default="good",
                        help="正常子目录名,默认 good")
    args = parser.parse_args()

    img_exts = tuple(e.lower() for e in args.img_exts)
    out_dir = os.path.join("dataset", "metadata", args.dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "full-shot.jsonl")

    total = 0
    missing_masks = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        # 顺序: train/good -> test/good -> test/bad
        # 训练样本只放 good,测试样本 good 和 bad 都放
        for cls in args.classes:
            # ---- train/good (label=0, 无 mask) ----
            train_good_dir = os.path.join(args.data_path, cls, "train", args.good_subdir)
            for fname in list_images(train_good_dir, img_exts):
                rec = {
                    "image_path": f"{cls}/train/{args.good_subdir}/{fname}",
                    "label": 0,
                    "class_name": cls,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

            # ---- test/good (label=0, 无 mask) ----
            test_good_dir = os.path.join(args.data_path, cls, "test", args.good_subdir)
            for fname in list_images(test_good_dir, img_exts):
                rec = {
                    "image_path": f"{cls}/test/{args.good_subdir}/{fname}",
                    "label": 0,
                    "class_name": cls,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

            # ---- test/<defect> (label=1, 需要 mask) ----
            test_bad_dir = os.path.join(args.data_path, cls, "test", args.defect_subdir)
            gt_bad_dir = os.path.join(args.data_path, cls, "ground_truth", args.defect_subdir)

            for fname in list_images(test_bad_dir, img_exts):
                # 推断 mask 文件名: pic1.png -> pic1_mask.png
                stem, ext = os.path.splitext(fname)
                mask_fname = f"{stem}{args.mask_suffix}{ext}"
                mask_rel_path = f"{cls}/ground_truth/{args.defect_subdir}/{mask_fname}"
                mask_abs_path = os.path.join(gt_bad_dir, mask_fname)

                if not os.path.exists(mask_abs_path):
                    missing_masks += 1
                    print(f"[WARN] 缺失 mask: {mask_abs_path} (图像 {fname})")
                    mask_field = ""
                else:
                    mask_field = mask_rel_path

                rec = {
                    "image_path": f"{cls}/test/{args.defect_subdir}/{fname}",
                    "label": 1,
                    "mask_path": mask_field,
                    "class_name": cls,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1

    print("=" * 60)
    print(f"[OK] 已生成: {out_path}")
    print(f"     共 {total} 条记录,缺失 mask {missing_masks} 条")
    print(f"     data_path    = {args.data_path}")
    print(f"     dataset_name = {args.dataset_name}")
    print(f"     classes      = {args.classes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
