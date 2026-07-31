"""
从 full-shot.jsonl 生成 few-shot.jsonl。

AA-CLIP 的 few-shot 约定：
    ./dataset/metadata/{dataset}/{shot}-shot.jsonl
每个 class 从 train/good 样本中随机抽取 shot 个（固定 seed 保证可复现）。
test 样本不包含（few-shot 仅限制训练样本数）。

用法：
    python tools/gen_few_shot_meta.py --dataset VisA --shot 4
    python tools/gen_few_shot_meta.py --dataset MVTec --shot 1 2 4 8 16 32
    python tools/gen_few_shot_meta.py --dataset Private --shot 4 --seed 111
"""
import os
import json
import random
import argparse


def load_full_shot(dataset_name):
    meta_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dataset",
        "metadata",
        dataset_name,
        "full-shot.jsonl",
    )
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"未找到 {meta_path}，请先生成 full-shot.jsonl")
    with open(meta_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f], meta_path


def split_by_class_and_split(meta):
    """按 class_name 分组，再按 train/test 分组。
    用 label 区分：label=0 是正常样本（用于 few-shot 训练），label=1 是异常样本。
    这样对 MVTec / VisA / Private 等不同目录结构都通用。
    """
    by_class = {}
    for m in meta:
        cls = m["class_name"]
        if cls not in by_class:
            by_class[cls] = {"train": [], "test": []}
        if m["label"] == 0:
            by_class[cls]["train"].append(m)
        else:
            by_class[cls]["test"].append(m)
    return by_class


def gen_few_shot(dataset_name, shot, seed=111):
    meta, full_path = load_full_shot(dataset_name)
    by_class = split_by_class_and_split(meta)

    out_path = os.path.join(os.path.dirname(full_path), f"{shot}-shot.jsonl")
    random.seed(seed)

    all_rows = []
    print(f"[INFO] dataset={dataset_name} shot={shot} seed={seed}")
    for cls in sorted(by_class.keys()):
        train_pool = by_class[cls]["train"]
        if len(train_pool) == 0:
            print(f"[WARN] {cls}: 无 train/good 样本，跳过")
            continue
        n_take = min(shot, len(train_pool))
        sampled = random.sample(train_pool, n_take)
        print(f"[INFO] {cls}: pool={len(train_pool)} sampled={n_take}")
        all_rows.extend(sampled)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] 已写入 {out_path}，共 {len(all_rows)} 条记录。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="数据集名，如 VisA / MVTec / Private")
    parser.add_argument(
        "--shot",
        type=int,
        nargs="+",
        required=True,
        help="shot 数，可传多个，如 --shot 1 2 4 8 16 32",
    )
    parser.add_argument("--seed", type=int, default=111, help="随机种子，默认 111（与 train.py 默认一致）")
    args = parser.parse_args()

    for s in args.shot:
        gen_few_shot(args.dataset, s, args.seed)


if __name__ == "__main__":
    main()
