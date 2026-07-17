"""Train a PatchCore model for a single MVTec-style custom class."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


LOGGER = logging.getLogger("train_foreign_object_patchcore")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train PatchCore with WR50/layer2+layer3 on one custom MVTec-style "
            "dataset. Supports both data/<class>/train and flat data/train layouts."
        )
    )
    parser.add_argument("--data_path", default="C:/Users/Administrator/Desktop/udesing/mllm/all-lpt", help="Dataset root.")
    parser.add_argument("--results_path", default="/root/model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_group", default="foreign_object_wr50_l2-3")
    parser.add_argument("--log_project", default="PatchCore")
    parser.add_argument("--subdatasets", nargs="+", default=["bottle"])
    parser.add_argument("--resize", type=int, default=332)
    parser.add_argument("--imagesize", type=int, default=332)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pretrain_embed_dimension", type=int, default=512)
    parser.add_argument("--target_embed_dimension", type=int, default=512)
    parser.add_argument("--anomaly_scorer_num_nn", type=int, default=3)
    parser.add_argument("--patchsize", type=int, default=5)
    parser.add_argument("--coreset_percentage", "-p", type=float, default=0.01)
    parser.add_argument("--faiss_on_gpu", action="store_true", default=False)
    parser.add_argument("--no_faiss_on_gpu", action="store_false", dest="faiss_on_gpu")
    parser.add_argument("--save_patchcore_model", action="store_true", default=True)
    parser.add_argument("--skip_evaluation", action="store_true", default=True)
    parser.add_argument("--save_segmentation_images", action="store_true")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    repo = _repo_root()
    try:
        import faiss  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit(
            "The current Python environment cannot import 'faiss'. Install it first, "
            "for example: conda install -c conda-forge faiss-cpu"
        )

    command = [
        sys.executable,
        str(repo / "bin" / "run_patchcore.py"),
        "--gpu",
        str(args.gpu),
        "--seed",
        str(args.seed),
        "--log_group",
        args.log_group,
        "--log_project",
        args.log_project,
    ]
    if args.save_patchcore_model:
        command.append("--save_patchcore_model")
    if args.skip_evaluation:
        command.append("--skip_evaluation")
    if args.save_segmentation_images:
        command.append("--save_segmentation_images")

    command.extend(
        [
            args.results_path,
            "patch_core",
            "-b",
            "wideresnet50",
            "-le",
            "layer2",
            "-le",
            "layer3",
        ]
    )
    if args.faiss_on_gpu:
        command.append("--faiss_on_gpu")
    command.extend(
        [
            "--pretrain_embed_dimension",
            str(args.pretrain_embed_dimension),
            "--target_embed_dimension",
            str(args.target_embed_dimension),
            "--anomaly_scorer_num_nn",
            str(args.anomaly_scorer_num_nn),
            "--patchsize",
            str(args.patchsize),
            "sampler",
            "-p",
            str(args.coreset_percentage),
            "approx_greedy_coreset",
            "dataset",
            "--resize",
            str(args.resize),
            "--imagesize",
            str(args.imagesize),
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
    )
    for subdataset in args.subdatasets:
        command.extend(["--subdatasets", subdataset])
    command.extend(["mvtec", args.data_path])

    env = os.environ.copy()
    pythonpath = str(repo / "src")
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")
    LOGGER.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True, cwd=str(repo), env=env)


if __name__ == "__main__":
    main()
