#!/usr/bin/env python3
"""
Driver for COS568-A1 Task 1 (Hyper-parameter tuning) on neuronic.

Runs the grid:
- Architectures: (cifar10, lottery, resnet20), (mnist, default, fc)
- Pruners: rand, mag, snip, grasp, synflow
- Compression exponent: 1 -> sparsity = 10%
- Pre-epochs: 200 for mag, 0 for others

Outputs:
- Pickled metrics under Results/data/singleshot/part1_*
- part1_summary.json listing commands/expids
"""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

ARCH_GRID = [
    ("cifar10", "lottery", "resnet20"),
    ("mnist", "default", "fc"),
]
PRUNERS = ["rand", "mag", "snip", "grasp", "synflow"]


def run_cmd(cmd, dry_run=False):
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Run COS568-A1 Task 1 grid.")
    p.add_argument("--python-bin", default="python")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--results-root", default="Results/data/singleshot/0.2Sparsity")
    p.add_argument("--compression", type=float, default=1.0)  # 10% sparsity
    p.add_argument("--post-epochs", type=int, default=10)
    p.add_argument("--pre-epochs-mag", type=int, default=200)
    p.add_argument("--pre-epochs-default", type=int, default=0)
    p.add_argument("--train-batch-size", type=int, default=256)
    p.add_argument("--test-batch-size", type=int, default=256)
    p.add_argument("--prune-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    summary = []

    for (dataset, model_class, model), pruner in itertools.product(ARCH_GRID, PRUNERS):
        expid = f"part1_{dataset}_{model}_{pruner}"
        pre_epochs = args.pre_epochs_mag if pruner == "mag" else args.pre_epochs_default
        cmd = [
            args.python_bin,
            str(Path(args.repo_root) / "main.py"),
            "--experiment", "singleshot",
            "--dataset", dataset,
            "--model-class", model_class,
            "--model", model,
            "--pruner", pruner,
            "--compression", str(args.compression),
            "--expid", expid,
            "--result-dir", str(args.results_root),
            "--post-epochs", str(args.post_epochs),
            "--pre-epochs", str(pre_epochs),
            "--train-batch-size", str(args.train_batch_size),
            "--test-batch-size", str(args.test_batch_size),
            "--prune-batch-size", str(args.prune_batch_size),
            "--workers", str(args.workers),
            "--seed", str(args.seed),
        ]
        run_cmd(cmd, dry_run=args.dry_run)
        summary.append({"expid": expid, "cmd": cmd})

    with open(Path(args.results_root) / "part1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", Path(args.results_root) / "part1_summary.json")


if __name__ == "__main__":
    main()
