#!/usr/bin/env python3
"""
Generate three figures for Task 1:
- Top-1 accuracy vs compression (CIFAR10 / ResNet20)
- Inference time vs compression (CIFAR10 / ResNet20)
- FLOP kept fraction vs compression (CIFAR10 / ResNet20)

Input: Results/data/singleshot/analysis_task1_all.csv
Output figs: Results/data/singleshot/fig_top1.png, fig_time.png, fig_flop.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INPUT_CSV = Path("Results/data/singleshot/analysis_task1_all.csv")
OUT_DIR = Path("Results/data/singleshot")


def plot_metric(df, metric, ylabel, outfile):
    plt.figure(figsize=(6, 4))
    for pruner, sub in df.groupby("pruner"):
        sub = sub.sort_values("compression")
        plt.plot(sub["compression"], sub[metric], marker="o", label=pruner)
    plt.xlabel("Compression exponent (a), sparsity = 10^-a")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Missing {INPUT_CSV}, run analyze_task1.py first.")

    df = pd.read_csv(INPUT_CSV)
    # Focus on CIFAR10 / ResNet20 as in README tables
    df = df[(df["dataset"] == "cifar10") & (df["model"] == "resnet20")]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric(df, "top1", "Top-1 Accuracy (%)", OUT_DIR / "fig_top1.png")
    plot_metric(df, "inference_time_s", "Inference Time (s)", OUT_DIR / "fig_time.png")
    plot_metric(df, "flop_sparsity", "FLOP kept (fraction)", OUT_DIR / "fig_flop.png")
    print("Saved figures to Results/data/singleshot/: fig_top1.png, fig_time.png, fig_flop.png")


if __name__ == "__main__":
    main()
