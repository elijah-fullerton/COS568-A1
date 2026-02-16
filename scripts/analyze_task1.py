#!/usr/bin/env python3
"""
Aggregate COS568-A1 Task 1 (singleshot) results into summary tables.

Expected layout (produced by scripts/run_hparam.py):
Results/data/singleshot/singleshot/part1_<dataset>_<model>_<pruner>/
    post-train.pkl
    compression.pkl

Outputs:
- Prints summary tables (top1 accuracy, inference time, param/flop sparsity)
- Writes CSV/JSON/Markdown snapshots to Results/data/singleshot/analysis_task1.*
"""
import json
import re
from pathlib import Path

import pandas as pd


def _safe_read_pickle(path: Path):
    """
    Some pickles were written with a pandas StringDtype signature that newer
    pandas balks at. Patch StringDtype.__init__ to accept the legacy args when
    needed.
    """
    try:
        return pd.read_pickle(path)
    except Exception:
        import pickle

        class _FakeStringDtype:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                return

        class _PatchedUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "pandas.arrays" and name == "StringDtype":
                    return _FakeStringDtype
                if module == "pandas.core.arrays.string_" and name == "StringDtype":
                    return _FakeStringDtype
                if module == "pandas._libs.arrays" and name == "StringDtype":
                    return _FakeStringDtype
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _PatchedUnpickler(f).load()


def _load_post(run_dir: Path):
    csv = run_dir / "post-train.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return _safe_read_pickle(run_dir / "post-train.pkl")


def _load_comp(run_dir: Path):
    csv = run_dir / "compression.csv"
    if csv.exists():
        return pd.read_csv(csv)
    return _safe_read_pickle(run_dir / "compression.pkl")


def load_run(run_dir: Path):
    post = _load_post(run_dir)
    comp = _load_comp(run_dir)

    final = post.tail(1).iloc[0]
    top1 = float(final["top1_accuracy"])
    top5 = float(final["top5_accuracy"])
    test_loss = float(final["test_loss"])
    inf_time = float(final["inference_time"])

    total_params = float((comp["sparsity"] * comp["size"]).sum())
    possible_params = float(comp["size"].sum())
    param_sparsity = total_params / possible_params

    total_flops = float((comp["sparsity"] * comp["flops"]).sum())
    possible_flops = float(comp["flops"].sum())
    flop_sparsity = total_flops / possible_flops

    return {
        "top1": top1,
        "top5": top5,
        "test_loss": test_loss,
        "inference_time_s": inf_time,
        "param_sparsity": param_sparsity,
        "flop_sparsity": flop_sparsity,
    }


def parse_log_fallback(log_path: Path, expids):
    import re

    text = log_path.read_text()
    parts = text.split("Train results:")[1:]
    rows = []
    for expid, part in zip(expids, parts):
        line = None
        for ln in part.splitlines():
            if ln.strip().startswith("Final"):
                line = ln
                break
        if not line:
            continue
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
        if len(nums) < 4:
            continue
        train_loss = nums[0]
        test_loss = nums[1]
        # top1 may be omitted in truncated display; handle length.
        if len(nums) >= 5:
            top1 = nums[2]
            top5 = nums[-2]
            inftime = nums[-1]
        else:
            top1 = None
            top5 = nums[-2]
            inftime = nums[-1]
        ps = re.search(r"Parameter Sparsity: [\d\.]+/[\d\.]+ \(([\d\.]+)\)", part)
        fs = re.search(r"FLOP Sparsity: [\d\.]+/[\d\.]+ \(([\d\.]+)\)", part)
        rows.append(
            {
                "expid": expid,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "top1": top1,
                "top5": top5,
                "inference_time_s": inftime,
                "param_sparsity": float(ps.group(1)) if ps else None,
                "flop_sparsity": float(fs.group(1)) if fs else None,
            }
        )
    return pd.DataFrame(rows)


def main():
    roots = [
        (Path("Results/data/singleshot/singleshot"), 1.0),
        (Path("Results/data/singleshot/0.5Sparsity/singleshot"), 0.5),
        (Path("Results/data/singleshot/0.2Sparsity/singleshot"), 0.2),
    ]
    runs = []
    for root, compression in roots:
        if not root.exists():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            m = re.match(r"part1_(?P<dataset>[^_]+)_(?P<model>[^_]+)_(?P<pruner>.+)", run_dir.name)
            if not m:
                continue
            try:
                info = load_run(run_dir)
            except Exception:
                info = None
            if info is not None:
                row = {"expid": run_dir.name, "compression": compression, **m.groupdict(), **info}
                runs.append(row)

    if not runs:
        # Fallback: try logs if nothing was parsed
        expids = [e["expid"] for e in json.load(open("Results/data/singleshot/part1_summary.json"))]
        log = Path("slurm_logs").glob("part1_*.out")
        log_path = next(log, None)
        if log_path is None:
            raise RuntimeError("No results parsed and no slurm log found.")
        df = parse_log_fallback(log_path, expids)
    else:
        df = pd.DataFrame(runs)
    df = df.sort_values(["compression", "expid"])

    # Derive dataset/model/pruner from expid if not already present.
    if "dataset" not in df.columns or "model" not in df.columns or "pruner" not in df.columns:
        parts = df["expid"].str.split("_", n=4, expand=True)
        df["dataset"] = parts[1]
        df["model"] = parts[2]
        df["pruner"] = parts[3]

    # Pivot tables for Task 1 reporting
    acc_table = df.pivot(index=["compression", "dataset", "model"], columns="pruner", values="top1")
    time_table = df.pivot(index=["compression", "dataset", "model"], columns="pruner", values="inference_time_s")
    param_table = df.pivot(index=["compression", "dataset", "model"], columns="pruner", values="param_sparsity")
    flop_table = df.pivot(index=["compression", "dataset", "model"], columns="pruner", values="flop_sparsity")

    out_base = Path("Results/data/singleshot/analysis_task1_all")
    df.to_csv(out_base.with_suffix(".csv"), index=False)
    with open(out_base.with_suffix(".json"), "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    # Save Markdown snapshots of the pivot tables
    with open(out_base.with_suffix(".md"), "w") as f:
        f.write("# Task 1 summary\n\n")
        f.write("## Top-1 Accuracy\n\n" + acc_table.to_markdown() + "\n\n")
        f.write("## Inference Time (s)\n\n" + time_table.to_markdown() + "\n\n")
        f.write("## Param Sparsity (kept fraction)\n\n" + param_table.to_markdown() + "\n\n")
        f.write("## FLOP Sparsity (kept fraction)\n\n" + flop_table.to_markdown() + "\n")

    # Print to stdout for convenience
    print("=== Summary (per run) ===")
    print(df)
    print("\n=== Top-1 Accuracy ===")
    print(acc_table)
    print("\n=== Inference Time (s) ===")
    print(time_table)
    print("\n=== Param Sparsity (kept fraction) ===")
    print(param_table)
    print("\n=== FLOP Sparsity (kept fraction) ===")
    print(flop_table)


if __name__ == "__main__":
    main()
