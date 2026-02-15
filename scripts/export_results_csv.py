#!/usr/bin/env python3
"""
Convert Task 1 singleshot pickles to CSV for portability.

Reads:
  Results/data/singleshot/singleshot/part1_*/post-train.pkl
  Results/data/singleshot/singleshot/part1_*/compression.pkl

Writes (alongside the originals):
  post-train.csv
  compression.csv
"""
from pathlib import Path

import pickle
import pandas as pd
import pandas._libs.arrays as sm_lib
import pandas.core.arrays.string_ as sm


class _FakeStringDtype:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        return


class _PatchedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas.core.arrays.string_" and name == "StringDtype":
            return _FakeStringDtype
        if module == "pandas._libs.arrays" and name == "StringDtype":
            return _FakeStringDtype
        if module == "pandas.arrays" and name == "StringDtype":
            return _FakeStringDtype
        if module == "pandas._libs.internals" and name == "_unpickle_block":
            # Bypass dtype handling by returning a no-op builder.
            def _noop_block(*args, **kwargs):
                return args

            return _noop_block
        if module == "pandas._libs.missing" and name == "NAType":
            class _FakeNA:
                pass

            return _FakeNA
        if module == "pandas._libs.arrays" and name == "NDArrayBacked":
            class _FakeArray:
                def __setstate__(self, state):
                    return

            return _FakeArray
        return super().find_class(module, name)


def convert_run(run_dir: Path):
    post_pkl = run_dir / "post-train.pkl"
    comp_pkl = run_dir / "compression.pkl"
    if post_pkl.exists():
        try:
            post = pd.read_pickle(post_pkl)
        except Exception:
            with open(post_pkl, "rb") as f:
                post = _PatchedUnpickler(f).load()
        post.to_csv(run_dir / "post-train.csv", index=False)
    if comp_pkl.exists():
        try:
            comp = pd.read_pickle(comp_pkl)
        except Exception:
            with open(comp_pkl, "rb") as f:
                comp = _PatchedUnpickler(f).load()
        comp.to_csv(run_dir / "compression.csv", index=False)


def main():
    root = Path("Results/data/singleshot/singleshot")
    if not root.exists():
        raise SystemExit(f"Not found: {root}")
    for run in sorted(root.iterdir()):
        if run.is_dir():
            convert_run(run)
    print("Done. CSVs written next to the pickles.")


if __name__ == "__main__":
    main()
