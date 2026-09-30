#!/usr/bin/env python3
"""Merge linear-clock and remaining-model run_records, then rebuild the figures."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from run_learning_curve import RUN_CSV_NAME, finalise


def main():
    parser = argparse.ArgumentParser(
        description="Combine learning-curve CSVs from multiple outdirs and replot all models."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=[
            os.path.join(_SCRIPT_DIR, "results_lasso"),
            os.path.join(_SCRIPT_DIR, "results_elasticnet"),
            os.path.join(_SCRIPT_DIR, "results_svr"),
            os.path.join(_SCRIPT_DIR, "results_rf"),
            os.path.join(_SCRIPT_DIR, "results_xgboost"),
            os.path.join(_SCRIPT_DIR, "results_moe"),
            os.path.join(_SCRIPT_DIR, "results"),
            os.path.join(_SCRIPT_DIR, "results_remaining"),
        ],
        help="directories that each contain run_records.csv",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(_SCRIPT_DIR, "results_all"),
        help="where to write the merged tables and figures",
    )
    parser.add_argument("--mode", default="learning_curve")
    args = parser.parse_args()

    frames = []
    for folder in args.dirs:
        path = os.path.join(folder, RUN_CSV_NAME)
        if not os.path.exists(path):
            print("Skip missing", path)
            continue
        df = pd.read_csv(path)
        df["source_dir"] = os.path.abspath(folder)
        frames.append(df)
        print(f"Loaded {len(df)} rows from {path}")
    if not frames:
        raise FileNotFoundError("No run_records.csv found in the given --dirs.")

    os.makedirs(args.outdir, exist_ok=True)
    merged = pd.concat(frames, ignore_index=True)
    merged_path = os.path.join(args.outdir, RUN_CSV_NAME)
    merged.to_csv(merged_path, index=False)
    print(f"Wrote {merged_path} ({len(merged)} rows)")
    finalise(args.outdir, args.mode)


if __name__ == "__main__":
    main()
