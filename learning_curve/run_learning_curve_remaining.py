#!/usr/bin/env python3
"""
Learning-curve analysis for the remaining clocks: Random Forest, XGBoost, MoE.

Same leak-free design as run_learning_curve.py (the LASSO / Elastic Net / SVR
script). Writes to a separate folder so the two jobs can run in parallel on
the server without overwriting each other.

Example
  python run_learning_curve_remaining.py \\
      --metadata ../model_training/total_metadata.txt \\
      --counts   ../model_training/total_count_sort_ortholog.txt \\
      --aging-genes ../model_training/aging_gene_list.csv \\
      --outdir   ./results_remaining \\
      --n-repeats 10 \\
      --resume

  # RF + XGBoost only (skip MoE if the GPU/Optuna budget is tight)
  python run_learning_curve_remaining.py --models rf,xgboost --resume

  # rebuild figures
  python run_learning_curve_remaining.py --plot-only --outdir ./results_remaining
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from run_learning_curve import build_parser, run_from_args


def main():
    parser = build_parser()
    parser.description = (
        "Leak-free learning curve for Random Forest, XGBoost, and MoE "
        "(the three models not covered by the default linear-clock script)."
    )
    parser.set_defaults(
        models="rf,xgboost,moe",
        outdir=os.path.join(_SCRIPT_DIR, "results_remaining"),
        moe_trials=25,
        moe_hpo_epochs=60,
        moe_refit_epochs=100,
    )
    args = parser.parse_args()
    run_from_args(args)


if __name__ == "__main__":
    main()
