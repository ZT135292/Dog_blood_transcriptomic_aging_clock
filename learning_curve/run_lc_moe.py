#!/usr/bin/env python3
"""Learning curve for MoE only. Output: ./results_moe/"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_learning_curve import build_parser, run_from_args


def main():
    parser = build_parser()
    parser.set_defaults(
        models="moe",
        outdir=os.path.join(_HERE, "results_moe"),
        moe_trials=25,
        moe_hpo_epochs=60,
        moe_refit_epochs=100,
    )
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
