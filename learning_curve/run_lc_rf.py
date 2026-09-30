#!/usr/bin/env python3
"""Learning curve for Random Forest only. Output: ./results_rf/"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from run_learning_curve import build_parser, run_from_args


def main():
    parser = build_parser()
    parser.set_defaults(
        models="rf",
        outdir=os.path.join(_HERE, "results_rf"),
    )
    run_from_args(parser.parse_args())


if __name__ == "__main__":
    main()
