#!/usr/bin/env bash
# Learning-curve analysis for LASSO, Elastic Net, and SVR.
# Paths default to the input files in ../model_training; override via env vars.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- edit these ----------
METADATA="${METADATA:-${SCRIPT_DIR}/../model_training/total_metadata.txt}"
COUNTS="${COUNTS:-${SCRIPT_DIR}/../model_training/total_count_sort_ortholog.txt}"
OUTDIR="${OUTDIR:-./results}"
RSCRIPT="${RSCRIPT:-Rscript}"
# Add ,rf,xgboost if you want the tree models as well.
MODELS="${MODELS:-lasso,elasticnet,svr}"
N_REPEATS="${N_REPEATS:-10}"
TRAIN_SIZES="${TRAIN_SIZES:-20,30,40,50,60}"
N_JOBS="${N_JOBS:-8}"
# --------------------------------

mkdir -p "${OUTDIR}"

python "${SCRIPT_DIR}/run_learning_curve.py" \
    --metadata "${METADATA}" \
    --counts "${COUNTS}" \
    --outdir "${OUTDIR}" \
    --models "${MODELS}" \
    --mode learning_curve \
    --n-repeats "${N_REPEATS}" \
    --train-sizes "${TRAIN_SIZES}" \
    --test-size 0.20 \
    --seed 66 \
    --n-jobs "${N_JOBS}" \
    --rscript "${RSCRIPT}" \
    --resume \
    "$@"

echo "Finished. Figures and tables are in ${OUTDIR}"
