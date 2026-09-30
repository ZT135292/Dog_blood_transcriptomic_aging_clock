#!/usr/bin/env bash
# Learning curve for Random Forest, XGBoost, and MoE.
# Can be submitted in parallel with run_learning_curve.sh (different --outdir).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

METADATA="${METADATA:-${SCRIPT_DIR}/../model_training/total_metadata.txt}"
COUNTS="${COUNTS:-${SCRIPT_DIR}/../model_training/total_count_sort_ortholog.txt}"
AGING_GENES="${AGING_GENES:-${SCRIPT_DIR}/../model_training/aging_gene_list.csv}"
OUTDIR="${OUTDIR:-./results_remaining}"
RSCRIPT="${RSCRIPT:-Rscript}"
# Default: all three remaining models. Use rf,xgboost to skip MoE.
MODELS="${MODELS:-rf,xgboost,moe}"
N_REPEATS="${N_REPEATS:-10}"
TRAIN_SIZES="${TRAIN_SIZES:-20,30,40,50,60}"
N_JOBS="${N_JOBS:-8}"
MOE_TRIALS="${MOE_TRIALS:-10}"
MOE_HPO_EPOCHS="${MOE_HPO_EPOCHS:-40}"

mkdir -p "${OUTDIR}"

python "${SCRIPT_DIR}/run_learning_curve_remaining.py" \
    --metadata "${METADATA}" \
    --counts "${COUNTS}" \
    --aging-genes "${AGING_GENES}" \
    --outdir "${OUTDIR}" \
    --models "${MODELS}" \
    --mode learning_curve \
    --n-repeats "${N_REPEATS}" \
    --train-sizes "${TRAIN_SIZES}" \
    --test-size 0.20 \
    --seed 66 \
    --n-jobs "${N_JOBS}" \
    --rscript "${RSCRIPT}" \
    --moe-trials "${MOE_TRIALS}" \
    --moe-hpo-epochs "${MOE_HPO_EPOCHS}" \
    --resume \
    "$@"

echo "Finished remaining models. See ${OUTDIR}"
