# Canine blood transcriptomic aging clocks

Code for building and evaluating canine blood RNA-seq aging clocks from 143 samples across seven studies (89 normal/control samples for training). Six models are compared: LASSO, Elastic Net, linear-kernel SVR, Random Forest, XGBoost, and a mixture-of-experts (MoE) network.

Fold-wise `filterByExpr` + TMM + logCPM and nested hyperparameter optimization are confined to each training partition to avoid leakage.

## Repository layout

| Folder | Contents |
|---|---|
| `preprocessing/` | SeqKit QC, SRA conversion, HISAT2, featureCounts |
| `model_training/` | Six clocks, 5-fold / DataSet-LOO / group-LOO plots |
| `body_size/` | Leave-one-size-out, size-augmented clocks, consensus-gene expression by size |
| `cell_composition/` | CIBERSORTx reference and leukocyte-composition analyses |
| `covariate_adjustment/` | Post-hoc OLS of chronological age on OOF predicted age |
| `aging_genes/` | Top-K consensus, stability selection, GO enrichment, ComBat visualization matrix |
| `intervention/` | Within-cohort Welch tests and covariate-adjusted comparisons |
| `ortholog/` | Dog–human–mouse protein extraction and MAFFT alignment |
| `learning_curve/` | Leak-free learning curves (test MAE / R² vs. training size) for all six clocks |

## Input files (place next to the training scripts)

- `model_training/total_metadata.txt`
- `model_training/total_count_sort_ortholog.txt`
- `model_training/aging_gene_list.csv` (MoE knowledge regularization)
- `body_size/total_metadata_weight.txt` (DAP size class)

## Run the clocks

```bash
cd model_training
python LASSO.py
python ElasticNet.py
python SVR.py
python random_forest.py
python XGBoost.py
python MoE.py
```

Each script writes `aging_clock_results_*_pipeline_foldwise_TMM_logCPM/` with `5fold_cv/`, `DataSet_loo/`, `group_loo/`, and `HPO/` outputs. Then:

```bash
python plot_total_5fold.py
python plot_total_Dataset_loo.py
python plot_total_group_loo.py
```

Downstream scripts in the other folders read these result directories (or the HPO gene rankings) and should be run from their own folder.

## Learning curves

For each of 10 repeats, a stratified ~20% test set is held out from the 89 normal samples, and nested training subsets (n = 20, 30, 40, 50, 60 and the full pool) are drawn from the remainder. `filterByExpr` + TMM + logCPM and hyperparameter tuning are fit on each training subset only. Test MAE is fitted with `MAE ≈ a + b / n` to estimate the error floor. Input files are read from `../model_training/` by default.

```bash
cd learning_curve
python run_lc_lasso.py        # also: run_lc_elasticnet.py, run_lc_svr.py,
                              #       run_lc_rf.py, run_lc_xgboost.py, run_lc_moe.py
python plot_all_learning_curves.py
```

Each `run_lc_<model>.py` writes `results_<model>/` (`run_records.csv`, `summary_by_size.csv`, `saturation_fit.csv`, report and figures); `--resume` skips finished cells. `plot_all_learning_curves.py` merges them into `results_all/`. `run_learning_curve.sh` (linear clocks) and `run_learning_curve_remaining.sh` (RF, XGBoost, MoE) are batch wrappers for a server.

## License

See `LICENSE`.
