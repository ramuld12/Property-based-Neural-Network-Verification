# Final Experiments Plan

Use this file as the checklist for the final experiment section.

## Dataset Readiness

- [ ] Verify `data/ciciot2023_preprocessed.tsv` is present and contains the expected labels.
- [ ] Verify `data/cicids2017_preprocessed.tsv` is present and contains the expected labels.
- [ ] Generate a richer CICIDS2017 dataset with more attack types, or decide to use CICIDS2018 instead.
- [ ] Confirm label names match the final experiment configs.
- [ ] Confirm engineered features exist in all final TSVs.
- [ ] Confirm the final TSVs contain the columns needed for property evaluation: `duration`, `orig_bytes`, `orig_pkts`, `uniq_dst_ports`, `scan_duration`, `fail_ratio`, `window_id`, and `id.orig_h`.

## Cluster Setup

- [ ] Use the cluster PyTorch environment instead of installing PyTorch wheels manually.
- [ ] Install project dependencies once, outside SLURM jobs.
- [ ] Confirm `python -c "import torch; print(torch.cuda.is_available())"` works.
- [ ] Confirm `python -c "import property_driven_ml; print('ok')"` works.
- [ ] Confirm `python -c "from thesis.experiments.properties import run_properties; print('ok')"` works.
- [ ] Confirm `GlobalBounds` is available from `property_driven_ml.constraints.preconditions`.
- [ ] Remove dependency installation from SLURM scripts before launching final grids.

## PGD Stability

- [ ] Run `configs/sweeps/property_pgd_stability_grid.yaml`.
- [ ] Generate the PGD stability command file.
- [ ] Set the SLURM array range to match the generated command count.
- [ ] Submit the PGD stability grid.
- [ ] Compare `attack_macro_f1`, `adv_dos_sat`, `adv_scan_sat`, `adv_dos_loss`, and `adv_scan_loss`.
- [ ] Select final PGD settings.
- [ ] Update final property configs with the selected PGD settings.

## Final Experiment Matrix

Final run manifest:

```bash
configs/sweeps/final_experiments.txt
```

### Binary Property Attacks

- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_binary_rf_engineered.yaml`
- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_binary_mlp_engineered.yaml`
- [ ] `python -m thesis.cli run properties --config configs/experiments/properties_binary.yaml`

### Binary All Attacks

- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_binary_all_attacks_rf_engineered.yaml`
- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_binary_all_attacks_mlp_engineered.yaml`
- [ ] `python -m thesis.cli run properties --config configs/experiments/properties_binary_all_attacks.yaml`

### 3-Class CICIOT To CICIDS

- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_multiclass_rf_engineered.yaml`
- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_multiclass_mlp_engineered.yaml`
- [ ] `python -m thesis.cli run properties --config configs/experiments/properties_multiclass.yaml`

### 3-Class CICIDS To CICIOT

- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_multiclass_cicids_rf_engineered.yaml`
- [ ] `python -m thesis.cli run baseline --config configs/experiments/baseline_multiclass_cicids_mlp_engineered.yaml`
- [ ] `python -m thesis.cli run properties --config configs/experiments/properties_multiclass_cicids.yaml`

## Multi-Seed Runs

- [ ] Run final configs with seed `1`.
- [ ] Run final configs with seed `2`.
- [ ] Run final configs with seed `3`.
- [ ] Run final configs with seed `4`.
- [ ] Run final configs with seed `5`.
- [ ] Save or record the output directory for every final run.
- [ ] Confirm every final run contains `config.yaml`, `metrics.json`, `training_history.csv` where applicable, `test/`, and `cross_eval/`.

## Result Aggregation

- [ ] Extract metrics from `outputs/runs`.
- [ ] Produce mean/std tables across seeds.
- [ ] Compare RF engineered, MLP engineered, and Property MLP engineered.
- [ ] Check test metrics for every task.
- [ ] Check cross-dataset metrics for every task.
- [ ] Check property metrics for property-trained models: `adv_dos_sat`, `adv_scan_sat`, `adv_dos_loss`, and `adv_scan_loss`.

## Thesis Reporting

- [ ] Write the final experiment setup section.
- [ ] Write the dataset setup and dataset limitation notes.
- [ ] Explain the CICIDS2017 richer-dataset task or the CICIDS2018 fallback.
- [ ] Write final result tables.
- [ ] Discuss whether properties help in binary property-attacks-only classification.
- [ ] Discuss whether properties help in binary all-attacks classification.
- [ ] Discuss whether properties help in 3-class concrete attack classification.
- [ ] Discuss limitations of using properties only for `DOS_HTTP_FLOOD` and `PORTSCAN`.
- [ ] Mention that CNNLSTM was excluded because MLPs are more suitable for Marabou-style verification.
