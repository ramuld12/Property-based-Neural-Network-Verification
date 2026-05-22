# Thesis experiments

This repository runs baseline and property-driven intrusion-detection experiments from preprocessed TSV files. The model pipeline starts from the files in `data/*_preprocessed.tsv`; regenerating those TSVs from PCAPs is optional and documented at the end.

## Project structure

- `src/thesis/`: experiment code for data loading, preprocessing, training, evaluation, and plotting.
- `configs/baseline/`: baseline model configs plus reproducibility command grids.
- `configs/properties/`: property-driven model configs plus reproducibility command grids.
- `data/*_preprocessed.tsv`: required experiment inputs.
- `outputs/`: generated run directories, models, metrics, plots, and saved configs.

## Setup

Install PyTorch first using the CUDA or CPU wheel appropriate for the machine, then install the remaining dependencies and the local package:

```bash
pip install -r requirements.txt
pip install -e .
```


The project expects these preprocessed input files to exist:

```text
data/cicids2017_preprocessed.tsv
data/ciciot2023_preprocessed_good.tsv
data/ciciot2023_preprocessed_bad.tsv
```

Small versions are also present for quick checks, but the experiment command grids use the full TSV files.
To generate the datasets, view the section about it below.

## Run one experiment

Run a single baseline experiment:

```bash
python -m thesis.cli run baseline --config configs/baseline/ex1_mlp.yaml
```

Run a single property-driven experiment:

```bash
python -m thesis.cli run properties --config configs/properties/ex1_prop_mlp.yaml
```

Any config value can be overridden from the command line with `--set key.path=value`:

```bash
python -m thesis.cli run properties \
  --config configs/properties/ex1_prop_mlp.yaml \
  --set experiment.name=run1 \
  --set experiment.seed=1 \
  --set output.root=outputs/ex1/properties/dl2/mlp_cicids2017_to_ciciot2023 \
  --set data.train_path=data/cicids2017_preprocessed.tsv \
  --set data.cross_eval_path=[data/ciciot2023_preprocessed_good.tsv] \
  --set properties.logic=dl2
```

`data.cross_eval_path` is always a list. Use one entry for a single external dataset, or multiple entries when the same trained model should be evaluated on several external datasets:

```yaml
data:
  cross_eval_path:
    - data/ciciot2023_preprocessed_good.tsv
    - data/ciciot2023_preprocessed_bad.tsv
```

CLI overrides should also pass a list:

```bash
--set data.cross_eval_path=[data/ciciot2023_preprocessed_good.tsv]
```

## Config parameters

The config files are the source of truth for experiment settings, and the same keys can be overridden with `--set`.

Common experiment keys:

- `experiment.name`: run name used in the output directory.
- `experiment.method`: configured method, usually matching the CLI method.
- `experiment.task`: task label such as `binary` or `multiclass`.
- `experiment.seed`: random seed.
- `output.root`: root directory for generated run outputs.

Data keys:

- `data.train_path`: preprocessed TSV used for training, validation, and test splits.
- `data.cross_eval_path`: list of preprocessed TSVs for cross-dataset evaluation.
- `data.labels`: class labels to train and evaluate.
- `data.attack_source_labels`: source attack labels collapsed into generic `ATTACK` for binary or mixed tasks.
- `data.test_size`: held-out test fraction.
- `data.val_size`: validation fraction taken from the training split.
- `data.windows_seconds`: window size used by the dataset loader.

Model keys:

- `model.type`: `random_forest` or `mlp` for baseline runs; property-driven runs use `mlp`.
- `model.n_estimators`: number of trees for `random_forest`.
- `model.n_jobs`: parallel workers for `random_forest`.
- `model.batch_size`: batch size for `mlp` training and prediction.
- `model.learning_rate`: Adam learning rate for `mlp`.
- `model.epochs`: maximum training epochs for `mlp`.
- `model.patience`: early-stopping patience.
- `model.min_epochs`: minimum epochs before early stopping in property-driven training.
- `model.min_delta`: minimum validation improvement for early stopping.

Property-driven keys:

- `properties.logic`: one of `goedel`, `boolean`, `dl2`, `lukasiewicz`, `reichenbach`, `yager`, or `stl`.
- `properties.lambda_dos`: weight for the DoS_HTTP_flood constraint loss.
- `properties.lambda_scan`: weight for the PORTSCAN constraint loss.
- `properties.pgd_steps`: PGD steps for adversarial constraint search.
- `properties.pgd_restarts`: PGD restarts for adversarial constraint search.
- `properties.pgd_step_size`: PGD step size.
- `attack_specs.dos_http_flood.*`: thresholds for the DoS_HTTP_flood property.
- `attack_specs.portscan.*`: thresholds for the portscan property.
- `preconditions.*`: property precondition definitions. The current configs use `GlobalBounds`.

## Experiment definitions

- `ex1`: binary `BENIGN` vs `ATTACK`, where `ATTACK` contains `DOS_HTTP_FLOOD` and `PORTSCAN`.
- `ex2`: binary `BENIGN` vs `ATTACK`, where `ATTACK` contains `DOS_HTTP_FLOOD`, `PORTSCAN`, `XSS`, `SQL_INJECTION`, and `BRUTE_FORCE`.
- `ex3`: three-class classification with `BENIGN`, `DOS_HTTP_FLOOD`, and `PORTSCAN`.
- `ex4`: mixed specific/generic classification with `BENIGN`, `DOS_HTTP_FLOOD`, `PORTSCAN`, and generic `ATTACK` consisting of `XSS`, `SQL_INJECTION`, and `BRUTE_FORCE`.

Baseline configs exist for random forest and MLP models.

## Recreate results

The reproducibility command grids contain the full experiment sweeps with explicit seeds, train/cross-evaluation directions, and output roots.

Baseline grids:

```text
configs/baseline/ex1_baseline_commands.txt
configs/baseline/ex2_baseline_commands.txt
configs/baseline/ex3_baseline_commands.txt
configs/baseline/ex4_baseline_commands.txt
```

Property grids:

```text
configs/properties/ex1_prop.txt
configs/properties/ex2_prop.txt
configs/properties/ex3_prop.txt
configs/properties/ex4_prop.txt
```


To recreate all baseline results, run the four baseline command grids. To recreate all property-driven results, run the four property command grids. These runs can take a long time because each command trains a model and writes a separate timestamped output directory.

## Evaluate an existing run

The CLI also exposes an evaluation command:

```bash
python -m thesis.cli evaluate --run outputs/path/to/run
python -m thesis.cli evaluate --run outputs/path/to/run --cross-data configs/path/to/cross_data.yaml
```

- `--run`: required path to an existing run directory.
- `--cross-data`: optional path for a requested cross-data config.

The current implementation only prints the requested paths and does not recompute metrics yet.

## Outputs

Each run writes to:

```text
<output.root>/<timestamp>_<experiment.name>/
```

The saved run directory includes:

- `config.yaml`: the effective config after command-line overrides.
- `model.joblib`: trained model and preprocessing artifacts.
- `metrics.json`: run-level metric summary.
- `test/metrics.json`: test metrics.
- `test/classification_report.csv`: per-class precision, recall, F1, support, and per-label accuracy.
- `test/confusion_matrix.csv`: test confusion matrix.
- `test/confusion_matrix.png`: plotted evaluation summary.
- `cross_eval/`: equivalent outputs for the cross-dataset evaluation. A one-item `data.cross_eval_path` list writes to `cross_eval/`; multiple entries write below `cross_eval/<dataset_stem>/`.
- `training_history.csv`: epoch history for MLP and property-driven runs.


## Regenerate preprocessed TSVs

To generate the preprocessed datasets, run the cells in `data/zeek_preprocessing_pipeline.ipynb` after having processed the PCAPs. The process differs a bit for each one:

### CICIDS2017

Create Zeek logs for each PCAP/day, for example:

```bash
zeek -C -r ./../pcaps/Wednesday-workingHours.pcap
```

Convert the Zeek connection log to TSV:

```bash
zeek-cut -m < conn.log >> conn.tsv
```

Then run the labeling notebook at `data/CICIDS2017/label_cicids2017.ipynb`.

### CICIoT2023

Download the CICIoT2023 PCAP files from CIC, then run the scripts in `label_ciciot2023_bad.sh` and `process_pcaps.sh` in `data/CICIoT2023/` against one or more PCAP files:

```bash
./process_pcaps.sh pcaps/file1.pcap pcaps/file2.pcap
./process_pcaps.sh pcaps/*.pcap
```

If needed, make the script executable first:

```bash
chmod +x process_pcaps.sh
```

For the "Bad" dataset, processing is finished.

For the "Good" dataset, run the labeling notebook at `data/CICIoT2023/label_ciciot2023.ipynb`.
