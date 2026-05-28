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

`data.cross_eval_path` is optional for training runs. Omit it, leave it blank, or set it to `[]` to skip training-time cross-dataset evaluation. When provided, it must be a list. Use one entry for a single external dataset, or multiple entries when the same trained model should be evaluated on several external datasets:

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

To skip cross-evaluation from the CLI, pass an empty list:

```bash
--set data.cross_eval_path=[]
```

## Config parameters

The config files are the source of truth for experiment settings, and the same keys can be overridden with `--set`.

Common experiment keys:

- `experiment.name`: run name used in the output directory.
- `experiment.task`: task label such as `binary` or `multiclass`.
- `experiment.seed`: random seed.
- `output.root`: root directory for generated run outputs.

Data keys:

- `data.train_path`: preprocessed TSV used for training, validation, and test splits.
- `data.cross_eval_path`: optional list of preprocessed TSVs for training-time cross-dataset evaluation; omit, leave blank, or set `[]` to skip.
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
  - `valid_packet_size_individual_min`: minimum original bytes per original packet for a flow to count as having valid DoS packet sizing.
  - `valid_pkt_size_total_min`: minimum total original bytes for a flow to count as having valid DoS traffic volume.
  - `mal_time_elapsed_min`: minimum allowed elapsed time between related flows for the DoS HTTP flood condition. The current tiny positive value excludes first-in-pair rows whose `time_elapsed` defaults to `0.0`.
  - `mal_time_elapsed_max`: maximum allowed elapsed time between related flows for the DoS HTTP flood condition.
- `attack_specs.portscan.*`: thresholds for the portscan property.
  - `mal_uniq_dst_ports_min`: minimum number of unique destination ports required for the portscan condition.
  - `mal_pkts_per_port_max`: maximum packets per destination port for the portscan condition.
  - `mal_scan_duration_max`: maximum scan duration for the portscan condition.
  - `mal_fail_ratio_min`: minimum failed-connection ratio for the portscan condition.
- `preconditions.*`: property precondition definitions. The current configs use `GlobalBounds`.
  - `default.name`: precondition class used when no property-specific precondition is configured.
  - `default.params.lower_bound`: lower bound for generated adversarial/property-search inputs.
  - `default.params.upper_bound`: upper bound for generated adversarial/property-search inputs.
  - `dos_http_flood` and `portscan`: optional property-specific precondition entries; when omitted, the property uses `preconditions.default`.

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

Use the evaluation command to load a saved model and run post-hoc cross-dataset evaluation without retraining:

```bash
python -m thesis.cli evaluate --model outputs/path/to/run/model.joblib
python -m thesis.cli evaluate --model outputs/path/to/run/model.joblib --cross-data data/ciciot2023_preprocessed_good.tsv
python -m thesis.cli evaluate --model outputs/path/to/run/model.joblib --cross-data data/ciciot2023_preprocessed_good.tsv data/ciciot2023_preprocessed_bad.tsv
python -m thesis.cli evaluate-tree --root outputs/high_lambda --cross-data data/ciciot2023_preprocessed_good.tsv data/ciciot2023_preprocessed_bad.tsv
```

- `--model`: required path to an existing `model.joblib` file.
- `--cross-data`: optional space-separated list of preprocessed TSV files to evaluate. If omitted, the command uses the saved run config's `data.cross_eval_path`.
- `evaluate-tree --root`: recursively evaluates every `model.joblib` below the root directory. `configs/properties/eval_high_lambda_models.txt` runs this for all models below `outputs/high_lambda`.

On SLURM, submit the matching batch script. It runs one model per array task and caps the array at 10 concurrent evaluations:

```bash
sbatch scripts/slurm/eval_high_lambda_models.sbatch
```

This behaves like training-time cross evaluation: each dataset is labeled and feature-engineered with the saved run config, transformed with the saved clipping bounds and scaler from the model payload, then evaluated with the saved model. Results are written below `<model_parent>/cross_eval/<dataset_stem>/`.

Older baseline models may need to be retrained before post-hoc evaluation if their saved `model.joblib` payload does not contain `scale_cols`, `clip_lower`, and `clip_upper`.

## Outputs

Each run writes to:

```text
<output.root>/<timestamp>_<experiment.name>/
```

The saved run directory includes:

- `config.yaml`: the effective config after command-line overrides.
- `model.joblib`: trained model and preprocessing artifacts. New runs include the scaler, scale columns, and clipping bounds needed for post-hoc cross-dataset evaluation.
- `metrics.json`: run-level metric summary.
- `test/metrics.json`: test metrics.
- `test/classification_report.csv`: per-class precision, recall, F1, support, and per-label accuracy.
- `test/confusion_matrix.csv`: test confusion matrix.
- `test/confusion_matrix.png`: plotted evaluation summary.
- `cross_eval/`: equivalent outputs for cross-dataset evaluation. Training-time single-dataset cross evaluation may write directly to `cross_eval/`; multiple datasets and post-hoc evaluation write below `cross_eval/<dataset_stem>/`.
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


## Marabou verification

Formal verification of trained MLP models is done with [Marabou] via three Jupyter notebooks. Random forest models cannot be verified because they cannot be exported to ONNX.

### Install Marabou

Marabou must be built from source. Clone the repository and follow the build instructions in its README:

```bash
git clone https://github.com/NeuralNetworkVerification/Marabou.git
cd Marabou
mkdir build && cd build
cmake ..
cmake --build .
```

The Python bindings (`maraboupy`) are part of the build output and do not require a separate `pip install`. After building, update the two `sys.path.insert` lines at the top of each verification notebook to point to your local Marabou clone and its `build/` directory:

```python
sys.path.insert(0, "/path/to/your/Marabou")
sys.path.insert(0, "/path/to/your/Marabou/build")
```

### Notebook overview
| `verify_baselines.ipynb` | Verifies all baseline MLP models (4 exercises × 5 runs = 20 models) |
| `verify_models.ipynb` | Verifies all property-driven MLP models (9 lambda × 4 exercises × 5 logics = 180 models) |
| `marabou.ipynb` | Single-model exploratory verification; useful for inspecting individual counterexamples |

Each notebook verifies two properties per model: `portscan` and `dos_http_flood`. For each property the solver checks whether there exists an input within the property's feature bounds where the attack class does **not** win — i.e. a counterexample to the property. The result per model/property is `UNSAT` (property holds), `SAT` (counterexample found), or `TIMEOUT` (solver exceeded the 120-second limit per rival class).

### Expected model directory layout

`verify_baselines.ipynb` expects:

```text
baselines/{ex}/mlp/{run}/model.joblib
```

`verify_models.ipynb` expects:

```text
final_models/lambda_{dos}_{scan}/{ex}/properties/{logic}/both/{run}/model.joblib
```

Place the notebooks at the root of the directory that contains `baselines/` and `final_models/`, or adjust the `Path(...)` expressions at the top of each notebook accordingly.

### Run verification

Open either batch notebook in Jupyter and run all cells:

```bash
jupyter notebook verify_baselines.ipynb
jupyter notebook verify_models.ipynb
```

Progress is printed after each model, including per-property CACC scores (fraction of property-region samples classified correctly), overall SAT/UNSAT/TIMEOUT status, and an ETA estimate.

### Outputs
Each batch notebook writes a timestamped JSON file to the current working directory:

```text
baseline_verification_results_YYYYMMDD_HHMMSS.json
verification_results_YYYYMMDD_HHMMSS.json
```

Each file contains three top-level keys:

- `verification`: per-model, per-property SAT/UNSAT/TIMEOUT results and per-rival-class breakdowns.
- `cacc`: constraint accuracy scores (fraction of 200 uniformly sampled property-region inputs classified as the attack class).
- `timing`: solver time in seconds per model/property.

Summary tables grouped by exercise, logic, and lambda value are printed at the end of `verify_models.ipynb`.