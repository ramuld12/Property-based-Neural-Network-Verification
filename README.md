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
data/ciciot2023_preprocessed.tsv
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
  --set data.cross_eval_path=data/ciciot2023_preprocessed.tsv \
  --set properties.logic=dl2
```

## Experiment definitions

- `ex1`: binary `BENIGN` vs `ATTACK`, where `ATTACK` contains `DOS_HTTP_FLOOD` and `PORTSCAN`.
- `ex2`: binary `BENIGN` vs `ATTACK`, where `ATTACK` contains `XSS`, `SQL_INJECTION`, and `BRUTE_FORCE`.
- `ex3`: three-class classification with `BENIGN`, `DOS_HTTP_FLOOD`, and `PORTSCAN`.
- `ex4`: mixed specific/generic classification with `BENIGN`, `DOS_HTTP_FLOOD`, `PORTSCAN`, and generic `ATTACK`.

Baseline configs exist for random forest and MLP models. Property configs currently use the MLP model with property constraints for DoS HTTP flood and portscan behavior.

## Recreate results

The reproducibility command grids contain the full experiment sweeps with explicit seeds, train/cross-evaluation directions, and output roots.

Baseline grids:

```text
configs/baseline/ex1_baseline_commands.txt
configs/baseline/ex2_baseline_commands.txt
configs/baseline/ex3_baseline_commands.txt
configs/baseline/ex4_baseline_commands.txt
```

Each baseline grid contains 40 commands: random forest and MLP, two train/cross-evaluation directions, and 10 seeds per setting.

Property grids:

```text
configs/properties/ex1_prop_commands.txt
configs/properties/ex2_prop_commands.txt
configs/properties/ex3_prop_commands.txt
configs/properties/ex4_prop_commands.txt
```

Each property grid contains 30 commands: five logics, two train/cross-evaluation directions, and three seeds per setting.


To recreate all baseline results, run the four baseline command grids. To recreate all property-driven results, run the four property command grids. These runs can take a long time because each command trains a model and writes a separate timestamped output directory.

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
- `cross_eval/`: equivalent outputs for the cross-dataset evaluation when `data.cross_eval_path` is configured.
- `training_history.csv`: epoch history for MLP and property-driven runs.

## Regenerate preprocessed TSVs

The generate the preprocessed datasets the process differs a bit for each one:

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

Download the CICIoT2023 PCAP files from CIC, then run the processing script in `data/CICIoT2023/` against one or more PCAP files:

```bash
./process_pcaps.sh pcaps/file1.pcap pcaps/file2.pcap
./process_pcaps.sh pcaps/*.pcap
```

If needed, make the script executable first:

```bash
chmod +x process_pcaps.sh
```

Then run the labeling notebook at `data/CICIoT2023/label_ciciot2023.ipynb`.
