
# Thesis experiments

This project uses preprocessed TSV files in `data/` as experiment inputs. The Zeek/PCAP-to-TSV pipeline is documented below and is not required when running model experiments.

## Project structure

- `src/thesis/`: reusable experiment code
  - `data/`: loading, label filtering, feature definitions, preprocessing
  - `models/`: MLP and CNN-LSTM model definitions
  - `properties/`: DL2 property constraints and attack specifications
  - `training/`: baseline and property-training loops
  - `results/`: metrics and plotting helpers
  - `experiments/`: end-to-end baseline and property runners
- `configs/experiments/`: YAML configs for baseline and property experiments
- `scripts/slurm/`: scripts for running experiments on a SLURM cluster
- `outputs/runs/`: generated models, metrics, plots, configs, and histories

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

## Run a property experiment

```bash
python -m thesis.cli run properties --config configs/experiments/properties_multiclass.yaml
```

## Run a baseline experiment

```bash
python -m thesis.cli run baseline --config configs/experiments/baseline_multiclass.yaml
```

## Override config values

```bash
python -m thesis.cli run properties \
  --config configs/experiments/properties_multiclass.yaml \
  --set properties.logic=dl2 \
  --set model.epochs=5 \
  --set properties.lambda_dos=0.5 \
  --set properties.lambda_scan=0.75 \
  --set attack_specs.portscan.min_uniq_dst_ports=20
```

Property logic and attack thresholds are defined in the property experiment configs. Edit the `properties.logic` field to switch logic, and edit `attack_specs` to change the rule thresholds used by the property constraints.

The property model does not use `orig_byte_rate`, `orig_pkt_rate`, or `pkts_per_port` as input features. The DoS rule still supports `mal_byte_rate_min` and `mal_pkt_rate_min`, and the Portscan rule still supports `max_pkts_per_port`; these values are calculated during property evaluation from adversarial packet/byte fields and window context instead of being perturbed directly by PGD.

Immutable context fields can be frozen during PGD:

```yaml
properties:
  frozen_features:
    - valid_tcp_handshake
    - valid_http_conn
    - time_elapsed
```

Property preconditions are also config-driven:

```yaml
preconditions:
  default:
    name: GlobalBounds
    params:
      lower_bound: 0.0
      upper_bound: 1.0

  portscan:
    name: EpsilonBall
    params:
      epsilon: 0.05
```

Precondition names are resolved from `property_driven_ml.constraints.preconditions`.

## Generate a property grid

Edit `configs/sweeps/property_grid.yaml`, then generate the command file:

```bash
python scripts/make_property_grid.py --sweep configs/sweeps/property_grid.yaml
```

This writes one command per parameter combination to `configs/sweeps/property_grid.txt`.

## Run on SLURM

Single experiment:

```bash
sbatch scripts/slurm/run_experiment.sbatch configs/experiments/properties_multiclass.yaml properties
```

Sweep:

```bash
sbatch scripts/slurm/run_array.sbatch configs/sweeps/property_grid.txt
```

Each run saves its config, model, metrics, classification reports, confusion matrices, and training history under `outputs/runs/`.

## Final experiment matrix

The main thesis comparison uses engineered features only:

- Random forest baseline
- MLP baseline
- Property-trained MLP

The final run list is in:

```bash
configs/sweeps/final_experiments.txt
```

It covers binary property-attacks-only, binary all-attacks, and 3-class concrete attack classification.

# Generate dataset from PCAP files
## CICIDS2017 and CICDDOS2019
For the combined dataset we process the pcaps individually before merging them together.
### CICIDS2017
To generate the tsv dataset from the PCAP files, run the following command in the zeek_logs folder:

```bash
zeek -C -r ./../pcaps/Wednesday-workingHours.pcap 
```
Afterward convert it to a tsv file using the following commands:
```bash
zeek-cut -m < conn.log >> conn.tsv
```
Then all that is left is to run all cells in the [`labelling`](data/CICIDS2017/label_cicids2017.ipynb) Jupyter Notebook file

### CICDDOS2019
Generate zeek flows by running the script: 
[process_pcaps_cicddos2019.sh](data/CICDDOS2019/process_pcaps_cicddos2019.sh).

Make sure the script is executable by running:

```bash
chmod -X process_pcaps_cicddos2019.sh
```

Then all that is left is to run all cells in the [`labelling`](data/CICDDOS2019/label_cicddos2019.ipynb) Jupyter Notebook file

## CICIoT2023
Since we have more direct naming from filenames, here we just need to run the script provided in [label_ciciot2023.sh](data/CICIoT2023/label_ciciot2023.sh).

Make sure to make it executable at first by running
```bash
chmod -X label_ciciot2023.sh
```
Then run it pointing to individual pcaps or all files ending with .pcap:

```bash
./label_ciciot2023.sh pcaps/file1.pcap pcaps/file2.pcap
./label_ciciot2023.sh pcaps/*.pcap
```
