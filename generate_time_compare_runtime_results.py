#!/usr/bin/env python3
import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

cache_root = Path(tempfile.gettempdir()) / "codex_matplotlib_cache"
cache_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


EXPERIMENTS = ["ex1", "ex2", "ex3", "ex4"]
MODEL_ORDER = ["rf", "baseline", "goedel", "lukasiewicz", "reichenbach", "yager", "dl2"]
MODEL_LABELS = {
    "rf": "Random Forest",
    "baseline": "Baseline MLP",
    "goedel": "Gödel",
    "lukasiewicz": "Łukasiewicz",
    "reichenbach": "Reichenbach",
    "yager": "Yager",
    "dl2": "DL2",
}
EXPERIMENT_COLORS = {
    "ex1": "#4e79a7",
    "ex2": "#f28e2b",
    "ex3": "#59a14f",
    "ex4": "#e15759",
}


@dataclass
class TimeCompareRecord:
    root: Path
    model_size: str
    experiment: str
    model_key: str
    model_label: str
    logic: str
    lambda_dos: str
    lambda_scan: str
    run_count: int
    run_seconds: list[float]
    mean_seconds: float
    std_seconds: float
    source_runs: list[Path]

    @property
    def mean_minutes(self):
        return self.mean_seconds / 60

    @property
    def lambda_label(self):
        if self.lambda_dos == "--" and self.lambda_scan == "--":
            return "--"
        return f"({self.lambda_dos},{self.lambda_scan})"


def read_yaml(path):
    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_rf_seconds(run):
    metrics_path = run / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = json.loads(metrics_path.read_text())
    fit_seconds = metrics.get("runtime", {}).get("fit_seconds")
    return float(fit_seconds) if fit_seconds is not None else None


def read_mlp_epoch_seconds(run):
    history_path = run / "training_history.csv"
    if not history_path.exists():
        return None

    with history_path.open(newline="") as f:
        reader = csv.DictReader(f)
        values = [
            float(row["epoch_seconds"])
            for row in reader
            if row.get("epoch_seconds") not in (None, "")
        ]
    return mean(values) if values else None


def model_size_from_root(root):
    name = root.name.lower()
    if "43k" in name:
        return "43k"
    if "186k" in name:
        return "186k"
    return root.name


def classify_run(run, config):
    model_type = str(config.get("model", {}).get("type", ""))
    properties = config.get("properties")

    if model_type == "random_forest":
        return "rf", "Random Forest", "--", "--", "--"

    if isinstance(properties, dict) and properties.get("logic"):
        logic = str(properties.get("logic"))
        lambda_dos = str(properties.get("lambda_dos", "--"))
        lambda_scan = str(properties.get("lambda_scan", "--"))
        return logic, MODEL_LABELS.get(logic, logic), logic, lambda_dos, lambda_scan

    return "baseline", "Baseline MLP", "--", "--", "--"


def collect_records(output_roots):
    grouped = {}
    skipped = []

    for root in output_roots:
        root = root.resolve()
        model_size = model_size_from_root(root)
        if not root.exists():
            skipped.append((root, "missing output root"))
            continue

        for experiment in EXPERIMENTS:
            experiment_root = root / experiment
            if not experiment_root.exists():
                skipped.append((experiment_root, "missing experiment folder"))
                continue

            for run in sorted(experiment_root.glob("*_run1")):
                if not run.is_dir():
                    continue

                config_path = run / "config.yaml"
                if not config_path.exists():
                    skipped.append((run, "missing config.yaml"))
                    continue

                config = read_yaml(config_path)
                model_key, model_label, logic, lambda_dos, lambda_scan = classify_run(run, config)
                runtime = read_rf_seconds(run) if model_key == "rf" else read_mlp_epoch_seconds(run)
                if runtime is None:
                    skipped.append((run, "missing runtime value"))
                    continue

                key = (
                    root,
                    model_size,
                    experiment,
                    model_key,
                    model_label,
                    logic,
                    lambda_dos,
                    lambda_scan,
                )
                grouped.setdefault(key, {"seconds": [], "runs": []})
                grouped[key]["seconds"].append(runtime)
                grouped[key]["runs"].append(run)

    records = []
    for (
        root,
        model_size,
        experiment,
        model_key,
        model_label,
        logic,
        lambda_dos,
        lambda_scan,
    ), values in grouped.items():
        seconds = values["seconds"]
        records.append(
            TimeCompareRecord(
                root=root,
                model_size=model_size,
                experiment=experiment,
                model_key=model_key,
                model_label=model_label,
                logic=logic,
                lambda_dos=lambda_dos,
                lambda_scan=lambda_scan,
                run_count=len(seconds),
                run_seconds=seconds,
                mean_seconds=mean(seconds),
                std_seconds=stdev(seconds) if len(seconds) > 1 else 0.0,
                source_runs=values["runs"],
            )
        )

    records.sort(
        key=lambda record: (
            str(record.root),
            EXPERIMENTS.index(record.experiment),
            MODEL_ORDER.index(record.model_key)
            if record.model_key in MODEL_ORDER
            else len(MODEL_ORDER),
        )
    )
    return records, skipped


def print_summary(records, skipped):
    current_root = None
    current_experiment = None

    for record in records:
        if record.root != current_root:
            current_root = record.root
            current_experiment = None
            print(f"\n===== {record.root} ({record.model_size}) =====")

        if record.experiment != current_experiment:
            current_experiment = record.experiment
            print(f"\n{record.experiment.upper()}")
            print(
                f"{'Model':<18} {'Lambda':>10} {'Runs':>5} "
                f"{'Mean sec':>10} {'Std sec':>9} {'Mean min':>10}"
            )
            print("-" * 68)

        print(
            f"{record.model_label:<18} {record.lambda_label:>10} "
            f"{record.run_count:>5} {record.mean_seconds:>10.2f} "
            f"{record.std_seconds:>9.2f} {record.mean_minutes:>10.2f}"
        )

    if skipped:
        print("\nSkipped paths:")
        for path, reason in skipped:
            print(f"- {path}: {reason}")


def write_csv(records, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "root",
                "model_size",
                "experiment",
                "model",
                "logic",
                "lambda_dos",
                "lambda_scan",
                "run_count",
                "mean_seconds",
                "mean_minutes",
                "source_run",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "root": str(record.root),
                    "model_size": record.model_size,
                    "experiment": record.experiment,
                    "model": record.model_label,
                    "logic": record.logic,
                    "lambda_dos": record.lambda_dos,
                    "lambda_scan": record.lambda_scan,
                    "run_count": record.run_count,
                    "mean_seconds": f"{record.mean_seconds:.6f}",
                    "mean_minutes": f"{record.mean_minutes:.6f}",
                    "source_run": ";".join(str(run) for run in record.source_runs),
                }
            )


def plot_root_records(records, path, model_size):
    by_key = {(record.experiment, record.model_key): record for record in records}
    group_centers = list(range(len(MODEL_ORDER)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    offsets = [(-1.5 + index) * width for index in range(len(EXPERIMENTS))]
    for center, key in zip(group_centers, MODEL_ORDER):
        for experiment, offset in zip(EXPERIMENTS, offsets):
            record = by_key.get((experiment, key))
            if record is None:
                continue
            ax.bar(
                center + offset,
                record.mean_seconds,
                yerr=record.std_seconds,
                width=width,
                color=EXPERIMENT_COLORS[experiment],
                edgecolor="#2f3b45",
                linewidth=0.7,
                capsize=3,
                label=experiment.upper() if center == 0 else None,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [MODEL_LABELS[key] for key in MODEL_ORDER],
        rotation=18,
        ha="right",
    )
    ax.set_ylabel("Average epoch/fit time (s)")
    ax.set_title(
        "Runtime for best lambda combinations across experiments "
        f"for model with {model_size} parameters"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Experiment", frameon=True, loc="upper left", ncols=1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_records(records, path):
    if not records:
        print("No records to plot.")
        return []

    roots = []
    for record in records:
        if record.root not in roots:
            roots.append(record.root)

    written_paths = []
    for root in roots:
        root_records = [record for record in records if record.root == root]
        model_size = model_size_from_root(root)
        root_path = (
            path
            if len(roots) == 1
            else path.with_name(f"{path.stem}_{model_size}{path.suffix}")
        )
        plot_root_records(root_records, root_path, model_size)
        written_paths.append(root_path)

    return written_paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate runtime plots for flat time_compare output folders."
    )
    parser.add_argument("output_roots", nargs="+", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--csv",
        type=Path,
        help="Optional CSV output path. Defaults to <out-dir>/time_compare_runtime_summary.csv.",
    )
    args = parser.parse_args()

    records, skipped = collect_records(args.output_roots)
    print_summary(records, skipped)

    csv_path = args.csv or args.out_dir / "time_compare_runtime_summary.csv"
    plot_path = args.out_dir / "time_compare_runtime_by_experiment.png"
    write_csv(records, csv_path)
    plot_paths = plot_records(records, plot_path)

    print(f"\nWrote CSV: {csv_path}")
    for path in plot_paths:
        print(f"Wrote plot: {path}")


if __name__ == "__main__":
    main()
