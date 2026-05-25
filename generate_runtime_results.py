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

from generate_report_results import (
    DATASETS,
    EXPERIMENTS,
    MODELS,
    baseline_folder,
    choose_best_candidate,
    complete_aggregate_runs,
    discover_baseline_roots,
    discover_property_roots,
    lambda_from_root,
    property_folder,
)


@dataclass
class RuntimeRecord:
    model_key: str
    model_label: str
    lambda_label: str
    run_count: int
    run_seconds: list[float]
    mean_seconds: float
    std_seconds: float

    @property
    def mean_minutes(self):
        return self.mean_seconds / 60


@dataclass
class LogicLambdaAverageRecord:
    model_key: str
    model_label: str
    lambda_count: int
    mean_seconds: float
    std_seconds: float

    @property
    def mean_minutes(self):
        return self.mean_seconds / 60


@dataclass
class ExperimentLogicRuntime:
    experiment: str
    records: list[LogicLambdaAverageRecord]


def fmt_seconds(value):
    return f"{value:.2f}"


def fmt_minutes(value):
    return f"{value:.2f}"


def tex_logic(label):
    return label


def runtime_model_label(model_key, model_label):
    return "Baseline MLP" if model_key == "baseline" else model_label


def plot_record_label(record, include_lambda=False):
    label = runtime_model_label(record.model_key, record.model_label)
    if include_lambda and record.lambda_label != "--":
        return f"{label}\n$\\lambda$={record.lambda_label}"
    return label


def compact_lambda(labels):
    labels = sorted({label for label in labels if label})
    if not labels:
        return "--"
    if len(labels) == 1:
        return labels[0]
    return "{" + ", ".join(labels) + "}"


def latex_lambda(label):
    if label.startswith("{") and label.endswith("}"):
        return r"\{" + label[1:-1] + r"\}"
    return label


def run_runtime_seconds(run, model_key):
    if model_key == "rf":
        metrics_path = run / "metrics.json"
        if not metrics_path.exists():
            return None
        metrics = json.loads(metrics_path.read_text())
        fit_seconds = metrics.get("runtime", {}).get("fit_seconds")
        return float(fit_seconds) if fit_seconds is not None else None

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


def dataset_runtime_summary(runs, model_key):
    values = [
        value
        for value in (run_runtime_seconds(run, model_key) for run in runs)
        if value is not None
    ]
    if not values:
        return None
    return values, mean(values)


def combine_dataset_summaries(summaries):
    summaries = [summary for summary in summaries if summary is not None]
    if not summaries:
        return None

    all_values = [value for values, _ in summaries for value in values]
    dataset_means = [dataset_mean for _, dataset_mean in summaries]
    avg_seconds = mean(dataset_means)
    std_seconds = stdev(all_values) if len(all_values) > 1 else 0.0
    return all_values, avg_seconds, std_seconds


def make_record(model_key, model_label, lambda_label, summaries):
    combined = combine_dataset_summaries(summaries)
    if combined is None:
        return None

    all_values, avg_seconds, std_seconds = combined
    return RuntimeRecord(
        model_key=model_key,
        model_label=runtime_model_label(model_key, model_label),
        lambda_label=lambda_label,
        run_count=len(all_values),
        run_seconds=all_values,
        mean_seconds=avg_seconds,
        std_seconds=std_seconds,
    )


def is_run2(run):
    return run.name.endswith("_run2")


def collect_best_lambda_records(output_root, experiment):
    baseline_roots = discover_baseline_roots(output_root)
    property_roots = discover_property_roots(output_root)
    records = []

    for model_key, model_label, is_property in MODELS:
        summaries = []
        lambda_labels = []

        for _, dataset_key in DATASETS:
            if is_property:
                candidates = [
                    (
                        property_folder(root, experiment, dataset_key, model_key),
                        lambda_from_root(root),
                    )
                    for root in property_roots
                ]
            else:
                candidates = [
                    (baseline_folder(root, experiment, dataset_key, model_key), "--")
                    for root in baseline_roots
                ]

            chosen = choose_best_candidate(candidates, dataset_key, run_filter=is_run2)
            if chosen is None:
                continue

            _, lambda_label, runs, _ = chosen
            summary = dataset_runtime_summary(runs, model_key)
            if summary is None:
                continue
            summaries.append(summary)
            lambda_labels.append(lambda_label)

        record = make_record(model_key, model_label, compact_lambda(lambda_labels), summaries)
        if record is not None:
            records.append(record)

    return records


def collect_all_lambda_records(output_root, experiment):
    baseline_roots = discover_baseline_roots(output_root)
    property_roots = discover_property_roots(output_root)
    records = []

    for model_key, model_label, is_property in MODELS:
        if not is_property:
            summaries = []
            for _, dataset_key in DATASETS:
                candidate_summaries = []
                for root in baseline_roots:
                    runs = complete_aggregate_runs(
                        baseline_folder(root, experiment, dataset_key, model_key),
                        dataset_key,
                    )
                    runs = [run for run in runs if is_run2(run)]
                    if runs:
                        candidate_summaries.append(dataset_runtime_summary(runs, model_key))
                summaries.extend(summary for summary in candidate_summaries if summary is not None)

            record = make_record(model_key, model_label, "--", summaries)
            if record is not None:
                records.append(record)
            continue

        for root in property_roots:
            lambda_label = lambda_from_root(root) or "--"
            summaries = []
            for _, dataset_key in DATASETS:
                runs = complete_aggregate_runs(
                    property_folder(root, experiment, dataset_key, model_key),
                    dataset_key,
                )
                runs = [run for run in runs if is_run2(run)]
                if not runs:
                    continue
                summary = dataset_runtime_summary(runs, model_key)
                if summary is not None:
                    summaries.append(summary)

            record = make_record(model_key, model_label, lambda_label, summaries)
            if record is not None:
                records.append(record)

    return records


def print_runtime_log(title, experiment, records):
    print(f"===== {title}: {experiment.upper()} =====")
    print(
        f"{'Logic':<16} {'Lambda':<24} {'Runs':>4} "
        f"{'Mean epoch/fit (s)':>18} {'Std seconds':>12} {'Mean min':>10}"
    )
    print("-" * 88)
    for record in records:
        print(
            f"{record.model_label:<16} {record.lambda_label:<24} "
            f"{record.run_count:>4} "
            f"{fmt_seconds(record.mean_seconds):>18} "
            f"{fmt_seconds(record.std_seconds):>12} "
            f"{fmt_minutes(record.mean_minutes):>10}"
        )
    print()


def print_latex_table(title, label, records):
    print(r"\begin{table}[t]")
    print(r"    \centering")
    print(r"    \setlength{\tabcolsep}{5pt}")
    print(r"    \renewcommand{\arraystretch}{1.08}")
    print(r"    \begin{tabular}{l c r r r}")
    print(r"    \toprule")
    print(
        r"    \textbf{Logic} & \textbf{$\lambda$} & \textbf{Runs} "
        r"& \textbf{Mean epoch/fit (s)} & \textbf{Mean epoch/fit (min)} \\"
    )
    print(r"    \midrule")
    for record in records:
        print(
            rf"    {tex_logic(record.model_label)} & {latex_lambda(record.lambda_label)} "
            rf"& {record.run_count} "
            rf"& {fmt_seconds(record.mean_seconds)} "
            rf"& {fmt_minutes(record.mean_minutes)} \\"
        )
    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(rf"    \caption{{{title}}}")
    print(rf"    \label{{{label}}}")
    print(r"\end{table}")
    print()


def csv_run_seconds(values):
    return ";".join(f"{value:.12g}" for value in values)


def parse_run_seconds(text):
    if not text:
        return []
    return [float(value) for value in text.split(";") if value]


def records_to_csv_rows(experiment, mode, records):
    return [
        {
            "experiment": experiment,
            "mode": mode,
            "model_key": record.model_key,
            "model_label": record.model_label,
            "lambda_label": record.lambda_label,
            "runs": str(record.run_count),
            "mean_seconds": f"{record.mean_seconds:.12g}",
            "std_seconds": f"{record.std_seconds:.12g}",
            "mean_minutes": f"{record.mean_minutes:.12g}",
            "run_seconds": csv_run_seconds(record.run_seconds),
        }
        for record in records
    ]


def record_from_csv_row(row):
    run_seconds = parse_run_seconds(row.get("run_seconds", ""))
    run_count = int(row["runs"])
    if not run_seconds:
        run_seconds = [float(row["mean_seconds"])] * run_count
    return RuntimeRecord(
        model_key=row["model_key"],
        model_label=runtime_model_label(row["model_key"], row["model_label"]),
        lambda_label=row["lambda_label"],
        run_count=run_count,
        run_seconds=run_seconds,
        mean_seconds=float(row["mean_seconds"]),
        std_seconds=float(row["std_seconds"]),
    )


def read_runtime_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def cached_records(rows, experiment):
    selected = [row for row in rows if row.get("experiment") == experiment]
    best = [record_from_csv_row(row) for row in selected if row.get("mode") == "best"]
    all_lambdas = [record_from_csv_row(row) for row in selected if row.get("mode") == "all"]
    if best and all_lambdas:
        return best, all_lambdas
    return None


def write_runtime_csv(path, experiment, best_records, all_records):
    existing = [
        row
        for row in read_runtime_csv(path)
        if row.get("experiment") != experiment
    ]
    rows = (
        existing
        + records_to_csv_rows(experiment, "best", best_records)
        + records_to_csv_rows(experiment, "all", all_records)
    )
    fieldnames = [
        "experiment",
        "mode",
        "model_key",
        "model_label",
        "lambda_label",
        "runs",
        "mean_seconds",
        "std_seconds",
        "mean_minutes",
        "run_seconds",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_best_lambda(records, experiment, path):
    labels = [plot_record_label(record, include_lambda=True) for record in records]
    means = [record.mean_seconds for record in records]
    errors = [record.std_seconds for record in records]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, means, yerr=errors, capsize=4, color="#6fa8dc", edgecolor="#2f4f6f")
    ax.set_ylabel("Average epoch/fit time (s)")
    ax.set_title(f"{experiment.upper()} average epoch/fit runtime")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_all_lambdas(records, experiment, path):
    sorted_records = sorted(records, key=lambda record: record.mean_seconds)
    labels = [
        f"{plot_record_label(record)} {record.lambda_label}"
        if record.lambda_label != "--"
        else plot_record_label(record)
        for record in sorted_records
    ]
    means = [record.mean_seconds for record in sorted_records]
    errors = [record.std_seconds for record in sorted_records]

    height = max(5.0, 0.28 * len(records))
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(labels, means, xerr=errors, capsize=3, color="#93c47d", edgecolor="#3f6331")
    ax.set_xlabel("Average epoch/fit time (s)")
    ax.set_title(f"{experiment.upper()} runtime by logic and lambda")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def collect_logic_lambda_average_records(all_records):
    records = []
    for model_key, model_label, is_property in MODELS:
        logic_records = [
            record
            for record in all_records
            if record.model_key == model_key
            and (is_property or record.lambda_label == "--")
        ]
        if not logic_records:
            continue

        lambda_means = [record.mean_seconds for record in logic_records]
        records.append(
            LogicLambdaAverageRecord(
                model_key=model_key,
                model_label=runtime_model_label(model_key, model_label),
                lambda_count=len(lambda_means) if is_property else 0,
                mean_seconds=mean(lambda_means),
                std_seconds=stdev(lambda_means) if len(lambda_means) > 1 else 0.0,
            )
        )
    return records


def print_logic_lambda_average_log(experiment, records):
    print(f"===== Logic runtime averaged over lambda combinations: {experiment.upper()} =====")
    print(
        f"{'Logic':<16} {'Lambdas':>7} "
        f"{'Mean epoch (s)':>16} {'Std across lambdas':>19} {'Mean min':>10}"
    )
    print("-" * 74)
    for record in records:
        print(
            f"{record.model_label:<16} {record.lambda_count or '--':>7} "
            f"{fmt_seconds(record.mean_seconds):>16} "
            f"{fmt_seconds(record.std_seconds):>19} "
            f"{fmt_minutes(record.mean_minutes):>10}"
        )
    print()


def plot_logic_lambda_average(records, experiment, path):
    labels = [record.model_label for record in records]
    means = [record.mean_seconds for record in records]
    errors = [record.std_seconds for record in records]

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.bar(labels, means, yerr=errors, capsize=4, color="#6fa8dc", edgecolor="#2f4f6f")
    ax.set_ylabel("Average epoch time (s)")
    ax.set_title(f"{experiment.upper()} runtime averaged over lambda combinations")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def collect_experiment_logic_runtimes(output_root):
    return [
        ExperimentLogicRuntime(
            experiment=experiment,
            records=collect_logic_lambda_average_records(
                collect_all_lambda_records(output_root, experiment)
            ),
        )
        for experiment in EXPERIMENTS
    ]


def print_combined_logic_runtime_log(experiment_records):
    print("===== Logic runtime averaged over lambda combinations: all experiments =====")
    for item in experiment_records:
        print_logic_lambda_average_log(item.experiment, item.records)


def plot_combined_logic_lambda_average(experiment_records, path):
    baseline_keys = [model_key for model_key, _, is_property in MODELS if not is_property]
    property_keys = [model_key for model_key, _, is_property in MODELS if is_property]
    labels_by_key = {
        model_key: runtime_model_label(model_key, model_label)
        for model_key, model_label, _ in MODELS
    }
    records_by_experiment = {
        item.experiment: {record.model_key: record for record in item.records}
        for item in experiment_records
    }

    baseline_records = {}
    for key in baseline_keys:
        experiment_baselines = [
            records_by_experiment[experiment].get(key)
            for experiment in EXPERIMENTS
            if records_by_experiment[experiment].get(key) is not None
        ]
        if not experiment_baselines:
            continue
        baseline_means = [record.mean_seconds for record in experiment_baselines]
        baseline_records[key] = LogicLambdaAverageRecord(
            model_key=key,
            model_label=labels_by_key[key],
            lambda_count=0,
            mean_seconds=mean(baseline_means),
            std_seconds=stdev(baseline_means) if len(baseline_means) > 1 else 0.0,
        )

    colors = {
        "ex1": "#4e79a7",
        "ex2": "#f28e2b",
        "ex3": "#59a14f",
        "ex4": "#e15759",
    }
    baseline_color = "#9da3a6"
    group_keys = baseline_keys + property_keys
    group_centers = list(range(len(group_keys)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    for center, key in zip(group_centers, group_keys):
        if key in baseline_keys:
            record = baseline_records.get(key)
            if record is None:
                continue
            ax.bar(
                center,
                record.mean_seconds,
                yerr=record.std_seconds,
                width=0.5,
                color=baseline_color,
                edgecolor="#4f565c",
                capsize=4,
                label="Baseline" if key == baseline_keys[0] else None,
            )
            continue

        offsets = [(-1.5 + index) * width for index in range(len(EXPERIMENTS))]
        for experiment, offset in zip(EXPERIMENTS, offsets):
            record = records_by_experiment[experiment].get(key)
            if record is None:
                continue
            ax.bar(
                center + offset,
                record.mean_seconds,
                yerr=record.std_seconds,
                width=width,
                color=colors[experiment],
                edgecolor="#2f3b45",
                linewidth=0.7,
                capsize=3,
                label=experiment.upper() if center == len(baseline_keys) else None,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([labels_by_key[key] for key in group_keys], rotation=18, ha="right")
    ax.set_ylabel("Average epoch/fit time (s)")
    ax.set_title("Runtime averaged over lambda combinations across experiments")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Experiment", frameon=True, loc="upper left", ncols=1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--runtime-csv",
        type=Path,
        help=(
            "Load cached runtime summaries from this CSV when possible; "
            "otherwise compute and write/update this CSV."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment is None:
        experiment_records = collect_experiment_logic_runtimes(args.output_root)
        print_combined_logic_runtime_log(experiment_records)
        path = args.out_dir / "runtime_all_experiments_logic_lambda_average.png"
        plot_combined_logic_lambda_average(experiment_records, path)
        print(f"Saved plot: {path}")
        return

    cached = None
    if args.runtime_csv is not None:
        cached = cached_records(read_runtime_csv(args.runtime_csv), args.experiment)

    if cached is not None:
        best_records, all_records = cached
        print(f"Loaded runtime summaries from: {args.runtime_csv}")
        print()
    else:
        best_records = collect_best_lambda_records(args.output_root, args.experiment)
        all_records = collect_all_lambda_records(args.output_root, args.experiment)
        if args.runtime_csv is not None:
            write_runtime_csv(args.runtime_csv, args.experiment, best_records, all_records)
            print(f"Wrote runtime summaries to: {args.runtime_csv}")
            print()

    logic_average_records = collect_logic_lambda_average_records(all_records)
    print_logic_lambda_average_log(args.experiment, logic_average_records)

    path = args.out_dir / f"runtime_{args.experiment}_logic_lambda_average.png"
    plot_logic_lambda_average(logic_average_records, args.experiment, path)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
