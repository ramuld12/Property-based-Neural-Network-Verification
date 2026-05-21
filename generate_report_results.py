#!/usr/bin/env python3
import csv
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

EXPERIMENTS = ["ex1", "ex2", "ex3", "ex4"]
PER_CLASS_EXPERIMENTS = ["ex4", "ex3", "ex2", "ex1"]

MODELS = [
    ("rf", "Random Forest", False),
    ("baseline", "Baseline", False),
    ("goedel", "Gödel", True),
    ("lukasiewicz", "Łukasiewicz", True),
    ("reichenbach", "Reichenbach", True),
    ("yager", "Yager", True),
    ("dl2", "DL2", True),
]

DATASETS = [
    ("Good", "good"),
    ("Bad", "bad"),
]

CLASSES = {
    "ex1": ["BENIGN", "ATTACK"],
    "ex2": ["BENIGN", "ATTACK"],
    "ex3": ["BENIGN", "PORTSCAN", "DOS_HTTP_FLOOD"],
    "ex4": ["BENIGN", "PORTSCAN", "DOS_HTTP_FLOOD", "ATTACK"],
}

CLASS_LABELS = {
    "BENIGN": "BENIGN",
    "PORTSCAN": "PORTSCAN",
    "DOS_HTTP_FLOOD": "DOS",
    "ATTACK": "ATTACK",
}


@dataclass
class ResultRow:
    experiment: str
    dataset_label: str
    dataset_key: str
    model_key: str
    model_label: str
    lambda_label: str
    aggregate: list[float] | None
    per_class: list[float] | None


def read_report(path):
    with path.open(newline="") as f:
        return {row[""]: row for row in csv.DictReader(f)}


def complete_runs(folder):
    if not folder.exists():
        return []
    return [
        run
        for run in sorted(folder.iterdir())
        if run.is_dir()
        and (run / "cross_eval" / "classification_report.csv").exists()
    ]


def complete_aggregate_runs(folder):
    return [
        run
        for run in complete_runs(folder)
        if (run / "test" / "classification_report.csv").exists()
    ]


def fmt(value):
    return f"{value:.3f}"


def tex_num(value, best=False):
    text = fmt(value)
    return rf"\textbf{{{text}}}" if best else text


def tex_logic(label, best=False):
    return rf"\textbf{{{label}}}" if best else label


LAMBDA_ROOT_RE = re.compile(r"^lambda_([0-9.]+)(?:_([0-9.]+))?$")


def lambda_from_root(root):
    match = LAMBDA_ROOT_RE.match(root.name)
    if match:
        dos = clean_lambda(match.group(1))
        scan = clean_lambda(match.group(2) or match.group(1))
        return f"({dos},{scan})"
    return None


def clean_lambda(value):
    return str(int(float(value))) if float(value).is_integer() else value


def lambda_from_config(run):
    config = run / "config.yaml"
    if not config.exists():
        return None

    text = config.read_text()
    dos = re.search(r"^\s*lambda_dos:\s*([0-9.]+)\s*$", text, re.MULTILINE)
    scan = re.search(r"^\s*lambda_scan:\s*([0-9.]+)\s*$", text, re.MULTILINE)
    if not dos or not scan:
        return None
    return f"({clean_lambda(dos.group(1))},{clean_lambda(scan.group(1))})"


def aggregate_values(runs):
    train_values = {"acc": [], "prec": [], "rec": [], "f1": []}
    cross_values = {"acc": [], "prec": [], "rec": [], "f1": []}

    for run in runs:
        train_report = read_report(run / "test" / "classification_report.csv")
        cross_report = read_report(run / "cross_eval" / "classification_report.csv")

        train_values["acc"].append(float(train_report["accuracy"]["accuracy"]))
        train_values["prec"].append(float(train_report["weighted avg"]["precision"]))
        train_values["rec"].append(float(train_report["weighted avg"]["recall"]))
        train_values["f1"].append(float(train_report["weighted avg"]["f1-score"]))

        cross_values["acc"].append(float(cross_report["accuracy"]["accuracy"]))
        cross_values["prec"].append(float(cross_report["weighted avg"]["precision"]))
        cross_values["rec"].append(float(cross_report["weighted avg"]["recall"]))
        cross_values["f1"].append(float(cross_report["weighted avg"]["f1-score"]))

    ordered = ["acc", "prec", "rec", "f1"]
    return [mean(train_values[key]) for key in ordered] + [
        mean(cross_values[key]) for key in ordered
    ]


def per_class_values(runs, classes):
    values = []
    for cls in classes:
        accs, f1s = [], []

        for run in runs:
            report = read_report(run / "cross_eval" / "classification_report.csv")
            if cls not in report:
                raise ValueError(f"{cls} missing in {run}")

            accs.append(float(report[cls]["accuracy"]))
            f1s.append(float(report[cls]["f1-score"]))

        values.extend([mean(accs), mean(f1s)])

    return values


def baseline_folder(root, experiment, dataset_key, model_key):
    folder_name = {
        "rf": f"rf_to_ciciot2023_{dataset_key}",
        "baseline": f"mlp_to_ciciot2023_{dataset_key}",
    }[model_key]
    return root / experiment / "baseline" / folder_name


def property_folder(root, experiment, dataset_key, model_key):
    return root / experiment / "properties" / model_key / dataset_key


def unique_existing(paths):
    seen = set()
    result = []
    for path in paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        result.append(path)
    return result


def discover_baseline_roots(output_root):
    candidates = [
        output_root / "final_baselines",
        output_root.parent / "final_baselines",
        Path("outputs") / "final_baselines",
    ]
    if output_root.name == "final_baselines":
        candidates.insert(0, output_root)

    for candidate in candidates:
        if candidate.exists():
            return [candidate]
    return []


def discover_property_roots(output_root):
    candidates = [output_root]
    candidates.extend(
        path for path in sorted(output_root.glob("lambda_*")) if lambda_from_root(path)
    )
    return unique_existing(candidates)


def choose_best_candidate(candidates):
    complete = []
    for folder, lambda_label in candidates:
        runs = complete_aggregate_runs(folder)
        if not runs:
            continue
        values = aggregate_values(runs)
        complete.append((values[7], values[4], folder, lambda_label, runs, values))

    if not complete:
        return None

    _, _, folder, lambda_label, runs, values = max(complete, key=lambda item: (item[0], item[1]))
    if runs:
        lambda_label = lambda_from_config(runs[0]) or lambda_label
    return folder, lambda_label or "--", runs, values


def collect_rows(output_root):
    baseline_roots = discover_baseline_roots(output_root)
    property_roots = discover_property_roots(output_root)
    rows = []

    for experiment in EXPERIMENTS:
        for dataset_label, dataset_key in DATASETS:
            for model_key, model_label, is_property in MODELS:
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

                chosen = choose_best_candidate(candidates)
                if chosen is None:
                    rows.append(
                        ResultRow(
                            experiment,
                            dataset_label,
                            dataset_key,
                            model_key,
                            model_label,
                            "--",
                            None,
                            None,
                        )
                    )
                    continue

                _, lambda_label, runs, aggregate = chosen
                rows.append(
                    ResultRow(
                        experiment,
                        dataset_label,
                        dataset_key,
                        model_key,
                        model_label,
                        lambda_label,
                        aggregate,
                        per_class_values(runs, CLASSES[experiment]),
                    )
                )

    return rows


def block_rows(rows, experiment, dataset_key):
    return [
        row
        for row in rows
        if row.experiment == experiment and row.dataset_key == dataset_key
    ]


def best_columns(rows, value_attr):
    populated = [getattr(row, value_attr) for row in rows if getattr(row, value_attr) is not None]
    if not populated:
        return []
    return [max(column) for column in zip(*populated)]


def is_best(value, best):
    return fmt(value) == fmt(best)


def print_overall_table(rows):
    print(r"\begin{table*}[p]")
    print(r"\centering")
    print(r"\footnotesize")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    print(r"\renewcommand{\arraystretch}{0.90}")
    print(r"\begin{adjustbox}{max totalsize={\textwidth}{0.96\textheight},center}")
    print(r"\begin{tabular}{ll l cccc cccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Exp.}} &")
    print(r"\multirow{2}{*}{\textbf{Dataset}} &")
    print(r"\multirow{2}{*}{\textbf{Logic}} &")
    print(r"\multicolumn{4}{c}{\textbf{Training Results}} &")
    print(r"\multicolumn{4}{c}{\textbf{Cross-Test Results}} \\")
    print(r"\cmidrule(lr){4-7}")
    print(r"\cmidrule(lr){8-11}")
    print(r"& & & \textbf{Acc} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1}")
    print(r"& \textbf{Acc} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\")
    print(r"\midrule")
    print()

    for exp_index, experiment in enumerate(EXPERIMENTS):
        exp_label = experiment.upper().replace("EX", "E")
        print(rf"\multirow{{14}}{{*}}{{{exp_label}}}")

        for dataset_index, (dataset_label, dataset_key) in enumerate(DATASETS):
            current_rows = block_rows(rows, experiment, dataset_key)
            best = best_columns(current_rows, "aggregate")

            for row_index, row in enumerate(current_rows):
                exp_prefix = "" if dataset_index or row_index else ""
                dataset_prefix = (
                    rf"& \multirow{{7}}{{*}}{{{dataset_label}}}"
                    if row_index == 0
                    else "&"
                )

                if row.aggregate is None:
                    print(
                        f"{exp_prefix}{dataset_prefix} & {row.model_label} "
                        r"& \multicolumn{8}{c}{No complete runs} \\"
                    )
                    continue

                values = [
                    tex_num(value, is_best(value, best[index]))
                    for index, value in enumerate(row.aggregate)
                ]
                print(
                    f"{exp_prefix}{dataset_prefix} & {row.model_label} & "
                    + " & ".join(values)
                    + r" \\"
                )

            if dataset_key == "good":
                print(r"\cmidrule(lr){2-11}")

        if exp_index != len(EXPERIMENTS) - 1:
            print()
            print(r"\midrule")

    print()
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{adjustbox}")
    print(
        r"\caption{Overall classification performance across experiments. "
        r"Bold values indicate the best score within each test-dataset block for the corresponding metric.}"
    )
    print(r"\label{tab:overallResults}")
    print(r"\end{table*}")


def per_class_column_spec(classes):
    return "llc " + " ".join(["cc"] * len(classes))


def per_class_logic_cells(row, logic_best):
    if logic_best and row.model_label == "Łukasiewicz":
        return rf"        & \textbf{{{row.model_label}}}" + "\n" + (
            f"                        & {row.lambda_label:<7} & "
        )
    if logic_best:
        return rf"        & \textbf{{{row.model_label}}}& {row.lambda_label:<7} & "
    return f"        & {row.model_label:<13} & {row.lambda_label:<7} & "


def print_per_class_table(rows, experiment):
    classes = CLASSES[experiment]
    exp_number = experiment.replace("ex", "")
    exp_label = experiment.upper().replace("EX", "E")
    is_wide = len(classes) >= 4
    table_env = "table*" if is_wide else "table"
    width = r"\textwidth" if is_wide else r"\linewidth"
    tabcolsep = "3pt" if len(classes) >= 3 else "4pt"

    print(rf"% =========================")
    print(rf"% Experiment {exp_label}: Per-class")
    print(rf"% =========================")
    print(rf"\begin{{{table_env}}}[t]")
    print(r"    \centering")
    print(rf"    \setlength{{\tabcolsep}}{{{tabcolsep}}}")
    print(r"    \renewcommand{\arraystretch}{1.12}")
    print(rf"    \resizebox{{{width}}}{{!}}{{%")
    print(rf"        \begin{{tabular}}{{{per_class_column_spec(classes)}}}")
    print(r"        \toprule")
    if experiment == "ex4":
        print(r"        \multirow{2}{*}{\textbf{Test-dataset}}")
    else:
        print(r"        \multirow{2}{*}{\textbf{\makecell{Test\\Dataset}}}")
    print(r"        & \multirow{2}{*}{\textbf{Logic}}")
    print(r"        & \multirow{2}{*}{\makecell{$\boldsymbol{\lambda}$\\$\boldsymbol{(\mathrm{dos},\mathrm{scan})}$}}")
    for cls_index, cls in enumerate(classes):
        suffix = r" \\" if cls_index == len(classes) - 1 else ""
        print(rf"        & \multicolumn{{2}}{{c}}{{\textbf{{{CLASS_LABELS[cls]}}}}}{suffix}")
    cmidrule_start = 4
    for cls_index, _ in enumerate(classes):
        start = cmidrule_start + 2 * cls_index
        print(rf"        \cmidrule(lr){{{start}-{start + 1}}}")
    print(r"        & &")
    for cls_index, _ in enumerate(classes):
        prefix = "        & " if cls_index == 0 else "        & "
        suffix = r" \\" if cls_index == len(classes) - 1 else ""
        print(prefix + r"\textbf{Acc} & \textbf{F1}" + suffix)
    print(r"        \midrule")
    print()

    for dataset_index, (dataset_label, dataset_key) in enumerate(DATASETS):
        if dataset_index:
            print()
            print(r"        \midrule")
            print()

        current_rows = block_rows(rows, experiment, dataset_key)
        best = best_columns(current_rows, "per_class")
        best_f1_sum = max(
            (
                sum(row.per_class[1::2])
                for row in current_rows
                if row.per_class is not None
            ),
            default=None,
        )

        print(rf"        \multirow{{7}}{{*}}{{{dataset_label}}}")
        for row_index, row in enumerate(current_rows):
            logic_best = (
                best_f1_sum is not None
                and row.per_class is not None
                and fmt(sum(row.per_class[1::2])) == fmt(best_f1_sum)
            )

            if row.per_class is None:
                ncols = 2 * len(classes)
                print(
                    f"        & {row.model_label:<13} & {row.lambda_label:<7} "
                    f"& \\multicolumn{{{ncols}}}{{c}}{{No complete runs}} \\\\"
                )
                continue

            values = [
                tex_num(value, is_best(value, best[index]))
                for index, value in enumerate(row.per_class)
            ]
            print(
                per_class_logic_cells(row, logic_best)
                + " & ".join(values)
                + r" \\"
            )

    print()
    print(r"        \bottomrule")
    if experiment == "ex4":
        print(r"        \end{tabular}")
        print(r"    }")
    else:
        print(r"        \end{tabular}}")
    print(
        rf"    \caption{{\textbf{{Experiment {exp_number}}}. "
        r"Bold values indicate the best score within each test-dataset block for the corresponding metric.}"
    )
    print(rf"    \label{{tab:e{exp_number}PerClassResults}}")
    print(rf"\end{{{table_env}}}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    output_root = args.output_root
    rows = collect_rows(output_root)

    print_overall_table(rows)
    print()

    for experiment in PER_CLASS_EXPERIMENTS:
        print_per_class_table(rows, experiment)
        print()
