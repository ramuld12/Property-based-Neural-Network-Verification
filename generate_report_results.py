#!/usr/bin/env python3
import csv
import json
import re
import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

EXPERIMENTS = ["ex1", "ex2", "ex3", "ex4"]

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

CROSS_EVAL_DATASETS = {
    "good": "ciciot2023_preprocessed_good",
    "bad": "ciciot2023_preprocessed_bad",
}

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

COMBINED_CLASSES = ["BENIGN", "PORTSCAN", "DOS_HTTP_FLOOD", "ATTACK"]


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
    constraints: dict[str, dict[str, float]] | None


def read_report(path):
    with path.open(newline="") as f:
        return {row[""]: row for row in csv.DictReader(f)}


def cross_eval_path(run, dataset_key, filename):
    cross_eval = run / "cross_eval"
    dataset_name = CROSS_EVAL_DATASETS[dataset_key]
    dataset_path = cross_eval / dataset_name / filename
    if dataset_path.exists() and "small" not in dataset_path.as_posix():
        return dataset_path

    legacy_path = cross_eval / filename
    if legacy_path.exists() and "small" not in legacy_path.as_posix():
        return legacy_path

    return None


def cross_eval_report_path(run, dataset_key):
    return cross_eval_path(run, dataset_key, "classification_report.csv")


def cross_eval_metrics_path(run, dataset_key):
    return cross_eval_path(run, dataset_key, "metrics.json")


def complete_runs(folder, dataset_key):
    if not folder.exists():
        return []
    return [
        run
        for run in sorted(folder.iterdir())
        if run.is_dir()
        and cross_eval_report_path(run, dataset_key) is not None
    ]


def complete_aggregate_runs(folder, dataset_key):
    return [
        run
        for run in complete_runs(folder, dataset_key)
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


def aggregate_values(runs, dataset_key):
    train_values = {"acc": [], "prec": [], "rec": [], "f1": []}
    cross_values = {"acc": [], "prec": [], "rec": [], "f1": []}

    for run in runs:
        train_report = read_report(run / "test" / "classification_report.csv")
        cross_report = read_report(cross_eval_report_path(run, dataset_key))

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


def per_class_values(runs, classes, dataset_key):
    values = []
    for cls in classes:
        accs, f1s = [], []

        for run in runs:
            report = read_report(cross_eval_report_path(run, dataset_key))
            if cls not in report:
                raise ValueError(f"{cls} missing in {run}")

            accs.append(float(report[cls]["accuracy"]))
            f1s.append(float(report[cls]["f1-score"]))

        values.extend([mean(accs), mean(f1s)])

    return values


def constraint_values(runs, dataset_key):
    values = {
        "dos": {"csec": [], "csat": []},
        "scan": {"csec": [], "csat": []},
        "attack": {"csec": [], "csat": []},
    }

    for run in runs:
        path = cross_eval_metrics_path(run, dataset_key)
        if path is None:
            continue

        metrics = json.loads(path.read_text())
        dos_csec = metrics.get("csec_dos")
        dos_csat = metrics.get("csat_dos")
        scan_csec = metrics.get("csec_scan")
        scan_csat = metrics.get("csat_scan")

        if dos_csec is not None:
            values["dos"]["csec"].append(float(dos_csec))
        if dos_csat is not None:
            values["dos"]["csat"].append(float(dos_csat))
        if scan_csec is not None:
            values["scan"]["csec"].append(float(scan_csec))
        if scan_csat is not None:
            values["scan"]["csat"].append(float(scan_csat))
        if dos_csec is not None and scan_csec is not None:
            values["attack"]["csec"].append((float(dos_csec) + float(scan_csec)) / 2)
        if dos_csat is not None and scan_csat is not None:
            values["attack"]["csat"].append((float(dos_csat) + float(scan_csat)) / 2)

    result = {}
    for constraint, metric_values in values.items():
        populated = {
            metric: mean(metric_list)
            for metric, metric_list in metric_values.items()
            if metric_list
        }
        if populated:
            result[constraint] = populated
    return result or None


def baseline_folder(root, experiment, dataset_key, model_key):
    folder_name = {
        "rf": f"rf_to_ciciot2023_{dataset_key}",
        "baseline": f"mlp_to_ciciot2023_{dataset_key}",
    }[model_key]
    return root / experiment / "baseline" / folder_name


def property_folder(root, experiment, dataset_key, model_key):
    root = root / experiment / "properties" / model_key
    dataset_folder = root / dataset_key
    if dataset_folder.exists():
        return dataset_folder

    both_folder = root / "both"
    if both_folder.exists():
        return both_folder

    return dataset_folder


def unique_existing(paths):
    seen = set()
    result = []
    for path in paths:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        result.append(path)
    return result


def has_experiment_subdir(root, subdir):
    return any((root / experiment / subdir).exists() for experiment in EXPERIMENTS)


def discover_baseline_roots(output_root):
    if has_experiment_subdir(output_root, "baseline"):
        return [output_root]

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
    if lambda_from_root(output_root):
        return [output_root]

    if has_experiment_subdir(output_root, "properties"):
        return [output_root]

    candidates = [
        path for path in sorted(output_root.glob("lambda_*")) if lambda_from_root(path)
    ]
    if candidates:
        return candidates

    final_roots = [
        output_root,
        output_root / "final_models",
        output_root.parent / "final_models",
        Path("outputs") / "final_models",
        output_root / "final_lambda_exp",
        output_root.parent / "final_lambda_exp",
        Path("outputs") / "final_lambda_exp",
    ]
    for root in final_roots:
        if root.name in {"final_models", "final_lambda_exp"} and root.exists():
            return [
                path
                for path in sorted(root.glob("lambda_*"))
                if lambda_from_root(path)
            ]

    return [output_root] if output_root.exists() else []


def choose_best_candidate(candidates, dataset_key, run_filter=None):
    complete = []
    for folder, lambda_label in candidates:
        runs = complete_aggregate_runs(folder, dataset_key)
        if run_filter is not None:
            runs = [run for run in runs if run_filter(run)]
        if not runs:
            continue
        values = aggregate_values(runs, dataset_key)
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

                chosen = choose_best_candidate(candidates, dataset_key)
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
                        per_class_values(runs, CLASSES[experiment], dataset_key),
                        constraint_values(runs, dataset_key),
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


def per_class_map(row):
    if row.per_class is None:
        return {}

    classes = CLASSES[row.experiment]
    return {
        cls: (row.per_class[2 * index], row.per_class[2 * index + 1])
        for index, cls in enumerate(classes)
    }


def constraint_metrics(row, cls):
    if row.constraints is None or cls == "BENIGN":
        return None
    if cls == "PORTSCAN":
        return row.constraints.get("scan")
    if cls == "DOS_HTTP_FLOOD":
        return row.constraints.get("dos")
    if cls == "ATTACK" and row.experiment in {"ex1", "ex2"}:
        return row.constraints.get("attack")
    return None


def constraint_value(row, cls, metric):
    metrics = constraint_metrics(row, cls)
    if metrics is None:
        return None
    return metrics.get(metric)


def constraint_cell(row, cls, metric, best):
    value = constraint_value(row, cls, metric)
    if value is None:
        return "{-}"
    best_value = best.get(cls, {}).get(metric)
    return tex_num(value, best_value is not None and is_best(value, best_value))


def combined_per_class_best(rows):
    best = {}
    for cls in COMBINED_CLASSES:
        accs, f1s, csecs, csats = [], [], [], []
        for row in rows:
            values = per_class_map(row).get(cls)
            if values is not None:
                accs.append(values[0])
                f1s.append(values[1])

            csec = constraint_value(row, cls, "csec")
            if csec is not None:
                csecs.append(csec)
            csat = constraint_value(row, cls, "csat")
            if csat is not None:
                csats.append(csat)
        if accs and f1s:
            best[cls] = {"acc": max(accs), "f1": max(f1s)}
        if csecs:
            best.setdefault(cls, {})["csec"] = max(csecs)
        if csats:
            best.setdefault(cls, {})["csat"] = max(csats)
    return best


def combined_per_class_values(row, best):
    values_by_class = per_class_map(row)
    cells = []
    for cls in COMBINED_CLASSES:
        values = values_by_class.get(cls)
        if values is None:
            cells.extend(["{-}", "{-}"])
            if cls != "BENIGN":
                cells.extend(
                    [
                        constraint_cell(row, cls, "csec", best),
                        constraint_cell(row, cls, "csat", best),
                    ]
                )
            continue

        best_acc = best[cls]["acc"]
        best_f1 = best[cls]["f1"]
        cells.extend(
            [
                tex_num(values[0], is_best(values[0], best_acc)),
                tex_num(values[1], is_best(values[1], best_f1)),
            ]
        )
        if cls != "BENIGN":
            cells.extend(
                [
                    constraint_cell(row, cls, "csec", best),
                    constraint_cell(row, cls, "csat", best),
                ]
            )
    return cells


def print_combined_table(rows):
    print(r"\begin{table*}[p]")
    print(r"    \centering")
    print(r"    \scriptsize")
    print(r"    \setlength{\tabcolsep}{1.7pt}")
    print(r"    \renewcommand{\arraystretch}{0.88}")
    print(r"    \begin{adjustbox}{max totalsize={\textwidth}{0.96\textheight},center}")
    print(
        r"    \begin{tabular}{lll "
        r"!{\color{black!25}\vrule width 0.35pt} cccc "
        r"!{\color{black!25}\vrule width 0.35pt} cc "
        r"!{\color{black!25}\vrule width 0.35pt} cccc "
        r"!{\color{black!25}\vrule width 0.35pt} cccc "
        r"!{\color{black!25}\vrule width 0.35pt} cccc}"
    )
    print(r"        \toprule")
    print(r"        \multirow{3}{*}{\textbf{Exp.}}")
    print(r"        & \multirow{3}{*}{\textbf{Dataset}}")
    print(r"        & \multirow{3}{*}{\textbf{Logic}}")
    print(r"        & \multicolumn{4}{c}{\textbf{Cross-Test Results}}")
    print(r"        & \multicolumn{14}{c}{\textbf{Per-Class Cross-Test Results}} \\")
    print(r"        \cmidrule(lr){4-7}")
    print(r"        \cmidrule(lr){8-21}")
    print(r"        & &")
    print(r"        & \textbf{Acc} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1}")
    print(r"        & \multicolumn{2}{c}{\textbf{BENIGN}}")
    print(r"        & \multicolumn{4}{c}{\textbf{PORTSCAN}}")
    print(r"        & \multicolumn{4}{c}{\textbf{DOS}}")
    print(r"        & \multicolumn{4}{c}{\textbf{ATTACK}} \\")
    print(r"        \cmidrule(lr){8-9}")
    print(r"        \cmidrule(lr){10-13}")
    print(r"        \cmidrule(lr){14-17}")
    print(r"        \cmidrule(lr){18-21}")
    print(r"        & &")
    print(r"        & & & &")
    print(r"        & \textbf{Acc} & \textbf{F1}")
    print(r"        & \textbf{Acc} & \textbf{F1} & \textbf{CSec} & \textbf{CSat}")
    print(r"        & \textbf{Acc} & \textbf{F1} & \textbf{CSec} & \textbf{CSat}")
    print(r"        & \textbf{Acc} & \textbf{F1} & \textbf{CSec} & \textbf{CSat} \\")
    print(r"        \midrule")
    print()

    for exp_index, experiment in enumerate(EXPERIMENTS):
        exp_label = experiment.upper().replace("EX", "E")
        print(rf"        \multirow{{14}}{{*}}{{{exp_label}}}")

        for dataset_index, (dataset_label, dataset_key) in enumerate(DATASETS):
            current_rows = block_rows(rows, experiment, dataset_key)
            aggregate_best = best_columns(current_rows, "aggregate")
            per_class_best = combined_per_class_best(current_rows)
            best_cross_f1 = max(
                (
                    row.aggregate[7]
                    for row in current_rows
                    if row.aggregate is not None
                ),
                default=None,
            )

            for row_index, row in enumerate(current_rows):
                exp_prefix = "" if dataset_index or row_index else ""
                dataset_prefix = (
                    rf"& \multirow{{7}}{{*}}{{{dataset_label}}}"
                    if row_index == 0
                    else "&"
                )

                if row.aggregate is None:
                    print(
                        f"        {exp_prefix}{dataset_prefix} & {row.model_label} "
                        r"& \multicolumn{18}{c}{No complete runs} \\"
                    )
                    continue

                logic_best = (
                    best_cross_f1 is not None
                    and fmt(row.aggregate[7]) == fmt(best_cross_f1)
                )
                logic_label = tex_logic(row.model_label, logic_best)
                cross_values = [
                    tex_num(value, is_best(value, aggregate_best[index]))
                    for index, value in enumerate(row.aggregate[4:], start=4)
                ]
                per_class_values = combined_per_class_values(row, per_class_best)
                print(
                    f"        {exp_prefix}{dataset_prefix} & {logic_label} & "
                    + " & ".join(cross_values + per_class_values)
                    + r" \\"
                )

            if dataset_key == "good":
                print(r"        \cmidrule(lr){2-21}")

        if exp_index != len(EXPERIMENTS) - 1:
            print()
            print(r"        \midrule")

    print()
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \end{adjustbox}")
    print(
        r"    \caption{Overall and per-class classification performance across experiments with "
        + lambda_caption(rows)
        + r".}"
    )
    print(r"\end{table*}")


def print_overall_table(rows):
    print_combined_table(rows)


def per_class_column_spec(classes):
    return "ll " + " ".join(["cc"] * len(classes))


def lambda_caption(rows, experiment=None):
    labels = sorted(
        {
            row.lambda_label
            for row in rows
            if (experiment is None or row.experiment == experiment)
            and row.lambda_label != "--"
        }
    )
    if not labels:
        return r"$\lambda=--$"
    if len(labels) == 1:
        return lambda_pair_caption(labels[0])
    return (
        r"$(\lambda_{\mathrm{dos}},\lambda_{\mathrm{scan}})\in\{"
        + ", ".join(labels)
        + r"\}$"
    )


def lambda_pair_caption(label, math=True):
    match = re.fullmatch(r"\(([^,]+),([^)]+)\)", label)
    if not match:
        return rf"$\lambda={label}$" if math else label

    text = (
        rf"\lambda_{{\mathrm{{dos}}}}={match.group(1)}, "
        rf"\lambda_{{\mathrm{{scan}}}}={match.group(2)}"
    )
    return f"${text}$" if math else text


def per_class_logic_cells(row, logic_best):
    if logic_best:
        return rf"        & \textbf{{{row.model_label}}}& "
    return f"        & {row.model_label:<13} & "


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
    for cls_index, cls in enumerate(classes):
        suffix = r" \\" if cls_index == len(classes) - 1 else ""
        print(rf"        & \multicolumn{{2}}{{c}}{{\textbf{{{CLASS_LABELS[cls]}}}}}{suffix}")
    cmidrule_start = 3
    for cls_index, _ in enumerate(classes):
        start = cmidrule_start + 2 * cls_index
        print(rf"        \cmidrule(lr){{{start}-{start + 1}}}")
    print(r"        &")
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
                    f"        & {row.model_label:<13} "
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
        rf"    \caption{{\textbf{{Experiment {exp_number}}} with "
        + lambda_caption(rows, experiment)
        + r".}"
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
