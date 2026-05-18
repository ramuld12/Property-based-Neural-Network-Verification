#!/usr/bin/env python3
import csv
import sys
from pathlib import Path
from statistics import mean

MODELS = [
    ("Random Forest", lambda root, ds: root / "baseline" / f"rf_to_ciciot2023_{ds}"),
    ("Baseline ", lambda root, ds: root / "baseline" / f"mlp_to_ciciot2023_{ds}"),
    ("Gödel ", lambda root, ds: root / "properties" / "goedel" / ds),
    ("Łukasiewicz ", lambda root, ds: root / "properties" / "lukasiewicz" / ds),
    ("Reichenbach ", lambda root, ds: root / "properties" / "reichenbach" / ds),
    ("Yager ", lambda root, ds: root / "properties" / "yager" / ds),
    ("DL2 ", lambda root, ds: root / "properties" / "dl2" / ds),
]

DATASETS = [
    ("CICIOT2023\\_good", "good"),
    ("CICIOT2023\\_bad", "bad"),
]

CLASSES = {
    "ex1": ["BENIGN", "ATTACK"],
    "ex2": ["BENIGN", "ATTACK"],
    "ex3": ["BENIGN", "PORTSCAN", "DOS_HTTP_FLOOD"],
    "ex4": ["BENIGN", "PORTSCAN", "DOS_HTTP_FLOOD", "ATTACK"],
}

def read_report(path):
    with path.open(newline="") as f:
        return {row[""]: row for row in csv.DictReader(f)}

def complete_runs(folder):
    if not folder.exists():
        return []
    return [
        run for run in sorted(folder.iterdir())
        if run.is_dir()
        and (run / "cross_eval" / "classification_report.csv").exists()
    ]

def complete_aggregate_runs(folder):
    return [
        run for run in complete_runs(folder)
        if (run / "test" / "classification_report.csv").exists()
    ]

def fmt(value):
    return f"{value:.3f}"

def aggregate_row_values(runs):
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
    return [fmt(mean(train_values[key])) for key in ordered] + [
        fmt(mean(cross_values[key])) for key in ordered
    ]

def print_aggregate_rows(output_root, experiments):
    print("% Overall aggregate rows")
    for experiment in experiments:
        root = output_root / experiment
        exp_label = experiment.upper().replace("EX", "E")

        for dataset_label, dataset_key in DATASETS:
            for model_label, folder_fn in MODELS:
                runs = complete_aggregate_runs(folder_fn(root, dataset_key))
                if not runs:
                    print(
                        f"        {exp_label} & {dataset_label} & {model_label} "
                        r"& \multicolumn{8}{c}{No complete runs} \\"
                    )
                    continue

                values = aggregate_row_values(runs)
                print(
                    f"        {exp_label} & {dataset_label} & {model_label} & "
                    + " & ".join(values)
                    + r" \\"
                )

            if dataset_key == "good":
                print(r"        \hline")

        if experiment != experiments[-1]:
            print(r"        \hline")

def print_per_class_rows(output_root, experiment):
    root = output_root / experiment
    classes = CLASSES[experiment]
    exp_label = experiment.upper().replace("EX", "E")

    print(f"% {exp_label} per-class rows")

    for dataset_label, dataset_key in DATASETS:
        for model_label, folder_fn in MODELS:
            runs = complete_runs(folder_fn(root, dataset_key))
            if not runs:
                ncols = 4 * len(classes)
                print(
                    f"        {dataset_label} & {model_label} "
                    f"& \\multicolumn{{{ncols}}}{{c}}{{No complete runs}} \\\\"
                )
                continue

            values = []
            for cls in classes:
                accs, precs, recs, f1s = [], [], [], []

                for run in runs:
                    report = read_report(run / "cross_eval" / "classification_report.csv")

                    if cls not in report:
                        raise ValueError(f"{cls} missing in {run}")

                    accs.append(float(report[cls]["accuracy"]))
                    precs.append(float(report[cls]["precision"]))
                    recs.append(float(report[cls]["recall"]))
                    f1s.append(float(report[cls]["f1-score"]))

                values.extend([
                    fmt(mean(accs)),
                    fmt(mean(precs)),
                    fmt(mean(recs)),
                    fmt(mean(f1s)),
                ])

            print(f"        {dataset_label} & {model_label} & " + " & ".join(values) + r" \\")

        if dataset_key == "good":
            print(r"        \hline")

if __name__ == "__main__":
    output_root = Path(sys.argv[1])
    experiments = ["ex1", "ex2", "ex3", "ex4"]
    print_aggregate_rows(output_root, experiments)
    print()

    for experiment in experiments:
        print_per_class_rows(output_root, experiment)
        print()
