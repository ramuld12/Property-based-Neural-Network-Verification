#!/usr/bin/env python3
import csv
from pathlib import Path
from statistics import mean

MODELS = [
    ("RF", lambda root, ds: root / "baseline" / f"rf_to_ciciot2023_{ds}"),
    ("Base MLP", lambda root, ds: root / "baseline" / f"mlp_to_ciciot2023_{ds}"),
    ("Gödel MLP", lambda root, ds: root / "properties" / "goedel" / ds),
    ("Łukasiewicz MLP", lambda root, ds: root / "properties" / "lukasiewicz" / ds),
    ("Reichenbach MLP", lambda root, ds: root / "properties" / "reichenbach" / ds),
    ("Yager MLP", lambda root, ds: root / "properties" / "yager" / ds),
    ("DL2 MLP", lambda root, ds: root / "properties" / "dl2" / ds),
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

def fmt(value):
    return f"{value:.3f}"

def latex_rows(experiment):
    root = Path("outputs") / experiment
    classes = CLASSES[experiment]

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
    for experiment in ["ex1", "ex2", "ex3", "ex4"]:
        print(f"% Experiment {experiment.upper()}")
        latex_rows(experiment)
        print()
