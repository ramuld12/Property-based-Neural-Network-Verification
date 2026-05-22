#!/usr/bin/env python3
import argparse
from pathlib import Path

from generate_best_lambda_report_results import constraint_cell
from generate_report_results import (
    CLASS_LABELS,
    CLASSES,
    DATASETS,
    EXPERIMENTS,
    block_rows,
    collect_rows,
    fmt,
    is_best,
    tex_logic,
    tex_num,
)


def class_width(cls):
    return 2 if cls == "BENIGN" else 4


def class_columns(classes):
    return sum(class_width(cls) for cls in classes)


def per_class_column_spec(classes):
    return "llc " + " ".join("cc" if cls == "BENIGN" else "cccc" for cls in classes)


def per_class_values_by_class(row):
    if row.per_class is None:
        return {}
    return {
        cls: (row.per_class[2 * index], row.per_class[2 * index + 1])
        for index, cls in enumerate(CLASSES[row.experiment])
    }


def per_class_best(rows, classes):
    best = {}
    for cls in classes:
        accs, f1s, csecs = [], [], []
        for row in rows:
            values = per_class_values_by_class(row).get(cls)
            if values is not None:
                accs.append(values[0])
                f1s.append(values[1])

            if row.constraints is None:
                continue
            if row.experiment in {"ex1", "ex2"}:
                csec = row.constraints.get("attack") if cls == "ATTACK" else None
            elif row.experiment in {"ex3", "ex4"}:
                if cls == "PORTSCAN":
                    csec = row.constraints.get("scan")
                elif cls == "DOS_HTTP_FLOOD":
                    csec = row.constraints.get("dos")
                else:
                    csec = None
            else:
                csec = None
            if csec is not None:
                csecs.append(csec)

        if accs and f1s:
            best[cls] = {"acc": max(accs), "f1": max(f1s)}
        if csecs:
            best.setdefault(cls, {})["csec"] = max(csecs)
    return best


def class_cells(row, cls, best):
    values = per_class_values_by_class(row).get(cls)
    if values is None:
        cells = ["{-}", "{-}"]
        if cls != "BENIGN":
            cells.extend(["{-}", "{}"])
        return cells

    cells = [
        tex_num(values[0], is_best(values[0], best[cls]["acc"])),
        tex_num(values[1], is_best(values[1], best[cls]["f1"])),
    ]
    if cls != "BENIGN":
        cells.extend([constraint_cell(row, cls, best), "{}"])
    return cells


def print_table_header(experiment, classes):
    exp_label = experiment.upper().replace("EX", "E")
    is_wide = len(classes) >= 4
    table_env = "table*" if is_wide else "table"
    width = r"\textwidth" if is_wide else r"\linewidth"
    tabcolsep = "2.5pt" if len(classes) >= 3 else "3.5pt"

    print(r"% =========================")
    print(rf"% Experiment {exp_label}: Per-class")
    print(r"% =========================")
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

    start = 4
    for index, cls in enumerate(classes):
        width = class_width(cls)
        suffix = r" \\" if index == len(classes) - 1 else ""
        print(rf"        & \multicolumn{{{width}}}{{c}}{{\textbf{{{CLASS_LABELS[cls]}}}}}{suffix}")

    start = 4
    for cls in classes:
        end = start + class_width(cls) - 1
        print(rf"        \cmidrule(lr){{{start}-{end}}}")
        start = end + 1

    print(r"        & &")
    for index, cls in enumerate(classes):
        suffix = r" \\" if index == len(classes) - 1 else ""
        if cls == "BENIGN":
            print(r"        & \textbf{Acc} & \textbf{F1}" + suffix)
        else:
            print(
                r"        & \textbf{Acc} & \textbf{F1} & \textbf{CSec} & \textbf{CSat}"
                + suffix
            )
    print(r"        \midrule")
    print()
    return table_env


def print_main_report_table(rows, experiment):
    classes = CLASSES[experiment]
    exp_number = experiment.replace("ex", "")
    table_env = print_table_header(experiment, classes)

    for dataset_index, (dataset_label, dataset_key) in enumerate(DATASETS):
        if dataset_index:
            print()
            print(r"        \midrule")
            print()

        current_rows = block_rows(rows, experiment, dataset_key)
        best = per_class_best(current_rows, classes)
        best_cross_f1 = max(
            (row.aggregate[7] for row in current_rows if row.aggregate is not None),
            default=None,
        )
        print(rf"        \multirow{{7}}{{*}}{{{dataset_label}}}")

        for row in current_rows:
            if row.per_class is None:
                print(
                    f"        & {row.model_label:<13} & {row.lambda_label:<7} "
                    rf"& \multicolumn{{{class_columns(classes)}}}{{c}}{{No complete runs}} \\"
                )
                continue

            logic_best = best_cross_f1 is not None and fmt(row.aggregate[7]) == fmt(best_cross_f1)
            logic_label = tex_logic(row.model_label, logic_best)
            values = []
            for cls in classes:
                values.extend(class_cells(row, cls, best))

            print(
                f"        & {logic_label} & {row.lambda_label:<7} & "
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
    print(rf"    \caption{{\textbf{{Experiment {exp_number}}}}}")
    print(rf"    \label{{tab:e{exp_number}PerClassResults}}")
    print(rf"\end{{{table_env}}}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.output_root)
    for experiment in EXPERIMENTS:
        print_main_report_table(rows, experiment)
        print()


if __name__ == "__main__":
    main()
