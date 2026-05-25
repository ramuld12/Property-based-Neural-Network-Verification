#!/usr/bin/env python3
import argparse
from pathlib import Path

from generate_report_results import (
    CLASSES,
    COMBINED_CLASSES,
    DATASETS,
    EXPERIMENTS,
    best_columns,
    block_rows,
    collect_rows,
    combined_per_class_best,
    combined_per_class_values,
    discover_property_roots,
    fmt,
    is_best,
    lambda_from_root,
    tex_logic,
    tex_num,
)


def per_class_map(row):
    if row.per_class is None:
        return {}

    classes = CLASSES[row.experiment]
    return {
        cls: (row.per_class[2 * index], row.per_class[2 * index + 1])
        for index, cls in enumerate(classes)
    }


def print_best_lambda_table(rows):
    print(r"\begin{table*}[p]")
    print(r"    \centering")
    print(r"    \scriptsize")
    print(r"    \setlength{\tabcolsep}{1.5pt}")
    print(r"    \renewcommand{\arraystretch}{0.88}")
    print(r"    \begin{adjustbox}{max totalsize={\textwidth}{0.96\textheight},center}")
    print(
        r"    \begin{tabular}{llll "
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
    print(r"        & \multirow{3}{*}{\textbf{Lambda}}")
    print(r"        & \multicolumn{4}{c}{\textbf{Cross-Test Results}}")
    print(r"        & \multicolumn{14}{c}{\textbf{Per-Class Cross-Test Results}} \\")
    print(r"        \cmidrule(lr){5-8}")
    print(r"        \cmidrule(lr){9-22}")
    print(r"        & & &")
    print(r"        & \textbf{Acc} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1}")
    print(r"        & \multicolumn{2}{c}{\textbf{BENIGN}}")
    print(r"        & \multicolumn{4}{c}{\textbf{PORTSCAN}}")
    print(r"        & \multicolumn{4}{c}{\textbf{DOS}}")
    print(r"        & \multicolumn{4}{c}{\textbf{ATTACK}} \\")
    print(r"        \cmidrule(lr){9-10}")
    print(r"        \cmidrule(lr){11-14}")
    print(r"        \cmidrule(lr){15-18}")
    print(r"        \cmidrule(lr){19-22}")
    print(r"        & & &")
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
                        rf"& {row.lambda_label} & \multicolumn{{18}}{{c}}{{No complete runs}} \\"
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
                    f"        {exp_prefix}{dataset_prefix} & {logic_label} "
                    f"& {row.lambda_label} & "
                    + " & ".join(cross_values + per_class_values)
                    + r" \\"
                )

            if dataset_key == "good":
                print(r"        \cmidrule(lr){2-22}")

        if exp_index != len(EXPERIMENTS) - 1:
            print()
            print(r"        \midrule")

    print()
    print(r"        \bottomrule")
    print(r"    \end{tabular}")
    print(r"    \end{adjustbox}")
    print(
        r"    \caption{Overall and per-class classification performance using the "
        r"best lambda value for each experiment, cross-test dataset, and logic.}"
    )
    print(r"\end{table*}")


def print_lambda_combinations(output_root):
    labels = [
        label
        for label in (
            lambda_from_root(root) for root in discover_property_roots(output_root)
        )
        if label is not None
    ]
    labels = sorted(set(labels))
    if not labels:
        print(r"% Possible lambda combinations: none discovered")
        return

    print(r"% Possible lambda combinations: " + ", ".join(labels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()

    rows = collect_rows(args.output_root)
    print_lambda_combinations(args.output_root)
    print()
    print_best_lambda_table(rows)


if __name__ == "__main__":
    main()
