from __future__ import annotations

from pathlib import Path

import pandas as pd

from thesis.data.datasets import prepare_features


DATA_PATH = Path("data/cicids2017_preprocessed.tsv")
TARGET_LABELS = ["BENIGN", "DOS_HTTP_FLOOD", "PORTSCAN"]
WINDOW_SECONDS = 5.0
PREFERRED_MATCH_INDEX = 500
INTERNAL_COLUMNS = ["is_failed_conn"]


def first_featured_row(featured_rows: pd.DataFrame, label: str, min_uniq_dst_ports: int = 1) -> pd.DataFrame:
    matches = featured_rows[
        (featured_rows["label"] == label) & (featured_rows["uniq_dst_ports"] >= min_uniq_dst_ports)
    ]
    if matches.empty:
        raise ValueError(f"Could not find a {label} row with uniq_dst_ports >= {min_uniq_dst_ports}")
    row_index = min(PREFERRED_MATCH_INDEX, len(matches) - 1)
    return matches.iloc[[row_index]]


def example_rows(path: Path, labels: list[str]) -> pd.DataFrame:
    featured_rows = prepare_features(
        pd.read_csv(path, sep="\t"),
        {"data": {"windows_seconds": WINDOW_SECONDS}},
    )
    rows = [
        first_featured_row(featured_rows, "BENIGN"),
        first_featured_row(featured_rows, "DOS_HTTP_FLOOD"),
        first_featured_row(featured_rows, "PORTSCAN", min_uniq_dst_ports=2),
    ]
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    featured_rows = example_rows(DATA_PATH, TARGET_LABELS)
    label_order = {label: index for index, label in enumerate(TARGET_LABELS)}
    featured_rows = featured_rows.assign(_label_order=featured_rows["label"].map(label_order))
    table = featured_rows.sort_values("_label_order").drop(columns=["_label_order", *INTERNAL_COLUMNS], errors="ignore")

    print(table.to_markdown(index=False, floatfmt=".6g"))


if __name__ == "__main__":
    main()
