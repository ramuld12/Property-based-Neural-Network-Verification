from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from thesis.data.features import filter_labels, make_binary_attack_df


@dataclass
class ExperimentData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cross_eval: pd.DataFrame | None


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip", delimiter="\t")


def prepare_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    labels = config["data"]["labels"]
    if labels == ["BENIGN", "ATTACK"]:
        return make_binary_attack_df(df, config["data"]["attack_source_labels"])
    return filter_labels(df, labels)


def load_experiment_data(config: dict) -> ExperimentData:
    data_cfg = config["data"]
    random_state = config["experiment"].get("seed", 42)

    full_df = prepare_labels(read_tsv(data_cfg["train_path"]), config)
    train_val_df, test_df = train_test_split(
        full_df,
        test_size=data_cfg.get("test_size", 0.3),
        random_state=random_state,
        stratify=full_df["label"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=data_cfg.get("val_size", 0.2),
        random_state=random_state,
        stratify=train_val_df["label"],
    )

    cross_eval = None
    if data_cfg.get("cross_eval_path"):
        cross_eval = prepare_labels(read_tsv(data_cfg["cross_eval_path"]), config)

    return ExperimentData(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        cross_eval=None if cross_eval is None else cross_eval.reset_index(drop=True),
    )
