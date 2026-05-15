from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from thesis.data.features import filter_labels, make_attack_label_df, recompute_portscan_window_features


@dataclass
class ExperimentData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cross_eval: pd.DataFrame | None


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip", delimiter="\t")


def attack_source_labels(config: dict) -> list[str]:
    data_cfg = config["data"]
    attack_labels = data_cfg.get("attack_source_labels")
    if not attack_labels:
        raise ValueError("data.attack_source_labels is required when data.labels includes ATTACK")

    labels = data_cfg["labels"]
    overlapping_labels = sorted(set(attack_labels) & (set(labels) - {"ATTACK"}))
    if overlapping_labels:
        raise ValueError(
            "data.attack_source_labels cannot also appear as explicit labels "
            f"when ATTACK is configured: {overlapping_labels}"
        )
    return attack_labels


def prepare_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    labels = config["data"]["labels"]
    if "ATTACK" in labels:
        return make_attack_label_df(df, labels, attack_source_labels(config))
    return filter_labels(df, labels)


def prepare_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    window_seconds = config["data"].get("portscan_window_seconds")
    if window_seconds is None:
        return df
    return recompute_portscan_window_features(df, float(window_seconds))


def load_experiment_data(config: dict) -> ExperimentData:
    data_cfg = config["data"]
    random_state = config["experiment"].get("seed", 42)

    full_df = prepare_labels(prepare_features(read_tsv(data_cfg["train_path"]), config), config)
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
        cross_eval = prepare_labels(prepare_features(read_tsv(data_cfg["cross_eval_path"]), config), config)

    return ExperimentData(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        cross_eval=None if cross_eval is None else cross_eval.reset_index(drop=True),
    )
