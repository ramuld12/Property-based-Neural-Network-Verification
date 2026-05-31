from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from thesis.data.features import (
    compute_portscan_window_features,
    compute_time_elapsed,
    compute_window_id,
    filter_labels,
    make_attack_label_df,
)


@dataclass
class CrossEvalData:
    name: str
    frame: pd.DataFrame


@dataclass
class ExperimentData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    cross_evals: list[CrossEvalData]


def read_tsv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip", delimiter="\t")


def cross_eval_paths(config: dict) -> list[Path]:
    paths = config["data"].get("cross_eval_path", [])
    if paths is None:
        return []
    if not isinstance(paths, list):
        raise TypeError("data.cross_eval_path must be a list, e.g. [data/ciciot2023_preprocessed_good.tsv].")
    return [Path(path) for path in paths]


def cross_eval_name(path: Path, used_names: set[str]) -> str:
    base_name = path.stem
    name = base_name
    counter = 2
    while name in used_names:
        name = f"{base_name}_{counter}"
        counter += 1
    used_names.add(name)
    return name


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
    window_seconds = config["data"].get("windows_seconds", 5.0)
    df = compute_window_id(df, window_seconds=window_seconds)
    df = compute_time_elapsed(df)
    return compute_portscan_window_features(df)


def split_experiment_frame(
    full_df: pd.DataFrame,
    config: dict,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    
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
    return train_df, val_df, test_df


def load_experiment_data(config: dict) -> ExperimentData:
    data_cfg = config["data"]
    random_state = config["experiment"].get("seed", 42)

    full_df = prepare_labels(read_tsv(data_cfg["train_path"]), config)
    train_df, val_df, test_df = split_experiment_frame(full_df, config, random_state)
    train_df = prepare_features(train_df, config)
    val_df = prepare_features(val_df, config)
    test_df = prepare_features(test_df, config)

    paths = cross_eval_paths(config)
    used_names = set()
    cross_evals = []
    for path in paths:
        cross_evals.append(
            CrossEvalData(
                name=cross_eval_name(path, used_names) if len(paths) > 1 else "cross_eval",
                frame=prepare_features(prepare_labels(read_tsv(path), config), config).reset_index(drop=True),
            )
        )

    return ExperimentData(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        cross_evals=cross_evals,
    )
