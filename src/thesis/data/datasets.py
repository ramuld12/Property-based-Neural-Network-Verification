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
    window_seconds = config["data"].get("windows_seconds", 5.0)
    df = compute_time_elapsed(df)
    return compute_portscan_window_features(df, float(window_seconds))


def group_labels(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return df.groupby(group_cols, dropna=False, as_index=False)["label"].agg(lambda labels: labels.mode().iat[0])


def stratify_if_possible(labels: pd.Series) -> pd.Series | None:
    if labels.value_counts().min() < 2:
        return None
    return labels


def split_by_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = group_labels(df, group_cols)
    try:
        train_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_if_possible(groups["label"]),
        )
    except ValueError as exc:
        print(f"Warning: falling back to unstratified group split: {exc}")
        train_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=random_state,
        )
    row_groups = pd.MultiIndex.from_frame(df[group_cols])
    test_group_index = pd.MultiIndex.from_frame(test_groups[group_cols])
    test_mask = row_groups.isin(test_group_index)
    return df.loc[~test_mask].copy(), df.loc[test_mask].copy()


def split_experiment_frame(
    full_df: pd.DataFrame,
    config: dict,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    split_group_cols = data_cfg.get("split_group_cols", ["id.orig_h", "window_id"])
    missing_group_cols = [col for col in split_group_cols if col not in full_df.columns]

    if missing_group_cols:
        print(
            "Warning: falling back to stratified random split; "
            f"missing data.split_group_cols={missing_group_cols}"
        )
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

    train_val_df, test_df = split_by_groups(
        full_df,
        split_group_cols,
        test_size=data_cfg.get("test_size", 0.3),
        random_state=random_state,
    )
    train_df, val_df = split_by_groups(
        train_val_df,
        split_group_cols,
        test_size=data_cfg.get("val_size", 0.2),
        random_state=random_state,
    )
    return train_df, val_df, test_df


def load_experiment_data(config: dict) -> ExperimentData:
    data_cfg = config["data"]
    random_state = config["experiment"].get("seed", 42)
    window_seconds = float(data_cfg.get("windows_seconds", 5.0))

    full_df = prepare_labels(read_tsv(data_cfg["train_path"]), config)
    full_df = compute_window_id(full_df, window_seconds)
    train_df, val_df, test_df = split_experiment_frame(full_df, config, random_state)
    train_df = prepare_features(train_df, config)
    val_df = prepare_features(val_df, config)
    test_df = prepare_features(test_df, config)

    cross_eval = None
    if data_cfg.get("cross_eval_path"):
        cross_eval = prepare_features(prepare_labels(read_tsv(data_cfg["cross_eval_path"]), config), config)

    return ExperimentData(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        cross_eval=None if cross_eval is None else cross_eval.reset_index(drop=True),
    )
