from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_CATEGORICAL_FEATURES = ["proto", "service", "conn_state", "history"]

MODEL_NUMERIC_FEATURES = [
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

ENGINEERED_FEATURES = [
    "time_elapsed",
    "valid_tcp_handshake",
    "valid_http_conn",
    "uniq_dst_ports",
    "scan_duration",
    "fail_ratio",
]

BOOLEAN_FEATURES = ["valid_tcp_handshake", "valid_http_conn"]


def property_features(config: dict) -> list[str]:
    return MODEL_NUMERIC_FEATURES + ENGINEERED_FEATURES


def baseline_features(config: dict) -> tuple[list[str], list[str], list[str]]:
    features = config["features"]
    categorical = features.get("categorical", MODEL_CATEGORICAL_FEATURES)
    numeric = features.get("numeric", MODEL_NUMERIC_FEATURES)
    engineered = features.get("engineered", []) if features.get("set") == "engineered" else []
    return categorical + numeric + engineered, categorical, numeric + engineered


def label_to_idx(labels: list[str]) -> dict[str, int]:
    return {label: i for i, label in enumerate(labels)}


def filter_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    mapping = label_to_idx(labels)
    df = df[df["label"].isin(labels)].copy()
    df["label_id"] = df["label"].map(mapping)
    return df


def make_binary_attack_df(df: pd.DataFrame, attack_labels: list[str]) -> pd.DataFrame:
    df = df[df["label"].isin(["BENIGN"] + attack_labels)].copy()
    df["attack_type"] = df["label"]
    df["label"] = np.where(df["label"] == "BENIGN", "BENIGN", "ATTACK")
    df["label_id"] = df["label"].map({"BENIGN": 0, "ATTACK": 1})
    return df
