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

PROPERTY_FROZEN_FEATURES = ["valid_tcp_handshake", "valid_http_conn", "time_elapsed"]

PORTSCAN_FAILED_STATES = {"S0", "REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR"}


def property_features(config: dict) -> list[str]:
    return MODEL_NUMERIC_FEATURES + ENGINEERED_FEATURES


def baseline_features(config: dict) -> tuple[list[str], list[str], list[str]]:
    categorical = MODEL_CATEGORICAL_FEATURES
    numeric = MODEL_NUMERIC_FEATURES
    engineered = ENGINEERED_FEATURES
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


def recompute_portscan_window_features(df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
    df["orig_pkts"] = pd.to_numeric(df["orig_pkts"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["flow_end_ts"] = df["ts"] + df["duration"]
    df["window_id"] = (df["ts"] // window_seconds).astype(int)
    df["is_failed_conn"] = df["conn_state"].astype(str).isin(PORTSCAN_FAILED_STATES).astype(int)

    agg = (
        df.groupby(["id.orig_h", "window_id"])
        .agg(
            uniq_dst_ports=("id.resp_p", "nunique"),
            total_orig_pkts=("orig_pkts", "sum"),
            window_start_ts=("ts", "min"),
            window_end_ts=("flow_end_ts", "max"),
            total_flows=("id.orig_h", "size"),
            failed_flows=("is_failed_conn", "sum"),
        )
        .reset_index()
    )
    agg["pkts_per_port"] = (agg["total_orig_pkts"] / agg["uniq_dst_ports"].replace(0, np.nan)).fillna(0.0)
    agg["scan_duration"] = agg["window_end_ts"] - agg["window_start_ts"]
    agg["fail_ratio"] = (agg["failed_flows"] / agg["total_flows"].replace(0, np.nan)).fillna(0.0)

    df = df.drop(columns=["flow_end_ts", "uniq_dst_ports", "pkts_per_port", "scan_duration", "fail_ratio"], errors="ignore")
    return df.merge(
        agg[["id.orig_h", "window_id", "uniq_dst_ports", "pkts_per_port", "scan_duration", "fail_ratio"]],
        on=["id.orig_h", "window_id"],
        how="left",
    )
