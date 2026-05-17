from __future__ import annotations

import numpy as np
import pandas as pd

FLOW_NUMERIC_FEATURES = [
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
    "pkts_per_port",
    "scan_duration",
    "fail_ratio",
]

BOOLEAN_FEATURES = ["valid_tcp_handshake", "valid_http_conn"]

SHARED_MODEL_FEATURES = FLOW_NUMERIC_FEATURES + ENGINEERED_FEATURES
DEFAULT_PROPERTY_FROZEN_FEATURES = [
    "valid_tcp_handshake",
    "valid_http_conn",
    "time_elapsed",
    "uniq_dst_ports",
    "pkts_per_port",
    "scan_duration",
    "fail_ratio",
]

PORTSCAN_FAILED_STATES = {"S0", "REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH", "SH", "SHR"}


def label_to_idx(labels: list[str]) -> dict[str, int]:
    return {label: i for i, label in enumerate(labels)}


def filter_labels(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    mapping = label_to_idx(labels)
    df = df[df["label"].isin(labels)].copy()
    df["label_id"] = df["label"].map(mapping)
    return df


def make_attack_label_df(df: pd.DataFrame, labels: list[str], attack_labels: list[str]) -> pd.DataFrame:
    explicit_labels = set(labels)
    generic_attack_sources = set(attack_labels) - explicit_labels
    df = df[df["label"].isin(explicit_labels | generic_attack_sources)].copy()
    df["attack_type"] = df["label"]
    df["label"] = np.where(df["label"].isin(generic_attack_sources), "ATTACK", df["label"])
    df["label_id"] = df["label"].map(label_to_idx(labels))
    return df


def make_binary_attack_df(df: pd.DataFrame, attack_labels: list[str]) -> pd.DataFrame:
    return make_attack_label_df(df, ["BENIGN", "ATTACK"], attack_labels)


def compute_window_id(df: pd.DataFrame, window_seconds: float) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
    df["window_id"] = (df["ts"] // window_seconds).astype(int)
    return df


def compute_time_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
    df = df.sort_values(["id.orig_h", "id.resp_h", "ts"]).reset_index(drop=True)
    df["time_elapsed"] = df.groupby(["id.orig_h", "id.resp_h"])["ts"].diff().fillna(999999.0)
    return df


def compute_portscan_window_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").fillna(0.0)
    df["orig_pkts"] = pd.to_numeric(df["orig_pkts"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["flow_end_ts"] = df["ts"] + df["duration"]
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
