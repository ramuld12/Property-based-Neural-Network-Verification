"""Data preprocessing utilities for network traffic classification."""

import numpy as np
import pandas as pd

MODEL_CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "conn_state",
    "history",
]

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
    "orig_pkt_rate",
    "orig_byte_rate",
    "time_elapsed",
    "valid_tcp_handshake",
    "valid_http_conn",

    # portscan features
    "uniq_dst_ports",
    "pkts_per_port",
    "scan_duration",
    "fail_ratio",
]

def filter_labels(df: pd.DataFrame, target_labels: list[str], label_to_idx: dict[str, int]):
    df = df[df["label"].isin(target_labels)].copy()
    df["label_id"] = df["label"].map(label_to_idx)
    return df

def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_df_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    print_section(f"{name} summary")
    print(f"rows: {len(df):,}")
    print(f"columns: {len(df.columns):,}")

    if "label" in df.columns:
        print("\nLabel distribution:")
        print_label_distribution(df)

def print_label_distribution(df: pd.DataFrame) -> None:
    total = len(df)
    counts = df["label"].value_counts().sort_index()

    for label, count in counts.items():
        print(f"{label:20s}  count={count:8d}  ratio={count / total:.4f}")