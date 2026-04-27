"""Data preprocessing utilities for network traffic classification."""

import numpy as np
import pandas as pd

BASE_ZEEK_FEATURES = [
    "ts",
    "uid",
    "id.orig_h",
    "id.orig_p",
    "id.resp_h",
    "id.resp_p",
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "local_orig",
    "local_resp",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "tunnel_parents",
    "ip_proto",
]

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
    "pkt_asymmetry",
    "byte_asymmetry",
    "time_elapsed",
    "flood_rate"
]


PROPERTY_BOOLEAN_FEATURES = [
    "is_tcp",
    "valid_http_conn",
    "valid_input",
    "valid_tcp_handshake",
    "valid_duration",
    "valid_packet_size",
    "valid_iat",
    "dos_http_mal_time_elapsed",
    "dos_http_mal_flood_rate",
    "portscan_many_ports",
    "portscan_few_pkts_per_port",
    "portscan_short_duration",
    "portscan_high_fail_ratio",
]

FEATURES = (
    MODEL_CATEGORICAL_FEATURES
    + MODEL_NUMERIC_FEATURES
    + PROPERTY_BOOLEAN_FEATURES
)

def balance_dataset(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Balance dataset by downsampling to minority class size.
    
    Args:
        X: Feature DataFrame
        y: Label Series
        random_state: Random seed for reproducibility
        
    Returns:
        Balanced (X, y) tuple
    """
    label_col = "label"
    df = X.copy()
    df[label_col] = y.values

    counts = df[label_col].value_counts()
    min_count = counts.min()

    balanced_parts = []
    for label in counts.index:
        sampled = df[df[label_col] == label].sample(
            n=min_count, random_state=random_state
        )
        balanced_parts.append(sampled)

    balanced_df = (
        pd.concat(balanced_parts, ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    X_bal = balanced_df.drop(columns=[label_col])
    y_bal = balanced_df[label_col]
    return X_bal, y_bal

def balance_df(df: pd.DataFrame, frac: float = 1.0) -> pd.DataFrame:
    min_count = df["label"].value_counts().min()
    min_count = int(min_count * frac)
    print(f"Balancing dataset to {min_count} rows per class")

    sampled_idx = (
        df.groupby("label")
        .sample(n=min_count, random_state=42)
        .index
    )

    balanced_df = (
        df.loc[sampled_idx]
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )

    return balanced_df

def filter_labels(df, target_labels):
    return df[df["label"].isin(target_labels)].copy()
