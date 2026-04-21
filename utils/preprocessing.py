"""Data preprocessing utilities for network traffic classification."""

import numpy as np
import pandas as pd

FEATURES = [
    "proto",
    "service",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "conn_state",
    "missed_bytes",
    "history",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",

    # shared engineered features
    "orig_pkt_rate",
    "orig_byte_rate",
    "pkt_asymmetry",
    "byte_asymmetry",
    "time_elapsed",
    "flood_rate",
    "valid_tcp_handshake_feature",
    "is_http",

    # portscan features
    "uniq_dst_ports",
    "pkts_per_port",
    "scan_duration",
    "fail_ratio",

    # UDP flood feature
    "is_udp",
    "udp_conn_count",
    "udp_packets",       
    "udp_rate",           
    "unique_src_ips",   

    # SYN flood features
    "syn_duration",
    "syn_conn_count",
    "syn_count",
    "syn_rate",
    "half_open_count",
    "source_ip_count",
]

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

def filter_labels(df, target_labels):
    return df[df["label"].isin(target_labels)].copy()
