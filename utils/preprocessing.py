"""Data preprocessing utilities for network traffic classification."""

import numpy as np
import pandas as pd

FEATURES = [
    "id.resp_p",
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
    "orig_pkt_rate", # New feature
    "orig_byte_rate", # New feature
    "pkt_asymmetry", # New feature
    "byte_asymmetry", # New feature
    "time_elapsed", # New feature
    "flood_rate", # New feature
]


def normalize_label_name(label: str) -> str:
    """Normalize label names by standardizing format.
    
    Args:
        label: Raw label string
        
    Returns:
        Normalized label string (uppercase with underscores)
    """
    label = str(label).strip()
    return (
        label.replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .upper()
    )

def load_cicids2017_data() -> pd.DataFrame:
    df_cicids2017_wednesday = load_and_preprocess_data("../data/CICIDS2017/wednesday_labeled.tsv")
    df_cicids2017_friday = load_and_preprocess_data("../data/CICIDS2017/friday_labeled.tsv")
    return pd.concat([df_cicids2017_wednesday, df_cicids2017_friday], ignore_index=True)

def load_ciciot2023_data() -> pd.DataFrame:
    return load_and_preprocess_data("../data/CICIoT2023/ciciot2023_labeled_conn.tsv")

def load_and_preprocess_data(datapath: str) -> pd.DataFrame:
    """Load and preprocess data from TSV file.
    
    Args:
        datapath: Path to TSV file
        
    Returns:
        Cleaned DataFrame with normalized labels
    """
    df = pd.read_csv(datapath, on_bad_lines="skip", delimiter="\t")
    df.columns = df.columns.str.strip()

    # Remove duplicates and bad numeric values
    df.drop_duplicates(inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["label"] = df["label"].astype(str).map(normalize_label_name)

    # Convert numeric columns and compute new features
    df.dropna(inplace=True)
    for col in [
            "ts",
            "duration",
            "orig_bytes",
            "resp_bytes",
            "missed_bytes",
            "orig_pkts",
            "orig_ip_bytes",
            "resp_pkts",
            "resp_ip_bytes",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = compute_and_add_time_elapsed(df)
    duration_safe = df["duration"].replace(0, 1e-6)

    df["orig_pkt_rate"] = df["orig_pkts"] / duration_safe
    df["orig_byte_rate"] = df["orig_bytes"] / duration_safe
    df["pkt_asymmetry"] = df["orig_pkts"] / (df["resp_pkts"] + 1.0)
    df["byte_asymmetry"] = df["orig_bytes"] / (df["resp_bytes"] + 1.0)
    df["flood_rate"] = df["orig_bytes"] / duration_safe

    return df


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

def compute_and_add_time_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["id.orig_h", "id.resp_h", "ts"]).reset_index(drop=True)
    df["time_elapsed"] = (
        df.groupby(["id.orig_h", "id.resp_h"])["ts"]
        .diff()
        .fillna(999999.0)
    )
    return df
