"""Data preprocessing utilities for network traffic classification."""

import numpy as np
import pandas as pd


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

    df.dropna(inplace=True)
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
