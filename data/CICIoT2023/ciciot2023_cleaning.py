

import os
import numpy as np
import pandas as pd


def list_csv_files(base_path):
    files = []
    print("Listing CSV files in:", base_path)
    for root, _, filenames in os.walk(base_path):
        for f in filenames:
            if f.lower().endswith(".csv"):
                files.append(os.path.join(root, f))

    print("Total files:", len(files))
    return files


def load_dataset(files):
    df_list = []

    for file in files:
        print("Reading:", file)
        df = pd.read_csv(file)

        # If Label isn't present, infer from path/filename
        if "Label" not in df.columns:
            lower = file.lower()
            if "benign" in lower:
                df["Label"] = "BENIGN"
            elif "dos_http" in lower or "dos-http" in lower or "dos http" in lower:
                df["Label"] = "dos_http"
            else:
                df["Label"] = "UNKNOWN"

        df_list.append(df)

    full_df = pd.concat(df_list, ignore_index=True)
    print("Full Shape:", full_df.shape)
    return full_df


def clean_dataset(full_df):
    raw_x, raw_y = full_df.shape
    print("Before cleaning:", full_df.shape)

    full_df.columns = full_df.columns.str.strip()

    # Remove duplicates
    full_df.drop_duplicates(inplace=True)
    print("Removed "+ str(raw_x - full_df.shape[0]) + " duplicates")

    # Replace infinity values with NaN
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop missing values
    full_df.dropna(inplace=True)
    print("Dropped " + str(raw_x - full_df.shape[0]) + " missing values")

    print("After cleaning:", full_df.shape)

    print(full_df["Label"].value_counts())

    return full_df


def load_and_clean(base_path):
    """
    - list files
    - load
    - clean
    """
    files = list_csv_files(base_path)
    full_df = load_dataset(files)
    full_df = clean_dataset(full_df)
    return full_df