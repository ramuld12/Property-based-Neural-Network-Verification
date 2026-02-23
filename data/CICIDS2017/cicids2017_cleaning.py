
import os
import numpy as np
import pandas as pd
import kagglehub


def download_dataset():
    base_path = kagglehub.dataset_download("chethuhn/network-intrusion-dataset")
    files = [os.path.join(base_path, f) for f in os.listdir(base_path)]
    print("Total files:", len(files))
    return base_path, files


def load_dataset(files):
    df_list = []
    for file in files:
        print("Reading:", file)
        df = pd.read_csv(file)
        df_list.append(df)
    full_df = pd.concat(df_list, ignore_index=True)
    print("Full Shape:", full_df.shape)
    return full_df

def clean_dataset(full_df):
    print("Before cleaning:", full_df.shape)
    full_df.columns = full_df.columns.str.strip()
    # Remove duplicates
    full_df.drop_duplicates(inplace=True)
    # Duplicate Column and Destination Port; Removal
    full_df.drop(columns=["Fwd Header Length.1", "Destination Port"], inplace=True)
    # Replace infinity values with NaN
    full_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Drop missing values
    full_df.dropna(inplace=True)
    print("After cleaning:", full_df.shape)
    print(full_df['Label'].value_counts())
    return full_df


def load_and_clean():
    base_path, files = download_dataset()
    full_df = load_dataset(files)
    full_df = clean_dataset(full_df)
    return full_df