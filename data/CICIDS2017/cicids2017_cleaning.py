import numpy as np
import pandas as pd


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

