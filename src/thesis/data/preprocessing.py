from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from thesis.data.features import BOOLEAN_FEATURES
from thesis.properties.specs import make_scaled_attack_specs


@dataclass
class PropertyData:
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    cross_eval_df: pd.DataFrame | None
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    cross_eval_loader: DataLoader | None
    features: list[str]
    labels: list[str]
    scaler: MinMaxScaler
    scaled_attack_specs: dict
    scale_cols: list[str]
    clip_lower: pd.Series
    clip_upper: pd.Series


@dataclass
class BaselineData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    x_cross_eval: np.ndarray | None
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    y_cross_eval: np.ndarray | None
    features: list[str]
    categorical_cols: list[str]
    continuous_cols: list[str]
    labels: list[str]
    scaler: StandardScaler
    ordinal_encoder: OrdinalEncoder


def make_loader(df: pd.DataFrame, feature_cols: list[str], batch_size: int, shuffle: bool = False) -> DataLoader:
    x = torch.tensor(df[feature_cols].to_numpy(), dtype=torch.float32)
    y = torch.tensor(df["label_id"].to_numpy(), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


def fit_property_data(data, config: dict, feature_cols: list[str]) -> PropertyData:
    batch_size = config["model"]["batch_size"]
    scale_cols = [col for col in feature_cols if col not in BOOLEAN_FEATURES]
    train_df = data.train.copy()
    val_df = data.val.copy()
    test_df = data.test.copy()
    cross_eval_df = None if data.cross_eval is None else data.cross_eval.copy()

    for df in [train_df, val_df, test_df] + ([] if cross_eval_df is None else [cross_eval_df]):
        df[scale_cols] = df[scale_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    clip_lower = train_df[scale_cols].quantile(0.01)
    clip_upper = train_df[scale_cols].quantile(0.99)
    scaler = MinMaxScaler()

    train_df[scale_cols] = train_df[scale_cols].clip(lower=clip_lower, upper=clip_upper, axis=1)
    val_df[scale_cols] = val_df[scale_cols].clip(lower=clip_lower, upper=clip_upper, axis=1)
    test_df[scale_cols] = test_df[scale_cols].clip(lower=clip_lower, upper=clip_upper, axis=1)

    train_df[scale_cols] = scaler.fit_transform(train_df[scale_cols])
    val_df[scale_cols] = scaler.transform(val_df[scale_cols])
    test_df[scale_cols] = scaler.transform(test_df[scale_cols])

    if cross_eval_df is not None:
        cross_eval_df[scale_cols] = cross_eval_df[scale_cols].clip(lower=clip_lower, upper=clip_upper, axis=1)
        cross_eval_df[scale_cols] = scaler.transform(cross_eval_df[scale_cols])

    return PropertyData(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cross_eval_df=cross_eval_df,
        train_loader=make_loader(train_df, feature_cols, batch_size, shuffle=True),
        val_loader=make_loader(val_df, feature_cols, batch_size),
        test_loader=make_loader(test_df, feature_cols, batch_size),
        cross_eval_loader=None if cross_eval_df is None else make_loader(cross_eval_df, feature_cols, batch_size),
        features=feature_cols,
        labels=config["data"]["labels"],
        scaler=scaler,
        scaled_attack_specs=make_scaled_attack_specs(config["attack_specs"], scaler, scale_cols),
        scale_cols=scale_cols,
        clip_lower=clip_lower,
        clip_upper=clip_upper,
    )


def fit_baseline_data(data, config: dict, feature_cols: list[str], categorical_cols: list[str], continuous_cols: list[str]) -> BaselineData:
    train_df = data.train.copy()
    val_df = data.val.copy()
    test_df = data.test.copy()
    cross_eval_df = None if data.cross_eval is None else data.cross_eval.copy()
    frames = [train_df, val_df, test_df] + ([] if cross_eval_df is None else [cross_eval_df])

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train_df[categorical_cols] = encoder.fit_transform(train_df[categorical_cols])
    for df in frames[1:]:
        df[categorical_cols] = encoder.transform(df[categorical_cols])

    for df in frames:
        df[continuous_cols] = df[continuous_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    for df in frames[1:]:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return BaselineData(
        x_train=train_df[feature_cols].to_numpy(dtype=np.float32),
        x_val=val_df[feature_cols].to_numpy(dtype=np.float32),
        x_test=test_df[feature_cols].to_numpy(dtype=np.float32),
        x_cross_eval=None if cross_eval_df is None else cross_eval_df[feature_cols].to_numpy(dtype=np.float32),
        y_train=train_df["label_id"].to_numpy(dtype=np.int64),
        y_val=val_df["label_id"].to_numpy(dtype=np.int64),
        y_test=test_df["label_id"].to_numpy(dtype=np.int64),
        y_cross_eval=None if cross_eval_df is None else cross_eval_df["label_id"].to_numpy(dtype=np.int64),
        features=feature_cols,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
        labels=config["data"]["labels"],
        scaler=scaler,
        ordinal_encoder=encoder,
    )


def torch_loaders_from_arrays(data: BaselineData, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train = TensorDataset(torch.tensor(data.x_train).unsqueeze(1), torch.tensor(data.y_train))
    val = TensorDataset(torch.tensor(data.x_val).unsqueeze(1), torch.tensor(data.y_val))
    test = TensorDataset(torch.tensor(data.x_test).unsqueeze(1), torch.tensor(data.y_test))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test, batch_size=batch_size),
    )
