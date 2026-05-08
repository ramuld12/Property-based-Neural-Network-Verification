from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_run_dir(config: dict) -> Path:
    root = Path(config["output"]["root"])
    run_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = root / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_config(config: dict, run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    print(f"Saved model to: {path}")
