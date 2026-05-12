from __future__ import annotations

from pathlib import Path


def evaluate_run(run_dir: Path, cross_data_config: Path | None = None) -> None:
    print(f"Existing run outputs are in: {run_dir}")
    if cross_data_config is not None:
        print(f"Cross-data config requested: {cross_data_config}")
