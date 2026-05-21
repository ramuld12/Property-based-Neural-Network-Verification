from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    for override in overrides:
        key, value = override.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = parse_value(value)
    return config


def parse_value(value: str):
    return yaml.safe_load(value)
