from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import yaml


def format_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def make_command(method: str, base_config: str, params: dict) -> str:
    overrides = " ".join(f"--set {key}={format_value(value)}" for key, value in params.items())
    return f"python -m thesis.cli run {method} --config {base_config} {overrides}".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", default="configs/sweeps/property_grid.yaml")
    parser.add_argument("--out")
    args = parser.parse_args()

    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))
    keys = list(sweep["grid"].keys())
    values = [sweep["grid"][key] for key in keys]
    out_path = Path(args.out or sweep["output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    commands = []
    for combination in product(*values):
        params = dict(zip(keys, combination))
        commands.append(make_command(sweep["method"], sweep["base_config"], params))

    out_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(commands)} commands to {out_path}")


if __name__ == "__main__":
    main()
