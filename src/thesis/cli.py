from __future__ import annotations

import argparse
from pathlib import Path

from thesis.config import apply_overrides, load_config
from thesis.experiments.evaluate import evaluate_run


def main() -> None:
    parser = argparse.ArgumentParser(prog="thesis")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("method", choices=["baseline", "properties"])
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--set", dest="overrides", action="append", default=[])

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--model", required=True, type=Path, help="Path to a saved model.joblib file")
    eval_parser.add_argument("--cross-data", nargs="+", type=Path)

    args = parser.parse_args()

    if args.command == "run":
        config = apply_overrides(load_config(args.config), args.overrides)
        if args.method == "baseline":
            from thesis.experiments.baseline import run_baseline

            run_baseline(config)
        else:
            from thesis.experiments.properties import run_properties

            run_properties(config)

    elif args.command == "evaluate":
        evaluate_run(args.model, args.cross_data)


if __name__ == "__main__":
    main()
