import argparse
import json
from pathlib import Path

import yaml

from src.latent_evaluation import evaluate_from_config
from src.training import train_from_config


def load_config(config_path: str | Path) -> dict:
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))


def run_train(config_path: str | Path) -> Path:
    return train_from_config(load_config(config_path))


def run_eval(config_path: str | Path) -> dict:
    return evaluate_from_config(load_config(config_path))


def run_train_eval(config_path: str | Path) -> dict:
    config = load_config(config_path)
    checkpoint = train_from_config(config)
    return evaluate_from_config(config, checkpoint_path=checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latent CoT benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "eval", "train-eval"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True)
    args = parser.parse_args()

    runners = {
        "train": run_train,
        "eval": run_eval,
        "train-eval": run_train_eval,
    }
    print(json.dumps(runners[args.command](args.config), default=str, indent=2))


if __name__ == "__main__":
    main()
