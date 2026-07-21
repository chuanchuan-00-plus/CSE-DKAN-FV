"""Command-line entry point for one preregistered Burgers training run."""

from __future__ import annotations

import argparse
import json

from cse_dkan.training import BurgersTrainingConfig, train_burgers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    config = BurgersTrainingConfig.from_json(arguments.config)
    result = train_burgers(config, arguments.output)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
