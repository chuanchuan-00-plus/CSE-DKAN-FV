from __future__ import annotations

import argparse
import json

from cse_dkan.euler_training import EulerTrainingConfig, train_euler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = train_euler(EulerTrainingConfig.from_json(arguments.config), arguments.output)
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
