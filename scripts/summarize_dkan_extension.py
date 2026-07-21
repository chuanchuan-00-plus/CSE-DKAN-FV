"""Summarize the canonical strong-form DKAN-PINN extension."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def stats(rows: list[dict], names: list[str]) -> dict:
    return {
        name: {
            "median": float(np.median([row[name] for row in rows])),
            "minimum": float(np.min([row[name] for row in rows])),
            "maximum": float(np.max([row[name] for row in rows])),
        }
        for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.protocol, encoding="utf-8") as handle:
        protocol = json.load(handle)
    result = {
        "protocol": protocol["protocol"],
        "status": "completed_descriptive",
        "configuration_audit_pass": True,
    }
    for equation in ("burgers", "euler"):
        rows = []
        pattern = str(
            Path(args.root) / equation / "batch_seed_*" / "dkan_pinn"
            / "seed_*" / "result.json"
        )
        for path in glob.glob(pattern):
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            config = payload["config"]
            if (
                config["seed"] not in protocol["seeds"]
                or config["dtype"] != "float64"
                or config["steps"] != protocol[equation]["steps"]
            ):
                raise ValueError(f"configuration mismatch: {path}")
            row = {"seed": config["seed"], **payload["metrics"]}
            if equation == "euler":
                row["primitive_geometric_mean_l2"] = math.exp(
                    np.mean(np.log([
                        row["density_l2"], row["velocity_l2"], row["pressure_l2"]
                    ]))
                )
            rows.append(row)
        if len(rows) != len(protocol["seeds"]):
            raise ValueError(f"missing {equation} seeds")
        names = (
            [
                "normalized_l2", "shock_width_10_90", "max_mass_drift",
                "cv_balance_rmse", "entropy_violation_max", "training_seconds",
                "inference_seconds",
            ]
            if equation == "burgers"
            else [
                "primitive_geometric_mean_l2", "density_l2", "velocity_l2",
                "pressure_l2", "shock_width_10_90", "max_mass_balance_error",
                "max_momentum_balance_error", "max_energy_balance_error",
                "shock_entropy_violation", "training_seconds", "inference_seconds",
            ]
        )
        result[equation] = {
            "optimized_parameters": int(rows[0]["optimized_parameters"]),
            "metrics": stats(rows, names),
            "raw": rows,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"protocol": result["protocol"], "status": result["status"]}))


if __name__ == "__main__":
    main()
