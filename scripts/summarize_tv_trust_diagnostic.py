"""Summarize the post-registration TV trust-budget diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


PRIMARY = "primitive_geometric_mean_l2"
FIXED = "hllc_fixed_mc_subcell"


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.mean(math.log(max(value, 1.0e-15)) for value in values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))["rows"]
    methods = sorted(
        method
        for method in {row["method"] for row in rows}
        if method == FIXED or "physics_gated_dkan_fv_engineering" in method
    )
    summary = []
    for group in ("in_distribution", "stress"):
        fixed = {
            row["case_id"]: float(row[PRIMARY])
            for row in rows
            if row["case_group"] == group and row["method"] == FIXED
        }
        for method in methods:
            selected = [
                row
                for row in rows
                if row["case_group"] == group and row["method"] == method
            ]
            errors = [float(row[PRIMARY]) for row in selected]
            gains = (
                [fixed[row["case_id"]] / float(row[PRIMARY]) for row in selected]
                if method != FIXED
                else []
            )
            widths = [
                float(row["mean_shock_width_10_90"])
                for row in selected
                if row["mean_shock_width_10_90"] is not None
            ]
            multipliers = [
                float(row["tv_trust_multiplier"])
                for row in selected
                if row.get("tv_trust_multiplier") is not None
            ]
            summary.append(
                {
                    "group": group,
                    "method": method,
                    "primary_error_median": statistics.median(errors),
                    "primary_error_geometric_mean": geometric_mean(errors),
                    "fixed_over_method_gain_median": statistics.median(gains) if gains else None,
                    "fixed_over_method_win_rate": (
                        sum(gain > 1.0 for gain in gains) / len(gains) if gains else None
                    ),
                    "density_tv_excess_median": statistics.median(
                        float(row["density_tv_excess"]) for row in selected
                    ),
                    "shock_width_median": statistics.median(widths) if widths else None,
                    "tv_trust_multiplier_median": (
                        statistics.median(multipliers) if multipliers else None
                    ),
                    "minimum_density": min(float(row["minimum_density"]) for row in selected),
                    "minimum_pressure": min(float(row["minimum_pressure"]) for row in selected),
                    "maximum_cell_average_change": max(
                        float(row["maximum_cell_average_change"] or 0.0) for row in selected
                    ),
                    "online_seconds_median": statistics.median(
                        float(row["online_seconds"]) for row in selected
                    ),
                }
            )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(summary[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "rows": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
