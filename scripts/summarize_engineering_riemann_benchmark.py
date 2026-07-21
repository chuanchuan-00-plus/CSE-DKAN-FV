"""Summarize the frozen engineering Riemann benchmark and evaluate its gates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


PRIMARY = "primitive_geometric_mean_l2"
ENGINEERING = "physics_gated_dkan_fv_engineering"
LEGACY = "physics_gated_dkan_fv_sod_neighbourhood"
FIXED = "hllc_fixed_mc_subcell"
FNO = "fno_operator"
FINE = "hllc_muscl_fine"


def geometric_mean(values: list[float]) -> float:
    return float(math.exp(statistics.mean(math.log(max(value, 1.0e-15)) for value in values)))


def percentile_interval(values: list[float]) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def median_by_case(rows: list[dict], method: str, group: str, metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["case_group"] == group and row[metric] is not None:
            grouped[row["case_id"]].append(float(row[metric]))
    return {case: float(statistics.median(values)) for case, values in grouped.items()}


def seed_geometric_means(rows: list[dict], method: str, group: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method and row["case_group"] == group:
            grouped[int(row["model_seed"])].append(float(row[PRIMARY]))
    return {seed: geometric_mean(values) for seed, values in grouped.items()}


def bootstrap_gain(
    numerator: dict[str, float],
    denominator: dict[str, float],
    *,
    statistic: str,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, list[float]]:
    cases = sorted(set(numerator) & set(denominator))
    ratios = np.asarray([numerator[case] / denominator[case] for case in cases])
    rng = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        sample = ratios[rng.integers(0, len(ratios), size=len(ratios))]
        if statistic == "median":
            bootstrap.append(float(np.median(sample)))
        elif statistic == "geometric_mean":
            bootstrap.append(float(np.exp(np.mean(np.log(np.maximum(sample, 1.0e-15))))))
        else:
            raise ValueError(f"unknown statistic: {statistic}")
    point = float(np.median(ratios)) if statistic == "median" else geometric_mean(ratios.tolist())
    return point, percentile_interval(bootstrap)


def bootstrap_median(values: list[float], *, seed: int, replicates: int = 10_000) -> tuple[float, list[float]]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = [
        float(np.median(array[rng.integers(0, len(array), size=len(array))]))
        for _ in range(replicates)
    ]
    return float(np.median(array)), percentile_interval(bootstrap)


def seeded_case_gains(rows: list[dict], method: str, baseline: str, group: str) -> list[float]:
    baseline_by_case = {
        row["case_id"]: float(row[PRIMARY])
        for row in rows
        if row["method"] == baseline and row["case_group"] == group
    }
    return [
        baseline_by_case[row["case_id"]] / float(row[PRIMARY])
        for row in rows
        if row["method"] == method and row["case_group"] == group
    ]


def aggregate_method(rows: list[dict], method: str, group: str) -> dict:
    selected = [row for row in rows if row["method"] == method and row["case_group"] == group]
    case_errors = median_by_case(rows, method, group, PRIMARY)
    case_tv = median_by_case(rows, method, group, "density_tv_excess")
    case_conservation = median_by_case(
        rows, method, group, "maximum_relative_conservation_error"
    )
    case_runtime = median_by_case(rows, method, group, "online_seconds")
    case_width = median_by_case(rows, method, group, "mean_shock_width_10_90")
    return {
        "method": method,
        "group": group,
        "rows": len(selected),
        "cases": len(case_errors),
        "primary_error_median": float(statistics.median(case_errors.values())),
        "primary_error_geometric_mean": geometric_mean(list(case_errors.values())),
        "density_tv_excess_median": float(statistics.median(case_tv.values())),
        "domain_integral_error_median": float(statistics.median(case_conservation.values())),
        "online_seconds_median": float(statistics.median(case_runtime.values())),
        "shock_width_median": float(statistics.median(case_width.values())) if case_width else None,
        "minimum_density": float(min(row["minimum_density"] for row in selected)),
        "minimum_pressure": float(min(row["minimum_pressure"] for row in selected)),
        "maximum_entropy_violation": float(
            max(row["maximum_shock_entropy_violation"] for row in selected)
        ),
        "maximum_rh_residual": float(
            max(row["maximum_normalized_rh_residual"] for row in selected)
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    rows = raw["rows"]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    methods = sorted({row["method"] for row in rows})
    groups = ("in_distribution", "stress")
    aggregates = [aggregate_method(rows, method, group) for group in groups for method in methods]

    fixed_id = median_by_case(rows, FIXED, "in_distribution", PRIMARY)
    engineering_id = median_by_case(rows, ENGINEERING, "in_distribution", PRIMARY)
    fno_id = median_by_case(rows, FNO, "in_distribution", PRIMARY)
    fixed_stress = median_by_case(rows, FIXED, "stress", PRIMARY)
    engineering_stress = median_by_case(rows, ENGINEERING, "stress", PRIMARY)

    same_ratios = seeded_case_gains(
        rows, ENGINEERING, FIXED, "in_distribution"
    )
    stress_ratios = seeded_case_gains(rows, ENGINEERING, FIXED, "stress")
    same_gain, same_ci = bootstrap_median(same_ratios, seed=2026071601)
    stress_gain, stress_ci = bootstrap_median(stress_ratios, seed=2026071602)
    fno_gain_case, fno_gain_ci = bootstrap_gain(
        fno_id, engineering_id, statistic="geometric_mean", seed=2026071603
    )
    fno_seeds = seed_geometric_means(rows, FNO, "in_distribution")
    engineering_seeds = seed_geometric_means(rows, ENGINEERING, "in_distribution")
    primary_gain = statistics.median(fno_seeds.values()) / statistics.median(
        engineering_seeds.values()
    )
    claim_methods = {ENGINEERING, FIXED, FNO}
    claim_rows = [row for row in rows if row["method"] in claim_methods]
    hard_change = max(
        float(row["maximum_cell_average_change"] or 0.0)
        for row in rows
        if row["method"] == ENGINEERING
    )
    all_positive = all(
        row["minimum_density"] > 0.0 and row["minimum_pressure"] > 0.0
        for row in claim_rows
    )
    gates = [
        {
            "gate": "primary_20_percent",
            "value": float(primary_gain),
            "threshold": 1.20,
            "passed": bool(primary_gain >= 1.20),
            "detail": "ratio of median seed-level geometric-mean errors, FNO / engineering DKAN-FV",
        },
        {
            "gate": "same_backbone_increment_gain",
            "value": same_gain,
            "threshold": 1.05,
            "passed": bool(same_gain >= 1.05),
            "ci95_low": same_ci[0],
            "ci95_high": same_ci[1],
        },
        {
            "gate": "same_backbone_increment_win_rate",
            "value": float(np.mean(np.asarray(same_ratios) > 1.0)),
            "threshold": 0.60,
            "passed": bool(np.mean(np.asarray(same_ratios) > 1.0) >= 0.60),
        },
        {
            "gate": "stress_robustness_gain",
            "value": stress_gain,
            "threshold": 1.00,
            "passed": bool(stress_gain >= 1.00),
            "ci95_low": stress_ci[0],
            "ci95_high": stress_ci[1],
        },
        {
            "gate": "stress_robustness_win_rate",
            "value": float(np.mean(np.asarray(stress_ratios) > 1.0)),
            "threshold": 0.50,
            "passed": bool(np.mean(np.asarray(stress_ratios) > 1.0) >= 0.50),
        },
        {
            "gate": "hard_conservation",
            "value": hard_change,
            "threshold": 1.0e-12,
            "passed": bool(hard_change <= 1.0e-12),
        },
        {
            "gate": "positivity",
            "value": int(all_positive),
            "threshold": 1,
            "passed": bool(all_positive),
        },
    ]
    case_rows = []
    for group in groups:
        fixed = median_by_case(rows, FIXED, group, PRIMARY)
        engineering = median_by_case(rows, ENGINEERING, group, PRIMARY)
        legacy = median_by_case(rows, LEGACY, group, PRIMARY)
        fno = median_by_case(rows, FNO, group, PRIMARY)
        fine = median_by_case(rows, FINE, group, PRIMARY)
        for case in sorted(fixed):
            case_rows.append(
                {
                    "case_id": case,
                    "case_group": group,
                    "fixed_mc_error": fixed[case],
                    "engineering_dkan_fv_error": engineering[case],
                    "legacy_dkan_fv_error": legacy[case],
                    "fno_error": fno[case],
                    "fine_hllc_error": fine[case],
                    "fixed_over_engineering_gain": fixed[case] / engineering[case],
                    "fno_over_engineering_gain": fno[case] / engineering[case],
                    "fine_over_engineering_gain": fine[case] / engineering[case],
                }
            )

    summary = {
        "protocol": protocol["protocol"],
        "protocol_status": protocol["status"],
        "row_count": len(rows),
        "methods": methods,
        "aggregates": aggregates,
        "seed_level_geometric_mean_errors": {
            "fno_in_distribution": fno_seeds,
            "engineering_dkan_fv_in_distribution": engineering_seeds,
        },
        "comparisons": {
            "primary_fno_over_engineering_dkan_fv": float(primary_gain),
            "case_bootstrap_fno_over_engineering_dkan_fv": {
                "value": fno_gain_case,
                "ci95": fno_gain_ci,
            },
            "fixed_mc_over_engineering_dkan_fv_in_distribution": {
                "paired_median": same_gain,
                "ci95": same_ci,
                "win_rate": float(np.mean(np.asarray(same_ratios) > 1.0)),
            },
            "fixed_mc_over_engineering_dkan_fv_stress": {
                "paired_median": stress_gain,
                "ci95": stress_ci,
                "win_rate": float(np.mean(np.asarray(stress_ratios) > 1.0)),
            },
        },
        "gates": gates,
        "all_predeclared_gates_passed": all(gate["passed"] for gate in gates),
        "interpretation_guardrails": [
            "Domain-integral error against the exact solution is not the same as hard reconstruction conservation.",
            "The pure-contact stress case has near-machine-zero finite-volume error and can destabilize geometric means; stress gates use paired medians.",
            "Fine-grid HLLC is an accuracy-cost reference and is not a superiority target.",
        ],
        "offline_seconds": {
            "fno": raw["fno_pretraining"],
            "hybrid": raw["hybrid_pretraining"],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(output / "method_group_summary.csv", aggregates)
    write_csv(output / "gate_results.csv", gates)
    write_csv(output / "case_comparison.csv", case_rows)
    write_csv(output / "all_metrics.csv", rows)
    print(
        json.dumps(
            {
                "output": str(output),
                "all_predeclared_gates_passed": summary["all_predeclared_gates_passed"],
                "primary_gain": primary_gain,
                "same_backbone_gain": same_gain,
                "stress_gain": stress_gain,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
