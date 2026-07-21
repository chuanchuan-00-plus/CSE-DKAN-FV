"""Audit 2048-cell mechanism references against the frozen 4096-cell check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_euler_mechanism_benchmark import (
    interpolate_primitive,
    load_protocol,
    normalized_l2,
    solve_snapshots,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-cache")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    protocol = load_protocol(Path(args.protocol))
    configuration = protocol["benchmark"]
    device = torch.device(args.device)
    target_points = int(configuration["evaluation_points"])
    target_x = (np.arange(target_points, dtype=np.float64) + 0.5) / target_points
    cache = Path(args.state_cache) if args.state_cache else None
    rows = []
    for case in protocol["cases"]:
        if case["kind"] not in {"shu_osher", "woodward_colella"}:
            continue
        times = np.asarray([0.0, float(case["time"])])
        coarse_states, coarse_seconds = solve_snapshots(
            case,
            int(configuration["fine_reference_cells"]),
            times,
            device,
            cache,
            float(configuration["cfl"]),
        )
        check_states, check_seconds = solve_snapshots(
            case,
            int(configuration["reference_convergence_cells"]),
            times,
            device,
            cache,
            float(configuration["cfl"]),
        )
        coarse = interpolate_primitive(coarse_states[-1:], target_x)[0]
        check = interpolate_primitive(check_states[-1:], target_x)[0]
        component = [normalized_l2(coarse[:, index], check[:, index]) for index in range(3)]
        range_normalized_linf = [
            float(
                np.max(np.abs(coarse[:, index] - check[:, index]))
                / max(np.ptp(check[:, index]), np.max(np.abs(check[:, index])), 1.0e-8)
            )
            for index in range(3)
        ]
        rows.append(
            {
                "case_id": case["id"],
                "reference_cells": int(configuration["fine_reference_cells"]),
                "check_cells": int(configuration["reference_convergence_cells"]),
                "density_normalized_l2": component[0],
                "velocity_normalized_l2": component[1],
                "pressure_normalized_l2": component[2],
                "primitive_rms_l2": float(np.sqrt(np.mean(np.square(component)))),
                "maximum_normalized_pointwise_disagreement": float(
                    np.max(np.abs(coarse - check) / np.maximum(np.abs(check), 1.0e-8))
                ),
                "maximum_range_normalized_linf_disagreement": max(range_normalized_linf),
                "reference_seconds": coarse_seconds,
                "check_seconds": check_seconds,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    Path(args.output).write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
