"""Audit reduced high-viscosity WENO references against doubled resolution."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cse_dkan.benchmarks import PeriodicBurgersProblem
from cse_dkan.metrics import normalized_lp


def cached_reference(
    amplitude: float, viscosity: float, cells: int, cache_root: Path
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    path = cache_root / f"a_{amplitude:.17g}_nu_{viscosity:.17g}_n_{cells}.npz"
    if path.exists():
        cached = np.load(path)
        return cached["x"], cached["u"], 0.0, True
    problem = PeriodicBurgersProblem(amplitude=amplitude, viscosity=viscosity)
    started = time.perf_counter()
    x, snapshots = problem.reference(cells)
    elapsed = time.perf_counter() - started
    u = snapshots[problem.final_time]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=x, u=u)
    return x, u, elapsed, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--amplitudes", nargs="+", type=float, required=True)
    parser.add_argument("--evaluation-cells", type=int, default=1024)
    parser.add_argument("--tolerance", type=float, default=5.0e-4)
    args = parser.parse_args()

    cache = Path(args.cache)
    evaluation_x = -1.0 + (
        np.arange(args.evaluation_cells, dtype=np.float64) + 0.5
    ) * (2.0 / args.evaluation_cells)
    rows = []
    for viscosity, coarse_cells, fine_cells in (
        (3.0e-3, 2048, 4096),
        (1.0e-2, 1024, 2048),
    ):
        for amplitude in args.amplitudes:
            coarse_x, coarse_u, coarse_seconds, coarse_cached = cached_reference(
                amplitude, viscosity, coarse_cells, cache
            )
            fine_x, fine_u, fine_seconds, fine_cached = cached_reference(
                amplitude, viscosity, fine_cells, cache
            )
            coarse = np.interp(evaluation_x, coarse_x, coarse_u, period=2.0)
            fine = np.interp(evaluation_x, fine_x, fine_u, period=2.0)
            disagreement = normalized_lp(coarse, fine, 2)
            rows.append(
                {
                    "amplitude": amplitude,
                    "viscosity": viscosity,
                    "coarse_reference_cells": coarse_cells,
                    "fine_reference_cells": fine_cells,
                    "normalized_l2_disagreement": disagreement,
                    "coarse_seconds": coarse_seconds,
                    "fine_seconds": fine_seconds,
                    "coarse_cache_hit": coarse_cached,
                    "fine_cache_hit": fine_cached,
                    "pass": disagreement <= args.tolerance,
                }
            )
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    result = {
        "tolerance": args.tolerance,
        "row_count": len(rows),
        "maximum_normalized_l2_disagreement": max(
            row["normalized_l2_disagreement"] for row in rows
        ),
        "all_pass": all(row["pass"] for row in rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
