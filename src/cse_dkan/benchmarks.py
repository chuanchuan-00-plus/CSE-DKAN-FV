"""Benchmark definitions and deterministic reference-data generation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from .reference import BurgersWENO


@dataclass(frozen=True)
class PeriodicBurgersProblem:
    x_left: float = -1.0
    x_right: float = 1.0
    final_time: float = 0.6
    mean: float = 0.2
    amplitude: float = 1.0
    origin: float = 0.0
    viscosity: float = 0.0

    @property
    def shock_formation_time(self) -> float:
        return 1.0 / (math.pi * self.amplitude)

    @property
    def shock_origin(self) -> float:
        return self.origin

    @property
    def initial_shock_speed(self) -> float:
        return self.mean

    def initial_numpy(self, x: np.ndarray) -> np.ndarray:
        return self.mean - self.amplitude * np.sin(math.pi * (x - self.origin))

    def initial_torch(self, x: Tensor) -> Tensor:
        return self.mean - self.amplitude * torch.sin(math.pi * (x - self.origin))

    def reference(
        self,
        n_cells: int = 4096,
        snapshot_times: tuple[float, ...] = (),
    ) -> tuple[np.ndarray, dict[float, np.ndarray]]:
        solver = BurgersWENO(
            x_left=self.x_left,
            x_right=self.x_right,
            n_cells=n_cells,
            viscosity=self.viscosity,
        )
        return solver.x, solver.solve(self.initial_numpy, self.final_time, snapshot_times)
