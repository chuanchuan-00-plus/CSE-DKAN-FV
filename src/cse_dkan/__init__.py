"""CSE-DKAN-PINN research package."""

from .metrics import normalized_lp, shock_width_10_90, total_variation_excess
from .benchmarks import PeriodicBurgersProblem
from .models import (
    AdaptiveFourierFeatures,
    BurgersInitialConditionAnsatz,
    ConservativeSpectralShockBurgersAnsatz,
    ViscousLayerConservativeSpectralBurgersAnsatz,
    DKAN,
    DKANLayer,
    MLP,
    LowRankShockBurgersAnsatz,
    ShockExplicitBurgersAnsatz,
)
from .reference import BurgersWENO, ExactSodSolver

__all__ = [
    "AdaptiveFourierFeatures",
    "BurgersInitialConditionAnsatz",
    "ConservativeSpectralShockBurgersAnsatz",
    "ViscousLayerConservativeSpectralBurgersAnsatz",
    "BurgersWENO",
    "DKAN",
    "DKANLayer",
    "ExactSodSolver",
    "MLP",
    "LowRankShockBurgersAnsatz",
    "PeriodicBurgersProblem",
    "ShockExplicitBurgersAnsatz",
    "normalized_lp",
    "shock_width_10_90",
    "total_variation_excess",
]
