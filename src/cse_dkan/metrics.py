"""Metrics for state accuracy, shock resolution, and conservation."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def normalized_lp(prediction: Array, reference: Array, order: int | float = 2, eps: float = 1.0e-14) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference must have identical shapes")
    error = prediction - reference
    if order == np.inf:
        return float(np.max(np.abs(error)) / (np.max(np.abs(reference)) + eps))
    return float(np.linalg.norm(error.ravel(), ord=order) / (np.linalg.norm(reference.ravel(), ord=order) + eps))


def total_variation(values: Array) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(np.sum(np.abs(np.diff(values))))


def total_variation_excess(prediction: Array, reference: Array, eps: float = 1.0e-14) -> float:
    tv_reference = total_variation(reference)
    return max(0.0, (total_variation(prediction) - tv_reference) / (tv_reference + eps))


def _crossing_location(x: Array, y: Array, level: float) -> float:
    shifted = y - level
    candidates = np.where(shifted[:-1] * shifted[1:] <= 0.0)[0]
    if candidates.size == 0:
        raise ValueError(f"profile does not cross level {level}")
    index = int(candidates[0])
    y0, y1 = y[index], y[index + 1]
    if abs(y1 - y0) < 1.0e-14:
        return float(0.5 * (x[index] + x[index + 1]))
    weight = (level - y0) / (y1 - y0)
    return float(x[index] + weight * (x[index + 1] - x[index]))


def shock_width_10_90(
    x: Array,
    profile: Array,
    left_state: float,
    right_state: float,
    window: tuple[float, float] | None = None,
) -> float:
    """Return the 10%-90% transition width in a monotone shock window."""
    x = np.asarray(x, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if x.ndim != 1 or profile.shape != x.shape:
        raise ValueError("x and profile must be matching one-dimensional arrays")
    if window is not None:
        mask = (x >= window[0]) & (x <= window[1])
        x, profile = x[mask], profile[mask]
    if x.size < 2:
        raise ValueError("shock window contains fewer than two points")
    low = left_state + 0.1 * (right_state - left_state)
    high = left_state + 0.9 * (right_state - left_state)
    x_low = _crossing_location(x, profile, low)
    x_high = _crossing_location(x, profile, high)
    return abs(x_high - x_low)


def periodic_mass(values: Array, domain_length: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    return float(domain_length * np.mean(values))


def shock_location_from_gradient(
    x: Array,
    profile: Array,
    window: tuple[float, float] | None = None,
) -> float:
    """Locate a compressive shock at the most negative resolved gradient."""
    x = np.asarray(x, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if x.ndim != 1 or profile.shape != x.shape:
        raise ValueError("x and profile must be matching one-dimensional arrays")
    gradient = np.gradient(profile, x)
    mask = np.ones_like(x, dtype=bool)
    if window is not None:
        mask = (x >= window[0]) & (x <= window[1])
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("shock window contains no points")
    return float(x[indices[np.argmin(gradient[indices])]])


def gradient_equivalent_width(
    x: Array,
    profile: Array,
    left_state: float,
    right_state: float,
    window: tuple[float, float] | None = None,
) -> float:
    """Jump magnitude divided by maximum compressive gradient magnitude."""
    x = np.asarray(x, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    gradient = np.gradient(profile, x)
    if window is not None:
        mask = (x >= window[0]) & (x <= window[1])
        gradient = gradient[mask]
    compression = max(float(-np.min(gradient)), 1.0e-14)
    return float(abs(left_state - right_state) / compression)


def overshoot_undershoot(prediction: Array, reference: Array) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    return float(max(0.0, np.max(prediction) - np.max(reference), np.min(reference) - np.min(prediction)))
