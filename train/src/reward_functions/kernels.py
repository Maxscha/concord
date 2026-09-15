
from __future__ import annotations

import numpy as np


_KERNEL_TYPES = ("gaussian", "laplace", "cauchy", "triangular")


def apply_kernel(error, sigma: float, kernel_type: str = "gaussian"):
    err = np.asarray(error, dtype=np.float64)
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    if kernel_type == "gaussian":
        return np.exp(-(err ** 2) / (2.0 * sigma ** 2))
    if kernel_type == "laplace":
        return np.exp(-np.abs(err) / sigma)
    if kernel_type == "cauchy":
        return 1.0 / (1.0 + (err / sigma) ** 2)
    if kernel_type == "triangular":
        return np.maximum(0.0, 1.0 - np.abs(err) / (3.0 * sigma))
    raise ValueError(
        f"Unknown kernel_type {kernel_type!r}. Expected one of {_KERNEL_TYPES}."
    )
