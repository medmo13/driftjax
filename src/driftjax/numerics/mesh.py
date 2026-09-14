"""1-D finite-volume mesh helpers (cm in, dimensionless out)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.units import thermal_scales


def make_uniform_mesh(
    total_thickness_cm: float, n_points: int, T: float = 300.0
) -> tuple[jax.Array, jax.Array]:
    """Uniform 1-D mesh: returns (x_dimless, dgrid_dimless) for given total thickness."""
    if n_points < 3:
        raise ValueError(f"n_points must be >=3, got {n_points}")
    sc = thermal_scales(T)
    x_cm = jnp.linspace(0.0, float(total_thickness_cm), int(n_points))
    x_dimless = x_cm / jnp.asarray(sc["length"], dtype=jnp.float64)
    dgrid = jnp.diff(x_dimless)
    return x_dimless, dgrid


def mesh_quality(dgrid: jax.Array) -> float:
    """Mesh quality metric: max(dgrid)/min(dgrid), 1.0 is uniform."""
    return float(jnp.max(dgrid) / jnp.maximum(jnp.min(dgrid), 1e-30))
