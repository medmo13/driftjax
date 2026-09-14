"""Runtime configuration for driftjax (backend selection, precision, tracing)."""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Global JAX setup
# ---------------------------------------------------------------------------

# Enabled at package import (see __init__.py) so every module can rely on
# jnp.float64 semantics without scatter-gun dtype arguments.

# ---------------------------------------------------------------------------
# Backend / numerical policy (overridable via env vars for CI)
# ---------------------------------------------------------------------------

USE_OPTIMISTIX = os.environ.get("DRIFTJAX_OPTIMISTIX", "0") == "1"
USE_BANDED = os.environ.get("DRIFTJAX_BANDED", "0") == "1"

# Newton convergence gates
NEWTON_STEP_TOL = 1e-8  # max |Δx| (dimensionless)
NEWTON_F_TOL_DEFAULT = 1e-8  # residual gate max|F|
EQ_TOL = 1e-10
LINRESID_OK = 1e-6  # acceptable relative linear-solve residual
DENSE_FALLBACK_COND = 1e14  # row-equilibrated cond above which CSR is dubious

# ---------------------------------------------------------------------------
# Mode descriptor for a simulation run
# ---------------------------------------------------------------------------
# The physics/algorithm axes every simulation dispatches on.  Each
# combination is a first-class, independently testable mode; the default

#   statistics: boltzmann | nilsson | exact   (carrier statistics)
#   optics:     beer-lambert | tmm
#   solver:     analytic | dense | ptc
#   fused:      bool   (fused kernels)
#   checkpoint: bool   (gradient checkpointing)


class Mode:
    """Frozen mode descriptor for a simulation run."""

    __slots__ = ("statistics", "optics", "solver", "fused", "checkpoint")

    def __init__(
        self,
        statistics="boltzmann",
        optics="table",
        solver="analytic",
        fused=False,
        checkpoint=False,
    ):
        # normalize hyphen/underscore
        optics_norm = (
            optics.strip().lower().replace("_", "-") if isinstance(optics, str) else optics
        )
        if optics_norm in ("beerlambert", "bl"):
            optics_norm = "beer-lambert"
        if statistics not in ("boltzmann", "nilsson", "exact"):
            raise ValueError(f"unknown statistics mode {statistics!r}")
        if optics_norm not in ("beer-lambert", "table", "tmm", "tauc", "fresnel"):
            raise ValueError(f"unknown optics mode {optics!r}")
        if solver not in ("analytic", "dense", "ptc"):
            raise ValueError(f"unknown solver mode {solver!r}")
        self.statistics = statistics
        self.optics = optics_norm
        self.solver = solver
        self.fused = bool(fused)
        self.checkpoint = bool(checkpoint)

    def __repr__(self):  # pragma: no cover - debug aid
        return (
            f"Mode(statistics={self.statistics!r}, optics={self.optics!r}, "
            f"solver={self.solver!r}, fused={self.fused}, "
            f"checkpoint={self.checkpoint})"
        )


DEFAULT_MODE = Mode()

# mapping to the simulate()/init_cell() keyword values
STATISTICS_ALIAS = {"boltzmann": "boltzmann", "nilsson": "nilsson", "exact": "exact"}
OPTICS_ALIAS = {
    "beer-lambert": "beer-lambert",
    "table": "table",
    "tmm": "tmm",
    "tauc": "tauc",
    "fresnel": "fresnel",
}


def mode_info() -> dict:
    """JSON-serializable summary of the current mode defaults + validity."""
    import jax

    return {
        "default_mode": {k: getattr(DEFAULT_MODE, k) for k in Mode.__slots__},
        "valid_statistics": ["boltzmann", "nilsson", "exact"],
        "valid_optics": ["beer-lambert", "table", "tmm", "tauc", "fresnel"],
        "valid_solver": ["analytic", "dense", "ptc"],
        "x64": bool(jax.config.jax_enable_x64),  # type: ignore[attr-defined]
        "version": "0.1.13",
    }
