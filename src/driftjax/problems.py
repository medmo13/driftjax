"""Simulation protocols (composable ``AbstractProblem`` objects).

Mirrors the Diffrax/Optimistix convention: the *what* to solve is a
pluggable object, not a keyword switch.
"""

from __future__ import annotations

import equinox as eqx


class AbstractProblem(eqx.Module):
    """Base class for simulation protocols."""


class Equilibrium(AbstractProblem):
    """Solve the thermal-equilibrium (V = 0) state only."""


class Sweep(AbstractProblem):
    """Full IV sweep from 0 to ``vmax`` in ``n_steps`` bias points."""

    vmax: float = 1.1
    n_steps: int = 41
    refinement: bool = False
    fused: bool = True
    batched: bool = False
    # Soft-maximum MPP temperature (power units, None = hard selection).
    # Opt-in for globally smooth efficiency objectives: replaces the
    # discrete winning-segment argmax (whose gradient jumps at switches)
    # with log-sum-exp over PCHIP candidates. Bias bounded by
    # tau*log(#candidates); report tau alongside any result using it.
    mpp_tau: float | None = None

    def __post_init__(self):
        # L2/M10: degenerate schedules break find_voc (empty sign-change
        # axis) and the MPP spline (length-1 diffs) — fail at construction.
        # Tolerant of tracers (concrete check only when convertible).
        try:
            ns = int(self.n_steps)
        except Exception:
            return
        if ns < 2:
            raise ValueError(f"Sweep needs n_steps >= 2, got {self.n_steps!r}")
        try:
            vm = float(self.vmax)
        except Exception:
            return
        if not (vm == vm and vm != float("inf") and vm != float("-inf")):
            raise ValueError(f"Sweep needs a finite vmax, got {self.vmax!r}")
