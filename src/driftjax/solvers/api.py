"""Solver / linear-solver API (composable ``AbstractSolver`` objects).

Mirrors Lineax (``AbstractLinearSolver``) and Optimistix (``AbstractSolver``):
the Newton loop is configured by a pluggable linear solver.
"""

from __future__ import annotations

import equinox as eqx


class AbstractLinearSolver(eqx.Module):
    """Base class for linear solvers used inside Newton."""


class BandedLapack(AbstractLinearSolver):
    """Pivoted banded solve (LAPACK dgbsv) — DriftJax default linear solver."""

    batched: bool = False


# Backwards-compatibility alias: v0.1.16 and earlier exposed this solver
# as ``BlockThomas``.  Retained so external scripts keep importing;
# new code should use ``BandedLapack``.
BlockThomas = BandedLapack


class AbstractSolver(eqx.Module):
    """Base class for nonlinear solvers."""


class Newton(AbstractSolver):
    """Damped Newton solver with analytic banded Jacobian + pivoted LAPACK banded.

    ``backend="auto"`` (default): LAPACK banded solve with truncated-SVD fallback.
    ``backend="native_ge_eq"`` (v0.1.18b megakernel): row-equilibrated native GE
    inlined into the compiled Newton step — zero LAPACK host callbacks, full XLA
    fusion across residual + Jacobian + GE + update. Falls back to dense solve
    when GE fails on singular blocks.
    """

    rtol: float = 1e-8
    max_steps: int = 100
    f_tol: float | None = None
    globalization: str = "auto"
    dense: bool = False
    refinement: bool = False
    fused: bool = True
    backend: str = "auto"
    linear_solver: AbstractLinearSolver = eqx.field(default_factory=BandedLapack)
