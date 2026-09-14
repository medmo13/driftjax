"""Pseudo-transient continuation (PTC) and Armijo line-search globalization.

The DDP Newton problem is stiff: out-of-basin guesses (flatband, degenerate
doping) make bare log-damped Newton stall or diverge.  PTC solves the
pseudo-time system

    (J + C/dt)·dx = −F,  x ← x + dx,  dt growing,

with C = clip(|diag J|) (Bank–Rose mass), an Armijo accept/reject on ‖F‖₂,
and a Newton tail once ‖F‖ < switch_f.  For M-matrix Jacobians (which the
DDP operator is), J + C/dt stays an M-matrix for every dt, so the damped
walk is a guaranteed descent direction and cannot park on a spurious ‖F‖
local minimum.

The line-search variant (``solve_newton_ls``) is the reconstruction-report
recommendation (Armijo on ‖F‖₂²) and doubles as ``auto`` recovery.
"""

from __future__ import annotations

import jax.numpy as jnp

from driftjax._util import is_tracer as _is_tracer
from driftjax._util import newton_stats
from driftjax.fields import pot2vec, vec2pot
from driftjax.numerics import linalg
from driftjax.numerics.mixed_precision import solve_refined
from driftjax.numerics.residual import F_jacobian, comp_F
from driftjax.solvers.newton import logdamp

_DIAG_CLIP = 1e-14


def _resid_pair(cell, bound, pot):
    """(‖F‖₂, max|F|) at a state — the two canonical residual keys (H2/L8)."""
    F = comp_F(cell, bound, pot)
    return float(jnp.linalg.norm(F)), float(jnp.max(jnp.abs(F)))


def _ptc_mass(J, F):
    """Bank–Rose mass: C = |diag J| clipped positive."""
    return jnp.clip(jnp.abs(jnp.diag(J)), min=_DIAG_CLIP)


def _linear_solve(L, rhs, refinement: bool, backend: str = "auto"):
    if refinement:
        x, rel, n_ref, conv = solve_refined(L, rhs, tol=1e-12)
        if _is_tracer(conv):
            # Traced context: no Python branch on tracer bool, no host fallback.
            return x, rel, "mixed_f32"
        if conv:
            return x, rel, f"mixed_f32/n={int(n_ref)}"
    if backend == "dense":
        x = jnp.linalg.solve(L, rhs)
        # AUDIT: relative (not absolute) residual — the CSR path returns
        # relative, and the gate below compares against 1e-4 relatively.
        return x, jnp.linalg.norm(L @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30), "dense"
    x, resid, tag = linalg.linsolve(L, rhs, backend=backend)
    if jnp.isnan(resid) or resid > 1e-4:
        x = jnp.linalg.solve(L, rhs)
        return x, jnp.linalg.norm(L @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30), "dense"
    return x, resid, tag


def solve_ptc(
    cell,
    bound,
    pot_ini,
    tol: float = 1e-8,
    f_tol: float | None = 1e-8,
    max_steps: int = 100,
    switch_f: float = 1e-3,
    dt0: float = 1e-2,
    dt_max: float = 1e2,
    growth: float = 2.0,
    shrink: float = 0.2,
    max_reject: int = 30,
    linear_solver: str = "auto",
    refinement: bool = False,
):
    """PTC-Newton: pseudo-time walk (Phase 1) + Newton tail (Phase 2)."""
    if _is_tracer(pot_ini.phi):
        return pot_ini, newton_stats(backend="tracer_passthrough")
    x = pot2vec(pot_ini)
    dt = float(dt0)
    rejects = 0
    F = comp_F(cell, bound, vec2pot(x))
    f0 = float(jnp.linalg.norm(F))
    diag_idx = jnp.arange(x.shape[0])

    it = 0
    while f0 >= switch_f and it < max_steps:
        it += 1
        J = F_jacobian(cell, bound, vec2pot(x))
        L = J.at[diag_idx, diag_idx].add(_ptc_mass(J, F) / dt)
        dx, _, _ = _linear_solve(L, -F, refinement, linear_solver)
        x_new = x + dx
        F_new = comp_F(cell, bound, vec2pot(x_new))
        f_new = float(jnp.linalg.norm(F_new))
        if jnp.isnan(f_new) or f_new >= f0:
            dt = max(dt * float(shrink), 1e-12)
            rejects += 1
            if rejects >= max_reject:
                break
            continue
        dt = min(dt * float(growth), dt_max)
        x, F, f0 = x_new, F_new, f_new

    for it2 in range(max_steps):
        F = comp_F(cell, bound, vec2pot(x))
        f0 = float(jnp.linalg.norm(F))
        if jnp.isnan(f0):
            break
        if f_tol is not None and f0 < f_tol:
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            return vec2pot(x), newton_stats(
                iters=it + it2 + 1,
                rejects=rejects,
                dt=dt,
                fmax=f0,
                converged=True,
                resid_gate=True,
                backend="ptc",
                resid=n2,
                resid_f=nf,
                error=f0,
                stagnated=False,
            )
        J = F_jacobian(cell, bound, vec2pot(x))
        dx, _, backend = _linear_solve(J, -F, refinement, linear_solver)
        x = x + dx
        err = float(jnp.max(jnp.abs(dx)))
        if f_tol is None and err < tol:
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            return vec2pot(x), newton_stats(
                iters=it + it2 + 1,
                rejects=rejects,
                dt=dt,
                err=err,
                fmax=f0,
                converged=True,
                backend=backend,
                resid=n2,
                resid_f=nf,
                error=err,
                stagnated=False,
            )
    n2, nf = _resid_pair(cell, bound, vec2pot(x))
    return vec2pot(x), newton_stats(
        iters=it + max_steps,
        rejects=rejects,
        fmax=n2,
        converged=False,
        backend="ptc",
        resid=n2,
        resid_f=nf,
        error=n2,
        stagnated=False,
    )


def solve_newton_ls(
    cell,
    bound,
    pot_ini,
    tol: float = 1e-8,
    f_tol: float | None = 1e-8,
    max_steps: int = 60,
    max_backtrack: int = 12,
    armijo_c: float = 1e-4,
    linear_solver: str = "auto",
    refinement: bool = False,
):
    """Newton + Armijo backtracking on ‖F‖₂² — globally convergent to a
    critical point, quadratic near the root."""
    if _is_tracer(pot_ini.phi):
        return pot_ini, newton_stats(backend="tracer_passthrough")
    x = pot2vec(pot_ini)
    F = comp_F(cell, bound, vec2pot(x))
    f0 = float(jnp.linalg.norm(F))
    n_backtrack = 0

    for it in range(max_steps):
        if jnp.isnan(f0):
            break
        if f_tol is not None and f0 < f_tol:
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            return vec2pot(x), newton_stats(
                iters=it,
                backtracks=n_backtrack,
                fmax=f0,
                converged=True,
                resid_gate=True,
                backend="ls",
                resid=n2,
                resid_f=nf,
                error=f0,
                stagnated=False,
            )
        J = F_jacobian(cell, bound, vec2pot(x))
        p, _, backend = _linear_solve(J, -F, refinement, linear_solver)
        p = logdamp(p)
        alpha = 1.0
        x_new = x + alpha * p
        F_new = comp_F(cell, bound, vec2pot(x_new))
        f_new = float(jnp.linalg.norm(F_new))
        while (
            jnp.isnan(f_new) or f_new * f_new > f0 * f0 * (1.0 - 2.0 * armijo_c * alpha)
        ) and alpha > 1e-6:
            alpha *= 0.5
            n_backtrack += 1
            x_new = x + alpha * p
            F_new = comp_F(cell, bound, vec2pot(x_new))
            f_new = float(jnp.linalg.norm(F_new))
        if alpha <= 1e-6:
            target = f_tol if f_tol is not None else tol
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            if f0 < target:
                return vec2pot(x), newton_stats(
                    iters=it + 1,
                    backtracks=n_backtrack,
                    fmax=f0,
                    converged=True,
                    resid_gate=True,
                    backend="ls",
                    resid=n2,
                    resid_f=nf,
                    error=f0,
                    stagnated=True,
                )
            return vec2pot(x), newton_stats(
                iters=it,
                backtracks=n_backtrack,
                fmax=f0,
                converged=False,
                stall=True,
                backend="ls",
                resid=n2,
                resid_f=nf,
                error=f0,
                stagnated=True,
            )
        x, F, f0 = x_new, F_new, f_new
        err = float(jnp.max(jnp.abs(alpha * p)))
        if f_tol is None and err < tol:
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            return vec2pot(x), newton_stats(
                iters=it + 1,
                backtracks=n_backtrack,
                fmax=f0,
                err=err,
                converged=True,
                backend=backend,
                resid=n2,
                resid_f=nf,
                error=err,
                stagnated=False,
            )
    n2, nf = _resid_pair(cell, bound, vec2pot(x))
    return vec2pot(x), newton_stats(
        iters=max_steps,
        backtracks=n_backtrack,
        fmax=n2,
        converged=False,
        backend="ls",
        resid=n2,
        resid_f=nf,
        error=n2,
        stagnated=False,
    )
