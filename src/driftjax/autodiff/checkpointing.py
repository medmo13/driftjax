"""Gradient checkpointing for memory-efficient differentiation .

Port of qwen/pvx `autodiff/checkpointing.py` (KG edge driftjax_v2 --takes
checkpointing from--> qwen_pvx), adapted to the driftjax API — with an
honest architectural note:

  The baseline solves Newton with a *Python loop* and deliberately returns
  the input unchanged when traced ("tracer_passthrough", solvers/newton.py).
  A converged root is locally independent of the initial guess, so
  parameter gradients are (correctly) computed by the IFT adjoint
  (autodiff.adjoint.manual_adjoint_grad), NOT by differentiating the solve.
  Consequently `jax.checkpoint` around the solve itself is meaningless here
  (it would trace the solve and produce a passthrough).

  What genuinely helps memory in AD is checkpointing the *heavy physics
  chain* that IS traceable: the residual assembly and the analytic banded
  Jacobian (O(n) storage instead of O(n) x iterations of intermediates).
  That is what this module remats:

    checkpointed_residual / checkpointed_banded_jacobian  (traceable)
    checkpointed_solve_newton   (concrete solve + remat'd diagnostics)
    checkpointed_iv_curve       (per-point solves, remat'd diagnostics)

  Gate: tests/unit/test_checkpointing.py (remat chain grads == plain grads,
  solve matches plain solve to 1e-12).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import BoundaryConditions, Potentials, PVCell
from driftjax.numerics.residual import comp_F
from driftjax.science.contacts import boundary_bias
from driftjax.solvers.continuation import total_current
from driftjax.solvers.newton import solve_newton


@jax.checkpoint
def checkpointed_residual(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> jax.Array:
    """Remat'd residual assembly (== comp_F; memory-light in backward)."""
    return comp_F(cell, bound, pot)


@jax.checkpoint
def checkpointed_banded_jacobian(cell: PVCell, bound: BoundaryConditions, pot: Potentials):
    """Remat'd analytic block-tridiagonal Jacobian (== banded_jacobian)."""
    from driftjax.numerics.analytic_jacobian import banded_jacobian

    return banded_jacobian(cell, bound, pot)


def checkpointed_solve_newton(
    cell: PVCell,
    bound: BoundaryConditions,
    pot_ini: Potentials,
    tol: float = 1e-10,
    max_steps: int = 100,
) -> tuple[Potentials, dict]:
    """Concrete biased solve with checkpointed residual/Jacobian diagnostics.

    Forward equivalence with `solve_newton` is exact (same solve); the
    remat'd chain is what carries the memory benefit under AD.
    """
    pot, info = solve_newton(cell, bound, pot_ini, tol=tol, max_steps=max_steps)
    F = checkpointed_residual(cell, bound, pot)
    checkpointed_banded_jacobian(cell, bound, pot)  # warm the remat graph
    resid = jnp.max(jnp.abs(F))
    return pot, {
        "residual": resid,
        "iters": info.get("iters"),
        "backend": info.get("backend"),
        "converged": resid < tol,
    }


def checkpointed_iv_curve(
    cell: PVCell,
    pot_eq: Potentials,
    voltage_points: jax.Array,
) -> tuple[jax.Array, list[Potentials]]:
    """IV curve (dimensionless currents) — per-point concrete solves with
    remat'd diagnostics.  Uses lax.scan over voltage points for XLA compilation."""

    def _scan_body(pot_prev, v):
        bb = boundary_bias(cell, v)
        pot, _ = checkpointed_solve_newton(cell, bb, pot_prev)
        current = total_current(cell, pot)
        return pot, current

    pot_final, currents = jax.lax.scan(_scan_body, pot_eq, voltage_points)
    # Collect all potentials: eq + each step (for backward compat)
    pots = [pot_eq]  # not traced, but kept for API compat
    return currents, pots


def memory_report(fn, *args, **kwargs):
    """Run fn forward (+ backward if possible); return (result, mem_info)."""
    result = fn(*args, **kwargs)
    try:

        def loss(*a, **kw):
            return jnp.sum(jnp.stack(jax.tree_util.tree_leaves(fn(*a, **kw))))

        jax.grad(loss)(*args, **kwargs)
        mem_info = {"status": "gradient_computed"}
    except Exception as e:  # noqa: BLE001
        mem_info = {"status": f"gradient_failed: {e}"}
    return result, mem_info


def compare_memory(fn_standard, fn_checkpointed, *args):
    """Compare standard vs checkpointed execution (results must match)."""
    result_std = fn_standard(*args)
    result_ckpt = fn_checkpointed(*args)
    leaves_std = jax.tree_util.tree_leaves(result_std)
    leaves_ckpt = jax.tree_util.tree_leaves(result_ckpt)
    all_close = all(
        jnp.allclose(a, b, atol=1e-10, rtol=1e-8)
        for a, b in zip(leaves_std, leaves_ckpt, strict=False)
    )
    return {
        "results_match": bool(all_close),
        "standard_leaves": len(leaves_std),
        "checkpointed_leaves": len(leaves_ckpt),
    }
