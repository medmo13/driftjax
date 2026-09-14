"""Batched (vmapped) solver machinery — NEW in driftjax.

Why: the serial bias sweep is a Python loop of jitted Newton solves, each
recompiling/relaunching.  The DDP block-tridiagonal Jacobian lets us solve
the *entire batch* of linear systems in one vmapped Block-Thomas pass, and
a lax.scan over a continuation schedule carries every bias in parallel
inside one XLA program.  On GPUs this converts per-point latency into bulk
tensor-core work (the merge-plan's "queued solver-path PR" item).

Structure
---------
* ``sweep_batched``: vmap a fixed-schedule Newton over the bias axis; the
  shared convergence test is max |Δx| over the whole batch (safe: biases
  are ordered and correlated, they converge together).
* Shared cell, batched boundary conditions; Jacobians computed by
  vmapped jacrev; linear solves by node-major batched Block-Thomas
  (exact, pure-jnp, differentiable).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import vmap

from driftjax.fields import Potentials, PVCell, vec2pot
from driftjax.numerics.analytic_jacobian import banded_jacobian
from driftjax.numerics.block_thomas import block_thomas_solve_batched
from driftjax.numerics.residual import comp_F
from driftjax.science.contacts import boundary_bias


def _pot2vec_b(pot) -> jax.Array:
    """Batched interleaved flat vector: (B, 3n)."""
    return jnp.stack([pot.phi_n, pot.phi_p, pot.phi], axis=-1).reshape(pot.phi.shape[0], -1)


def _vec2pot_b(v: jax.Array) -> Potentials:
    """Inverse of ``_pot2vec_b`` (v: (B, 3n))."""
    reshaped = v.reshape(v.shape[0], -1, 3)
    return Potentials(reshaped[..., 0], reshaped[..., 1], reshaped[..., 2])


def _batched_jacobian(cell, bound_b, pot_b):
    """(B, 3n, 3n) Jacobian via per-bias jacrev (batch axis is the vmap)."""

    def f(b, v):
        return jax.jacrev(lambda vv: comp_F(cell, b, vec2pot(vv)))(v)

    return vmap(f)(bound_b, _pot2vec_b(pot_b))


def _batched_banded_jacobian(cell, bound_b, pot_b):
    """(A, B, C) analytic banded blocks, vmapped over the bias axis.

    A: (B, n, 3, 3) diagonal, B: (B, n-1, 3, 3) super, C: (B, n-1, 3, 3)
    sub — identical math to the serial analytic banded Jacobian, one
    vmapped XLA program instead of per-bias jacrev (P3 fix: the previous
    dense-jacrev batch was ~1200x slower than serial on CPU at n=500).
    """
    return vmap(lambda b, pb: banded_jacobian(cell, b, pb))(bound_b, pot_b)


def _batched_step(cell, bound_b, pot_b, err_prev, f_tol, tol, damping=True, analytic=True):
    """One damped Newton step across the batch; returns (pot_b_new, err_b, resid_b).

    Per-member convergence freezing: a member whose FRESH residual (measured
    at its current state) is already <= f_tol, or whose previous step norm was
    <= tol, is frozen (dx = 0). This is essential near V = 0 where the coupled
    Jacobian is near-degenerate (cond ~ 1e18): without freezing, a converged
    member takes a junk step dx = J^-1 F with |F| ~ 1e-11 but |dx| ~ 1e-3,
    displacing itself O(1e-3) in current from the true (equilibrium) state.
    """
    B = pot_b.phi.shape[0]
    F_b = vmap(lambda b, p: comp_F(cell, b, p))(bound_b, pot_b)
    resid_b = jnp.max(jnp.abs(F_b), axis=1)  # (B,) fresh, at current states
    active = (resid_b > f_tol) & (err_prev > tol)  # (B,)
    if analytic:
        A, Bk, C = _batched_banded_jacobian(cell, bound_b, pot_b)
    else:
        J_b = _batched_jacobian(cell, bound_b, pot_b)
        A, Bk, C = vmap(_blocks)(J_b)
    # batched block-Thomas: node-major block layout
    A = A.transpose(1, 0, 2, 3)  # (N, B, 3, 3)
    Bk = Bk.transpose(1, 0, 2, 3)
    C = C.transpose(1, 0, 2, 3)
    rhs3 = (-F_b).reshape(B, -1, 3).transpose(1, 0, 2)  # (N, B, 3)
    dx3 = block_thomas_solve_batched(A, Bk, C, rhs3)  # (N, B, 3)
    dx = dx3.transpose(1, 0, 2).reshape(B, -1)  # (B, 3n)
    if damping:
        dx = jnp.where(jnp.abs(dx) > 1, jnp.log(1 + jnp.abs(dx) * 1.72) * jnp.sign(dx), dx)
    dx = jnp.where(active[:, None], dx, 0.0)  # freeze converged members
    x_b = _pot2vec_b(pot_b) + dx
    err_b = jnp.max(jnp.abs(dx), axis=1)  # (B,) per-member step norm
    return _vec2pot_b(x_b), err_b, resid_b


def _blocks(J):
    """3×3 block diagonals of one (3N, 3N) interleaved Jacobian."""
    N = J.shape[0] // 3
    Jr = J.reshape(N, 3, N, 3)
    idx = jnp.arange(N)
    return (Jr[idx, :, idx, :], Jr[idx[:-1], :, idx[1:], :], Jr[idx[1:], :, idx[:-1], :])


def sweep_batched(
    cell: PVCell,
    vmax: float,
    n_steps: int = 20,
    tol: float = 1e-8,
    max_newton: int = 60,
    damping: bool = True,
    analytic: bool | None = None,
):
    """IV sweep with all biases solved together (one XLA program).

    Returns (voltages, currents, pots) where pots is a (B, 3, n) stack.
    The first (equilibrium) bias is solved serially, then the batch is
    warm-started from a QFL-split + linear-prediction guess.
    Fully jit-compatible via lax.while_loop (no Python float(err) break).
    """
    # M10: degenerate single-point schedules break downstream reductions.
    if int(n_steps) < 2:
        raise ValueError(f"sweep_batched needs n_steps >= 2, got {n_steps!r}")
    voltages = jnp.linspace(0.0, vmax, n_steps)
    from driftjax.solvers.continuation import equilibrium_guess, qfl_hotstart
    from driftjax.solvers.newton import _is_tracer, solve_eq

    # equilibrium start (single system) — uses trace-safe path when under jit
    bound_eq = boundary_bias(cell, 0.0)
    # AUDIT: allow_trace detection is passed through (was missing): under
    # jit/grad the guess is a tracer and solve_eq otherwise returns the
    # unsolved guess as pot_eq, degrading every warm start in the batch.
    pot_eq = solve_eq(
        cell, bound_eq, equilibrium_guess(cell).phi, allow_trace=bool(_is_tracer(cell))
    )

    bounds = vmap(lambda v: boundary_bias(cell, v))(voltages)

    pots0 = vmap(lambda v: qfl_hotstart(pot_eq, v))(voltages)

    if analytic is None:
        analytic = cell.statistics == "boltzmann"
    # lax.while_loop: trace-safe, works under jit and eager.
    # Per-member convergence: each bias member freezes once its own residual
    # (measured fresh at its current state) is <= f_tol, or its own step
    # norm is <= tol. The residual gate is essential on stiff
    # heterostructures where the step norm cannot certify convergence on a
    # near-degenerate Jacobian; the per-member freezing prevents converged
    # members (notably V = 0) from being displaced by junk steps.
    # AUDIT: f_tol was hardcoded to 1e-10, ignoring the caller's tol — a
    # request for tol=1e-12 "converged" without meeting it. The gate is now
    # min(tol, 1e-10): identical behaviour at the default tol=1e-8 while
    # honouring tighter requests (at the cost of honest non-convergence if
    # the residual floor cannot reach them). Degenerate tol inputs (None,
    # non-positive, non-finite) fall back to 1e-10 instead of disabling the
    # freeze gate (f_tol <= 0 would freeze nothing and burn max_newton).
    # M9: broad guard — a traced tol raises concretization errors outside
    # (TypeError, ValueError); any failure falls back to 1e-10.
    try:
        tol_f = float(tol) if tol is not None else 1e-10
    except Exception:
        tol_f = 1e-10
    if not (tol_f > 0.0 and float("-inf") < tol_f < float("inf")):
        tol_f = 1e-10
    tol = tol_f
    f_tol = min(tol_f, 1e-10)

    def cond_fn(state):
        it, _, err_b, resid_b = state
        return (it < max_newton) & jnp.any((resid_b > f_tol) & (err_b > tol))

    def body_fn(state):
        it, pot_b, err_b, _ = state
        pot_b_next, err_next, resid_next = _batched_step(
            cell, bounds, pot_b, err_prev=err_b, f_tol=f_tol, tol=tol,
            damping=damping, analytic=analytic,
        )
        return (it + 1, pot_b_next, err_next, resid_next)

    init_state = (
        jnp.array(0, dtype=jnp.int32),
        pots0,
        jnp.full((voltages.shape[0],), 1e10, dtype=jnp.float64),
        jnp.full((voltages.shape[0],), 1e10, dtype=jnp.float64),
    )
    _, pot_b, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # AUDIT: currents go through total_current (was raw mean) so the
    # serial and batched paths share one primal AND one custom-JVP
    # derivative instead of differing by construction under grad.
    from driftjax.solvers.continuation import total_current

    currents = vmap(lambda p: total_current(cell, p))(pot_b)
    return voltages, currents, pot_b
