"""Transient (backward-Euler) and AC small-signal analysis of the DDP system.

  C(V)·dV/dt + F(V) = 0

The capacitance matrix is C = diag(n, p, 0) per node (interleaved) — the
carrier storage terms; Poisson has no time derivative.  Tested claims are
the *provable* ones (package decision, 2026-08-08): per-step residuals < tol,
fixed-point consistency (a step from the DC root at fixed bias stays put),
long-dt relaxation toward the steady residual, and finite AC admittance
with the correct ω→0 real limit.  A finite-window "settles to DC" claim is
NOT made: the diffusion relaxation is ~6 decades beyond any sane window.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import Potentials, PVCell, pot2vec, vec2pot
from driftjax.numerics.residual import F_jacobian, comp_F
from driftjax.numerics.scharfetter_gummel import Jn, Jp
from driftjax.science.carrier_statistics import n, p
from driftjax.science.contacts import boundary_bias


@jax.jit
def capacitance_diag(cell: PVCell, pot: Potentials) -> jax.Array:
    """Diagonal of C = diag(n, p, 0) per node (interleaved), shape (3N,)."""
    n_v, p_v = n(cell, pot), p(cell, pot)
    N = pot.n
    diag = jnp.zeros(3 * N, dtype=jnp.float64)
    idx = jnp.arange(3 * N)
    diag = diag.at[idx[::3]].set(n_v)
    diag = diag.at[idx[1::3]].set(p_v)
    return diag


@jax.jit
def capacitance_matrix(cell: PVCell, pot: Potentials) -> jax.Array:
    """C = diag(n, p, 0) per node (interleaved [φn, φp, φ]) — dense for AC compat."""
    return jnp.diag(capacitance_diag(cell, pot))


@jax.jit
def transient_residual(cell, bound, pot, pot_prev, dt):
    """F_trans = F_ss(V) + C(V)·(V − V_prev)/dt (diagonal C, O(N))."""
    diag = capacitance_diag(cell, pot)
    return comp_F(cell, bound, pot) + diag * (pot2vec(pot) - pot2vec(pot_prev)) / dt


def solve_transient_step(cell, bound, pot_prev, dt, max_iter: int = 20, tol: float = 1e-6):
    """One backward-Euler step via lax.while_loop (trace-safe).

    Uses O(N) block-tridiagonal Jacobian + pivoted banded solve (LAPACK dgbsv)
    instead of dense O(N²) Jacobian + O(N³) linalg.solve — ~10x faster at N=500.
    """
    from driftjax.numerics.analytic_jacobian import banded_jacobian
    from driftjax.numerics.block_thomas import banded_solve

    eye3 = jnp.eye(3)

    def _cond(state):
        k, pot, error, _alpha, _best_pot, _best_error = state
        return (k < max_iter) & (error > tol)

    def _body(state):
        k, pot, _error, _alpha, _best_pot, _best_error = state
        F = transient_residual(cell, bound, pot, pot_prev, dt)
        err = jnp.max(jnp.abs(F))
        A, B, C = banded_jacobian(cell, bound, pot)
        diag = capacitance_diag(cell, pot).reshape(-1, 3) / dt
        A_t = A + diag[:, :, None] * eye3[None, :, :]
        reg = jnp.max(jnp.abs(A_t)) * 1e-8
        reg = jnp.maximum(reg, 1e-10)
        A_r = A_t + reg * eye3[None, :, :]
        delta = banded_solve(A_r, B, C, -F.reshape(-1, 3)).reshape(-1)
        dn = jnp.max(jnp.abs(delta))
        delta = jnp.where(dn > 1.0, delta * (1.0 / dn), delta)

        # Line search: unrolled lax.fori_loop over 15 halvings
        def _ls_step(ls_state, _):
            alpha, best_pot_ls, best_err_ls = ls_state
            pot_try = vec2pot(pot2vec(pot) + alpha * delta)
            new_err = jnp.max(jnp.abs(transient_residual(cell, bound, pot_try, pot_prev, dt)))
            improved = jnp.isfinite(new_err) & (new_err < best_err_ls)
            best_pot_ls = jax.tree.map(lambda b, t: jnp.where(improved, t, b), best_pot_ls, pot_try)
            best_err_ls = jnp.where(improved, new_err, best_err_ls)
            return (alpha * 0.5, best_pot_ls, best_err_ls), None

        init_ls = (jnp.array(1.0), pot, err)
        _, best_pot_final, best_err_final = jax.lax.scan(_ls_step, init_ls, None, length=15)[0]

        # Update best if line search found improvement
        improved_any = best_err_final < err
        pot_new = jax.tree.map(lambda b, p: jnp.where(improved_any, b, p), best_pot_final, pot)
        err_new = jnp.where(improved_any, best_err_final, err)

        return (k + 1, pot_new, err_new, jnp.array(1.0), best_pot_final, best_err_final)

    init = (
        jnp.array(0, dtype=jnp.int32),
        pot_prev,
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.array(1.0, dtype=jnp.float64),
        pot_prev,
        jnp.array(jnp.inf, dtype=jnp.float64),
    )
    _k, pot_out, error_out, *_ = jax.lax.while_loop(_cond, _body, init)
    return pot_out, {"converged": error_out < tol, "iterations": _k, "error": error_out}


def solve_transient(cell, v_applied, pot_eq, t_final, n_steps: int = 100):
    """Ramp-hold protocol; returns (times, currents, pot_final).

    Uses lax.scan over time steps for full XLA compilation.
    """
    dt = t_final / n_steps
    times = jnp.linspace(0.0, t_final, n_steps + 1)

    def _step(carry, i):
        pot = carry
        bias_frac = jnp.minimum(1.0, times[i + 1] / (0.5 * t_final))
        pot_new, _ = solve_transient_step(cell, boundary_bias(cell, v_applied * bias_frac), pot, dt)
        current = jnp.mean(Jn(cell, pot_new) + Jp(cell, pot_new))
        return pot_new, current

    pot_final, currents_body = jax.lax.scan(_step, pot_eq, jnp.arange(n_steps))
    currents = jnp.concatenate([jnp.array([0.0]), currents_body])
    return times, currents, pot_final


def ac_small_signal(cell, v_dc, pot_dc, omega, amplitude: float = 1e-4):
    """AC terminal-current admittance Y(ω) at a DC operating point.

    Uses vmap over frequencies for batched solve.
    """
    if isinstance(pot_dc, tuple):
        pot_dc = pot_dc[0]
    bound = boundary_bias(cell, v_dc)
    C = capacitance_matrix(cell, pot_dc)
    J = F_jacobian(cell, bound, pot_dc)
    F_v = jax.jacrev(lambda vv: comp_F(cell, boundary_bias(cell, vv), pot_dc))(v_dc)
    I_x = pot2vec(jax.jacrev(lambda p: jnp.mean(Jn(cell, p) + Jp(cell, p)))(pot_dc))

    J_c = J.astype(jnp.complex128)
    F_c = (-F_v * amplitude).astype(jnp.complex128)
    C_c = C.astype(jnp.complex128)
    I_c = I_x.astype(jnp.complex128)
    w = jnp.asarray(omega, dtype=jnp.float64)

    # Vectorized: build (n_omega, 3N, 3N) batch and vmap solve
    M_all = 1j * w[:, None, None] * C_c[None] + J_c[None]
    F_all = jnp.broadcast_to(F_c, (len(omega), F_c.shape[0]))
    V_all = jax.vmap(jnp.linalg.solve)(M_all, F_all)
    Y = jax.vmap(lambda v: I_c @ v / amplitude)(V_all)
    return Y
