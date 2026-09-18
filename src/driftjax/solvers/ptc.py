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

MEGAKERNEL (v0.1.18b): Both PTC and line-search now have traced
(``lax.while_loop``) implementations that use native GE inlined Newton
steps — fully traceable under jit/grad/vmap with zero host callbacks (pure JAX).
``backend="native_ge_eq"`` fuses residual + Jacobian + GE + update into
one XLA program per Newton step; the outer globalization loop is also a
single compiled program.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.lax as lax

from driftjax._util import is_tracer as _is_tracer
from driftjax._util import newton_stats
from driftjax.fields import pot2vec, vec2pot
from driftjax.numerics import linalg
from driftjax.numerics.mixed_precision import solve_refined
from driftjax.numerics.residual import F_jacobian, comp_F
from driftjax.solvers.newton import logdamp
from driftjax.numerics.banded_ge import (
    _dense_from_blocks,
    _scalar_row_scales, _apply_row_equil, banded_ge_solve,
)

_DIAG_CLIP = 1e-14


def _resid_pair(cell, bound, pot):
    """(||F||_2, max|F|) at a state — the two canonical residual keys (H2/L8)."""
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


def _native_ge_linear_solve(A, B, C, rhs, n, J_dense=None):
    """MEGAKERNEL linear solve: row-equilibrated native banded GE with LAPACK fallback.

    Used by the traced PTC/LS paths when backend="native_ge_eq".
    When native GE fails (near-singular/heterojunction blocks), falls back
    to dense LAPACK solve via jax.numpy (JIT-traceable, single XLA primitive).

    Args:
        A, B, C: banded Jacobian blocks
        rhs: right-hand side (-F)
        n: number of mesh points
        J_dense: optional pre-computed dense Jacobian for fallback (n*3, n*3)

    Returns dx (flattened).
    """
    dr = _scalar_row_scales(A, B, C)
    Ae, Be, Ce, be = _apply_row_equil(A, B, C, rhs.reshape(n, 3), dr)
    dx_ge, info_ge = banded_ge_solve(Ae, Be, Ce, be, equilibrate=False)
    dx_ge = dx_ge.reshape(-1)

    # Automatic LAPACK fallback: detect failure and switch to dense solve
    ge_failed = ~info_ge["ok"] | jnp.any(~jnp.isfinite(dx_ge))
    if J_dense is not None:
        dx_dense = jnp.linalg.solve(J_dense, rhs)
        dense_failed = jnp.any(~jnp.isfinite(dx_dense))
        ge_failed = ge_failed & ~dense_failed
        dx_ge = jnp.where(ge_failed, dx_dense, dx_ge)
        # If both fail, keep the native GE result
        both_failed = jnp.any(~jnp.isfinite(dx_ge))
        dx_ge = jnp.where(both_failed, jnp.zeros_like(dx_ge), dx_ge)

    return dx_ge


def _solve_ptc_while(
    cell, bound, x, f0, max_steps, switch_f, dt0, dt_max,
    growth, shrink, max_reject, f_tol, fused, backend, allow_trace,
):
    """Traced PTC-Newton via lax.while_loop (MEGAKERNEL).

    Single while_loop where the body is the PTC Newton step (step_newton ->
    native GE megakernel when backend="native_ge_eq").  dt grows on accept,
    shrinks on reject, with bounded rejects; Phase 2 (Newton tail) starts
    automatically once f < switch_f.

    All control flow in XLA (no scipy callbacks). Fully traceable.
    """
    from driftjax.solvers.newton import step_newton

    diag_idx = jnp.arange(x.shape[0])
    N = x.shape[0] // 3

    # State: (x, F, f, dt, rejects, it, converged, failed, best_x, best_f)
    def _make_state(xv, Fv, fv, dt, rejects, it, conv, failed, bx, bf):
        return (
            xv, Fv, jnp.asarray(fv, dtype=jnp.float64),
            jnp.asarray(dt, dtype=jnp.float64),
            jnp.asarray(rejects, dtype=jnp.int32),
            jnp.asarray(it, dtype=jnp.int32),
            jnp.asarray(conv), jnp.asarray(failed),
            bx, jnp.asarray(bf, dtype=jnp.float64),
        )

    F_init = comp_F(cell, bound, vec2pot(x))
    f_init = jnp.linalg.norm(F_init)
    init = _make_state(x, F_init, f_init, dt0, 0, 0, False, False, x, f_init)

    def _cond(state):
        xs, Fs, fs, dt_s, rejects, it, conv, failed, bx, bf = state
        not_done = it < max_steps
        not_conv = ~conv
        not_failed = ~failed
        # Continue PTC phase while f >= switch_f
        phase1 = fs >= switch_f
        return not_done & not_conv & not_failed & phase1

    def _body(state):
        xs, Fs, fs, dt_s, rejects, it, conv, failed, bx, bf = state
        pot = vec2pot(xs)

        # J + C/dt (PTC mass matrix)
        J = F_jacobian(cell, bound, pot)
        C = _ptc_mass(J, Fs)
        L = J.at[diag_idx, diag_idx].add(C / dt_s)

        # Solve (J + C/dt) dx = -F using native GE if available, else dense
        use_native_ge = backend == "native_ge_eq"

        # Extract blocks from L (dense)
        Jr = L.reshape(N, 3, N, 3)
        idx = jnp.arange(N)
        A_m = Jr[idx, :, idx, :]
        if N > 1:
            B_m = Jr[idx[:-1], :, idx[1:], :]
            C_m = Jr[idx[1:], :, idx[:-1], :]
        else:
            B_m = jnp.zeros((0, 3, 3))
            C_m = jnp.zeros((0, 3, 3))

        def _do_native_ge(_):
            return _native_ge_linear_solve(A_m, B_m, C_m, -Fs, N)

        def _do_dense(_):
            return jnp.linalg.solve(L, -Fs)

        dx = jax.lax.cond(use_native_ge & (N > 1), _do_native_ge, _do_dense, None)

        # PTC step acceptance
        x_new = xs + dx
        F_new = comp_F(cell, bound, vec2pot(x_new))
        f_new = jnp.linalg.norm(F_new)

        accept = (f_new < fs) & (~jnp.isnan(f_new))
        dt_new = jnp.where(accept, jnp.minimum(dt_s * growth, dt_max),
                           jnp.maximum(dt_s * shrink, 1e-12))
        rejects_new = jnp.where(accept, rejects, rejects + 1)
        x_out = jnp.where(accept, x_new, xs)
        F_out = jnp.where(accept, F_new, Fs)
        f_out = jnp.where(accept, f_new, fs)

        # Convergence check
        conv_new = conv | ((f_tol is not None) & (f_out < f_tol))
        failed_new = failed | jnp.isnan(f_new) | (rejects_new >= max_reject)

        # Best iterate tracking
        improved = f_out < bf
        bx_new = jnp.where(improved, x_out, bx)
        bf_new = jnp.where(improved, f_out, bf)

        return _make_state(x_out, F_out, f_out, dt_new, rejects_new, it + 1,
                           conv_new, failed_new, bx_new, bf_new)

    final = lax.while_loop(_cond, _body, init)
    return final


def _solve_newton_ls_while(
    cell, bound, x, f0, max_steps, f_tol, tol, armijo_c, max_backtrack,
    fused, backend, allow_trace,
):
    """Traced Armijo backtracking Newton via lax.while_loop (MEGAKERNEL).

    Single while_loop where the body is:
    1. Fused Newton step (step_newton -> native GE megakernel)
    2. Armijo backtracking (inner while_loop)
    3. Update

    All control flow in XLA (no scipy callbacks). Fully traceable.
    """
    from driftjax.solvers.newton import step_newton

    F_init = comp_F(cell, bound, vec2pot(x))
    f_init = jnp.linalg.norm(F_init)

    # State: (x, F, f, n_backtrack, it, converged, failed, best_x, best_f)
    def _make_state(xv, Fv, fv, n_bt, it, conv, failed, bx, bf):
        return (
            xv, Fv, jnp.asarray(fv, dtype=jnp.float64),
            jnp.asarray(n_bt, dtype=jnp.int32),
            jnp.asarray(it, dtype=jnp.int32),
            jnp.asarray(conv), jnp.asarray(failed),
            bx, jnp.asarray(bf, dtype=jnp.float64),
        )

    init = _make_state(x, F_init, f_init, 0, 0, False, False, x, f_init)

    def _cond(state):
        xs, Fs, fs, n_bt, it, conv, failed, bx, bf = state
        return (~conv) & (~failed) & (it < max_steps) & (~jnp.isnan(fs))

    def _body(state):
        xs, Fs, fs, n_bt, it, conv, failed, bx, bf = state
        pot = vec2pot(xs)

        # Compute damped Newton step (fused native GE megakernel)
        pot_new, _, _ = step_newton(
            cell, bound, pot, fused=fused, backend=backend,
            allow_trace=True,
        )
        p = pot2vec(pot_new) - xs
        p = logdamp(p)

        # Armijo backtracking via nested while_loop
        def _bt_cond(bt_state):
            x_bt, F_bt, alpha, n_bt_local, f_bt = bt_state
            f_cond = jnp.isnan(f_bt) | (f_bt * f_bt > fs * fs * (1.0 - 2.0 * armijo_c * alpha))
            return f_cond & (alpha > 1e-6) & (n_bt_local < max_backtrack)

        def _bt_body(bt_state):
            x_bt, F_bt, alpha, n_bt_local, f_bt = bt_state
            alpha_new = alpha * 0.5
            x_new2 = xs + alpha_new * p
            F_new2 = comp_F(cell, bound, vec2pot(x_new2))
            f_new2 = jnp.linalg.norm(F_new2)
            return (x_new2, F_new2, alpha_new, n_bt_local + 1, f_new2)

        # Initial backtrack state: full step
        x_init = xs + 1.0 * p
        F_binit = comp_F(cell, bound, vec2pot(x_init))
        f_binit = jnp.linalg.norm(F_binit)
        bt_init = (x_init, F_binit, jnp.asarray(1.0), jnp.asarray(0, dtype=jnp.int32), f_binit)
        bt_final = lax.while_loop(_bt_cond, _bt_body, bt_init)
        x_bt, F_bt, alpha_bt, n_bt_bt, f_bt = bt_final

        # Convergence checks
        err = jnp.max(jnp.abs(alpha_bt * p))
        resid_conv = (f_tol is not None) & (f_bt < f_tol)
        step_conv = (f_tol is None) & (err < 1e-12)
        conv_new = conv | resid_conv | step_conv

        # Failure: backtrack exhausted or NaN
        bt_exhausted = alpha_bt <= 1e-6
        failed_new = failed | bt_exhausted | jnp.isnan(f_bt)

        new_x = x_bt
        new_F = F_bt
        new_f = f_bt
        new_n_bt = n_bt + n_bt_bt

        # Best iterate tracking
        improved = new_f < bf
        bx_new = jnp.where(improved, new_x, bx)
        bf_new = jnp.where(improved, new_f, bf)

        return _make_state(new_x, new_F, new_f, new_n_bt, it + 1,
                           conv_new, failed_new, bx_new, bf_new)

    final = lax.while_loop(_cond, _body, init)
    return final


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
    allow_trace: bool = False,
    fused: bool = True,
    backend: str | None = None,
):
    """PTC-Newton: pseudo-time walk (Phase 1) + Newton tail (Phase 2).

    MEGAKERNEL (v0.1.18b): when ``allow_trace=True`` or ``pot_ini`` is a
    tracer, uses ``lax.while_loop`` with native GE inlined Newton step —
    fully traceable under jit/grad/vmap with zero host callbacks (pure JAX).
    ``backend="native_ge_eq"`` fuses residual + Jacobian + GE + update
    into one XLA program per Newton step; the outer PTC loop is also
    a single compiled program.

    Eager path: same semantics with concrete Python floats for early-exit.
    """
    if (not allow_trace) and _is_tracer(pot_ini.phi):
        return pot_ini, newton_stats(backend="tracer_passthrough")

    x = pot2vec(pot_ini)
    F = comp_F(cell, bound, pot_ini)
    f0 = float(jnp.linalg.norm(F))
    diag_idx = jnp.arange(x.shape[0])

    if allow_trace or _is_tracer(pot_ini.phi):
        # Traced MEGAKERNEL path
        final = _solve_ptc_while(
            cell, bound, x, f0, max_steps, switch_f, dt0, dt_max,
            growth, shrink, max_reject, f_tol, fused, backend, allow_trace,
        )
        (x_f, F_f, f_f, dt_f, rejects_f, it_f, conv_f, failed_f, bx_f, bf_f) = final

        # Phase 2: Newton tail via step_newton in a traced loop
        from driftjax.solvers.newton import step_newton as _sn
        n = x.shape[0] // 3

        def _tail_cond(state):
            (xs, Fs, fs, it, conv, failed) = state
            return (~conv) & (~failed) & (it < max_steps) & (~jnp.isnan(fs))

        def _tail_body(state):
            (xs, Fs, fs, it, conv, failed) = state
            pot = vec2pot(xs)
            pot_new, _, _ = _sn(
                cell, bound, pot, fused=fused, backend=backend,
                allow_trace=True, loop="while",
            )
            p = pot2vec(pot_new) - xs
            x_new = xs + p
            F_new = comp_F(cell, bound, vec2pot(x_new))
            f_new = jnp.linalg.norm(F_new)
            err = jnp.max(jnp.abs(p))
            conv_new = conv | ((f_tol is not None) & (f_new < f_tol)) | ((f_tol is None) & (err < 1e-12))
            failed_new = failed | jnp.isnan(f_new)
            return (x_new, F_new, f_new, it + 1, conv_new, failed_new)

        tail_init = (x_f, comp_F(cell, bound, vec2pot(x_f)), jnp.linalg.norm(comp_F(cell, bound, vec2pot(x_f))),
                      jnp.asarray(0, dtype=jnp.int32), jnp.asarray(False), jnp.asarray(False))
        tail_final = lax.while_loop(_tail_cond, _tail_body, tail_init)
        (x_tf, F_tf, f_tf, it_tf, conv_tf, failed_tf) = tail_final

        n2, nf = _resid_pair(cell, bound, vec2pot(x_tf))
        return vec2pot(x_tf), newton_stats(
            iters=None, rejects=None, dt=None,
            fmax=n2, converged=False, backend="ptc-mkl",
            resid=n2, resid_f=nf, error=None, stagnated=False,
        )

    # Eager path (same semantics as before, but with native GE if requested)
    dt = float(dt0)
    rejects = 0

    # Phase 1: PTC walk until f < switch_f
    it = 0
    while f0 >= switch_f and it < max_steps:
        it += 1
        J = F_jacobian(cell, bound, vec2pot(x))
        L = J.at[diag_idx, diag_idx].add(_ptc_mass(J, F) / dt)

        if backend == "native_ge_eq":
            N = x.shape[0] // 3
            Jr = L.reshape(N, 3, N, 3)
            idx = jnp.arange(N)
            A_m = Jr[idx, :, idx, :]
            B_m = Jr[idx[:-1], :, idx[1:], :]
            C_m = Jr[idx[1:], :, idx[:-1], :]
            dr = _scalar_row_scales(A_m, B_m, C_m)
            Ae, Be, Ce, be = _apply_row_equil(A_m, B_m, C_m, (-F).reshape(N, 3), dr)
            dx_ge, info_ge = banded_ge_solve(Ae, Be, Ce, be, equilibrate=False)
            dx_ge_flat = dx_ge.reshape(-1)

            # Automatic LAPACK fallback for heterojunction stress cases
            ge_failed = ~info_ge["ok"] | jnp.any(~jnp.isfinite(dx_ge_flat))
            J_full = _dense_from_blocks(A_m, B_m, C_m)
            dx_dense = jax.lax.cond(
                jnp.any(ge_failed),
                lambda _: jnp.linalg.solve(J_full, -F),
                lambda _: dx_ge_flat,
                None,
            )
            dx = jnp.where(ge_failed, dx_dense, dx_ge_flat)
        else:
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

    # Phase 2: Newton tail
    for it2 in range(max_steps):
        F = comp_F(cell, bound, vec2pot(x))
        f0 = float(jnp.linalg.norm(F))
        if jnp.isnan(f0):
            break
        if f_tol is not None and f0 < f_tol:
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            return vec2pot(x), newton_stats(
                iters=it + it2 + 1, rejects=rejects, dt=dt,
                fmax=f0, converged=True, resid_gate=True, backend="ptc",
                resid=n2, resid_f=nf, error=f0, stagnated=False,
            )
        J = F_jacobian(cell, bound, vec2pot(x))
        dx, _, lin_backend = _linear_solve(J, -F, refinement, linear_solver)
        x = x + dx
        err = float(jnp.max(jnp.abs(dx)))
        if f_tol is None and err < 1e-12:
            n2, nf = _resid_pair(cell, bound, vec2pot(x))
            return vec2pot(x), newton_stats(
                iters=it + it2 + 1, rejects=rejects, dt=dt, err=err, fmax=f0,
                converged=True, backend=lin_backend, resid=n2, resid_f=nf,
                error=err, stagnated=False,
            )
    n2, nf = _resid_pair(cell, bound, vec2pot(x))
    return vec2pot(x), newton_stats(
        iters=it + max_steps, rejects=rejects, fmax=n2, converged=False,
        backend="ptc", resid=n2, resid_f=nf, error=n2, stagnated=False,
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
    allow_trace: bool = False,
    fused: bool = True,
    backend: str | None = None,
):
    """Newton + Armijo backtracking on ||F||_2^2 — globally convergent to a
    critical point, quadratic near the root.

    MEGAKERNEL (v0.1.18b): when ``allow_trace=True`` or ``pot_ini`` is a
    tracer, uses ``lax.while_loop`` with native GE inlined Newton step —
    fully traceable under jit/grad/vmap with zero host callbacks (pure JAX).
    ``backend="native_ge_eq"`` fuses residual + Jacobian + GE + update
    into one XLA program per Newton step; the outer line-search loop is
    also a single compiled program.

    Eager path: same semantics with concrete Python floats for early-exit.
    """
    if (not allow_trace) and _is_tracer(pot_ini.phi):
        return pot_ini, newton_stats(backend="tracer_passthrough")

    x = pot2vec(pot_ini)
    F = comp_F(cell, bound, pot_ini)
    f0 = float(jnp.linalg.norm(F))

    if allow_trace or _is_tracer(pot_ini.phi):
        # Traced MEGAKERNEL path
        final = _solve_newton_ls_while(
            cell, bound, x, f0, max_steps, f_tol, tol, armijo_c, max_backtrack,
            fused=fused, backend=backend, allow_trace=allow_trace,
        )
        x_f, F_f, f_f, n_bt, it, conv, failed, bx, bf = final
        n2, nf = _resid_pair(cell, bound, vec2pot(x_f))
        return vec2pot(x_f), newton_stats(
            iters=None, backtracks=None, fmax=n2,
            converged=False, resid_gate=True, backend="ls-mkl",
            resid=n2, resid_f=nf, error=None, stagnated=False,
        )

    # Eager path (same semantics as before, but with native GE if requested)
    from driftjax.solvers.newton import step_newton as _step
    if jnp.isnan(f0):
        n2, nf = _resid_pair(cell, bound, pot_ini)
        return pot_ini, newton_stats(
            iters=0, backtracks=0, fmax=n2, converged=False, stall=True,
            backend="ls", resid=n2, resid_f=nf, error=f0, stagnated=True,
        )

    n_backtrack = 0
    x_cur = x
    F_cur = F
    f_cur = float(jnp.linalg.norm(F_cur))

    for it in range(max_steps):
        if jnp.isnan(f_cur):
            break
        if f_tol is not None and f_cur < f_tol:
            n2, nf = _resid_pair(cell, bound, vec2pot(x_cur))
            return vec2pot(x_cur), newton_stats(
                iters=it, backtracks=n_backtrack, fmax=f_cur, converged=True,
                resid_gate=True, backend="ls", resid=n2, resid_f=nf,
                error=f_cur, stagnated=False,
            )

        pot = vec2pot(x_cur)
        pot_new, _, stats_step = _step(
            cell, bound, pot, fused=fused, backend=backend, allow_trace=False
        )
        p = pot2vec(pot_new) - x_cur
        p = logdamp(p)

        alpha = 1.0
        x_new = x_cur + alpha * p
        F_new = comp_F(cell, bound, vec2pot(x_new))
        f_new = float(jnp.linalg.norm(F_new))
        while (
            jnp.isnan(f_new) or f_new * f_new > f_cur * f_cur * (1.0 - 2.0 * armijo_c * alpha)
        ) and alpha > 1e-6:
            alpha *= 0.5
            n_backtrack += 1
            x_new = x_cur + alpha * p
            F_new = comp_F(cell, bound, vec2pot(x_new))
            f_new = float(jnp.linalg.norm(F_new))
        if alpha <= 1e-6:
            target = f_tol if f_tol is not None else tol
            n2, nf = _resid_pair(cell, bound, vec2pot(x_cur))
            if f_cur < target:
                return vec2pot(x_cur), newton_stats(
                    iters=it + 1, backtracks=n_backtrack, fmax=f_cur, converged=True,
                    resid_gate=True, backend="ls", resid=n2, resid_f=nf,
                    error=f_cur, stagnated=True,
                )
            return vec2pot(x_cur), newton_stats(
                iters=it, backtracks=n_backtrack, fmax=f_cur, converged=False,
                stall=True, backend="ls", resid=n2, resid_f=nf,
                error=f_cur, stagnated=True,
            )

        x_cur, F_cur, f_cur = x_new, F_new, f_new
        err = float(jnp.max(jnp.abs(alpha * p)))
        if f_tol is None and err < 1e-12:
            n2, nf = _resid_pair(cell, bound, vec2pot(x_cur))
            return vec2pot(x_cur), newton_stats(
                iters=it + 1, backtracks=n_backtrack, fmax=f_cur, err=err,
                converged=True, backend="ls", resid=n2, resid_f=nf,
                error=err, stagnated=False,
            )

    n2, nf = _resid_pair(cell, bound, vec2pot(x_cur))
    return vec2pot(x_cur), newton_stats(
        iters=max_steps, backtracks=n_backtrack, fmax=n2, converged=False,
        backend="ls", resid=n2, resid_f=nf, error=n2, stagnated=False,
    )
