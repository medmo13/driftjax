"""Damped Newton solvers for the DDP system (equilibrium + full coupled).

Path (package-validated): jacfwd(comp_F) → block-tridiagonal analytic Jacobian
→ pivoted banded solve (LAPACK dgbsv) — with a residual gate, a globalization
fallback (line search / PTC), and a dense last resort.  Anti-regressions
kept from the package: never lax.scan over spsolve on CPU; Python loops
for Newton; tracer-safe early returns so grad-through-solver stays concrete.

NEW: ``solve_newton(..., refinement="auto")`` routes the linear solve to
the FP32-Core + FP64-refinement path (mixed_precision.solve_refined) when
the equilibration suggests it will converge, else CSR/dense.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax._util import is_tracer as _is_tracer
from driftjax._util import newton_stats
from driftjax.fields import BoundaryConditions, Potentials, PVCell, pot2vec, vec2pot
from driftjax.numerics import linalg
from driftjax.numerics.mixed_precision import solve_refined
from driftjax.numerics.residual import F_eq, F_eq_jacobian, F_jacobian, comp_F


def logdamp(move):
    """Component-wise log damping of oversized Newton update steps."""
    return jnp.where(jnp.abs(move) > 1, jnp.log(1 + jnp.abs(move) * 1.72) * jnp.sign(move), move)


# L1: covers the full solve_refined range (MAX_REFINE = 8); anything else
# maps to 9 (unknown) instead of silently colliding with a real backend.
_BACKEND_CODE = {
    "analytic-banded": 0,
    "csr": 1,
    "dense": 2,
    "mixed_f32": 3,
    **{f"mixed_f32/n={i}": 3 for i in range(9)},
}

# Best-iterate certification thresholds shared by both Newton loops (AUDIT R3:
# hoisted — was copy-pasted REBOUND_CONV = 1e-8 plus a hardcoded 10.0 tail
# factor in each loop).
_REBOUND_CONV = 1e-8  # package residual criterion (Newton.rtol convention)
_REBOUND_TAIL = 10.0  # tail materially worse than best ⇔ rebound (not settle)


def _rebound_criterion(f_tol) -> float:
    """Residual threshold below which a best iterate certifies convergence.

    Honors an explicitly tighter caller gate: with f_tol active, only a best
    below min(1e-8, f_tol) certifies (previously the fixed 1e-8 could certify
    against an unmet tighter gate). Unset f_tol keeps pristine 1e-8 semantics.
    """
    if f_tol is None:
        return _REBOUND_CONV
    return min(_REBOUND_CONV, float(f_tol))


def _linear_solve(J, rhs, tol=1e-6, dense=False, backend=None, refinement=False):
    if dense:
        x = jnp.linalg.solve(J, rhs)
        # AUDIT: +1e-30 floor (was missing) — at an exact root rhs=0 the
        # residual was 0/0=NaN, poisoning stats and solver-selection gates.
        return x, jnp.linalg.norm(J @ x - rhs) / (jnp.linalg.norm(rhs) + 1e-30), "dense"
    if refinement:
        x, rel, n_ref, conv = solve_refined(J, rhs, tol=1e-12)
        if _is_tracer(conv):
            # Traced (jit/grad/while_loop): host CSR fallback is unavailable
            # and Python branching on a tracer bool would raise
            # TracerBoolConversionError. Keep the refined result; the
            # Newton loop gates on the residual.
            return x, rel, "mixed_f32"
        if conv:
            return x, rel, f"mixed_f32/n={int(n_ref)}"
        # refinement failed (ill-conditioned): fall back to CSR
    return linalg.linsolve(J, rhs, backend=backend or "auto", tol=tol)


# ---------------------------------------------------------------------------
# Equilibrium (Poisson-only)
# ---------------------------------------------------------------------------


def _eq_tridiag(cell, pot):
    """Row-indexed (sub, diag, sup) of the equilibrium Poisson Jacobian.

    F_eq = [φ0−φ0b, pois₁..pois_{n−2}, φL−φLb], pois_k = (flux[k−1] −
    flux[k])/ave[k] − charge[k] (harmonic-ε faces; Boltzmann partials
    ∂charge/∂φ = −(n+p)).  sub[i] = ∂F_i/∂φ_{i−1} etc.
    """
    from driftjax.numerics.poisson import harmonic_ave_eps
    from driftjax.science.carrier_statistics import n, p

    dgl = cell.dgrid
    eps_face = harmonic_ave_eps(cell)
    ave = (dgl[:-1] + dgl[1:]) / 2  # ave[j] = mean at node j+1
    n_v = n(cell, pot)
    p_v = p(cell, pot)
    eL = eps_face[:-1] / dgl[:-1] / ave  # row k (k=j+1): e_{k-1} term
    eR = eps_face[1:] / dgl[1:] / ave  # row k (k=j+2): e_k term
    rc = (eL + eR) + n_v[1:-1] + p_v[1:-1]  # interior diagonal (n−2,)
    sub = jnp.concatenate([jnp.zeros(1), -eL, jnp.zeros(1)])
    diag = jnp.concatenate([jnp.ones(1), rc, jnp.ones(1)])
    sup = jnp.concatenate([jnp.zeros(1), -eR, jnp.zeros(1)])
    return sub, diag, sup


def _thomas(sub, diag, sup, b):
    """Scalar Thomas solve: sub[i]·x[i−1] + diag[i]·x[i] + sup[i]·x[i+1] = b[i].

    fori_loop implementation (unrolled Python loops re-enter the dispatch
    trap; this stays inside one XLA trace and is jit-compiled once).
    """
    n = b.shape[0]
    if n == 1:
        # L7: only reachable via direct calls (Device/mesh enforce
        # n_points >= 3); kept as the correct scalar base case.
        return b / diag

    def fwd(i, carry):
        cp, dp_ = carry
        m = diag[i] - sub[i] * cp[i - 1]
        cp = cp.at[i].set(jnp.where(i < n - 1, sup[i] / m, 0.0))
        dp_ = dp_.at[i].set((b[i] - sub[i] * dp_[i - 1]) / m)
        return cp, dp_

    cp0 = jnp.zeros(n).at[0].set(sup[0] / diag[0])
    dp0 = jnp.zeros(n).at[0].set(b[0] / diag[0])
    cp, dp_ = jax.lax.fori_loop(1, n, fwd, (cp0, dp0))

    def bwd(j, x):
        i = n - 2 - j
        return x.at[i].set(dp_[i] - cp[i] * x[i + 1])

    x0 = jnp.zeros(n).at[n - 1].set(dp_[n - 1])
    return jax.lax.fori_loop(0, n - 1, bwd, x0)


def _eq_step_impl(cell, bound, phi):
    """One damped Newton step of the equilibrium solve (jittable)."""
    pot = Potentials(jnp.zeros_like(phi), jnp.zeros_like(phi), phi)
    F = F_eq(cell, bound, pot)
    sub, diag, sup = _eq_tridiag(cell, pot)
    p = _thomas(sub, diag, sup, -F)
    dx = logdamp(p)
    return phi + dx, jnp.max(jnp.abs(dx))


_eq_step_jit = jax.jit(_eq_step_impl)


def solve_eq(
    cell: PVCell,
    bound: BoundaryConditions,
    phi_ini,
    tol: float = 1e-10,
    max_iter: int = 60,
    dense: bool = False,
    analytic: bool = True,
    iter_cb=None,
    allow_trace: bool = False,
    loop=None,
    backend: str | None = None,
) -> Potentials:
    """Solve F_eq(φ) = 0 with damped Newton (φn = φp = 0 frozen).

    analytic=True (default): exact O(n) Thomas solve on the tridiagonal
    Poisson Jacobian (harmonic-ε, Boltzmann carrier partials), one XLA
    compile, ~40× faster than the jacrev path; else jacrev+dense.

    L6: returns the bare ``Potentials`` (no iters/resid/converged) — unlike
    ``solve_newton``, which returns ``(pot, stats)``. Kept for
    backward compatibility; equilibrium callers needing diagnostics should
    evaluate ``comp_F``/norms on the returned state.
    """
    if (not allow_trace) and _is_tracer(phi_ini):
        return Potentials(jnp.zeros_like(phi_ini), jnp.zeros_like(phi_ini), phi_ini)
    if loop is None:
        loop = "python" if not allow_trace else "while"
    if loop == "while":
        # Trace-safe: ONE compiled body, runtime early-exit -> small jaxpr.
        return _solve_eq_while(
            cell, bound, phi_ini, tol, max_iter, dense, analytic, allow_trace, backend
        )
    # "python" (eager): fast for-loop with early-exit (mirrors v0.0.x). Only
    # used when not tracing, so float()/break are concrete-safe.
    return _solve_eq_python(
        cell, bound, phi_ini, tol, max_iter, dense, analytic, allow_trace, iter_cb, backend
    )


# ---------------------------------------------------------------------------
# Full coupled Newton
# ---------------------------------------------------------------------------


def _step_newton_impl(cell, bound, x, dense, refinement, analytic, fused, backend=None):
    """Pure per-iteration body (jittable): F, Jacobian, damped step, stats."""
    from driftjax.numerics.analytic_jacobian import banded_jacobian
    from driftjax.numerics.fused_kernels import (
        fused_jacobian_banded,
        fused_residual,
        fused_residual_and_jacobian,
    )
    from driftjax.numerics.residual import comp_F_precomputed

    pot = vec2pot(x)
    use_analytic = (
        analytic
        and not dense
        and not refinement
        and getattr(cell, "statistics", "boltzmann") == "boltzmann"
    )
    use_native_ge = backend == "native_ge_eq"
    _fused_pair = None
    _pre_n = _pre_p = _pre_ni = None
    if fused and use_analytic:
        F, A, B, C = fused_residual_and_jacobian(cell, bound, pot)
        _fused_pair = (F, A, B, C)
    else:
        if fused:
            F = fused_residual(cell, bound, pot)
        elif use_analytic:
            # Fused path: compute carrier stats once, share between residual and Jacobian
            F, _pre_n, _pre_p, _pre_ni = comp_F_precomputed(cell, bound, pot)
        else:
            F = comp_F(cell, bound, pot)
    if use_analytic:
        n = pot.n
        if _fused_pair is not None:
            _, A, B, C = _fused_pair  # F already taken
        else:
            if fused:
                A, B, C = fused_jacobian_banded(cell, bound, pot)
            elif _pre_n is not None:
                # Reuse carrier stats from comp_F_precomputed
                A, B, C = banded_jacobian(cell, bound, pot, n_v=_pre_n, p_v=_pre_p, ni_v=_pre_ni)
            else:
                A, B, C = banded_jacobian(cell, bound, pot)

        if use_native_ge:
            # v0.1.18b megakernel: native GE with row equilibration — zero
            # LAPACK host callbacks, full XLA fusion across residual +
            # Jacobian + GE + update. Entire Newton step is one XLA program.
            from driftjax.numerics.analytic_jacobian import blockwise_residual
            from driftjax.numerics.banded_ge import (
                _apply_row_equil,
                _scalar_row_scales,
                _dense_from_blocks,
                banded_ge_solve,
            )

            blocks_finite = (
                jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(B)) & jnp.all(jnp.isfinite(C))
            )

            def _do_native_ge(_):
                dr = _scalar_row_scales(A, B, C)
                Ae, Be, Ce, be = _apply_row_equil(A, B, C, (-F).reshape(n, 3), dr)
                p_b, info_ge = banded_ge_solve(Ae, Be, Ce, be, equilibrate=False)
                p_b = p_b.reshape(-1)
                lin_b = jnp.linalg.norm(blockwise_residual(A, B, C, F, p_b)) / (
                    jnp.linalg.norm(F) + 1e-30
                )
                ge_ok = info_ge["ok"] & jnp.isfinite(lin_b) & blocks_finite
                return p_b, lin_b, ge_ok

            def _do_dense_fallback(_):
                # MEGAKERNEL: retry GE without row equilibration, then LAPACK
                from driftjax.numerics.analytic_jacobian import blockwise_residual
                dr2 = jnp.ones_like(_scalar_row_scales(A, B, C))
                Ae2, Be2, Ce2, be2 = _apply_row_equil(A, B, C, (-F).reshape(n, 3), dr2)
                p_b2, _info2 = banded_ge_solve(Ae2, Be2, Ce2, be2, equilibrate=False)
                p_b2 = p_b2.reshape(-1)
                lin2 = jnp.linalg.norm(blockwise_residual(A, B, C, F, p_b2)) / (
                    jnp.linalg.norm(F) + 1e-30
                )
                # If unequilibrated GE also failed, fall back to dense LAPACK
                ge_ok2 = _info2["ok"] & jnp.all(jnp.isfinite(p_b2)) & jnp.isfinite(lin2)
                ge_ok2 = ge_ok2 & (lin2 < 1e-4)

                # LAPACK dense fallback: JIT-traceable via jax.numpy
                J_dense = _dense_from_blocks(A, B, C)
                p_b3 = jnp.linalg.solve(J_dense, -F.reshape(-1))
                lin3 = jnp.linalg.norm(J_dense @ p_b3 + F.reshape(-1)) / (
                    jnp.linalg.norm(F) + 1e-30
                )
                lapack_ok = jnp.all(jnp.isfinite(p_b3)) & (lin3 < 1e-4)

                # Priority: unequilibrated GE -> LAPACK -> zero
                use_lapack = (~ge_ok2) & lapack_ok
                p_b2 = jnp.where(use_lapack, p_b3, p_b2)
                lin2 = jnp.where(use_lapack, lin3, lin2)
                # If all fail, return zero step
                all_failed = (~ge_ok2) & (~lapack_ok)
                p_b2 = jnp.where(all_failed, jnp.zeros_like(p_b2), p_b2)
                lin2 = jnp.where(all_failed, jnp.inf, lin2)
                return p_b2, lin2

            # P0-solver contract: zero/nonfinite GE step -> gate -> dense fallback
            p_a, lin_a, _ge_ok = jax.lax.cond(
                blocks_finite,
                _do_native_ge,
                lambda _: (jnp.zeros_like(F), jnp.inf, jnp.array(False)),
                None,
            )
            use_dense = (~_ge_ok) | (~jnp.isfinite(lin_a)) | (lin_a > 1e-4)
            # Both branches return (p, lin) — no LAPACK anywhere
            p_f, lin_f = jax.lax.cond(
                use_dense,
                _do_dense_fallback,
                lambda _: (p_a, lin_a),
                None,
            )
            lstsq_used = (~use_dense) & (~_ge_ok)
            backend_code = jnp.where(use_dense, 2, 0)
        else:
            # Default path: LAPACK banded solve with truncated-SVD lstsq fallback.
            from driftjax.numerics.analytic_jacobian import blockwise_residual
            from driftjax.numerics.banded_solve import banded_solve_with_info

            blocks_finite = (
                jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(B)) & jnp.all(jnp.isfinite(C))
            )

            def _do_banded(_):
                p_b, _info = banded_solve_with_info(A, B, C, (-F).reshape(n, 3))
                p_b = p_b.reshape(-1)
                lin_b = jnp.linalg.norm(blockwise_residual(A, B, C, F, p_b)) / (
                    jnp.linalg.norm(F) + 1e-30
                )
                return p_b, lin_b, _info["used_lstsq"]

            def _do_fallback(_):
                Jf = F_jacobian(cell, bound, pot)
                pf = jnp.linalg.solve(Jf, -F)
                lin = jnp.linalg.norm(Jf @ pf + F) / (jnp.linalg.norm(F) + 1e-30)
                return pf, lin, jnp.array(False)

            p_a, lin_a, lstsq_a = jax.lax.cond(blocks_finite, _do_banded, _do_fallback, None)
            use_dense = (~jnp.isfinite(lin_a)) | (lin_a > 1e-4)
            p_f, lin_f, _ = jax.lax.cond(
                use_dense, _do_fallback, lambda _: (p_a, lin_a, lstsq_a), None
            )
            lstsq_used = lstsq_a & (~use_dense)
            backend_code = jnp.where(use_dense, 2, 0)

        dx = logdamp(p_f)
        x_new = x + dx
        return (
            x_new,
            jnp.max(jnp.abs(dx)),
            jnp.linalg.norm(F),
            jnp.max(jnp.abs(F)),
            lin_f,
            backend_code,
            lstsq_used,
        )
    else:
        J = F_jacobian(cell, bound, pot)
        # FD statistics or tracer context requires jit-safe dense solve (linsolve auto uses host spsolve)
        is_fd = getattr(cell, "statistics", "boltzmann") != "boltzmann"
        if is_fd or _is_tracer(J):
            p = jnp.linalg.solve(J, -F)
            linresid = jnp.linalg.norm(J @ p + F) / (jnp.linalg.norm(F) + 1e-30)
            backend = "dense-fd" if is_fd else "dense-tracer"
        else:
            p, linresid, backend = _linear_solve(J, -F, dense=dense, refinement=refinement)
    dx = logdamp(p)
    x_new = x + dx
    # backend is a concrete Python str (static branches) — pass an int code
    backend_code = _BACKEND_CODE.get(backend, 9)
    # R2: dense path never touches the banded lstsq fallback.
    return (
        x_new,
        jnp.max(jnp.abs(dx)),
        jnp.linalg.norm(F),
        jnp.max(jnp.abs(F)),
        linresid,
        backend_code,
        jnp.array(False),
    )


# ---------------------------------------------------------------------------
# Fully-fused Newton step with native GE (v0.1.18b)
#
# Collapses residual + Jacobian + row-equilibrated GE solve + damped update
# into a SINGLE XLA program — zero host callbacks, optimal for GPU residency.
# Use via: solver = Newton(backend="native_ge_eq")
# Falls back to LAPACK path by default (unchanged behavior).
# ---------------------------------------------------------------------------


def _fused_newton_step_ge(cell, bound, x, n):
    """Fully-fused Newton step: F + analytic Jacobian + native GE + damped update.

    ALL operations (residual assembly, Jacobian assembly, row-equilibrated
    banded GE, logdamp, x update) are inlined into one XLA program.
    No host callbacks, no scipy LAPACK round-trips.

    The row equilibration (D from block max-abs) is computed from the Jacobian
    blocks directly — O(N*bw) with no dense Jacobian materialization — and
    applied to both A/B/C blocks and the RHS before the scan-based GE.

    Args:
        cell: PVCell with boltzmann statistics
        bound: BoundaryConditions
        x: dimensionless potential vector (3n,)
        n: number of mesh points (static)

    Returns:
        (x_new, step_norm, residual_norm, max_abs_F, linresid, backend_code, lstsq_used)
    """
    from driftjax.numerics.analytic_jacobian import banded_jacobian, blockwise_residual
    from driftjax.numerics.banded_ge import (
        _dense_from_blocks,
        _apply_row_equil,
        _scalar_row_scales,
        banded_ge_solve,
    )
    from driftjax.numerics.residual import comp_F_precomputed

    pot = vec2pot(x)

    # --- Fused residual + Jacobian (shared carrier statistics) ---
    F, _pre_n, _pre_p, _pre_ni = comp_F_precomputed(cell, bound, pot)
    A, B, C = banded_jacobian(cell, bound, pot, n_v=_pre_n, p_v=_pre_p, ni_v=_pre_ni)

    # --- Native GE solve with row equilibration (entirely in-XLA) ---
    dr = _scalar_row_scales(A, B, C)  # (N,) per-scalar-row
    Ae, Be, Ce, be = _apply_row_equil(A, B, C, (-F).reshape(n, 3), dr)
    p_b, info_ge = banded_ge_solve(Ae, Be, Ce, be, equilibrate=False)  # already equilibrated
    p_b = p_b.reshape(-1)

    # --- Residual certification on ORIGINAL blocks (un-equilibrated check) ---
    linresid = jnp.linalg.norm(blockwise_residual(A, B, C, F, p_b)) / (jnp.linalg.norm(F) + 1e-30)

    # --- Damped update + stats ---
    lstsq_used = ~info_ge["ok"]
    backend_code = jnp.where(info_ge["ok"], jnp.int32(0), jnp.int32(9))
    dx = logdamp(p_b)
    x_new = x + dx

    return (
        x_new,
        jnp.max(jnp.abs(dx)),
        jnp.linalg.norm(F),
        jnp.max(jnp.abs(F)),
        linresid,
        backend_code,
        lstsq_used,
    )


_fused_newton_step_ge_jit = jax.jit(_fused_newton_step_ge, static_argnames=("n",))


_step_newton_jit = jax.jit(
    _step_newton_impl, static_argnames=("dense", "refinement", "analytic", "fused", "backend")
)


def step_newton(
    cell,
    bound,
    pot,
    tol=1e-12,
    dense=False,
    refinement=False,
    analytic=True,
    fused=False,
    allow_trace: bool = False,
    backend: str | None = None,
):
    """One damped Newton step; returns (pot_new, error, stats).

    analytic=True (default): Jacobian from the pinned fast backend
    (numerics.analytic_jacobian.banded_jacobian, block-tridiagonal, solved
    via pivoted LAPACK banded).  The canonical jacfwd Jacobian
    remains available via dense=True and is the unit-test reference.
    The step body is XLA-compiled once and reused across iterations and
    bias steps (fixes the historical per-op dispatch overhead that made
    the un-jitted loop ~4× slower than prime1/previous's `@jit step`).
    """
    x_new, err, resid, resid_f, linres, backend_code, lstsq_used = _step_newton_jit(
        cell, bound, pot2vec(pot), dense, refinement, analytic, fused, backend
    )
    pot_new = vec2pot(x_new)
    stats = {
        "error": err,
        "resid": resid,
        "resid_f": resid_f,
        "linresid": linres,
        # R2 provenance: True when any banded step in this Newton step used
        # the truncated-SVD (lstsq) fallback (trace-safe boolean array).
        "lstsq": lstsq_used,
        # backend label is built statically by the caller (solve_newton); the
        # per-step backend_code is a tracer inside lax.while_loop, so it cannot
        # be int()-ed here. Keep a constant placeholder.
        "backend": "other",
    }
    return pot_new, stats["error"], stats


def solve_newton(
    cell,
    bound,
    pot_ini,
    tol: float = 1e-8,
    max_steps: int = 100,
    f_tol: float | None = None,
    globalization: str = "auto",
    dense: bool = False,
    refinement: bool = False,
    fused: bool = False,
    iter_cb=None,
    allow_trace: bool = False,
    loop=None,
    backend: str | None = None,
):
    """Solve comp_F = 0.

    globalization:
      * "logdamp" — plain damped Newton (historical path)
      * "auto"    — logdamp; on failure, Armijo line-search Newton (robust
                    out-of-basin recovery) before the dense fallback
      * "ls"      — Armijo line search from the start
      * "ptc"     — pseudo-transient continuation from the start

    refinement=True uses the FP32+refinement linear path (NEW).
    f_tol enables the residual gate: stop on max|F| < f_tol too (never
    trust the step norm alone on a degenerate Jacobian).

    Convergence-gating division of labor (N1): the in-loop gate is
    step-norm-only (``step_ok``: error <= tol) in both the traced and eager
    loops, because the truncated-SVD (lstsq) fallback exists precisely for
    systems where step-norm progress decouples from conditioning floors — a
    residual gate inside the loop would re-break the case the fallback
    fixes. Residual certification is deferred to the post-hoc absolute
    (1e-6) audit in ``simulate()``, which warns loudly on unconverged
    biases. ``last_stats["lstsq"]`` records whether the fallback fired.

    iter_cb(it, err, resid): per-iteration observer (0-based it; err = step
    norm, resid = max|F|) — used by console.DebugLog for post-hoc traces.
    No-op when None; exceptions swallowed. Eager loop only: the traced
    ``while_loop`` path never calls it (a per-iteration host callback would
    serialize the compiled loop; M1), so progress bars stay dark under jit.
    """
    # Float64 is mandatory (Jacobians ~1e19, Nc*Nv ~1e38 overflow float32).
    if pot_ini.phi.dtype != jnp.float64:
        raise TypeError(
            f"solve_newton requires float64 potentials, got {pot_ini.phi.dtype}; ensure JAX_ENABLE_X64=1"
        )
    if (not allow_trace) and _is_tracer(pot_ini.phi):
        return pot_ini, {"backend": "tracer_passthrough"}

    if globalization == "ls":
        from driftjax.solvers.ptc import solve_newton_ls

        return solve_newton_ls(
            cell, bound, pot_ini, tol=tol, f_tol=f_tol, max_steps=max_steps,
            refinement=refinement, allow_trace=allow_trace, fused=fused, backend=backend,
        )
    if globalization == "ptc":
        from driftjax.solvers.ptc import solve_ptc

        return solve_ptc(
            cell, bound, pot_ini, tol=tol, f_tol=f_tol, max_steps=max_steps,
            refinement=refinement, allow_trace=allow_trace, fused=fused, backend=backend,
        )

    if loop is None:
        loop = "python" if not allow_trace else "while"

    if loop == "while":
        pot, last_stats = _solve_newton_while(
            cell,
            bound,
            pot_ini,
            tol,
            max_steps,
            f_tol,
            dense,
            refinement,
            fused,
            allow_trace,
            iter_cb,
            backend,
        )
    else:  # "python" (eager): fast for-loop with early-exit (v0.0.x path)
        pot, last_stats = _solve_newton_python(
            cell,
            bound,
            pot_ini,
            tol,
            max_steps,
            f_tol,
            dense,
            refinement,
            fused,
            allow_trace,
            iter_cb,
            backend,
        )

    # Eager-only line-search fallback (robust out-of-basin recovery). Skipped
    # under trace, where the conditional would concretize. M5: every eager
    # return path sets a concrete Python-bool "converged", so `not` never
    # sees a tracer here; default False (missing flag ⇒ try recovery).
    if (not allow_trace) and (not last_stats.get("converged", False)) and globalization == "auto":
        from driftjax.solvers.ptc import solve_newton_ls

        pot_ls, sls = solve_newton_ls(
            cell,
            bound,
            pot_ini,
            tol=tol,
            f_tol=f_tol if f_tol is not None else tol,
            max_steps=max(max_steps, 300),
            refinement=refinement,
            allow_trace=allow_trace,
            fused=fused,
            backend=backend,
        )
        if sls.get("converged"):
            return pot_ls, {**sls, "fallback": "ls"}
    return pot, last_stats


def _solve_eq_while(
    cell, bound, phi_ini, tol, max_iter, dense, analytic, allow_trace, backend=None
):
    """Trace-safe equilibrium solve via lax.while_loop (ONE compiled body,
    runtime early-exit -> small jaxpr). Used under jit/grad."""
    phi = jnp.asarray(phi_ini, dtype=jnp.float64)

    def _cond(state):
        it, phi, err = state
        return (it < max_iter) & (err > tol)

    def _body(state):
        it, phi, err = state
        if analytic and not dense:
            phi_new, err_new = _eq_step_jit(cell, bound, phi)
        else:
            pot = Potentials(jnp.zeros_like(phi), jnp.zeros_like(phi), phi)
            F = F_eq(cell, bound, pot)
            J = F_eq_jacobian(cell, bound, pot)
            p, _, _ = _linear_solve(J, -F, dense=dense)
            phi_new = phi + logdamp(p)
            err_new = jnp.max(jnp.abs(logdamp(p)))
        return (it + 1, phi_new, err_new)

    init = (jnp.array(0, dtype=jnp.int32), phi, jnp.array(jnp.inf, dtype=jnp.float64))
    it, phi, err = jax.lax.while_loop(_cond, _body, init)
    return Potentials(jnp.zeros_like(phi), jnp.zeros_like(phi), phi)


def _solve_eq_python(
    cell, bound, phi_ini, tol, max_iter, dense, analytic, allow_trace, iter_cb, backend=None
):
    """Eager equilibrium solve: Python for-loop with early-exit (v0.0.x path).
    Only used when not tracing, so float()/break are concrete-safe and fast."""
    phi = jnp.asarray(phi_ini, dtype=jnp.float64)
    for it in range(max_iter):
        if analytic and not dense:
            phi, err = _eq_step_jit(cell, bound, phi)
        else:
            pot = Potentials(jnp.zeros_like(phi), jnp.zeros_like(phi), phi)
            F = F_eq(cell, bound, pot)
            J = F_eq_jacobian(cell, bound, pot)
            p, _, _ = _linear_solve(J, -F, dense=dense)
            phi = phi + logdamp(p)
            err = jnp.max(jnp.abs(logdamp(p)))
        if iter_cb is not None:
            try:
                iter_cb(it, float(err), float(err))
            except Exception:
                pass
        if float(err) < tol:
            break
    return Potentials(jnp.zeros_like(phi), jnp.zeros_like(phi), phi)


def _solve_newton_while(
    cell,
    bound,
    pot_ini,
    tol,
    max_steps,
    f_tol,
    dense,
    refinement,
    fused,
    allow_trace,
    iter_cb,
    backend=None,
):
    """Trace-safe coupled-Newton solve via lax.while_loop (ONE compiled body,
    runtime early-exit -> small jaxpr, trace-safe). Used under jit/grad.

    Best-iterate tracking with rebound detection (same semantics as the eager
    `_solve_newton_python` loop): the loop carries the lowest-residual iterate
    seen. When the residual REBOUNDS from its best value (junk Newton steps on
    a near-degenerate Jacobian kick the iterate out after the residual bottomed
    out at the machine floor), the loop stops and returns the best iterate,
    certifying convergence; a settled tail comparable to best is returned as
    converged, exactly as the eager exhaustion branch does. Step-norm
    certification additionally requires the residual gate (S6/H1), so
    previously-good trajectories are bit-identical whenever the residual
    was already below criterion at step convergence (the normal case).
    NaN steps freeze the iterate and exit via the rebound branch instead
    of poisoning the carry.
    """
    crit = _rebound_criterion(f_tol)
    f_tol_active = f_tol is not None
    # Best-iterate state: (best_pot, best_resid) ride along in the loop carry
    # (two extra 3N-vectors of device-local traffic per iteration — negligible
    # next to one Jacobian assembly + banded solve). `failed` freezes the
    # iterate on NaN steps (eager returns best immediately; here we flag exit
    # and let the post-loop rebound logic select best).

    # S6/H1: step-norm certification (same as the eager loop).
    # Residual-based gating is deferred to the post-hoc audit in
    # simulate(), which is condition-number-aware and does not reject
    # solutions that are the best achievable under ill-conditioning.
    def step_ok(error, resid_f):
        return error <= tol

    def _cond(state):
        it, pot, error, resid_f, _best_pot, _best_resid, failed, _any_lstsq = state
        step_converged = step_ok(error, resid_f)
        resid_converged = (resid_f < f_tol) if f_tol_active else jnp.array(False)
        return (it < max_steps) & (~(step_converged | resid_converged)) & (~failed)

    def _body(state):
        it, pot, error, resid_f, best_pot, best_resid, failed, any_lstsq = state
        pot_new, error_new, stats = step_newton(
            cell,
            bound,
            pot,
            tol,
            dense=dense,
            refinement=refinement,
            fused=fused,
            allow_trace=allow_trace,
            backend=backend,
        )
        # resid_f is max|F| at the INPUT state (the state stepped FROM), so it
        # eager loop, which updates best before checking for NaN. A NaN
        # resid_f itself is never an improvement (NaN < x is False), so the
        # clean best survives NaN tails.
        r_new = stats["resid_f"]
        step_failed = (
            (~jnp.isfinite(error_new)) | (~jnp.isfinite(stats["resid"])) | (~jnp.isfinite(r_new))
        )
        improved = r_new < best_resid
        best_pot_new = jax.tree.map(lambda b, p: jnp.where(improved, p, b), best_pot, pot)
        best_resid_new = jnp.where(improved, r_new, best_resid)
        # Freeze the iterate on failure but preserve the NaN error for
        # diagnostics (matches eager `last_stats["error"]`); the gate below
        # stays safe because NaN <= tol is False.
        pot_out = jax.tree.map(lambda new, old: jnp.where(step_failed, old, new), pot_new, pot)
        return (
            it + 1,
            pot_out,
            error_new,
            r_new,
            best_pot_new,
            best_resid_new,
            failed | step_failed,
            any_lstsq | jnp.asarray(stats.get("lstsq", False)),
        )

    init = (
        jnp.array(0, dtype=jnp.int32),
        pot_ini,
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.array(jnp.inf, dtype=jnp.float64),
        pot_ini,
        jnp.array(jnp.inf, dtype=jnp.float64),
        jnp.array(False),
        jnp.array(False),
    )
    it, pot, error, resid_f, best_pot, best_resid, failed, any_lstsq = jax.lax.while_loop(
        _cond, _body, init
    )
    step_converged = step_ok(error, resid_f)
    resid_converged = (resid_f < f_tol) if f_tol_active else jnp.array(False)
    converged = step_converged | resid_converged
    # Rebound / settled certification (mirrors the eager exhaustion/NaN
    # branches one-for-one): exhausted or failed with a usable best whose
    # tail is materially worse (or non-finite) → return best as converged +
    # stagnated; exhausted with a settled tail comparable to best → return
    # the tail as converged (degenerate endgame wandering, not divergence).
    # `stagnated` also covers failed-but-unconverged exits (NaN-in garbage),
    # matching the eager NaN branch. M3: the rebound/settled comparison uses
    # a FRESH residual at the returned tail iterate — the loop-carried
    # `resid_f` is max|F| at the state the last step was taken FROM (one
    # iterate stale). Converged solves keep the carried value (pristine
    # reporting); `jnp.fmin` (not minimum) keeps a NaN tail from poisoning
    # the reported settled residual.
    F_tail = comp_F(cell, bound, pot)
    r_tail = jnp.max(jnp.abs(F_tail))
    best_usable = best_resid < crit
    tail_worse = (r_tail > _REBOUND_TAIL * best_resid) | failed | (~jnp.isfinite(r_tail))
    rebound = (~converged) & best_usable & tail_worse
    settled = (~converged) & (~rebound) & best_usable & (~failed)
    out_pot = jax.tree.map(lambda p, b: jnp.where(rebound, b, p), pot, best_pot)
    out_resid_f = jnp.where(
        rebound,
        best_resid,
        jnp.where(settled, jnp.fmin(best_resid, r_tail), jnp.where(converged, resid_f, r_tail)),
    )
    backend_label = "dense" if dense else ("refined" if refinement else "analytic-banded")
    last_stats = newton_stats(
        backend=backend_label,
        iters=it,
        error=error,
        resid=jnp.linalg.norm(comp_F(cell, bound, out_pot)),
        resid_f=out_resid_f,
        converged=converged | rebound | settled,
        stagnated=rebound | (failed & (~converged)),
        fallback=None,
        # R2 provenance: trace-safe flag — True if any banded step in this
        # solve used the truncated-SVD (lstsq) fallback.
        lstsq=any_lstsq,
    )
    return out_pot, last_stats


def _solve_newton_python(
    cell,
    bound,
    pot_ini,
    tol,
    max_steps,
    f_tol,
    dense,
    refinement,
    fused,
    allow_trace,
    iter_cb,
    backend=None,
):
    """Eager coupled-Newton solve: Python for-loop with early-exit (v0.0.x
    path). Only used when not tracing, so float()/break are concrete-safe and
    fast; matches the converged result of the while_loop path.

    Best-iterate tracking with rebound detection (mirrors
    ``_solve_newton_while``): the lowest-residual iterate is kept; when the
    residual rebounds from its best value (junk endgame steps on a
    near-degenerate Jacobian) the best iterate is returned as converged.
    """
    crit = _rebound_criterion(f_tol)  # shared with the traced loop (AUDIT R3)
    pot = pot_ini
    best_pot = pot_ini
    best_resid = None
    last_stats = {}
    # R2 provenance: OR-accumulates the per-step lstsq flag into last_stats.
    any_lstsq = False
    for it in range(max_steps):
        pot_prev = pot
        pot, error, stats = step_newton(
            cell,
            bound,
            pot,
            tol,
            dense=dense,
            refinement=refinement,
            fused=fused,
            allow_trace=allow_trace,
            backend=backend,
        )
        try:
            any_lstsq = any_lstsq or bool(stats.get("lstsq", False))
        except Exception:
            pass
        last_stats = {
            **stats,
            "iters": it,
            "fallback": None,
            "backend": "dense" if dense else ("refined" if refinement else "analytic-banded"),
            "lstsq": any_lstsq,
        }
        r = float(stats["resid_f"])  # max|F| at pot_prev (the state stepped FROM)
        if best_resid is None or r < best_resid:
            best_resid, best_pot = r, pot_prev
        err = float(error)
        if iter_cb is not None:
            try:
                # M2: the observer contract is max|F| (resid_f), not ‖F‖₂.
                rf = float(stats["resid_f"])
                iter_cb(it, err, rf if not jnp.isnan(rf) else None)
            except Exception:
                pass
        # Step-norm convergence (same criterion as the traced loop).
        # Residual-based gating is deferred to the post-hoc audit in
        # simulate(), which is condition-number-aware.
        if err < tol:
            last_stats = newton_stats(**{**last_stats, "converged": True, "stagnated": False})
            return pot, last_stats
        if jnp.isnan(error) or jnp.isnan(stats["resid"]):
            # junk/NaN step: the best iterate seen is the closest-to-root state
            F_best = comp_F(cell, bound, best_pot)
            last_stats = newton_stats(
                **{
                    **last_stats,
                    "converged": bool(best_resid is not None and best_resid < crit),
                    "stagnated": True,
                    "resid": float(jnp.linalg.norm(F_best)),
                    "resid_f": best_resid,
                }
            )
            return best_pot, last_stats
    # exhausted: return the best iterate only when the tail is materially
    # worse than the best (genuine endgame wandering); a settled tail whose
    # residual is comparable to the best is returned as-is (pristine
    # semantics — the last iterate of a converged-in-step-norm solve).
    if best_resid is not None and best_resid < crit:
        if not jnp.isnan(float(jnp.linalg.norm(pot2vec(pot)))):
            F_last = comp_F(cell, bound, pot)
            resid_last = float(jnp.max(jnp.abs(F_last)))
            norm_last = float(jnp.linalg.norm(F_last))
        else:
            resid_last, norm_last = best_resid * float("inf"), float("inf")
        if resid_last > _REBOUND_TAIL * best_resid:
            F_best = comp_F(cell, bound, best_pot)
            last_stats = newton_stats(
                **{
                    **last_stats,
                    "converged": True,
                    "stagnated": True,
                    "resid": float(jnp.linalg.norm(F_best)),
                    "resid_f": best_resid,
                }
            )
            return best_pot, last_stats
        last_stats = newton_stats(
            **{
                **last_stats,
                "converged": True,
                "stagnated": False,
                "resid": norm_last,
                "resid_f": min(best_resid, resid_last),
            }
        )
        return pot, last_stats
    last_stats = newton_stats(**{**last_stats, "converged": last_stats.get("converged", False)})
    return pot, last_stats
