"""Bias continuation: warm starts and IV sweep."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from driftjax.fields import Potentials, PVCell
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.solvers.newton import solve_eq, solve_newton


def equilibrium_guess(cell: PVCell) -> Potentials:
    """φ = −E_F guess; φn = φp = 0 (equilibrium reference)."""
    from driftjax.science.carrier_statistics import EF_zero

    eq_phi = EF_zero(cell)
    return Potentials(jnp.zeros_like(eq_phi), jnp.zeros_like(eq_phi), eq_phi)


def qfl_hotstart(pot_eq: Potentials, v: float) -> Potentials:
    half = 0.5 * v
    return Potentials(pot_eq.phi_n + half, pot_eq.phi_p - half, pot_eq.phi)


def linear_extrapolation(pot_a, pot_b, v_a, v_b, v_target) -> Potentials:
    # AUDIT: guard the degenerate v_b == v_a case (n_steps=1 / vmax=0 gave
    # 0/0 → NaN guess); fall back to the latest state. Pure jnp ops (not a
    # Python branch) so the guard is also tracer-safe under jit/grad — note
    # the division itself must be sanitized first, because Python-float
    # 0.0-division raises before any where() could select away from it.
    denom = jnp.asarray(v_b - v_a)
    safe_denom = jnp.where(denom == 0.0, 1.0, denom)
    scale = jnp.where(denom == 0.0, 0.0, (v_target - v_b) / safe_denom)
    return Potentials(
        pot_b.phi_n + scale * (pot_b.phi_n - pot_a.phi_n),
        pot_b.phi_p + scale * (pot_b.phi_p - pot_a.phi_p),
        pot_b.phi + scale * (pot_b.phi - pot_a.phi),
    )


from jax import custom_jvp as _custom_jvp


def _total_current_impl(cell: PVCell, pot: Potentials):
    from driftjax.numerics.scharfetter_gummel import Jn, Jp

    return jnp.mean(Jn(cell, pot) + Jp(cell, pot))


total_current = _custom_jvp(_total_current_impl)


@total_current.defjvp
def _total_current_jvp(primals, tangents):
    cell, pot = primals
    dcell, dpot = tangents
    y = _total_current_impl(cell, pot)
    # Analytic pot part (hot path for g_x) — mean of Jn/Jp phi derivatives
    try:
        from driftjax.numerics.analytic_jacobian import _Jn_deriv, _Jp_deriv

        DJn_n0, DJn_n1, DJn_0, DJn_1 = _Jn_deriv(cell, pot)
        DJp_p0, DJp_p1, DJp_0, DJp_1 = _Jp_deriv(cell, pot)
        # dpot leaves: phi_n, phi_p, phi
        dJn_pot = (
            DJn_n0 * dpot.phi_n[:-1]
            + DJn_n1 * dpot.phi_n[1:]
            + DJn_0 * dpot.phi[:-1]
            + DJn_1 * dpot.phi[1:]
        )
        dJp_pot = (
            DJp_p0 * dpot.phi_p[:-1]
            + DJp_p1 * dpot.phi_p[1:]
            + DJp_0 * dpot.phi[:-1]
            + DJp_1 * dpot.phi[1:]
        )
        dI_pot = jnp.mean(dJn_pot + dJp_pot)
    except Exception:
        # Fallback to autodiff for pot if analytic unavailable
        _, dI_pot = jax.jvp(lambda p: _total_current_impl(cell, p), (pot,), (dpot,))
        dI_pot = dI_pot if jnp.ndim(dI_pot) == 0 else jnp.mean(dI_pot)
        dI_pot = jnp.asarray(dI_pot)
    # Cell part via jvp (covers mn/mp/Chi etc., less hot).
    # AUDIT: failures used to be swallowed as dI_cell=0, silently dropping
    # the entire optics/doping gradient channel. Propagate instead — a wrong
    # gradient is worse than an exception.
    _, dI_cell = jax.jvp(lambda c: _total_current_impl(c, pot), (cell,), (dcell,))
    return y, dI_pot + dI_cell


def sweep(
    cell: PVCell,
    vmax: float,
    n_steps: int = 40,
    tol=1e-10,
    refinement: bool = False,
    fused: bool = False,
    allow_trace: bool = False,
    loop=None,
    progress=None,
    v_scale=None,
    init=None,
    batched: bool = False,
):
    """Serial bias sweep 0 → vmax with QFL hot-start and linear prediction.

    progress: optional reporting hook — an object with
    ``update(i, *, v_v, iters, resid, backend, fallback, converged, dt,
    compiling)`` and ``close()`` (e.g. ``driftjax.console.SweepProgress``),
    or a plain callable ``fn(i, info_dict)``.  Never affects numerics.
    v_scale: convert dimensionless bias to volts (``sc["energy"]``).
    """
    # M10: degenerate single-point schedules break find_voc (empty
    # sign-change axis) and the MPP spline (length-1 diffs) — fail loudly.
    if int(n_steps) < 2:
        raise ValueError(f"sweep needs n_steps >= 2, got {n_steps!r}")
    # Batched path: one vmapped banded-solver pass over bias axis (opt-in,
    # Boltzmann statistics / no warm-start / no progress reporting).  Jit-compatible
    # (lax.while_loop) so it works under jax.jit and the custom_vjp primal.
    # Fused flag propagates for optics+Jacobian fusion (single XLA program).
    if batched and init is None and not refinement and progress is None:
        from driftjax.solvers.batched import sweep_batched as _batched_sweep

        volts_b, curs_b, pots_b = _batched_sweep(cell, vmax, n_steps=n_steps, tol=tol)
        # Normalise to the serial path's outputs: list[Potentials] per bias.
        # sweep_batched returns one batched Potentials pytree (fields (B, n)).
        pots_list = [
            jax.tree_util.tree_map(lambda a, i=i: a[i], pots_b) for i in range(pots_b.phi.shape[0])
        ]
        # R2: the vmapped batched path carries no per-bias Newton stats, so
        # fallback usage is unknown here (None = unverified, not False).
        return volts_b, curs_b, pots_list, [None] * int(n_steps)
    voltages = jnp.linspace(0.0, vmax, n_steps)
    currents: list[jax.Array] = []
    pots: list[Potentials] = []
    # R2 provenance: per-bias truncated-SVD fallback flags (True/False on the
    # concrete serial path; True means dgbsv failed and lstsq stepped).
    fallbacks: list = []
    init_pots = None
    if init is not None:
        if hasattr(init, "potentials"):
            init_pots = list(init.potentials)
        elif isinstance(init, (list, tuple)):
            init_pots = list(init)
        else:
            init_pots = [init]
    have_iter = progress is not None and hasattr(progress, "iter")

    def _iter_phase(phase):
        def cb(it, err, resid):
            try:
                progress.iter(it, err=err, resid=resid, phase=phase)
            except Exception:
                pass

        return cb

    for i, v in enumerate(voltages):
        bound = boundary_bias(cell, v)
        # R2: per-bias fallback flag for this bias (OR over main + sub-steps).
        bias_lstsq = False
        if init_pots is not None and i < len(init_pots) and init_pots[i] is not None:
            guess = init_pots[i]
        elif i == 0:
            guess = solve_eq(
                cell,
                boundary_eq(cell),
                equilibrium_guess(cell).phi,
                iter_cb=_iter_phase("equilibrium") if have_iter else None,
                allow_trace=allow_trace,
                loop=loop,
            )
        elif (not allow_trace) and i == 1:
            guess = qfl_hotstart(pots[0], float(v))
        elif not allow_trace:
            guess = linear_extrapolation(
                pots[i - 2], pots[i - 1], float(voltages[i - 2]), float(voltages[i - 1]), float(v)
            )
        else:
            # trace-safe hot-start: continue from the previous converged solution
            guess = pots[i - 1]
        if have_iter and (not allow_trace):
            vtag = (
                f"V = {float(v) * v_scale:.3f} V" if v_scale is not None else f"v = {float(v):.3f}"
            )
            pot, stats = solve_newton(
                cell,
                bound,
                guess,
                tol=tol,
                refinement=refinement,
                fused=fused,
                iter_cb=_iter_phase(vtag),
                allow_trace=allow_trace,
                loop=loop,
            )
        else:
            pot, stats = solve_newton(
                cell,
                bound,
                guess,
                tol=tol,
                refinement=refinement,
                fused=fused,
                allow_trace=allow_trace,
                loop=loop,
            )
        # Adaptive step-halving continuation (eager path only): if the
        # direct solve from the scheduled guess failed to converge (stiff
        # heterostructures, large bias steps, near-degenerate Jacobians),
        # re-approach v_i from the last converged state through 2, 4, ...,
        # 32 sub-steps. Deterministic ladder; zero overhead when the
        # direct solve converges.
        # NOTE: the tracer check MUST come first — bool() on the converged
        # flag raises TracerBoolConversionError under jit/grad.
        if not allow_trace:
            # AUDIT: all three fields must be finite (was phi only —
            # phi_n/phi_p NaNs slipped through as "converged").
            _conv = bool(stats.get("converged", False)) and bool(
                jnp.all(jnp.isfinite(pot.phi))
                and jnp.all(jnp.isfinite(pot.phi_n))
                and jnp.all(jnp.isfinite(pot.phi_p))
            )
            try:
                bias_lstsq = bias_lstsq or bool(stats.get("lstsq", False))
            except Exception:
                pass
            # i >= 1: halving re-approaches v_i from the previous converged
            # state; at i == 0 there is no prior state to sub-step from.
            if not _conv and i >= 1:
                v_a = float(voltages[i - 1]) if i >= 1 else 0.0
                pot_ref = pots[i - 1]
                for n_sub in (2, 4, 8, 16, 32):
                    sub_vs = jnp.linspace(jnp.asarray(v_a), v, n_sub + 1)[1:]
                    pot_cur = pot_ref
                    ok = True
                    for v_s in sub_vs:
                        pot_s, st_s = solve_newton(
                            cell,
                            boundary_bias(cell, v_s),
                            pot_cur,
                            tol=tol,
                            refinement=refinement,
                            fused=fused,
                            allow_trace=allow_trace,
                            loop=loop,
                        )
                        try:
                            bias_lstsq = bias_lstsq or bool(st_s.get("lstsq", False))
                        except Exception:
                            pass
                        if bool(st_s.get("converged", False)) and bool(
                            jnp.all(jnp.isfinite(pot_s.phi))
                            and jnp.all(jnp.isfinite(pot_s.phi_n))
                            and jnp.all(jnp.isfinite(pot_s.phi_p))
                        ):
                            pot_cur = pot_s
                        else:
                            ok = False
                            break
                    if ok:
                        # AUDIT: carry the sub-step residual (was missing, so
                        # the progress report below claimed converged=False
                        # for a converged step-halving recovery). H2: full
                        # canonical keys via newton_stats (‖F‖₂ + max|F|).
                        from driftjax._util import newton_stats as _ns
                        from driftjax.numerics.residual import comp_F as _comp_F

                        _res = _comp_F(cell, boundary_bias(cell, v), pot_cur)
                        _n2 = float(jnp.linalg.norm(_res))
                        _nf = float(jnp.max(jnp.abs(_res)))
                        pot, stats = (
                            pot_cur,
                            _ns(
                                converged=True,
                                fallback="step-halving",
                                iters=None,
                                resid=_n2,
                                resid_f=_nf,
                                error=_n2,
                                stagnated=False,
                                backend=None,
                                lstsq=bias_lstsq,
                            ),
                        )
                        break
        pots.append(pot)
        currents.append(total_current(cell, pot))
        # R2: record this bias's fallback flag alongside the state.
        fallbacks.append(bias_lstsq if not allow_trace else None)
        if progress is not None and (not allow_trace):
            resid = stats.get("resid")
            info = {
                "v_v": float(v) * v_scale if v_scale is not None else float(v),
                "iters": stats.get("iters"),
                "resid": None if resid is None or jnp.isnan(resid) else float(resid),
                "backend": stats.get("backend"),
                "fallback": stats.get("fallback"),
                # H1: the solver's own flag is authoritative — a finite
                # residual alone must not report convergence (an
                # unconverged-but-finite tail previously lit the bar green).
                "converged": bool(stats.get("converged", False)),
                "dt": 0.0,
                "compiling": False,
            }
            try:
                if hasattr(progress, "update"):
                    progress.update(i, **info)
                else:
                    progress(i, info)
            except Exception:
                pass  # reporting must never break the sweep
    # R2: 4-tuple — the 4th element is the per-bias truncated-SVD fallback
    # flags (True/False concrete serial; None entries when unverified).
    return voltages, jnp.array(currents), pots, fallbacks


def find_voc(voltages, currents) -> jax.Array:
    """Open-circuit voltage by linear interpolation of the current sign change.

    NaN when the sweep does not reach Voc (currents stay positive), i.e. the
    true Voc lies above the sweep's last bias point.

    Written to be ``jit``/``vmap``-safe: all reductions act on the trailing
    axis and use only statically-shaped primitives.
    """
    c = jnp.asarray(currents)
    v = jnp.asarray(voltages)
    cond = c[..., :-1] * c[..., 1:] < 0.0
    # first sign-change index along the trailing axis (argmax of a boolean
    # array returns the index of the first True, or 0 if none)
    idx = jnp.argmax(cond, axis=-1)
    found = jnp.any(cond, axis=-1)
    idx = jnp.where(found, idx, 0)
    idx = jnp.clip(idx, 0, c.shape[-1] - 2)
    I1 = jnp.take_along_axis(c, idx[..., None], axis=-1)[..., 0]
    I2 = jnp.take_along_axis(c, (idx + 1)[..., None], axis=-1)[..., 0]
    V1 = jnp.take_along_axis(v, idx[..., None], axis=-1)[..., 0]
    V2 = jnp.take_along_axis(v, (idx + 1)[..., None], axis=-1)[..., 0]
    # AUDIT: the 1e-300 floor is absolute — denormal opposite-sign currents
    # (|I2−I1| ~ 1e-320, i.e. both legs numerically zero) made it dominate
    # the denominator and fabricated a huge Voc. Require the crossing legs
    # to carry a resolvable current, else report no crossing (NaN). The
    # reference scale floors at 1e-100, not 1e-300: 1e-30 * 1e-300 = 1e-330
    # underflows to 0.0 in float64 (min subnormal ~5e-324), which would
    # disarm the guard for exactly the denormal case it protects.
    dI = I2 - I1
    tiny = jnp.abs(dI) < 1e-30 * jnp.maximum(jnp.maximum(jnp.abs(I1), jnp.abs(I2)), 1e-100)
    voc = V1 - I1 * (V2 - V1) / (dI + 1e-300)
    found = found & (~tiny)
    # No zero crossing in range -> NaN (truthful "Voc above sweep" signal).
    # Callers that need a finite number handle it explicitly (sweep FoM
    # guards, Solution vmax fallback, console "> vmax" rendering). Never
    # silently substitute vmax here: that masks truncation as a real Voc.
    return jnp.where(found, voc, jnp.nan)


def voc_bracket(voltages, currents):
    """First sign-change bracket (v_a, v_b, found) mirroring find_voc.

    Concrete-path helper for secant refinement: returns Python floats and
    a bool. Shares find_voc's denormal guard so bracket existence agrees
    exactly with a finite find_voc value.
    """
    import numpy as _np

    c = _np.asarray(currents, dtype=float)
    v = _np.asarray(voltages, dtype=float)
    for i in range(len(c) - 1):
        if c[i] * c[i + 1] < 0.0:
            dI = c[i + 1] - c[i]
            tiny = abs(dI) < 1e-30 * max(max(abs(c[i]), abs(c[i + 1])), 1e-100)
            if not tiny:
                return float(v[i]), float(v[i + 1]), True
    return float("nan"), float("nan"), False


def refine_voc(cell, v_a, v_b, pot_a, tol=1e-10, max_iter=3):
    """Secant-refine the implicit root J(Voc) = 0 inside a bracket.

    Uses the sweep's own bracket-leg currents (no re-solve: re-solving a
    leg from a foreign guess can land on a different Newton branch, which
    is exactly what broke naive secant here — branch dependence is real
    on stiff devices). Fresh evaluations are secant proposals CLAMPED to
    [v_a, v_b], each warm-started from the nearest evaluated state; the
    best |J| seen wins. Returns (voc, slope_ok); on any failure
    (non-convergence, out-of-bracket proposal, no improvement over the
    legs) returns (nan, False) and the caller keeps the interpolation.
    slope_ok additionally requires |dJ/dV| above the resolvability floor
    the implicit sensitivity dVoc/dtheta = -J_theta/J_V needs.
    """
    from driftjax.solvers.newton import solve_newton

    def _current_at(vv, guess):
        pot, st = solve_newton(cell, boundary_bias(cell, vv), guess, tol=tol)
        if not bool(st.get("converged", False)):
            return None, None
        return float(total_current(cell, pot)), pot

    j_a, pot_a2 = _current_at(v_a, pot_a)
    if j_a is None:
        return float("nan"), False
    j_b, pot_b = _current_at(v_b, pot_a2)
    if j_b is None:
        return float("nan"), False
    if j_a * j_b >= 0.0:
        # Legs re-solved onto the same branch side: the sweep bracket does
        # not reproduce under re-solve (branch dependence). Keep sweep
        # interpolation rather than chasing a ghost root.
        return float("nan"), False
    lo, hi = (v_a, v_b) if v_a < v_b else (v_b, v_a)
    best_v, best_j = v_a, j_a
    if abs(j_b) < abs(best_j):
        best_v, best_j = v_b, j_b
    v_prev, j_prev = v_a, j_a
    v_cur, j_cur, pot_cur = v_b, j_b, pot_b
    for _ in range(max_iter):
        denom = j_cur - j_prev
        if denom == 0.0:
            break
        v_next = v_cur - j_cur * (v_cur - v_prev) / denom
        # Clamp to bracket: never extrapolate (branch jumps live outside).
        v_next = min(max(v_next, lo), hi)
        j_next, pot_next = _current_at(v_next, pot_cur)
        if j_next is None:
            break
        if abs(j_next) < abs(best_j):
            best_v, best_j = v_next, j_next
        if abs(j_next) == 0.0:
            break
        v_prev, j_prev = v_cur, j_cur
        v_cur, j_cur, pot_cur = v_next, j_next, pot_next
    if abs(best_j) >= min(abs(j_a), abs(j_b)):
        return float("nan"), False
    slope = abs((j_cur - j_prev) / (v_cur - v_prev)) if v_cur != v_prev else 0.0
    return float(best_v), bool(slope > 1e-30)
