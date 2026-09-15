"""The single, composable entry point: ``simulate``.

``simulate(device, protocol, *, solver, optics, adjoint, progress)`` returns a
:class:`~driftjax.solution.Solution`.  With the default ``ImplicitAdjoint`` it is
differentiable via :func:`jax.grad` using the implicit-function-theorem
adjoint (no manual gradient function required), and is ``jit``/``vmap``-safe.

Progress reporting is opt-in via ``progress=`` so *any* calculation shows its
progress when run:

* ``progress=None`` (default) — silent.
* ``progress=True`` / ``"live"`` — a live progress bar with per-bias solver
  info on a terminal (equilibrium iterations, V / iters / residual per step),
  log-friendly lines when piped, and a framed summary box (Jsc / Voc / FF /
  eta / MPP) printed from the result at the end.
* ``progress=<object>`` — your own reporter with ``update(i, **info)`` (per
  bias step) and/or ``iter(it, err, resid, phase)`` (per Newton step), or a
  plain callable ``fn(i, info_dict)``.  Reporting never affects numerics.
"""

from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import custom_vjp

from driftjax.adjoint.api import DirectAdjoint, ImplicitAdjoint
from driftjax.fields import DeviceDesign, pot2vec, vec2pot
from driftjax.numerics.residual import F_jacobian, comp_F
from driftjax.optics.api import BeerLambert
from driftjax.problems import Equilibrium, Sweep
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.science.spectrum import spectrum
from driftjax.simulator import _mpp, init_cell
from driftjax.solution import Solution
from driftjax.solvers.api import Newton
from driftjax.solvers.continuation import equilibrium_guess, find_voc, total_current
from driftjax.solvers.continuation import sweep as _continuation_sweep
from driftjax.solvers.newton import _is_tracer, solve_eq, solve_newton
from driftjax.units import thermal_scales

# ---------------------------------------------------------------------------
# Progress reporting (opt-in; never affects numerics)
# ---------------------------------------------------------------------------


def _resolve_progress(progress, protocol, design=None, optics=None):
    """Normalise the ``progress`` argument into a reporter for ``sweep``.

    ``True``/``"live"`` builds :class:`driftjax.console.SweepProgress`: a live
    bar with per-bias solver info on a terminal, log-friendly per-step lines
    when piped, and a framed summary box (Jsc / Voc / FF / eta / MPP) printed
    from the result at ``close(res)``.  Visible both on a real terminal *and*
    when piped/subprocessed — inspectable everywhere.
    """
    if progress is None or progress is False:
        return None
    if progress is True or progress == "live":
        from driftjax.console import SweepProgress

        n = getattr(protocol, "n_steps", 41) if protocol is not None else 41
        label = "equilibrium" if isinstance(protocol, Equilibrium) else "IV sweep"
        return SweepProgress(
            n,
            vmax_v=getattr(protocol, "vmax", None),
            label=label,
            design=design,
            alpha_mode=getattr(optics, "alpha_mode", None),
        )
    return progress


def _adjoint_solve(cell, bound, pot, g_x):
    """Solve ``J^T lambda = g_x`` for the IFT adjoint state.

    Pivoted dense LU only. (An O(N) banded-transpose attempt with residual
    gate used to live here; removed in v0.1.12 because the gate cannot
    certify accuracy under ill-conditioning — kappa ~ 3e14 amplifies the
    ~1e-12 banded-vs-true discrepancy into ~30% errors at r ~ 1e-14.
    See CHANGELOG.)

    ``F_jacobian`` is the exact state Jacobian, so ``jnp.linalg.solve``
    gives the exact adjoint.  ``jnp.linalg.solve`` is a pure-JAX primitive,
    so unlike the original scipy ``spsolve`` path it is jit- and
    vmap-compatible (the old path raised ``TracerBoolConversionError``
    under jit and was non-batchable).

    Opt-in structured path: with ``DRIFTJAX_BANDED_ADJOINT=1``, the solve
    is first attempted via row-equilibrated pivoted banded transpose
    (:func:`adjoint_banded_solve`); a residual gate routes failures to
    dense LU. Default (unset) is dense LU.

    Numerical note: for some multi-layer stacks the state Jacobian becomes
    *numerically singular* near open circuit (condition number ~1e14,
    smallest singular value at round-off); there the forward Newton solve
    still converges (its residual lies in the well-conditioned range) but
    the adjoint right-hand side excites the null space, so the adjoint is
    ill-posed.  Gradient-based optimisation of such stacks should use a
    finite-difference Jacobian instead -- which is exactly what the
    3-layer optimisation example does (FD fallback).
    """
    from driftjax.numerics.mixed_precision import adjoint_dense_solve as _ads3

    J = F_jacobian(cell, bound, pot)
    if _banded_adjoint_enabled():
        from driftjax.numerics.banded_solve import adjoint_banded_solve as _abs3

        lam_b, fb = _abs3(J, g_x)
        return jax.lax.cond(fb, lambda _: _ads3(J.T, g_x), lambda _: lam_b, None)
    return _ads3(J.T, g_x)


def _banded_adjoint_enabled() -> bool:
    """Whether the adjoint may attempt the equilibrated banded transpose.

    Opt-in via ``DRIFTJAX_BANDED_ADJOINT=1`` (default off); evaluated at
    trace time from static context, so only the taken path compiles.
    """
    import os

    return os.environ.get("DRIFTJAX_BANDED_ADJOINT", "0") == "1"


def _audit_sweep_solution(sol):
    """Concrete-path post-hoc residual audit (H1 failure semantics).

    Re-measures max|F| per bias directly: K cheap residual evals, no
    solves. Returns (converged, max_resid, per_bias_residuals).

    An absolute threshold of 1e-6 is used.  For severely ill-conditioned
    systems (cond(J) >> 1/eps) the solver may do its best yet still
    have a large residual — in that case converged is False but the
    efficiency value is still returned (not nan-gated) so the user can
    inspect the result.  The warning message explains the situation.

    Must only be called with concrete (non-traced) solutions; under
    jit/grad the audit cannot concretize and the fields stay at
    unverified defaults.
    """
    max_resid = 0.0
    per_bias = []
    try:
        e_scale = float(thermal_scales(float(sol.cell.T))["energy"])
        volts = sol.voltages
        for pot_b, vb in zip(sol.potentials, volts, strict=False):
            bound = boundary_bias(sol.cell, float(vb) / e_scale)
            r_b = float(jnp.max(jnp.abs(comp_F(sol.cell, bound, pot_b))))
            per_bias.append(r_b)
            if not jnp.isfinite(r_b):
                return False, float("inf"), per_bias
            max_resid = max(max_resid, r_b)
    except Exception:
        return False, float("inf"), per_bias
    if max_resid > 1e-6:
        return False, max_resid, per_bias
    return True, max_resid, per_bias


def _iter_cb_for(progress, phase):
    """Wrap ``progress.iter`` into a Newton ``iter_cb`` (per Newton step)."""
    if progress is None or not hasattr(progress, "iter"):
        return None

    def cb(it, err, resid):
        try:
            if _is_tracer(resid):
                return None
            progress.iter(it, err=err, resid=resid, phase=phase)
        except Exception:
            pass
        return None

    return cb


# ---------------------------------------------------------------------------
# Forward (shared, traceable).  Differentiates w.r.t. ``design``.
# ---------------------------------------------------------------------------


def _forward(
    design, protocol, solver, optics, loop=None, progress=None, ls=None, statistics=None, init=None
):
    allow_trace = _is_tracer(design)
    if allow_trace:
        progress = None  # reporting is a concrete-path feature; never during grad/vmap
    ls = (
        ls if ls is not None else spectrum(normalize=False)
    )  # raw AM1.5G (matches canonical examples)
    statistics = statistics or "boltzmann"
    # Effective strategy flags. The ``Newton`` fields are authoritative;
    # the legacy ``Sweep.refinement`` / ``Sweep.fused`` flags act as OR-fallbacks
    # so existing protocols keep working. ``BandedLapack(batched=True)`` (or the
    # opt-in ``Sweep.batched`` flag) routes the sweep through the vmapped
    # batched banded-solver path where admissible.
    refinement = bool(solver.refinement or getattr(protocol, "refinement", False))
    fused = bool(solver.fused or getattr(protocol, "fused", False))
    dense = bool(getattr(solver, "dense", False))
    globalization = getattr(solver, "globalization", "auto")
    batched = bool(
        getattr(getattr(solver, "linear_solver", None), "batched", False)
        or getattr(protocol, "batched", False)
    )
    newton_kw = dict(
        tol=solver.rtol,
        # No f_tol residual gate here: on stiff heterostructures the step-norm
        # criterion alone cannot certify convergence (residual reaches machine
        # zero while |dx| stays above tol), which used to exhaust max_steps and
        # propagate NaN. That case is now handled by best-iterate tracking in
        # the Newton loops (the lowest-residual iterate is returned on
        # exhaustion/NaN), so normal solves keep pristine step-norm semantics
        # and an early gate cannot inject noise-selected states.
        max_steps=solver.max_steps,
        dense=dense,
        refinement=refinement,
        fused=fused,
        globalization=globalization,
    )
    cell = init_cell(
        design, ls, alpha_mode=optics.alpha_mode, statistics=statistics, optics=optics, fused=fused
    )
    sc = thermal_scales(design.T)
    _eq_iter = (
        _iter_cb_for(progress, "equilibrium")
        if (progress is not None and hasattr(progress, "iter") and not allow_trace)
        else None
    )

    if isinstance(protocol, Equilibrium):
        if init is not None and getattr(init, "eq_pot", None) is not None:
            pot_eq, _ = solve_newton(
                cell,
                boundary_eq(cell),
                init.eq_pot,
                allow_trace=allow_trace,
                loop=loop,
                iter_cb=_eq_iter,
                **newton_kw,
            )
            pot_eq0 = pot_eq
        else:
            pot_eq0 = solve_eq(
                cell,
                boundary_eq(cell),
                equilibrium_guess(cell).phi,
                allow_trace=allow_trace,
                loop=loop,
                iter_cb=_eq_iter,
            )
            pot_eq, _ = solve_newton(
                cell,
                boundary_eq(cell),
                pot_eq0,
                allow_trace=allow_trace,
                loop=loop,
                iter_cb=_eq_iter,
                **newton_kw,
            )
        return (
            Solution(
                voltages=jnp.array([]),
                current=jnp.array([]),
                potentials=[pot_eq],
                cell=cell,
                eff=0.0,
                voc=0.0,
                ff=0.0,
                jsc=0.0,
                pmax=0.0,
                eq_pot=pot_eq,
                protocol="equilibrium",
                P_in=ls.P_in,
            ),
            (pot_eq,),
        )

    # Equilibrium is solved ONCE here.  It seeds both the v = 0 sweep guess
    # (via the injected warm-start list below) and ``Solution.eq_pot``; the
    # previous flow solved it a second time AFTER the sweep — one full
    # redundant Poisson + coupled-Newton solve per ``simulate`` call.
    user_init_pots = None
    if init is not None:
        if hasattr(init, "potentials"):
            user_init_pots = list(init.potentials)
        elif isinstance(init, (list, tuple)):
            user_init_pots = list(init)
    if init is not None and getattr(init, "eq_pot", None) is not None:
        pot_eq, _ = solve_newton(
            cell,
            boundary_eq(cell),
            init.eq_pot,
            allow_trace=allow_trace,
            iter_cb=_eq_iter,
            **newton_kw,
        )
        pot_eq0 = pot_eq
    else:
        pot_eq0 = solve_eq(
            cell,
            boundary_eq(cell),
            equilibrium_guess(cell).phi,
            allow_trace=allow_trace,
            loop=loop,
            iter_cb=_eq_iter,
        )
        pot_eq, _ = solve_newton(
            cell,
            boundary_eq(cell),
            pot_eq0,
            allow_trace=allow_trace,
            iter_cb=_eq_iter,
            **newton_kw,
        )

    # Inject the converged equilibrium as the v = 0 guess (the remaining biases
    # keep the sweep's own QFL / linear-extrapolation schedule).  Skipped when
    # the caller supplied per-bias potentials or requested the batched path
    # (which warm-starts internally).
    sweep_init = init
    if user_init_pots is None and not batched:
        sweep_init = [pot_eq] + [None] * max(protocol.n_steps - 1, 0)

    vmax_dim = protocol.vmax / sc["energy"]
    voltages_dim, currents_dim, pots, _sweep_fallbacks = _continuation_sweep(
        cell,
        vmax_dim,
        protocol.n_steps,
        tol=1e-10,
        refinement=refinement,
        fused=fused,
        batched=batched,
        v_scale=sc["energy"],
        allow_trace=allow_trace,
        loop=loop,
        progress=progress,
        init=sweep_init,
    )
    # R2 provenance: per-bias lstsq flags ride alongside the sweep states.
    # The serial concrete path reports True/False; fused-scan, batched and
    # traced paths report None entries (unverified, not "clean").
    try:
        _sweep_fallbacks = list(_sweep_fallbacks)
    except Exception:
        _sweep_fallbacks = [None] * len(pots)
    voc_dim = find_voc(voltages_dim, currents_dim)
    jsc_dim = jnp.abs(currents_dim[0])
    pmax_dim, _ = _mpp(voltages_dim, currents_dim)
    # NOTE: implicit-Voc secant refinement runs once at the public-simulate
    # choke point (all paths: serial, fused-scan, batched), not here, so
    # every path reports the same refined root and bracket status.
    voc_bracketed: bool = False
    ff = jnp.where(jnp.isfinite(voc_dim), pmax_dim / (voc_dim * jsc_dim + 1e-30), jnp.nan)
    v_volts = voltages_dim * sc["energy"]
    j_phys = currents_dim * sc["current"]
    pmax_phys_wm2 = pmax_dim * sc["energy"] * sc["current"] * 1e4
    eff = pmax_phys_wm2 / jnp.sum(ls.P_in)
    # Keep the scalars as tracers under differentiation/vmap (valid pytree
    # leaves for the custom_vjp primal) but expose Python floats on the
    # concrete path so the public API stays unchanged.
    f = (lambda x: x) if allow_trace else float
    return Solution(
        voltages=v_volts,
        current=j_phys,
        potentials=pots,
        cell=cell,
        eff=f(eff),
        voc=f(voc_dim * sc["energy"]),
        ff=f(ff),
        jsc=f(jsc_dim * sc["current"]),
        pmax=f(pmax_dim),
        eq_pot=pot_eq,
        protocol="sweep",
        P_in=ls.P_in,
        # R2 provenance: per-bias lstsq flags (True/False concrete serial;
        # None entries when unverified: traced, fused-scan, batched paths).
        fallback_used=list(_sweep_fallbacks),
        voc_bracketed=bool(voc_bracketed),
    ), (voltages_dim, currents_dim, pots)


# ---------------------------------------------------------------------------
# Sweep: differentiable core (ImplicitAdjoint via custom_vjp over ``design``)
# ---------------------------------------------------------------------------


@partial(custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7, 8))
def _simulate_sweep(
    design, solver, optics, protocol, progress, ls, statistics, init=None, fused=False
):
    if (
        fused
        and progress is None
        and init is None
        and not _is_tracer(design)
        and not getattr(protocol, "batched", False)
    ):
        try:
            pot_eq, cell, ls2, sc, vd, cd, pa, fb_arr = _forward_fused_scan(
                design, solver, optics, protocol, ls, statistics
            )
            # The whole-sweep scan compiles to one XLA program whose Newton
            # trajectory can diverge (non-finite currents) where the serial
            # sweep converges (rounding-level path differences amplified at
            # ill-conditioned high-bias points). Validate before trusting it.
            fast_ok = bool(jnp.isfinite(vd).all()) and bool(jnp.isfinite(cd).all())
        except Exception:
            fast_ok = False
        if fast_ok:
            voc_dim = find_voc(vd, cd)
            jsc_dim = jnp.abs(cd[0])
            pmax_dim, _ = _mpp(vd, cd)
            ff = jnp.where(jnp.isfinite(voc_dim), pmax_dim / (voc_dim * jsc_dim + 1e-30), jnp.nan)
            v_volts = vd * sc["energy"]
            j_phys = cd * sc["current"]
            pmax_phys_wm2 = pmax_dim * sc["energy"] * sc["current"] * 1e4
            eff = pmax_phys_wm2 / jnp.sum(ls2.P_in)
            pots_list = [
                jax.tree_util.tree_map(lambda a, i=i: a[i], pa) for i in range(pa.phi.shape[0])
            ]
            return Solution(
                voltages=v_volts,
                current=j_phys,
                potentials=pots_list,
                cell=cell,
                eff=float(eff),
                voc=float(voc_dim * sc["energy"]),
                ff=float(ff),
                jsc=float(jsc_dim * sc["current"]),
                pmax=float(pmax_dim),
                eq_pot=pot_eq,
                protocol="sweep",
                P_in=ls2.P_in,
                # R2: fused-scan per-bias lstsq flags (concrete bools).
                fallback_used=[bool(x) for x in list(fb_arr)],
            )
        warnings.warn(
            "Fused forward sweep produced non-finite currents (Newton divergence "
            "in the compiled scan); falling back to the serial sweep.",
            UserWarning,
            stacklevel=2,
        )
    sol, _ = _forward(
        design, protocol, solver, optics, progress=progress, ls=ls, statistics=statistics, init=init
    )
    return sol


def _static_key(optics, ls):
    """Hashable fingerprint of everything the fused scan closes over.

    Array payloads are hashed by value (shapes + bytes), scalars by repr,
    so equal configurations share one compilation while any difference
    (spectrum, optics type/mode, solver/protocol scalars) compiles
    separately. Never keys on object identity (id-reuse would be unsound).
    """
    import numpy as _np

    parts = [type(optics).__name__, type(ls).__name__]
    for obj in (optics, ls):
        try:
            fields = vars(obj)
        except TypeError:
            fields = {}
        for k in sorted(fields):
            v = fields[k]
            if isinstance(v, _np.ndarray):
                parts += [k, str(v.shape), str(v.dtype), v.tobytes()]
            else:
                try:
                    import jax as _jax

                    if isinstance(v, _jax.Array):
                        vv = _np.asarray(v)
                        parts += [k, str(vv.shape), str(vv.dtype), vv.tobytes()]
                    else:
                        parts.append((k, repr(v)))
                except Exception:
                    parts.append((k, repr(v)))
    return tuple(parts)


_SCAN_JIT_CACHE: dict = {}
_SCAN_JIT_CACHE_MAX = 64


def _forward_fused_scan(design, solver, optics, protocol, ls, statistics):
    """One-XLA forward: init_cell + eq + bias scan via lax.scan (no Python loops).

    The scan closure is cached per static configuration (solver/protocol
    scalars + optics/spectrum fingerprint): recreating ``jax.jit`` closures
    per call defeats JAX's compilation cache and re-traces + recompiles the
    whole sweep on every evaluation (fatal for SLSQP/FD/Sobol loops).
    """
    from driftjax.science.spectrum import spectrum as _spec
    from driftjax.solvers.continuation import equilibrium_guess as _eg
    from driftjax.solvers.continuation import total_current as _tc
    from driftjax.solvers.newton import solve_eq as _solve_eq_fused
    from driftjax.solvers.newton import solve_newton as _solve_newton_fused

    if ls is None:
        ls = _spec(normalize=False)
    statistics = statistics or "boltzmann"
    fused = bool(solver.fused or getattr(protocol, "fused", False))
    rtol = float(solver.rtol)
    max_steps = int(solver.max_steps)
    alpha_mode = optics.alpha_mode
    n_steps = int(protocol.n_steps)
    vmax = float(protocol.vmax)
    key = None
    try:
        key = (rtol, max_steps, n_steps, vmax, statistics, fused, _static_key(optics, ls))
    except Exception:
        key = None
    fn = _SCAN_JIT_CACHE.get(key) if key is not None else None
    if fn is None:
        if len(_SCAN_JIT_CACHE) >= _SCAN_JIT_CACHE_MAX:
            _SCAN_JIT_CACHE.clear()

        @jax.jit
        def fn(d):
            cell = init_cell(
                d, ls, alpha_mode=alpha_mode, statistics=statistics, optics=optics, fused=fused
            )
            sc = thermal_scales(d.T)
            pot_eq0 = _solve_eq_fused(
                cell, boundary_eq(cell), _eg(cell).phi, allow_trace=True, loop="while"
            )
            pot_eq, _ = _solve_newton_fused(
                cell, boundary_eq(cell), pot_eq0, allow_trace=True, loop="while", fused=fused
            )
            vmax_dim = vmax / sc["energy"]
            vs = jnp.linspace(0.0, vmax_dim, n_steps)

            def scan_body(pot_prev, v):
                bound = boundary_bias(cell, v)
                # f_tol=rtol: residual gate (see _forward newton_kw note).
                pot_new, st_ = _solve_newton_fused(
                    cell,
                    bound,
                    pot_prev,
                    allow_trace=True,
                    loop="while",
                    fused=fused,
                    tol=rtol,
                    max_steps=max_steps,
                )
                cur = _tc(cell, pot_new)
                # R2: per-bias lstsq flag rides the scan outputs (tracer
                # bool; concretized by the eager jit call below, never
                # bool()-converted inside the trace).
                return pot_new, (v, cur, pot_new, st_.get("lstsq", False))

            _, (voltages_dim, currents_dim, pots_arr, lstsq_arr) = jax.lax.scan(
                scan_body, pot_eq, vs
            )
            return pot_eq, cell, sc, voltages_dim, currents_dim, pots_arr, lstsq_arr

        if key is not None:
            _SCAN_JIT_CACHE[key] = fn

    pot_eq, cell, sc, voltages_dim, currents_dim, pots_arr, lstsq_arr = fn(design)
    return pot_eq, cell, ls, sc, voltages_dim, currents_dim, pots_arr, lstsq_arr


def _sweep_fwd(design, solver, optics, protocol, progress, ls, statistics, init=None, fused=False):
    # Fused-everything fast path: only for concrete forward (not grad) — keeps grad at 2.1s vs 41s
    if (
        fused
        and progress is None
        and init is None
        and not getattr(protocol, "batched", False)
        and not _is_tracer(design)
    ):
        try:
            pot_eq, cell, ls2, sc, vd, cd, pa, fb_arr = _forward_fused_scan(
                design, solver, optics, protocol, ls, statistics
            )
            # Same divergence guard as _simulate_sweep: validate the compiled
            # scan before trusting it; fall back to the serial sweep on NaN.
            fast_ok = bool(jnp.isfinite(vd).all()) and bool(jnp.isfinite(cd).all())
        except Exception:
            fast_ok = False
        if fast_ok:
            # pa is batched Potentials (B,n); build per-bias list for residuals via pot2vec
            pot_arr = jax.vmap(pot2vec)(pa)
            voc_dim = find_voc(vd, cd)
            jsc_dim = jnp.abs(cd[0])
            pmax_dim, _ = _mpp(vd, cd)
            # AUDIT: NaN-safe ff (matches the serial _forward guard) — beyond
            # the sweep range voc_dim is NaN and must propagate as NaN, not
            # as NaN/eps noise.
            ff = jnp.where(jnp.isfinite(voc_dim), pmax_dim / (voc_dim * jsc_dim + 1e-30), jnp.nan)
            v_volts = vd * sc["energy"]
            j_phys = cd * sc["current"]
            pmax_phys_wm2 = pmax_dim * sc["energy"] * sc["current"] * 1e4
            eff = pmax_phys_wm2 / jnp.sum(ls2.P_in)
            # potentials as list for Solution compat (keep tracers)
            pots_list = [
                jax.tree_util.tree_map(lambda a, i=i: a[i], pa) for i in range(pa.phi.shape[0])
            ]
            sol = Solution(
                voltages=v_volts,
                current=j_phys,
                potentials=pots_list,
                cell=cell,
                eff=eff,
                voc=voc_dim * sc["energy"],
                ff=ff,
                jsc=jsc_dim * sc["current"],
                pmax=pmax_dim,
                eq_pot=pot_eq,
                protocol="sweep",
                P_in=ls2.P_in,
                # R2: fused-scan per-bias lstsq flags (concrete bools from the
                # eager jit call; True = dgbsv failed and lstsq stepped).
                fallback_used=[bool(x) for x in list(fb_arr)],
            )
            residuals = (design, cell, pot_arr, vd, cd, ls2.P_in, fused)
            return sol, residuals
        warnings.warn(
            "Fused forward sweep produced non-finite currents (Newton divergence "
            "in the compiled scan); falling back to the serial sweep.",
            UserWarning,
            stacklevel=2,
        )
    sol, (voltages_dim, currents_dim, pots) = _forward(
        design, protocol, solver, optics, progress=progress, ls=ls, statistics=statistics, init=init
    )
    pot_arr = jnp.stack([pot2vec(p) for p in pots])
    residuals = (design, sol.cell, pot_arr, voltages_dim, currents_dim, sol.P_in, fused)
    return sol, residuals


def _warn_uncertified_primal_if(max_resid):
    """Runtime warning for gradients on uncertified primal states (P0-5).

    Called via jax.debug.callback from the backward pass with the concrete
    max|F| recomputed from the forward residuals. Silent when the primal is
    certified (max|F| <= 1e-6, same threshold as the concrete-path audit);
    warns loudly otherwise, so a normal-looking gradient is never returned
    silently on an unconverged state.
    """
    import warnings

    try:
        mr = float(max_resid)
    except Exception:
        return
    if mr > 1e-6 or mr != mr:  # NaN-safe comparison
        warnings.warn(
            f"DriftJax backward pass on uncertified primal state (max|F|={mr:.3e} "
            f"> 1e-6 audit threshold); returned gradient is UNRELIABLE.",
            UserWarning,
            stacklevel=2,
        )


def _sweep_bwd(solver, optics, protocol, progress, ls, statistics, init, fused, residuals, g_sol):
    design, cell, pot_arr, voltages_dim, currents_dim, P_in, fused_val = residuals
    alpha_mode = optics.alpha_mode
    # P0-5: the custom VJP must not differentiate silently through an
    # uncertified primal. Recompute max|F| per bias from the forward
    # residuals (K cheap residual evals, no solves; trace-safe pure JAX)
    # and warn at runtime when it exceeds the audit threshold. The IFT
    # requires a converged root; without it the gradient is UNRELIABLE.
    from driftjax.numerics.residual import comp_F as _comp_F_bwd

    _F_all = jax.vmap(lambda pv, vb: _comp_F_bwd(cell, boundary_bias(cell, vb), vec2pot(pv)))(
        pot_arr, voltages_dim
    )
    jax.debug.callback(_warn_uncertified_primal_if, jnp.max(jnp.abs(_F_all)))

    # Fold ALL Solution cotangents (eff/voc/ff/jsc/current) back onto the
    # primitive sweep outputs (voltages_dim, currents_dim) AND capture the
    # explicit T-dependence of the physical-unit scaling ``sc`` (which depends
    # on ``design.T``).  A plain ``g_cdim = g_sol.current * sc["current"]`` fold
    # would miss the ``currents_dim * d sc/dT`` term; wrapping the full physical
    # map ``(design, vdim, cdim) -> (eff, voc, ff, jsc, current)`` in one VJP
    # yields both the folded cotangent (for the IFT) and the explicit ``sc``
    # design-cotangent (added directly to g_design below).
    def postprocess_full(design_d, vdim, cdim):
        scx = thermal_scales(design_d.T)
        voc_raw = find_voc(vdim, cdim)
        # NaN-safe wrapper: when the IV curve never crosses zero (Voc beyond
        # the swept range) ``find_voc`` returns NaN. Wrapping it in
        # ``where(isfinite, x, vmax)`` keeps the VJP finite (zero on the NaN
        # branch) so differentiating eff/voc/ff is well-defined even for
        # devices whose Voc lies outside the sweep (e.g. some 3-layer stacks).
        # The forward-facing ``Solution.voc`` still reports the true NaN.
        voc_d = jnp.where(jnp.isfinite(voc_raw), voc_raw, vdim[-1])
        jsc_d = jnp.abs(cdim[0])
        pmax_d, _ = _mpp(vdim, cdim)
        ff_d = pmax_d / (voc_d * jsc_d + 1e-30)
        eff_d = pmax_d * scx["energy"] * scx["current"] * 1e4 / jnp.sum(P_in)
        return (eff_d, voc_d * scx["energy"], ff_d, jsc_d * scx["current"], cdim * scx["current"])

    ct_eff = g_sol.eff if g_sol.eff is not None else 0.0
    ct_voc = g_sol.voc if g_sol.voc is not None else 0.0
    ct_ff = g_sol.ff if g_sol.ff is not None else 0.0
    ct_jsc = g_sol.jsc if g_sol.jsc is not None else 0.0
    ct_cur = 0.0 if g_sol.current is None else g_sol.current
    _, vjp_post = jax.vjp(postprocess_full, design, voltages_dim, currents_dim)
    g_design_from_sc, _g_vdim, g_cdim = vjp_post((ct_eff, ct_voc, ct_ff, ct_jsc, ct_cur))

    # ---- vectorised adjoint over biases (single trace, batched) ----
    # The old implementation re-traced ``init_cell``/optics/newton once per bias
    # (a Python ``for`` loop), which made ``jax.grad(simulate)`` compile for
    # minutes.  ``jax.vmap`` over the bias axis turns the whole sweep-adjoint
    # into ONE traced graph that is batched across biases; the per-bias adjoint
    # itself is the exact, stable dense solve ``J^T lam = g_x`` (the
    # block-tridiagonal transpose sweep is numerically
    # unstable on real DD Jacobians, see commit 56ee78d, so dense is kept).
    #
    # Bias-independence hoist: the design -> cell map (optics + generation
    # included) does NOT depend on the bias, so it is traced ONCE here
    # (``cell_of_d``) and pulled back ONCE at the end (``vjp_cell``).  The
    # vmapped kernel only forms cell-space cotangents — the direct current
    # channel minus the implicit IFT channel, scaled by the folded output
    # cotangent.  (Previously ``init_cell`` was re-traced twice per bias inside
    # the vmap, which dominated backward trace size for spectral optics — TMM
    # worst case.)

    cell_of_d, vjp_cell = jax.vjp(
        lambda d: init_cell(
            d, ls, alpha_mode=alpha_mode, statistics=statistics, optics=optics, fused=fused
        ),
        design,
    )

    _analytic_ok = fused and getattr(cell_of_d, "statistics", "boltzmann") == "boltzmann"

    # NOTE (measured 2026-09): a mid-sweep pilot (banded-vs-dense lam
    # agreement) was tried as a banded/dense selector and REJECTED:
    # agreement varies 18x across biases on the same device (p-i-n mid
    # 2e-4 vs worst 3.5e-3), and no threshold separates the zoo cleanly
    # (hetero-mid ~1e-4 vs p-i-n-mid 2e-4). A single pilot cannot certify
    # a sweep; per-bias pilots cost as much as dense. Dense stays default;
    # banded is explicit opt-in with FD validation. See CHANGELOG v0.1.12.

    @jax.checkpoint
    def per_bias(potv, vb, gcb):
        pot = vec2pot(potv)
        bound = boundary_bias(cell_of_d, vb)
        if _analytic_ok:
            from driftjax.numerics.analytic_adjoint import analytic_g_x as _agx
            from driftjax.numerics.analytic_adjoint import analytic_per_bias as _apb
            from driftjax.numerics.analytic_jacobian import banded_jacobian as _bj
            from driftjax.numerics.analytic_jacobian import dense_from_blocks as _dfb
            from driftjax.numerics.mixed_precision import adjoint_dense_solve as _ads

            g_x = _agx(cell, pot)
            Ab, Bb, Cb = _bj(cell, bound, pot)
            J = _dfb(Ab, Bb, Cb)
            if _banded_adjoint_enabled():
                # Blocks-direct: no dense extraction (Phase B). Falls back
                # to dense LU on the same gate as the dense-J API.
                from driftjax.numerics.banded_solve import (
                    adjoint_banded_solve_blocks as _absb,
                )

                lam_b, fb = _absb(Ab, Bb, Cb, g_x)
                lam = jax.lax.cond(fb, lambda _: _ads(J.T, g_x), lambda _: lam_b, None)
            else:
                lam = _ads(J.T, g_x)
            return _apb(cell_of_d, pot, vb, lam, gcb)
        # g_x = d current_b / d x  at the fixed forward cell
        g_x = jax.grad(lambda x: total_current(cell, vec2pot(x)))(potv)
        from driftjax.numerics.mixed_precision import adjoint_dense_solve as _ads2

        J = F_jacobian(cell, bound, pot)
        if _banded_adjoint_enabled():
            from driftjax.numerics.banded_solve import adjoint_banded_solve as _abs2

            lam_b, fb = _abs2(J, g_x)
            lam = jax.lax.cond(fb, lambda _: _ads2(J.T, g_x), lambda _: lam_b, None)
        else:
            lam = _ads2(J.T, g_x)
        # direct channel:  d current_b / d(cell leaves), pot fixed
        u_cell = jax.grad(lambda c: total_current(c, pot))(cell_of_d)
        # implicit channel:  (dF_b/d(cell leaves))^T lam
        _, vjp_Fc = jax.vjp(lambda c: comp_F(c, boundary_bias(c, vb), pot), cell_of_d)
        lamF_c = vjp_Fc(lam)[0]
        return jax.tree.map(lambda a, b: (a - b) * gcb, u_cell, lamF_c)

    t_cell = jax.vmap(per_bias, in_axes=(0, 0, 0))(pot_arr, voltages_dim, g_cdim)
    t_cell = jax.tree.map(lambda leaf: jnp.sum(leaf, axis=0), t_cell)
    g_design = vjp_cell(t_cell)[0]
    g_design = jax.tree.map(lambda a, b: a + b, g_design, g_design_from_sc)
    return (g_design,)


_simulate_sweep.defvjp(_sweep_fwd, _sweep_bwd)


# ---------------------------------------------------------------------------
# Equilibrium: differentiable core (ImplicitAdjoint via custom_vjp)
# ---------------------------------------------------------------------------


@partial(custom_vjp, nondiff_argnums=(1, 2, 3, 4, 5, 6, 7))
def _simulate_eq(design, solver, optics, protocol, progress, ls, statistics, init=None):
    sol, _ = _forward(
        design, protocol, solver, optics, progress=progress, ls=ls, statistics=statistics, init=init
    )
    return sol


def _eq_fwd(design, solver, optics, protocol, progress, ls, statistics, init=None):
    sol, _ = _forward(
        design, protocol, solver, optics, progress=progress, ls=ls, statistics=statistics, init=init
    )
    residuals = (design, sol.cell, sol.potentials[0], sol.P_in)
    return sol, residuals


def _eq_bwd(solver, optics, protocol, progress, ls, statistics, init, residuals, g_sol):
    design, cell, pot_eq, P_in = residuals
    alpha_mode = optics.alpha_mode
    _fused = bool(getattr(solver, "fused", False) or getattr(protocol, "fused", False))

    def _cell_of_d(d):
        return init_cell(
            d, ls, alpha_mode=alpha_mode, statistics=statistics, optics=optics, fused=_fused
        )

    g_x = jnp.zeros_like(pot2vec(pot_eq))
    if g_sol.potentials is not None and len(g_sol.potentials) > 0:
        gp = g_sol.potentials[0]
        if gp is not None:
            g_x = g_x + pot2vec(gp)
    if g_sol.eq_pot is not None:
        g_x = g_x + pot2vec(g_sol.eq_pot)
    lam = _adjoint_solve(cell, boundary_eq(cell), pot_eq, g_x)

    def F_eq_d(d, pot_eq=pot_eq):
        # AUDIT: must mirror the forward cell exactly — the old call dropped
        # `statistics` (FD/Blakemore equilibrium gradients silently used the
        # Boltzmann VJP cell) and `fused`.
        c = _cell_of_d(d)
        return comp_F(c, boundary_eq(c), pot_eq)

    _, vjp_F = jax.vjp(F_eq_d, design)
    lamF = vjp_F(lam)[0]
    # AUDIT: the direct channel is NOT zero in general. The old code assumed
    # the equilibrium "loss" is the potential itself (dL/dd = -lam^T dF/dd),
    # but any objective with explicit cell dependence — e.g. sum(n(cell,pot))
    # — carries dL/dcell · dcell/dd, which was silently dropped (measured:
    # d(sum n)/dChi[0] off by ~1e5x on a degenerate device). Pull the
    # Solution's cell cotangent back through init_cell (identity map).
    g_cell = getattr(g_sol, "cell", None)
    if g_cell is not None:
        _, vjp_cell = jax.vjp(_cell_of_d, design)
        (g_direct,) = vjp_cell(g_cell)
        g_design = jax.tree.map(lambda a, b: a - b, g_direct, lamF)
    else:
        g_design = jax.tree.map(lambda a: -a, lamF)
    return (g_design,)


_simulate_eq.defvjp(_eq_fwd, _eq_bwd)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def simulate(
    device,
    protocol=None,
    *,
    solver=None,
    optics=None,
    adjoint=None,
    progress=None,
    statistics=None,
    T=None,
    ls=None,
    init=None,
):
    """Simulate a ``Device`` and return a :class:`~driftjax.solution.Solution`.

    Parameters
    ----------
    device : Device
    protocol : Equilibrium | Sweep
        What to solve. Defaults to ``Sweep()``.
    solver : Newton
        Nonlinear solver configuration. Defaults to ``Newton()``.
    optics : BeerLambert | TMM
        Optical generation model. Defaults to ``BeerLambert(device.alpha_mode)``.
    adjoint : ImplicitAdjoint | DirectAdjoint
        Differentiation mode. Defaults to ``ImplicitAdjoint`` (so
        ``jax.grad(simulate)`` is correct and fast).
        ``DirectAdjoint`` is an explicit alias for ``ImplicitAdjoint``
        (a ``UserWarning`` is emitted when it is selected); unrolled
        Newton differentiation is not supported.
    progress : None | True | "live" | object | callable
        Opt-in progress reporting (see module docstring).  When ``True``/``"live"``,
        every calculation prints per-bias / per-Newton-step progress so long
        runs are inspectable.
    statistics : str
        Carrier-statistics model for the cell ("boltzmann" | "fd" |
        "blakemore", where "blakemore" is a legacy A-B-only approximation
        with known ~60% degenerate error, never a production choice).
    T : float
        Override the device temperature (K); ``None`` uses ``device.T``.
    ls : LightSource
        Optional incident-light source; ``None`` uses the built-in AM1.5G spectrum.

    Examples
    --------
    >>> sol = simulate(dev)                          # Sweep + Newton + Beer-Lambert + IFT
    >>> float(sol.efficiency)                        # power conversion efficiency
    >>> sol = simulate(dev, progress=True)           # show sweep progress live
    >>> grad = jax.grad(lambda d: simulate(d).efficiency)(dev)   # transparent IFT gradient
    """
    protocol = protocol if protocol is not None else Sweep()
    if not isinstance(protocol, (Equilibrium, Sweep)):
        raise TypeError("protocol must be Equilibrium() or Sweep()")
    solver = solver if solver is not None else Newton()
    optics = (
        optics if optics is not None else BeerLambert(getattr(device, "alpha_mode", "beer-lambert"))
    )
    adjoint = adjoint if adjoint is not None else ImplicitAdjoint()
    if isinstance(adjoint, DirectAdjoint):
        # ``DirectAdjoint`` is an explicit ALIAS for the stable IFT path, NOT
        # an unrolled Newton differentiation mode: differentiating through the
        # Newton iterations directly is numerically unstable on this
        # ill-conditioned DDP Jacobian (kappa >> 1), while the IFT adjoint is
        # mathematically identical for a converged root.  Warn loudly so the
        # routing is never silent (see driftjax.adjoint.api.DirectAdjoint).
        warnings.warn(
            "DirectAdjoint is an alias for ImplicitAdjoint (IFT); unrolled "
            "Newton differentiation is not supported and was not used.",
            UserWarning,
            stacklevel=2,
        )
        adjoint = ImplicitAdjoint()
    design = device if isinstance(device, DeviceDesign) else device.design()
    progress = _resolve_progress(progress, protocol, design=design, optics=optics)
    fused = bool(solver.fused or getattr(protocol, "fused", False))
    statistics = statistics or "boltzmann"
    if str(statistics).strip().lower() == "blakemore":
        # AUDIT R3: the Blakemore branch is a legacy provenance approximation
        # (60%+ error at small η, discontinuous at η=0; envelope pinned in
        # tests/unit/test_statistics.py). It is kept for A-B checks only —
        # use statistics="exact" (or "fd") for real degenerate physics.
        warnings.warn(
            'statistics="blakemore" is a legacy provenance approximation with '
            "~60% error at degeneracy eta=1 and a discontinuity at eta=0 "
            "(envelope pinned in tests/unit/test_statistics.py); it is kept "
            "for A-B checks only and is NOT a validated degenerate model. "
            'Use statistics="exact" for degenerate Newton solves.',
            UserWarning,
            stacklevel=2,
        )
    ls = ls if ls is not None else spectrum(normalize=False)
    if T is not None:
        design = design.with_temperature(float(T))
    # close() runs in a finally so a mid-sweep exception cannot leak the
    # reporter's file handle (DebugLog/SweepProgress); on success the
    # reporter receives the Solution, on failure it is closed bare (res=None
    # triggers the TypeError fallback, mirroring the original contract).
    res = None
    try:
        if isinstance(protocol, Equilibrium):
            res = _simulate_eq(
                design, solver, optics, protocol, progress, ls, statistics, init=init
            )
        else:
            res = _simulate_sweep(
                design, solver, optics, protocol, progress, ls, statistics, init=init, fused=fused
            )
        # H1: post-hoc residual audit at the single choke point covering all
        # sweep paths (serial, fused-scan, batched). Concrete path only;
        # under jit/grad/vmap the fields stay at unverified defaults.
        if not isinstance(protocol, Equilibrium) and not _is_tracer(res.voltages):
            _conv, _mr, _per_bias = _audit_sweep_solution(res)
            if not _conv:
                warnings.warn(
                    f"DriftJax sweep did not converge (max|F|={_mr:.3e}); "
                    f"eff/voc/ff are unreliable.",
                    UserWarning,
                    stacklevel=2,
                )
            import equinox as _eqx

            # R2: keep construction-time fallback flags when present (serial
            # concrete path); otherwise mark every bias unverified (None) so
            # "unknown" is never confused with "clean" (False).
            _fb = getattr(res, "fallback_used", []) or [None] * len(res.potentials)
            # Implicit-Voc refinement at the single choke point (all paths:
            # serial, fused-scan, batched). When the sweep brackets a sign
            # change, secant-refine J(Voc) = 0 on converged evaluations
            # instead of trusting the coarse interpolation; on failure (or
            # no bracket) keep the interpolation and report voc_bracketed.
            _voc_val = res.voc
            _voc_bracketed = bool(getattr(res, "voc_bracketed", False))
            try:
                from driftjax.solvers.continuation import refine_voc as _refine_voc
                from driftjax.solvers.continuation import voc_bracket as _voc_bracket

                _e_scale = float(thermal_scales(float(res.cell.T))["energy"])
                _vd = [float(v) / _e_scale for v in res.voltages]
                _cd = [float(j) for j in res.current]
                _va, _vb, _found = _voc_bracket(_vd, _cd)
                _voc_bracketed = bool(_found)
                if _found:
                    _ia = int(
                        min(
                            range(len(res.potentials)),
                            key=lambda i: abs(_vd[i] - _va),
                        )
                    )
                    _vr, _sok = _refine_voc(res.cell, _va, _vb, res.potentials[_ia])
                    if bool(_sok) and _vr == _vr:
                        _voc_val = _vr * _e_scale
            except Exception:
                pass
            res = _eqx.tree_at(
                lambda s: (
                    s.converged,
                    s.max_residual,
                    s.per_bias_residuals,
                    s.fallback_used,
                    s.voc,
                    s.voc_bracketed,
                ),
                res,
                (_conv, _mr, _per_bias, list(_fb), float(_voc_val), _voc_bracketed),
            )
        return res
    finally:
        if progress is not None and hasattr(progress, "close"):
            try:
                progress.close(res)
            except TypeError:
                progress.close()
            except Exception:
                pass  # never mask the primary exception with a close() error
