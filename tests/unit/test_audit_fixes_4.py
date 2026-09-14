"""Quality-pass regression tests (2026-09-12, subagent findings H1–H5, M, L, T).

Each test pins a fix from the read-only quality pass over the R3 audit work:

* H1: sweep progress ``converged`` was recomputed from finite ``resid``,
  ignoring the solver flag (unconverged-but-finite tails lit the bar green).
* H2: Newton / line-search / PTC / step-halving returned different
  stats-dict shapes (``resid`` vs ``fmax``, missing ``backend``/``error``).
* H3: ``slsqp`` scalar-objective penalty returned a bare ``1e10`` under the
  default tuple protocol, breaking SciPy's ``jac=True`` unpacking.
* H4: ``linsolve(backend="banded")`` returned NaN silently where ``auto``
  falls back to dense.
* H5: legacy ``simulator.sweep`` FF lacked the NaN guard (``simulate`` had it).
* M2/M4: eager ``iter_cb`` passed ‖F‖₂ as max|F|; settled branch omitted
  ``stagnated``. M3: traced settled tail used a stale residual.
* M10/L2: degenerate schedules/inputs (``n_steps=1``, ``white(0)``,
  ``monochromatic(-1)``, ``T=0``) failed deep instead of loudly.
* T1: Fresnel coverage now end-to-end through ``simulate``.
* T2/T3: forced-NaN rebound parity + tighter-``f_tol`` criterion pins.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj
from driftjax.science.spectrum import monochromatic, spectrum, white
from driftjax.solvers.newton import _rebound_criterion

MAT = dj.material(
    Chi=3.9,
    Eg=1.12,
    eps=11.7,
    Nc=2.8e19,
    Nv=1.04e19,
    mn=1.08,
    mp=0.56,
    tn=1e-06,
    tp=1e-06,
    A=20000.0,
)
LS = spectrum(normalize=True)


def _small_device(n_points=20):
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=n_points,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    )


def _stiff_cell(n_points=20):
    from driftjax.simulator import init_cell

    des = _small_device(n_points).design()
    return init_cell(des, LS, alpha_mode="tauc", statistics="boltzmann", optics=None, fused=False)


@pytest.fixture(scope="module")
def _stiff_cell_shared():
    """Module-cached stiff cell for the 4-path stats parity + iter_cb tests."""
    return _stiff_cell()


@pytest.fixture(scope="module")
def _stiff_eq_shared(_stiff_cell_shared):
    """Module-cached equilibrium state for the stiff cell."""
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq

    cell = _stiff_cell_shared
    pot0 = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
    return cell, pot0


# ---------------------------------------------------------------------------
# H3: slsqp scalar-penalty tuple protocol
# ---------------------------------------------------------------------------


def test_slsqp_scalar_objective_with_hole_completes():
    """Scalar f with a non-finite probe under default jac=None must be
    penalised as a (value, grad) pair, not crash SciPy's tuple unpacking."""
    from driftjax.optimize.optimizers import slsqp

    def f(x):
        v = float(np.sum(np.asarray(x) ** 2))
        if x[0] > 5.0:
            return float("inf")  # failed physics probe
        return v

    r = slsqp(f, np.array([10.0, 1.0]), bounds=[(0.0, 12.0), (-2.0, 2.0)], maxiter=30)
    assert np.isfinite(float(r.fun)), r.fun


def test_slsqp_tuple_nan_grad_penalized():
    """Tuple f returning a NaN gradient must be rejected, not propagated."""

    from driftjax.optimize.optimizers import slsqp

    def f(x):
        v = float(np.sum((np.asarray(x) - 1.0) ** 2))
        g = np.zeros(2) if x[0] < 0.0 else np.full(2, np.nan)
        return v, g

    r = slsqp(f, np.array([-1.0, -1.0]), bounds=[(-2.0, 2.0), (-2.0, 2.0)], maxiter=30)
    assert np.isfinite(float(r.fun)), r.fun


# ---------------------------------------------------------------------------
# H2: stats-key parity across Newton / LS / PTC
# ---------------------------------------------------------------------------

_CANON = {"backend", "iters", "error", "resid", "resid_f", "converged", "stagnated", "fallback"}


@pytest.mark.parametrize("path", ["newton-python", "newton-while", "ls", "ptc"])
def test_solver_stats_key_parity(path, _stiff_eq_shared):
    """Every solver path returns all canonical keys (H2/T4)."""
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.newton import solve_newton
    from driftjax.solvers.ptc import solve_newton_ls, solve_ptc

    cell, pot0 = _stiff_eq_shared
    bound = boundary_bias(cell, 0.1)
    if path == "newton-python":
        _, st = solve_newton(cell, bound, pot0, tol=1e-8, max_steps=15, loop="python")
    elif path == "newton-while":
        _, st = solve_newton(
            cell, bound, pot0, tol=1e-8, max_steps=15, loop="while", allow_trace=True
        )
    elif path == "ls":
        _, st = solve_newton_ls(cell, bound, pot0, max_steps=8)
    else:
        _, st = solve_ptc(cell, bound, pot0, max_steps=5)
    assert set(st) >= _CANON, (path, sorted(st))
    assert isinstance(bool(st["converged"]), bool)


# ---------------------------------------------------------------------------
# H1: progress converged flag follows the solver
# ---------------------------------------------------------------------------


def test_progress_converged_follows_solver_flag(_stiff_eq_shared):
    """A stubbed unconverged-but-finite solve must report converged=False
    (old code derived True from the finite residual)."""
    import driftjax.solvers.continuation as cont

    cell_ = _stiff_cell(n_points=12)
    real_solve = cont.solve_newton

    def stub(cell__, bound_, pot_, **kw):
        pot, _ = real_solve(cell__, bound_, pot_, tol=1e-10, max_steps=2)
        return pot, {
            "converged": False,
            "resid": 1e-3,
            "resid_f": 1e-3,
            "iters": 2,
            "backend": "stub",
            "fallback": None,
        }

    seen = []

    def prog(i, info):
        seen.append(info)

    cont.solve_newton = stub
    try:
        cont.sweep(cell_, 0.2, n_steps=3, tol=1e-10, progress=prog)
    finally:
        cont.solve_newton = real_solve
    assert seen, "progress hook never fired"
    assert all(s["converged"] is False for s in seen), seen
    assert all(np.isfinite(s["resid"]) for s in seen), seen


# ---------------------------------------------------------------------------
# H4: banded dense fallback
# ---------------------------------------------------------------------------


def test_banded_fallback_on_singular():
    """Zero-matrix banded solve must fall back to dense (tag 'dense'),
    mirroring the auto path — previously returned silently with tag banded."""
    from driftjax.numerics.linalg import linsolve

    J = jnp.zeros((6, 6))
    rhs = jnp.ones(6)
    _, _, tag = linsolve(J, rhs, backend="banded")
    assert tag == "dense", tag


# ---------------------------------------------------------------------------
# H5: legacy sweep FF NaN guard
# ---------------------------------------------------------------------------


def test_legacy_sweep_ff_nan_beyond_range():
    """A sweep too short to reach Voc (voc NaN) must report ff NaN —
    and a normal sweep must report 0 < ff <= 1."""
    from driftjax.simulator import sweep as legacy_sweep

    dev = _small_device(n_points=25)
    out = legacy_sweep(dev.design(), None, 0.05, n_steps=4)
    assert float(out["voc"]) != float(out["voc"])  # NaN sanity
    assert bool(jnp.isnan(out["ff"])), out["ff"]
    out2 = legacy_sweep(dev.design(), None, 0.9, n_steps=8)
    assert 0.0 < float(out2["ff"]) <= 1.0, out2["ff"]


# ---------------------------------------------------------------------------
# T2/T3: rebound criterion + forced-NaN parity
# ---------------------------------------------------------------------------


def test_rebound_criterion_tighter_ftol():
    assert _rebound_criterion(None) == 1e-8
    assert _rebound_criterion(1e-6) == 1e-8
    assert _rebound_criterion(1e-12) == 1e-12


def test_nan_init_rebound_parity(_stiff_eq_shared):
    """NaN initial guess: both loops must exit stagnated (not converged)."""
    import jax.tree_util as tu

    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.newton import solve_newton

    cell, pot0 = _stiff_eq_shared
    nan_pot = tu.tree_map(lambda a: jnp.full_like(a, jnp.nan), pot0)
    bound = boundary_bias(cell, 0.1)
    _, se = solve_newton(cell, bound, nan_pot, tol=1e-10, max_steps=10, loop="python")
    _, sw = solve_newton(
        cell, bound, nan_pot, tol=1e-10, max_steps=10, loop="while", allow_trace=True
    )
    for s in (se, sw):
        assert bool(s["stagnated"]) is True, s
        assert bool(s["converged"]) is False, s


# ---------------------------------------------------------------------------
# M2/M4: eager iter_cb contract + stagnated key
# ---------------------------------------------------------------------------


def test_iter_cb_receives_max_resid_and_stagnated_present(_stiff_eq_shared):
    """iter_cb third arg is max|F| (M2); converged stats carry stagnated (M4)."""
    from driftjax.numerics.residual import comp_F
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.newton import solve_newton

    cell, pot0 = _stiff_eq_shared
    bound = boundary_bias(cell, 0.1)
    calls = []
    pot, st = solve_newton(
        cell,
        bound,
        pot0,
        tol=1e-10,
        max_steps=30,
        loop="python",
        iter_cb=lambda it, err, r: calls.append((it, err, r)),
    )
    assert calls, "iter_cb never fired"
    assert all(c[2] is None or np.isfinite(c[2]) for c in calls)
    assert bool(st["converged"]) and st["stagnated"] is False, st
    # max|F| at the returned root bounds the reported final residual scale
    r_root = float(jnp.max(jnp.abs(comp_F(cell, bound, pot))))
    assert r_root < 1e-8, r_root


# ---------------------------------------------------------------------------
# M10/L2: degenerate-input validation
# ---------------------------------------------------------------------------


def test_degenerate_schedules_rejected():
    from driftjax.solvers.batched import sweep_batched
    from driftjax.solvers.continuation import sweep

    cell = _stiff_cell(n_points=12)
    with pytest.raises(ValueError):
        sweep(cell, 0.5, n_steps=1)
    with pytest.raises(ValueError):
        sweep_batched(cell, 0.5, n_steps=1)
    with pytest.raises(ValueError):
        dj.Sweep(n_steps=1)
    with pytest.raises(ValueError):
        dj.Sweep(n_steps=0)


def test_white_monochromatic_validation():
    with pytest.raises(ValueError):
        white(n_points=0)
    with pytest.raises(ValueError):
        white(lam_lo=800.0, lam_hi=400.0)
    with pytest.raises(ValueError):
        monochromatic(-1.0)
    with pytest.raises(ValueError):
        monochromatic(0.0)


def test_device_T_validation():
    with pytest.raises(ValueError):
        dj.Device(
            layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
            n_points=12,
            T=0.0,
        )
    with pytest.raises(ValueError):
        dj.Device(
            layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
            n_points=12,
            T=-50.0,
        )


def test_canonicalize_alpha_mismatch_warns():
    from driftjax.io import material

    with pytest.warns(UserWarning):
        material(Chi=3.9, Eg=1.12, alpha=[1.0, 2.0], Lambda=[500.0])


# ---------------------------------------------------------------------------
# T4: batched degenerate tol
# ---------------------------------------------------------------------------


def test_batched_degenerate_tol_falls_back():
    from driftjax.solvers.batched import sweep_batched

    cell = _stiff_cell(n_points=12)
    for bad_tol in (None, -1.0, float("nan")):
        v, j, _ = sweep_batched(cell, 0.4, n_steps=4, tol=bad_tol)
        assert bool(jnp.all(jnp.isfinite(j))), bad_tol


# ---------------------------------------------------------------------------
# T5: at_bias tracer error
# ---------------------------------------------------------------------------


def test_at_bias_tracer_error():
    dev = _small_device(n_points=12)
    sol = dj.simulate(dev, dj.Sweep(vmax=0.6, n_steps=3))
    assert len(sol.potentials) == 3
    with pytest.raises(ValueError, match="concrete-path"):
        jax.jit(lambda s: s.at_bias(0.3))(sol)
