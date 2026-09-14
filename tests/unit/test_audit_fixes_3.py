"""Third-round audit regression tests (2026-09-11).

Each test fails on the pre-fix code and passes after:

* Fresnel ``rear_reflectance`` was accepted but never forwarded — every call
  silently used a perfect rear mirror (Rb=1).
* ``linsolve`` dense/banded/auto returned NaN residuals at exact roots
  (``0/0``) and silently accepted unknown backends.
* ``_eq_bwd`` dropped the direct cell channel (wrong equilibrium gradients
  for any cell-dependent objective) and the forward ``statistics``.
* ``white()`` integrated to ~20 suns instead of 1 sun.
* ``load_material``/optics loaders interpolated ``name`` into paths
  (traversal) and used ``np.load`` with default pickle settings.
* ``find_voc`` fabricated huge Voc from denormal current legs.
* ``linear_extrapolation`` produced NaN guesses for degenerate schedules.
* ``slsqp`` discarded an explicitly passed Jacobian.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj
from driftjax.numerics.linalg import linsolve
from driftjax.science.optics import (
    _fresnel_per_lambda,
    _refractive_index,
    alpha_tauc,
    beer_lambert_G,
    fresnel_generation,
    photonflux,
)
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import find_voc, linear_extrapolation
from driftjax.units import length as _L

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


def _design(n_points=80):
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [MAT, MAT], [1e16, -1e16], strict=False)),
        n_points=n_points,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


# ---------------------------------------------------------------------------
# Fresnel rear reflectance (was silently Rb=1 always)
# ---------------------------------------------------------------------------


def test_fresnel_rear_reflectance_honored():
    """G must grow monotonically with Rb; Rb=0 < Beer–Lambert (front loss)."""
    des = _design()
    sums = [
        float(
            np.sum(np.asarray(fresnel_generation(des, LS, alpha_mode="tauc", rear_reflectance=rb)))
        )
        for rb in (0.0, 0.9, 1.0)
    ]
    assert sums[0] < sums[1] < sums[2], f"Rb ordering violated: {sums}"
    g_bl = float(np.sum(np.asarray(beer_lambert_G(des, LS, alpha_mode="tauc"))))
    assert sums[0] < g_bl, "Rb=0 (front loss) must sit below plain Beer–Lambert"


def test_fresnel_rb0_absorptance_identity():
    """Public Rb=0 output matches the single-pass analytic absorptance
    A=(1−Rf)(1−e^{−τL}) — failed pre-fix (backward flux always included)."""
    des = _design(n_points=2000)
    x_m = np.asarray(des.x) * float(_L) * 0.01
    dx = x_m[1] - x_m[0]
    alpha = np.asarray(alpha_tauc(des, LS))
    n_real, k = _refractive_index(des, LS, "tauc")
    n_tilde = n_real[:, 0] + 1j * k[:, 0]
    phi0 = np.asarray(photonflux(LS))
    tauL = np.sum(alpha[:, :-1] * dx, axis=1)
    R_f = np.abs((1.0 - n_tilde) / (1.0 + n_tilde)) ** 2
    A_an = (1.0 - R_f) * (1.0 - np.exp(-tauL))
    G = np.asarray(
        _fresnel_per_lambda(jnp.asarray(x_m), jnp.asarray(phi0), jnp.asarray(alpha), n_tilde, 0.0)
    )
    A = (np.sum(G, axis=1) - 0.5 * (G[:, 0] + G[:, -1])) * dx / phi0
    m = A_an > 1e-06
    assert float(np.max(np.abs(A[m] / A_an[m] - 1.0))) < 0.0025


# ---------------------------------------------------------------------------
# linsolve: zero-rhs + strict backends
# ---------------------------------------------------------------------------


def test_linsolve_zero_rhs_finite_residual():
    """At an exact root (rhs=0) the relative residual was 0/0=NaN."""
    J2 = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    # banded backend needs a (3N, 3N) block-tridiagonal Jacobian (N=2 here)
    J6 = (
        jnp.diag(jnp.full(6, 2.0))
        + jnp.diag(jnp.full(5, 0.25), 1)
        + jnp.diag(jnp.full(5, 0.25), -1)
    )
    for backend, J in (("dense", J2), ("banded", J6), ("auto", J2), ("csr", J2)):
        rhs = jnp.zeros(J.shape[0])
        x, resid, _tag = linsolve(J, rhs, backend=backend)
        assert bool(jnp.all(jnp.isfinite(x))), backend
        assert bool(jnp.isfinite(resid)), backend


def test_linsolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown linsolve backend"):
        linsolve(jnp.eye(3), jnp.ones(3), backend="bogus")


def test_newton_and_ptc_linear_solve_zero_rhs():
    """Newton/PTC private linear solves share the 0/0-NaN hazard and the
    absolute-vs-relative inconsistency — both must return finite relative
    residuals at an exact root."""
    from driftjax.solvers.newton import _linear_solve as newton_lin
    from driftjax.solvers.ptc import _linear_solve as ptc_lin

    J = jnp.array([[2.0, 0.5], [0.5, 3.0]])
    rhs = jnp.zeros(2)
    x, resid, tag = newton_lin(J, rhs, dense=True)
    assert bool(jnp.all(jnp.isfinite(x))) and bool(jnp.isfinite(resid)), (x, resid)
    assert tag == "dense"
    x, resid, tag = ptc_lin(J, rhs, False, backend="dense")
    assert bool(jnp.all(jnp.isfinite(x))) and bool(jnp.isfinite(resid)), (x, resid)
    # relative scale: residual of the exact zero solution must be ~0, not ||rhs||
    assert float(resid) < 1e-12, float(resid)


# ---------------------------------------------------------------------------
# Equilibrium adjoint: direct cell channel + statistics (vs FD ground truth)
# ---------------------------------------------------------------------------


def _degenerate_design():
    mat = MAT
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [mat, mat], [5e20, -5e20], strict=False)),
        n_points=25,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


@pytest.mark.slow  # 3x jax.grad(simulate) through equilibrium — each is a full adjoint backward
@pytest.mark.parametrize("stats", ["boltzmann", "fd", "exact"])
def test_eq_adjoint_cell_objective_matches_fd(stats):
    """d(sum n(cell,pot))/dChi[0] via the IFT adjoint must match central FD.

    Pre-fix the direct cell channel was dropped (error ~1e5x here) and the
    backward cell always used Boltzmann statistics. eps=0.2 absolute on
    Chi[0] (~150 dimensionless, so ~0.1% relative): large enough to clear
    Newton solver noise (~1e-10 on the objective), small enough that the
    smooth equilibrium response is FD-linear (verified by halving eps)."""
    import equinox as eqx

    from driftjax.science.carrier_statistics import n as nstat

    des = _degenerate_design()

    def obj(dd):
        s = dj.simulate(dd, dj.Equilibrium(), statistics=stats, ls=LS)
        return jnp.sum(nstat(s.cell, s.eq_pot))

    g = jax.grad(obj)(des)
    ad = float(g.Chi[0])
    c0 = float(des.Chi[0])
    eps = 0.2
    f1 = float(obj(eqx.tree_at(lambda m: m.Chi, des, des.Chi.at[0].set(c0 + eps))))
    f0 = float(obj(eqx.tree_at(lambda m: m.Chi, des, des.Chi.at[0].set(c0 - eps))))
    fd = (f1 - f0) / (2 * eps)
    assert abs(ad - fd) < 1e-4 * max(1.0, abs(fd)), f"{stats}: AD={ad} FD={fd}"


@pytest.mark.slow  # jax.grad(simulate) through equilibrium — full adjoint backward
def test_eq_adjoint_potential_objective_matches_fd():
    """Pure-potential equilibrium objective (no explicit cell dependence):
    covers the `g_cell is None` branch of _eq_bwd (direct channel inactive)."""
    import equinox as eqx

    des = _degenerate_design()

    def obj(dd):
        s = dj.simulate(dd, dj.Equilibrium(), statistics="fd", ls=LS)
        return jnp.sum(s.eq_pot.phi)

    g = jax.grad(obj)(des)
    ad = float(g.Chi[0])
    c0 = float(des.Chi[0])
    eps = 0.2
    f1 = float(obj(eqx.tree_at(lambda m: m.Chi, des, des.Chi.at[0].set(c0 + eps))))
    f0 = float(obj(eqx.tree_at(lambda m: m.Chi, des, des.Chi.at[0].set(c0 - eps))))
    fd = (f1 - f0) / (2 * eps)
    assert abs(ad - fd) < 1e-6 * max(1.0, abs(fd)), f"AD={ad} FD={fd}"


# ---------------------------------------------------------------------------
# Spectrum / io / continuation / optimizers
# ---------------------------------------------------------------------------


def test_load_material_rejects_traversal():
    from driftjax.io import _load_alpha_table, _load_optics_from_db, load_material

    for bad in ("../../etc/x", "/abs/path", "..", "a/b", "x\\y"):
        with pytest.raises(ValueError, match="invalid material name"):
            load_material(bad)
        with pytest.raises(ValueError, match="invalid material name"):
            _load_optics_from_db(bad)
        with pytest.raises(ValueError, match="invalid material name"):
            _load_alpha_table(bad)


def test_find_voc_denormal_legs_report_no_crossing():
    """Denormal opposite-sign legs (±1e-320, both numerically zero) must not
    fabricate a Voc via the absolute 1e-300 floor."""
    v = jnp.array([0.0, 0.5])
    c = jnp.array([1e-320, -1e-320])
    assert bool(jnp.isnan(find_voc(v, c))), float(find_voc(v, c))


def test_linear_extrapolation_degenerate_schedule():
    """v_a == v_b (n_steps=1 / vmax=0) must fall back to pot_b, not NaN."""
    from driftjax.fields import Potentials

    n = 8
    pa = Potentials(jnp.zeros(n), jnp.zeros(n), jnp.zeros(n))
    pb = Potentials(jnp.ones(n), 2 * jnp.ones(n), 3 * jnp.ones(n))
    out = linear_extrapolation(pa, pb, 0.5, 0.5, 0.7)
    for field in (out.phi_n, out.phi_p, out.phi):
        assert bool(jnp.all(jnp.isfinite(field)))
    np.testing.assert_allclose(np.asarray(out.phi), 3 * np.ones(n))
    np.testing.assert_allclose(np.asarray(out.phi_n), np.ones(n))


def test_slsqp_explicit_jac_is_forwarded():
    """An explicitly passed jac callable must be used (was discarded)."""
    from driftjax.optimize.optimizers import slsqp

    calls = []

    def f(x):
        return float(np.sum((x - 2.0) ** 2))

    def jac(x):
        calls.append(1)
        return 2.0 * (np.asarray(x) - 2.0)

    r = slsqp(f, np.array([0.0, 0.0]), jac=jac, maxiter=50)
    assert calls, "explicit jac was never called"
    np.testing.assert_allclose(r.x, [2.0, 2.0], atol=1e-6)


def test_provenance_record_has_utc_timestamp():
    from driftjax.runtime import provenance

    assert provenance.UTC is not None
    rec = provenance.record()
    assert "timestamp_utc" in rec and rec["timestamp_utc"]


# ---------------------------------------------------------------------------
# R3 "best things": blakemore warning, newton while parity
# ---------------------------------------------------------------------------


@pytest.mark.slow  # Equilibrium solve to trigger the warning path
def test_blakemore_statistics_warns():
    """statistics='blakemore' must warn toward 'exact' (legacy provenance
    approximation, not an accurate FD model)."""
    import warnings

    des = _degenerate_design()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            dj.simulate(des, dj.Equilibrium(), statistics="blakemore", ls=LS)
        except UserWarning as e:
            assert "blakemore" in str(e) and "exact" in str(e)
        else:
            raise AssertionError("blakemore UserWarning was not raised")


def _stiff_cell(n_points=40):
    cds = dj.material(
        Chi=4.0,
        Eg=2.4,
        eps=9.0,
        Nc=2.2e18,
        Nv=1.8e19,
        mn=100,
        mp=25,
        tn=1e-8,
        tp=1e-8,
        A=1e4,
    )
    cdte = dj.material(
        Chi=3.9,
        Eg=1.5,
        eps=9.4,
        Nc=8e17,
        Nv=1.8e19,
        mn=100,
        mp=40,
        tn=1e-8,
        tp=1e-8,
        A=1e4,
    )
    return dj.Device(
        layers=list(zip([4e-05, 0.0001], [cds, cdte], [1e17, -1e15], strict=False)),
        n_points=n_points,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


@pytest.mark.slow  # 2x solve_newton on stiff heterojunction (N=40) + residual check
def test_newton_while_matches_eager():
    """Traced loop must reproduce the eager loop on clean solves (primal
    unchanged for previously-good trajectories)."""
    from driftjax.fields import pot2vec
    from driftjax.numerics.residual import comp_F
    from driftjax.science.contacts import boundary_eq
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_newton

    des = _stiff_cell()
    cell = init_cell(des, LS, alpha_mode="tauc", statistics="boltzmann", optics=None, fused=False)
    bound = boundary_eq(cell)
    pot_ini = equilibrium_guess(cell)
    kw = dict(tol=1e-10, max_steps=60, globalization="logdamp")
    pw, sw = solve_newton(cell, bound, pot_ini, loop="while", **kw)
    pe, se = solve_newton(cell, bound, pot_ini, loop="python", **kw)
    assert bool(sw["converged"]) and bool(se["converged"])
    assert float(jnp.max(jnp.abs(pot2vec(pw) - pot2vec(pe)))) < 1e-8
    for s in (sw, se):
        r = float(jnp.max(jnp.abs(comp_F(cell, bound, pw if s is sw else pe))))
        assert r < 1e-8, r


@pytest.mark.slow  # solve_newton with tol=0.0 on stiff heterojunction (40 iterations)
def test_newton_while_exhaustion_certifies_settled_tail():
    """Forced exhaustion (unreachable tol) on a well-conditioned problem:
    the traced loop must certify the settled machine-floor tail as converged
    (eager parity) instead of reporting failure."""
    from driftjax.numerics.residual import comp_F
    from driftjax.science.contacts import boundary_eq
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_newton

    des = _stiff_cell()
    cell = init_cell(des, LS, alpha_mode="tauc", statistics="boltzmann", optics=None, fused=False)
    bound = boundary_eq(cell)
    pot_ini = equilibrium_guess(cell)
    pw, sw = solve_newton(
        cell, bound, pot_ini, tol=0.0, max_steps=40, loop="while", globalization="logdamp"
    )
    assert bool(sw["converged"]), sw
    r = float(jnp.max(jnp.abs(comp_F(cell, bound, pw))))
    assert r < 1e-8, r
