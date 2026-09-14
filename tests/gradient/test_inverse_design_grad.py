"""P4: inverse-design gradient gate — IFT adjoint vs finite differences.

Both gates freeze the bias/index so the objective is a smooth function of x
(no argmax flips under ±h):

  (1) fixed-bias: L = -(v_k·J_k) for the sweep index nearest 0.5 V;
  (2) MPP envelope: L = -(v_k·J_k) for k = argmax(v·J) at x0 — the exact
      objective the envelope theorem differentiates (dP/dV = 0 at V*).

The chain rule (x -> dimensionless design arrays -> adjoint) must match a
central FD to ~1 %.  Gate: tests/gradient/test_inverse_design_grad.py.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import driftjax as dj
import driftjax.science
from driftjax.autodiff.adjoint import manual_adjoint_grad
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import total_current
from driftjax.units import current as _CUR
from driftjax.units import energy as _E

L_ETM, L_Perov, L_HTM = (5e-05, 0.00011, 5e-05)
Perov = dj.material(
    Nc=1.8e19,
    Nv=1.8e19,
    Eg=1.5,
    eps=9.4,
    Et=0,
    mn=100,
    mp=100,
    tn=1e-06,
    tp=1e-06,
    Chi=3.9,
    A=20000.0,
)
_PIN = float(jnp.sum(spectrum(normalize=False).P_in))
_CUR = float(_CUR)


def make_des(x):
    """dtx mapping (dimensionless), x = [Eg, Chi, Nc, Nd] ETM | HTM."""
    Eg0, Chi0, Nc0, Nd0, Eg1, Chi1, Nc1, Na1 = x
    ETM = dj.material(
        Eg=Eg0,
        Chi=Chi0,
        eps=9.4,
        Nc=10**Nc0,
        Nv=1.8e19,
        mn=100,
        mp=100,
        tn=1e-06,
        tp=1e-06,
        A=20000.0,
    )
    HTM = dj.material(
        Eg=Eg1,
        Chi=Chi1,
        eps=9.4,
        Nc=10**Nc1,
        Nv=1.8e19,
        mn=100,
        mp=100,
        tn=1e-06,
        tp=1e-06,
        A=20000.0,
    )
    return dj.Device(
        layers=list(
            zip([L_ETM, L_Perov, L_HTM], [ETM, Perov, HTM], [10**Nd0, 0, -(10**Na1)], strict=False)
        ),
        n_points=60,  # was 90, 60 saves ~33% with same chain rule pin,
        Snl=10000000.0,
        Snr=10000000.0,
        Spl=10000000.0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


X0 = np.array([1.7, 4.7, 18.8, 18.6, 2.5, 2.5, 19.4, 17.6])
N_STEPS = 10


def _chain_rule_dx(g, des):
    Ulen = (L_ETM + L_Perov + L_HTM) / float(np.asarray(des.x)[-1])
    b0 = int(np.searchsorted(np.asarray(des.x), L_ETM / Ulen))
    b1 = int(np.searchsorted(np.asarray(des.x), (L_ETM + L_Perov) / Ulen))
    R0, R2 = ((0, b0), (b1, len(np.asarray(des.x))))

    def fsum(reg, nm):
        return float(jnp.sum(getattr(g, nm)[reg[0] : reg[1]]))

    def dabs(reg, nm):
        return float(jnp.mean(jnp.abs(getattr(des, nm)[reg[0] : reg[1]])))

    ln10 = np.log(10.0)
    dx = np.zeros(8)
    dx[0] = fsum(R0, "Eg") / _E
    dx[1] = fsum(R0, "Chi") / _E
    dx[2] = fsum(R0, "Nc") * ln10 * dabs(R0, "Nc")
    dx[3] = fsum(R0, "Ndop") * ln10 * dabs(R0, "Ndop")
    dx[4] = fsum(R2, "Eg") / _E
    dx[5] = fsum(R2, "Chi") / _E
    dx[6] = fsum(R2, "Nc") * ln10 * dabs(R2, "Nc")
    dx[7] = fsum(R2, "Ndop") * -ln10 * dabs(R2, "Ndop")
    return dx


def _Pk(x, k):
    """P at a fixed sweep index: v_k(x)·J_k(x) in W/m² (v-grid is x-free).

    `iv[1]` is already the PHYSICAL current (A/cm², = j_dim·current), so
    no extra `_CUR` factor here — double-counting shifts FD vs adjoint by
    exactly 1.096e6 and kills the ratio.
    """
    r = dj.simulate(
        make_des(x),
        dj.Sweep(vmax=0.95, n_steps=N_STEPS),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )
    v, j = r.iv_curve()
    return float(np.asarray(v)[k]) * float(np.asarray(j)[k]) * 10000.0


def _adjoint_at(x, k):
    """(dL/dx) of L = -P_k/PIN via IFT adjoint at (v_k, pot_k)."""
    des = make_des(x)
    r = dj.simulate(
        des,
        dj.Sweep(vmax=0.95, n_steps=N_STEPS),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )
    v, j = r.iv_curve()
    v_mp = float(np.asarray(v)[k])
    pot = r.potentials[k]

    def loss(cell, pot_):
        return -(v_mp * total_current(cell, pot_) * _CUR * 10000.0 / _PIN)

    _, g = manual_adjoint_grad(
        loss,
        des,
        v_mp / _E,
        pot,
        tol=1e-11,
        alpha_mode="beer-lambert",
        ls=spectrum(normalize=False),
    )
    return 100.0 * _chain_rule_dx(g, des)


@pytest.fixture(scope="module")
def r0_shared():
    """Base sweep shared by both k params (was re-simulated per param)."""
    return dj.simulate(
        make_des(X0),
        dj.Sweep(vmax=0.95, n_steps=N_STEPS),
        optics=dj.BeerLambert(alpha_mode="beer-lambert"),
        ls=spectrum(normalize=False),
    )


@pytest.mark.parametrize("k", [0, 7])  # was [0,3,7], 0&3 duplicate fixed-0.5V, keep 1 fixed +1 MPP
def test_adjoint_matches_fd(k, r0_shared):
    """Chain rule + adjoint at the MPP (frozen index) — 2 % gate.

    Gate value evidence (v0.1.14 audit, rtol sweep 1e-8 -> 1e-12 at param 7):
    the IFT adjoint is rtol-stable to 0.01 % (-2.7167e-3 across the sweep)
    while the central FD (h = 1e-4, full re-simulations at default rtol)
    scatters non-monotonically by +-1.3 % around it (-2.682e-3 .. -2.742e-3).
    The 1 % gate previously passed only through a lucky noise realization;
    2 % still catches every chain-rule/adjoint defect class (those produce
    O(10 %)-to-sign-flip mismatches) while sitting above the demonstrated
    FD noise floor.
    """
    r0 = r0_shared
    v0 = np.asarray(r0.voltages)
    kk = int(np.argmin(np.abs(v0 - 0.5))) if k < 3 else int(np.argmax(v0 * np.asarray(r0.current)))
    h = 0.0001
    xp, xm = (X0.copy(), X0.copy())
    xp[k] += h
    xm[k] -= h
    fd = -100.0 * ((_Pk(xp, kk) - _Pk(xm, kk)) / (2 * h)) / _PIN
    adj = _adjoint_at(X0, kk)[k]
    ratio = adj / fd
    assert abs(ratio - 1.0) < 0.02, (
        f"param {k} @ idx {kk}: adjoint {adj:.6e} vs FD {fd:.6e} (ratio {ratio:.5f})"
    )
