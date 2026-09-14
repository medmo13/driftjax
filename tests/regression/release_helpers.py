"""Production-owned helpers for the release validation workflows.

The helpers keep the regression suite self-contained: the release package
does not require archived external comparison artifacts to run its tests.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import jacobian, value_and_grad

from driftjax import BeerLambert, Device, Newton, Sweep, material, simulate

L_ETM, L_PEROV, L_HTM = 5e-5, 1.1e-4, 5e-5
N_DEFAULT, A, TAU, S = 120, 2e4, 1e-6, 1e7
ABSORBER = dict(Nc=3.9e18, Nv=2.7e18, Eg=1.5, eps=10, Et=0, mn=2,
                mp=2, tn=TAU, tp=TAU, Chi=3.9, Br=2.3e-9, A=A)
VL = np.array([1, 1, 1, 17, 17, 0, 0, 1, 1, 1, 17, 17, 0, 0, 17, 17], float)
VU = np.array([5, 5, 20, 20, 20, 3, 3, 5, 5, 20, 20, 20, 3, 3, 20, 20], float)


def x2des(x, n_points=N_DEFAULT):
    x = jnp.asarray(x, dtype=jnp.float64)
    etm = material(Eg=x[0], Chi=x[1], eps=x[2], Nc=10**x[3], Nv=10**x[4],
                   mn=10**x[5], mp=10**x[6], tn=TAU, tp=TAU, A=A)
    htm = material(Eg=x[7], Chi=x[8], eps=x[9], Nc=10**x[10], Nv=10**x[11],
                   mn=10**x[12], mp=10**x[13], tn=TAU, tp=TAU, A=A)
    return Device(n_points=n_points,
                  layers=[(L_ETM, etm, 10**x[14]),
                          (L_PEROV, material(**ABSORBER), 0.0),
                          (L_HTM, htm, -10**x[15])],
                  Snl=S, Snr=S, Spl=S, Spr=S)


def pce_jax(x, vmax=1.0, n_steps=20, n_points=N_DEFAULT, max_steps=400):
    sol = simulate(x2des(x, n_points), Sweep(vmax=vmax, n_steps=n_steps),
                   solver=Newton(max_steps=max_steps), optics=BeerLambert())
    return sol.efficiency * 100.0


def pce(x, **kwargs):
    return float(pce_jax(x, **kwargs))


def _flatband_work_function(Nc, Nv, Eg, Chi, doping):
    ni = jnp.sqrt(Nc * Nv) * jnp.exp(-Eg / 2)
    efi = -Chi - Eg / 2 + 0.5 * jnp.log(Nc / Nv)
    d_ef = jnp.where(doping > 0, jnp.log(jnp.abs(doping) / ni),
                     -jnp.log(jnp.abs(doping) / ni))
    return -efi - d_ef


def constraints_jax(x):
    x = jnp.asarray(x, dtype=jnp.float64)
    p0 = _flatband_work_function(10**x[3], 10**x[4], x[0], x[1], 10**x[14])
    pl = _flatband_work_function(10**x[10], 10**x[11], x[7], x[8], -10**x[15])
    return -jnp.array([x[1] - p0, x[8] - ABSORBER["Chi"],
                       pl - x[8] - x[7], x[8] + x[7] - ABSORBER["Chi"] - ABSORBER["Eg"],
                       ABSORBER["Chi"] - x[1]])


_constraints_jac = jax.jit(jacobian(constraints_jax))


def g_np(x):
    return np.asarray(constraints_jax(x))


def g_jac_np(x):
    return np.asarray(_constraints_jac(x))


def _polar(x, y):
    return jnp.arctan2(y, x), jnp.sqrt(x**2 + y**2)


def iv_distance_jax(y1, y2, norm=2.0):
    y1, y2 = 10.0 * y1, 10.0 * y2
    x1 = jnp.arange(y1.shape[0], dtype=jnp.float64) * 0.05
    x2 = jnp.arange(y2.shape[0], dtype=jnp.float64) * 0.05
    t1, r1 = _polar(x1, y1)
    t2, r2 = _polar(x2, y2)
    theta = jnp.linspace(0.0, jnp.pi / 2.0, 100)
    return jnp.sum((jnp.abs(jnp.interp(theta, t1, r1) -
                           jnp.interp(theta, t2, r2))) ** norm)


def iv_distance(y1, y2, norm=2.0):
    return float(iv_distance_jax(y1, y2, norm))


def _make_objective(fn):
    compiled = jax.jit(value_and_grad(fn))

    def objective(x):
        value, gradient = compiled(jnp.asarray(x, dtype=jnp.float64))
        return float(value), np.asarray(gradient, dtype=float)

    return objective


def make_psc_jac(**kwargs):
    return _make_objective(lambda x: -pce_jax(x, **kwargs))


def make_hol_jac(**kwargs):
    return _make_objective(lambda x: -pce_jax(x, **kwargs))
