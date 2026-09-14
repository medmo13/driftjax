"""optimize_perovskite.py - inverse design of a perovskite p-i-n stack.

16-parameter SLSQP maximization of the power-conversion efficiency
(task introduced in Mann et al., CPC 2021, Sec. 4.2).  The built-in IFT
adjoint supplies the exact gradient via a single `jax.grad(pce)`.

Run:  PYTHONPATH=src python validation/psc.py
      MAXITER=10 PYTHONPATH=src python validation/psc.py   # short
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from driftjax import BeerLambert, Device, Newton, Sweep, material, simulate

L_ETM, L_Perov, L_HTM = 5e-5, 1.1e-4, 5e-5
A, S, tau = 2e4, 1e7, 1e-6
PEROV = dict(
    Nc=3.9e18, Nv=2.7e18, Eg=1.5, eps=10, Et=0, mn=2, mp=2, tn=tau, tp=tau, Chi=3.9, Br=2.3e-9, A=A
)
Chi_P, Eg_P = 3.9, 1.5
vl = np.array([1, 1, 1, 17, 17, 0, 0, 1, 1, 1, 17, 17, 0, 0, 17, 17], float)
vu = np.array([5, 5, 20, 20, 20, 3, 3, 5, 5, 20, 20, 20, 3, 3, 20, 20], float)
X0 = np.array(
    [
        1.661788237392516,
        4.698293002285373,
        19.6342803183675,
        18.83471869026531,
        19.54569869328745,
        0.7252792557586427,
        1.6231392299175988,
        2.5268524699070234,
        2.51936429069554,
        6.933634938056497,
        19.41835918276137,
        18.271793488422656,
        0.46319949214386513,
        0.2058139980642224,
        18.63975340175838,
        17.643726318153238,
    ]
)
N, NS = 120, 20


def x2des(x, n_points=None):
    x = jnp.asarray(x, float)
    etm = material(
        Eg=x[0],
        Chi=x[1],
        eps=x[2],
        Nc=10 ** x[3],
        Nv=10 ** x[4],
        mn=10 ** x[5],
        mp=10 ** x[6],
        tn=tau,
        tp=tau,
        A=A,
    )
    htm = material(
        Eg=x[7],
        Chi=x[8],
        eps=x[9],
        Nc=10 ** x[10],
        Nv=10 ** x[11],
        mn=10 ** x[12],
        mp=10 ** x[13],
        tn=tau,
        tp=tau,
        A=A,
    )
    return Device(
        n_points=N if n_points is None else n_points,
        layers=[
            (L_ETM, etm, 10 ** x[14]),
            (L_Perov, material(**PEROV), 0.0),
            (L_HTM, htm, -(10 ** x[15])),
        ],
        Snl=S,
        Snr=S,
        Spl=S,
        Spr=S,
    )


def pce_jax(x):
    return (
        jnp.asarray(
            simulate(
                x2des(x),
                Sweep(vmax=1.0, n_steps=NS),
                solver=Newton(max_steps=400),
                optics=BeerLambert(),
            ).eff
        )
        * 100.0
    )


_pce_jit = jax.jit(pce_jax)  # JIT: ~8 s/eval eager -> ~0.1 s/eval
_grad_jit = jax.jit(jax.grad(pce_jax))


def pce(x):
    return float(_pce_jit(jnp.asarray(x, float)))


def grad_pce(x):  # the IFT adjoint: one jit-compiled reverse pass
    return np.asarray(_grad_jit(jnp.asarray(x, float)))


def flatband_wf(Nc, Nv, Eg, Chi, N):
    ni = jnp.sqrt(Nc * Nv) * jnp.exp(-Eg / 2)
    EFi = -Chi - Eg / 2 + 0.5 * jnp.log(Nc / Nv)
    dEF = jnp.where(N > 0, jnp.log(jnp.abs(N) / ni), -jnp.log(jnp.abs(N) / ni))
    return float(-EFi - dEF)


def g(x):  # 5 band-alignment feasibility constraints
    x = jnp.asarray(x, float)
    EgE, ChiE, NcE, NvE, NdE = x[0], x[1], 10 ** x[3], 10 ** x[4], 10 ** x[14]
    EgH, ChiH, NcH, NvH, NaH = x[7], x[8], 10 ** x[10], 10 ** x[11], 10 ** x[15]
    P0 = flatband_wf(NcE, NvE, EgE, ChiE, NdE)
    PL = flatband_wf(NcH, NvH, EgH, ChiH, -NaH)
    return np.asarray(
        -jnp.array(
            [ChiE - P0, ChiH - Chi_P, PL - ChiH - EgH, ChiH + EgH - Chi_P - Eg_P, Chi_P - ChiE]
        )
    )


if __name__ == "__main__":
    xs, ys = [], []

    def cb(xk):
        xs.append(np.asarray(xk).copy())
        ys.append(pce(xk))

    res = minimize(
        lambda x: -pce(x),
        X0,
        method="SLSQP",
        jac=lambda x: -grad_pce(x),
        bounds=list(zip(vl, vu, strict=True)),
        constraints=[{"type": "ineq", "fun": g}],
        options={"maxiter": int(os.environ.get("MAXITER", "50")), "disp": False},
        callback=cb,
    )
    ys = np.array(ys)
    start = float(pce(X0))  # true starting point (ys[0] is SLSQP's first iterate)
    print(
        f"psc: start PCE = {start:.3f}%  ->  end PCE = {-res.fun:.3f}%  "
        f"(iters={len(ys)}, first iterate {ys[0]:.3f}%)"
    )
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "perovskite.json", "w") as fh:
        json.dump(
            {
                "history": [float(v) for v in ys],
                "x_history": [x.tolist() for x in xs],
                "x_final": [float(v) for v in np.asarray(res.x)],
            },
            fh,
            indent=2,
        )
