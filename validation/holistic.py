"""optimize_holistic.py - holistic device optimization of a perovskite stack.

Maximize the PCE of the full perovskite p-i-n stack by tuning all 16 ETM/HTM
design parameters under band-alignment feasibility constraints
(task introduced in Mann et al., CPC 2021).

The MPP voltage needs no extra optimization variable: `simulate` already
returns the MPP efficiency `sol.eff`
*with the operating point differentiated through*: jax.grad(sol.eff) IS the
envelope gradient d(eff)/dx (the gradient of the max-power point w.r.t. the
design, holding the operating voltage at its optimum). So no extra variable is
needed -- the IFT adjoint computes exactly the envelope gradient
d(eff)/dx at the optimum.

This script is the FULLY-adjoint formulation: we supply the exact IFT gradient
for BOTH the objective (-dPCE/dx) AND the five band-alignment feasibility
constraints (their Jacobian, d g/dx). psc.py, by contrast, only supplies the
objective gradient and lets SLSQP estimate the constraint Jacobian numerically;
holistic.py uses the complete first-order information from the adjoint.

Run:  PYTHONPATH=src python validation/holistic.py
      MAXITER=10 PYTHONPATH=src python validation/holistic.py     # short
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import psc as _psc
from scipy.optimize import minimize

Chi_P, Eg_P = _psc.Chi_P, _psc.Eg_P


def g_jax(x):
    """Trace-safe feasibility constraints (pure jnp, so jax.jacobian works)."""
    x = jnp.asarray(x, float)
    EgE, ChiE, NcE, NvE, NdE = x[0], x[1], 10 ** x[3], 10 ** x[4], 10 ** x[14]
    EgH, ChiH, NcH, NvH, NaH = x[7], x[8], 10 ** x[10], 10 ** x[11], 10 ** x[15]

    def fbw(Nc, Nv, Eg, Chi, N):
        ni = jnp.sqrt(Nc * Nv) * jnp.exp(-Eg / 2)
        EFi = -Chi - Eg / 2 + 0.5 * jnp.log(Nc / Nv)
        dEF = jnp.where(N > 0, jnp.log(jnp.abs(N) / ni), -jnp.log(jnp.abs(N) / ni))
        return -EFi - dEF

    P0 = fbw(NcE, NvE, EgE, ChiE, NdE)
    PL = fbw(NcH, NvH, EgH, ChiH, -NaH)
    return -jnp.array(
        [ChiE - P0, ChiH - Chi_P, PL - ChiH - EgH, ChiH + EgH - Chi_P - Eg_P, Chi_P - ChiE]
    )


_jac_g = jax.jit(jax.jacobian(g_jax))  # exact constraint Jacobian


def g_np(x):
    return np.asarray(g_jax(jnp.asarray(x, float)))


def g_jac_np(x):
    return np.asarray(_jac_g(jnp.asarray(x, float)))


if __name__ == "__main__":
    xs, ys = [], []

    def cb(xk):
        xs.append(np.asarray(xk).copy())
        ys.append(_psc.pce(xk))

    res = minimize(
        lambda x: -_psc.pce(x),
        _psc.X0,
        method="SLSQP",
        jac=lambda x: -_psc.grad_pce(x),
        bounds=list(zip(_psc.vl, _psc.vu, strict=False)),
        constraints=[{"type": "ineq", "fun": g_np, "jac": g_jac_np}],
        options={"maxiter": int(os.environ.get("MAXITER", "50")), "disp": False},
        callback=cb,
    )
    ys = np.array(ys)
    start = float(_psc.pce(_psc.X0))  # true starting point (ys[0] is SLSQP's first iterate)
    print(
        f"holistic: start PCE = {start:.3f}%  ->  end PCE = {ys[-1]:.3f}%  "
        f"(iters={len(ys)}, first iterate {ys[0]:.3f}%)"
    )
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "holistic.json", "w") as fh:
        json.dump(
            {
                "history": [float(v) for v in ys],
                "x_history": [x.tolist() for x in xs],
                "x_final": [float(v) for v in np.asarray(res.x)],
            },
            fh,
            indent=2,
        )
