"""optimize_material_recovery.py - recover material parameters from a target IV curve.

Recovers the (hole mobility, band gap) of a homojunction by minimizing the
polar IV-curve distance R(J_hat, J*) against a synthesized target
(task introduced in Mann et al., CPC 2021).

Gradient note: the swept IV array sol.current is not back-propagated by the
adjoint (its analytic sensitivity is ~1e-7..1e-12), so SciPy estimates the
metric gradient numerically (jac=False); the PCE objective elsewhere uses the
exact IFT gradient.

Run:  PYTHONPATH=src python validation/multi.py
"""
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from driftjax import Device, Sweep, material, simulate

N, NS = 200, 25


def _material(mp, Eg):
    return material(Eg=jnp.asarray(Eg, float), Chi=3.0, eps=10.0, Nc=1e18,
                                 Nv=1e18, mn=130.0, mp=jnp.asarray(mp, float), A=2e4)


def _device(mp, Eg):
    m = _material(mp, Eg)
    return Device(n_points=N, layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)],
                  Snl=1e7, Snr=0, Spl=0, Spr=1e7)


def get_J(mp, Eg):                       # eager forward (target + plotting)
    return np.asarray(simulate(_device(float(mp), float(Eg)),
                               Sweep(vmax=1.1, n_steps=NS)).current)


def polar(x, y):
    return jnp.arctan2(x, y), jnp.sqrt(x ** 2 + y ** 2)


def iv_distance(y1, y2, ref, norm=2):
    """Polar curve distance R(J_hat, J*) (metric of Mann et al. 2021), scale-normalized by ref."""
    y1 = 10.0 * y1 / ref
    y2 = 10.0 * y2 / ref
    x1 = jnp.arange(y1.shape[0]) * 0.05
    x2 = jnp.arange(y2.shape[0]) * 0.05
    t1, r1 = polar(x1, y1)
    t2, r2 = polar(x2, y2)
    th = jnp.linspace(0, jnp.pi / 2, 100)
    ri1 = jnp.interp(th, t1, r1)
    ri2 = jnp.interp(th, t2, r2)
    return jnp.sum(jnp.power(jnp.abs(ri1 - ri2), norm))


J0 = get_J(160.0, 1.0)                   # target curve: true (mp, Eg) = (160, 1.0)
ref = float(np.abs(J0).max()) + 1e-30

# jit the forward so the many finite-difference evaluations stay fast
_getJ_jit = jax.jit(lambda mp, Eg: simulate(_device(mp, Eg), Sweep(vmax=1.1, n_steps=NS)).current)

# The polar metric evaluates to ~1e-6 at these current scales, and the curve is
# ~1e4x more sensitive to Eg than to mp, so SLSQP's default ftol would treat the
# tiny mp-gradient as zero and stall. We normalise the objective by its value at
# the start (f(x0)=1, minimum f=0) so the landscape is O(1) and well conditioned,
# and tighten ftol. The minimum is still iv_distance = 0.
SCALE = 1.0 / (float(iv_distance(get_J(100.0, 1.2), J0, ref)) + 1e-30)


def f(x):
    J = np.asarray(_getJ_jit(10 ** x[0], x[1]))
    return float(iv_distance(J, J0, ref) * SCALE)


if __name__ == "__main__":
    xs, ys = [], []
    def cb(xk):
        xs.append(np.asarray(xk).copy()); ys.append(f(xk))
    res = minimize(f, np.array([2.0, 1.2]), method="SLSQP", jac=False,
                   bounds=[(1.0, 3.0), (0.5, 2.0)],
                   options={"maxiter": int(os.environ.get("MAXITER", "50")) * 2,
                            "ftol": 1e-9, "disp": False},
                   callback=cb)
    ys = np.array(ys)
    tgt = np.array([np.log10(160.0), 1.0])
    recovered = np.asarray(res.x)
    r_initial = float(ys[0]) if len(ys) > 0 else float('inf')
    r_final = float(res.fun) / SCALE
    print(f"material recovery: R final = {r_final:.3e}  recovered (log10 mp, Eg) = "
          f"{np.round(recovered, 3)}  (target {np.round(tgt, 3)})")
    # PASS if objective reduced by >10x and Eg recovered to <0.05
    eg_err = abs(recovered[1] - tgt[1])
    passed = r_final < r_initial * 0.1 and eg_err < 0.05
    print(f"PASS" if passed else "FAIL")
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    with open(out / "material_recovery.json", "w") as fh:
        json.dump({"history": [float(v) for v in ys],
                   "x_history": [x.tolist() for x in xs],
                   "x_final": [float(v) for v in recovered],
                   "r_final": r_final, "passed": bool(passed)}, fh, indent=2)
