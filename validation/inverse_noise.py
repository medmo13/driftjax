"""Practical identifiability probe: noisy IV recovery distribution (Obs 12).

Same 2-parameter (thickness um, log-doping) matched-model problem as
examples/research/13_target_iv_structure.py but reduced (N=200, 21-step
sweep) so that K=8 independent 5%-noise refits are affordable. Each
realization: Gaussian noise at 5% of |J| added to the target curve,
Nelder-Mead refit from the published start with capped budget. Reports
recovered-parameter mean/std, MSE distribution, and success rate
(thickness within 5% of truth). A probe of practical identifiability,
not a production fit. Writes JSON to docs/paper/records/inverse_noise.json.
"""
# ruff: noqa: E402 -- sys.path bootstrap precedes package imports

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftjax as dj
from driftjax.optimize.objectives import iv_mse

POINTS, STEPS, K, NOISE_FRAC, MAXITER = 200, 21, 8, 0.05, 60
TRUTH = np.array([1.8, 16.0])
START = np.array([0.85, 15.5])
BOUNDS = ((0.5, 3.0), (15.0, 17.0))


def device_from(x):
    material = dj.material(
        Chi=3.9,
        Eg=1.55,
        eps=12.0,
        Nc=2.2e19,
        Nv=2.2e19,
        mn=1000.0,
        mp=300.0,
        tn=1e-6,
        tp=1e-6,
        A=3e4,
    )
    return dj.Device(
        n_points=POINTS,
        layers=[(5e-6, material, 10.0 ** x[1]), (x[0] * 1e-4, material, -(10.0 ** x[1]))],
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )


def curve(x):
    sol = dj.simulate(device_from(jnp.asarray(x, dtype=float)), dj.Sweep(vmax=1.5, n_steps=STEPS))
    return np.asarray(sol.voltages), np.asarray(sol.currents)


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    _, target = curve(TRUTH)
    rng = np.random.default_rng(20260915)
    recs = []
    t0 = time.perf_counter()
    for k in range(K):
        noisy = target + rng.normal(size=target.shape) * NOISE_FRAC * np.abs(target)
        history = []

        def objective(x, _t=noisy, _h=history):  # noqa: B023
            val = float(iv_mse(jnp.asarray(curve(np.asarray(x))[1]), jnp.asarray(_t)))
            _h.append(val)
            return val

        res = dj.optimize.nelder_mead(objective, START, bounds=BOUNDS, maxiter=MAXITER)
        xf = np.asarray(res.x)
        recs.append(
            {
                "fitted_um": xf.tolist(),
                "thickness_rel_err": float(abs(xf[0] - TRUTH[0]) / TRUTH[0]),
                "final_mse": float(res.fun),
                "n_evals": len(history),
                "success": bool(abs(xf[0] - TRUTH[0]) / TRUTH[0] < 0.05),
            }
        )
        print(
            f"realization {k}: th={xf[0]:.4f} dopedec={xf[1]:.3f} "
            f"relerr={recs[-1]['thickness_rel_err']:.3e} evals={len(history)} "
            f"success={recs[-1]['success']}",
            flush=True,
        )
    th = np.array([r["fitted_um"][0] for r in recs])
    out = {
        "K": K,
        "noise_fraction": NOISE_FRAC,
        "maxiter": MAXITER,
        "truth_um": TRUTH.tolist(),
        "start_um": START.tolist(),
        "thickness_mean": float(th.mean()),
        "thickness_std": float(th.std()),
        "success_rate": float(np.mean([r["success"] for r in recs])),
        "wall_s": time.perf_counter() - t0,
        "realizations": recs,
    }
    dest = (
        Path(__file__).resolve().parent.parent / "docs" / "paper" / "records" / "inverse_noise.json"
    )
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
