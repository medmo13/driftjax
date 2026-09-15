"""Identifiability + Voc gradient evidence (referee Steps 9-matrix, 38).

(a) Sensitivity matrix S_ka = dJ(V_k)/dtheta_a at the heterojunction
    operating point (3 builder params, 11 biases): singular values and
    sigma_min/sigma_max ratio — the structural identifiability number.
(b) Voc FD-vs-adjoint on a homojunction thickness direction at N=100
    and N=200 (fills the Voc cell of the verification matrix).

Writes JSON to docs/paper/records/identifiability.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import driftjax as dj  # noqa: E402

BASE_ABS_W = 4e-4


def hetero_dev(x, n):
    CdS = dj.material(
        Nc=2.2e18, Nv=1.8e19, Eg=2.4, eps=10, Et=0, mn=100, mp=25,
        tn=1e-8, tp=1e-13, Chi=4.0, A=1e4,
    )
    CdTe = dj.material(
        Nc=8e17, Nv=1.8e19, Eg=1.5, eps=9.4, Et=0, mn=320, mp=40,
        tn=5e-9, tp=5e-9, Chi=3.9, A=1e4,
    )
    return dj.Device(
        n_points=n,
        layers=[(2.5e-6, CdS, 10.0 ** x[0]), (x[2] * BASE_ABS_W, CdTe, -(10.0 ** x[1]))],
        Snl=1.16e7, Snr=1.16e7, Spl=1.16e7, Spr=1.16e7,
    )


def homo_dev(s, n):
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100,
        tn=1e-8, tp=1e-8, A=1e4,
    )
    return dj.Device(
        n_points=n, layers=[(1e-4, mat, 1e17), (s * 1e-4, mat, -1e17)],
        Snl=1e7, Snr=0, Spl=0, Spr=1e7,
    )


def main():
    out = {}
    x0 = jnp.array([17.0, 15.0, 1.0])
    # (a) sensitivity SVD at N=100, 11 biases
    t0 = time.perf_counter()

    def currents(x):
        sol = dj.simulate(hetero_dev(x, 100), dj.Sweep(vmax=1.0, n_steps=11))
        return jnp.asarray(sol.current)

    S = np.asarray(jax.jacobian(currents)(x0))
    sv = np.linalg.svd(np.asarray(S), compute_uv=False)
    out["sensitivity"] = {
        "K_biases": 11, "P_params": 3,
        "singular_values": [float(v) for v in sv],
        "sigma_min_over_max": float(sv[-1] / sv[0]),
        "wall_s": time.perf_counter() - t0,
    }
    print("sv:", ["%.3e" % v for v in sv], flush=True)
    # (b) Voc check at two meshes
    out["voc"] = {}
    for n in (100, 200):
        f = lambda s: dj.simulate(homo_dev(s, n), dj.Sweep(vmax=1.1, n_steps=11)).voc
        g = float(jax.grad(f)(1.0))
        rows = []
        for h in (1e-2, 1e-3, 1e-4):
            fd = (float(f(1.0 + h)) - float(f(1.0 - h))) / (2 * h)
            rows.append({"h": h, "fd": fd,
                         "rel_err": abs(fd - g) / (abs(g) + 1e-30)})
        out["voc"][str(n)] = {"g_adj": g, "steps": rows}
        print(f"N={n} g={g:.4e} best_rel={min(r['rel_err'] for r in rows):.1e}", flush=True)
    dest = (Path(__file__).resolve().parent.parent / "docs" / "paper"
            / "records" / "identifiability.json")
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
