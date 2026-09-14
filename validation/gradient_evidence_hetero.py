"""Heterojunction adjoint evidence: multi-direction Taylor remainders, two meshes.

Device: CdS/CdTe heterojunction (same materials/geometry as
examples.support.ex2_device). Design vector (differentiable, builder-threaded):

    x = [log10(window donor doping), log10(absorber |acceptor| doping),
         absorber thickness scale factor]

Objective: power density at a fixed bias index near V*=0.7 V (fixed index
keeps the objective smooth; the index is resolved once per mesh at x0,
outside any differentiation).

For each mesh (N=200, 500) and each coordinate direction: adjoint
directional derivative vs central FD over a step-size sequence, plus the
directional Taylor remainder. Writes JSON to docs/paper/records/.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftjax as dj  # noqa: E402  (sys.path bootstrap above)

VSTAR = 0.7
N_STEPS = 12
H_SEQ = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 1e-6, 1e-7]
X0 = np.array([17.0, 15.0, 1.0])  # log10 N_D,win, log10 N_A,abs, thickness scale
BASE_ABS_W = 4e-4  # cm, matches ex2_device absorber width


def device_from_x(x, n_points):
    CdS = dj.material(
        Nc=2.2e18,
        Nv=1.8e19,
        Eg=2.4,
        eps=10,
        Et=0,
        mn=100,
        mp=25,
        tn=1e-8,
        tp=1e-13,
        Chi=4.0,
        A=1e4,
    )
    CdTe = dj.material(
        Nc=8e17,
        Nv=1.8e19,
        Eg=1.5,
        eps=9.4,
        Et=0,
        mn=320,
        mp=40,
        tn=5e-9,
        tp=5e-9,
        Chi=3.9,
        A=1e4,
    )
    return dj.Device(
        n_points=n_points,
        layers=[
            (2.5e-6, CdS, 10.0 ** x[0]),
            (x[2] * BASE_ABS_W, CdTe, -(10.0 ** x[1])),
        ],
        Snl=1.16e7,
        Snr=1.16e7,
        Spl=1.16e7,
        Spr=1.16e7,
    )


def main():
    out = {
        "VSTAR": VSTAR,
        "n_steps": N_STEPS,
        "h_sequence": H_SEQ,
        "x0": X0.tolist(),
        "coords": ["log10_window_doping", "log10_absorber_doping", "absorber_thickness_scale"],
        "meshes": {},
    }
    for n in (200, 500):
        dev0 = device_from_x(jnp.asarray(X0), n)
        sol0 = dj.simulate(dev0, dj.Sweep(vmax=0.9, n_steps=N_STEPS))
        idx = int(np.argmin(np.abs(np.asarray(sol0.voltages) - VSTAR)))

        def f(x, n_points=n, idx=idx):
            sol = dj.simulate(device_from_x(x, n_points), dj.Sweep(vmax=0.9, n_steps=N_STEPS))
            return -jnp.asarray(sol.voltages)[idx] * jnp.asarray(sol.current)[idx]

        x0 = jnp.asarray(X0)
        f0 = float(f(x0))
        t0 = time.perf_counter()
        g = np.asarray(jax.grad(f)(x0))
        grad_wall = time.perf_counter() - t0
        mesh_rec = {
            "fixed_bias_index": idx,
            "f0": f0,
            "grad_wall_s": grad_wall,
            "g_adj": g.tolist(),
            "directions": {},
        }
        for a, name in enumerate(out["coords"]):
            e = np.zeros(3)
            e[a] = 1.0
            rows = []
            for h in H_SEQ:
                fp = float(f(x0 + h * e))
                fm = float(f(x0 - h * e))
                fd = (fp - fm) / (2 * h)
                taylor = abs(fp - f0 - h * g[a]) / (abs(h * g[a]) + 1e-30)
                rel_fd = abs(fd - g[a]) / (abs(g[a]) + 1e-30)
                rows.append({"h": h, "fd": fd, "rel_fd_err": rel_fd, "taylor_remainder": taylor})
            mesh_rec["directions"][name] = {"g_adj": float(g[a]), "steps": rows}
        out["meshes"][str(n)] = mesh_rec
    dest = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "paper"
        / "records"
        / "gradient_evidence_hetero.json"
    )
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}")
    for n, m in out["meshes"].items():
        for name, d in m["directions"].items():
            best = min(d["steps"], key=lambda r: r["rel_fd_err"])
            print(
                f"N={n} {name}: g_adj={d['g_adj']:.6e} "
                f"best_rel_fd={best['rel_fd_err']:.2e} at h={best['h']}"
            )


if __name__ == "__main__":
    main()
