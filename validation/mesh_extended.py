"""Extended mesh refinement: homojunction to N=2000 + Richardson (Obs 7).

Runs the ex1 homojunction at N = 500, 1000, 2000 (uniform, interface at a
node by construction: 2x1um symmetric junction) and reports Jsc, Voc, FF,
PCE plus Richardson-extrapolated PCE from the three levels. Appends to
docs/paper/records/mesh_extended.json (ex01 record holds N<=1000).
"""
# ruff: noqa: E402 -- sys.path bootstrap precedes package imports

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import driftjax as dj
from examples.support import ex1_device


def run(n):
    t0 = time.perf_counter()
    sol = dj.simulate(ex1_device(n_points=n), dj.Sweep(vmax=1.1, n_steps=41))
    return {
        "jsc": float(abs(sol.jsc)),
        "voc": float(sol.voc),
        "ff": float(sol.ff),
        "pce_pct": float(sol.efficiency) * 100.0,
        "wall_s": time.perf_counter() - t0,
    }


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    import os

    out = {}
    ns = (500, 1000, 2000) if not os.environ.get("N4000_ONLY") else (4000,)
    if os.environ.get("N4000_ONLY"):
        prev = json.load(
            open(
                Path(__file__).resolve().parent.parent
                / "docs" / "paper" / "records" / "mesh_extended.json"
            )
        )
        out.update(prev)
    for n in ns:
        out[str(n)] = run(n)
        print(n, {k: round(v, 6) for k, v in out[str(n)].items()}, flush=True)
    # Richardson extrapolation for PCE assuming order p from the two
    # successive ratios (uniform refinement factor 2).
    e = np.array([out["500"]["pce_pct"], out["1000"]["pce_pct"], out["2000"]["pce_pct"]])
    r1, r2 = e[0] - e[1], e[1] - e[2]
    p = float(np.log2(abs(r1 / r2))) if r2 != 0 else float("nan")
    out["richardson"] = {
        "observed_order": p,
        "extrapolated_pce_pct": float(e[2] + r2 / (2**p - 1)) if np.isfinite(p) else None,
    }
    print(out["richardson"], flush=True)
    dest = (
        Path(__file__).resolve().parent.parent / "docs" / "paper" / "records" / "mesh_extended.json"
    )
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
