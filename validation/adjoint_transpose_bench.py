"""Dense vs equilibrated-banded transpose adjoint benchmark (Steps 6/12/13).

Compares the production dense-LU transpose solve against the opt-in
equilibrated banded transpose (adjoint_banded_solve, DRIFTJAX_BANDED_ADJOINT=1)
on saved device Jacobians, measuring wall time, backward residual, and
forward error against the longdouble reference from solver_causality.

Devices: homojunction N=100, CdS/CdTe heterojunction N=100 (V=0.5 V),
3-layer perovskite N=15 (V=0.25 V; rank-deficient — expected to route to
the dense fallback or report honestly if it cannot).

Writes docs/paper/records/adjoint_transpose_bench.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import solver_causality as sc_mod

import driftjax as dj
from driftjax.numerics.banded_solve import adjoint_banded_solve
from examples.support import ex2_device

RECORD = (
    Path(__file__).resolve().parent.parent
    / "docs"
    / "paper"
    / "records"
    / "adjoint_transpose_bench.json"
)


def _homo(n_points=100):
    return dj.Device(
        n_points=n_points,
        layers=[
            (
                1e-4,
                dj.material(
                    Chi=3.9,
                    Eg=1.5,
                    eps=9.4,
                    Nc=8e17,
                    Nv=1.8e19,
                    mn=100,
                    mp=100,
                    tn=1e-8,
                    tp=1e-8,
                    A=1e4,
                ),
                1e17,
            ),
            (
                1e-4,
                dj.material(
                    Chi=3.9,
                    Eg=1.5,
                    eps=9.4,
                    Nc=8e17,
                    Nv=1.8e19,
                    mn=100,
                    mp=100,
                    tn=1e-8,
                    tp=1e-8,
                    A=1e4,
                ),
                -1e17,
            ),
        ],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def _perovskite3():
    mt = dj.material(Eg=1.6, Chi=3.9, eps=20.0, Nc=1e18, Nv=1e18, mn=100.0, mp=100.0, A=2e4)
    mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
    return dj.Device(
        n_points=15,
        layers=[(2e-5, mt, 1e18), (6e-5, mat, -1e18), (2e-5, mat, 1e18)],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )


def bench_case(Jn, g, label, reps=5):
    n = Jn.shape[0]
    JeT, DrT = sc_mod.equilibrate(Jn.T)
    try:
        lam_ref, ref_floor = sc_mod.lu_longdouble(JeT, DrT @ np.asarray(g))
        lam_ref = np.asarray(lam_ref, dtype=float)
        ref_ok = True
    except Exception as e:  # singular transpose: no reference computable
        lam_ref, ref_floor, ref_ok = None, None, False
        ref_error = f"{type(e).__name__}: {e}"

    def fwd_err(lam):
        if not ref_ok:
            return None
        lam = np.asarray(lam, dtype=float)
        return float(np.linalg.norm(lam - lam_ref) / (np.linalg.norm(lam_ref) + 1e-30))

    def bwd_res(lam):
        lam = np.asarray(lam, dtype=float)
        gg = np.asarray(g, dtype=float)
        return float(np.linalg.norm(Jn.T @ lam - gg) / (np.linalg.norm(gg) + 1e-30))

    # Dense production path (timed, best of reps after warmup).
    t_dense, lam_dense, dense_ok = None, None, True
    try:
        for _ in range(reps + 1):
            t0 = time.perf_counter()
            lam_dense = np.linalg.solve(Jn.T, np.asarray(g))
            dt = time.perf_counter() - t0
            t_dense = dt if t_dense is None else min(t_dense, dt)
    except Exception as e:
        dense_ok, dense_error = False, f"{type(e).__name__}: {e}"

    # Opt-in equilibrated banded transpose (timed likewise).
    t_band, lam_band, used_fb = None, None, None
    try:
        for _ in range(reps + 1):
            t0 = time.perf_counter()
            lam_j, fb = adjoint_banded_solve(jnp.asarray(Jn), jnp.asarray(np.asarray(g)), tol=1e-8)
            dt = time.perf_counter() - t0
            if t_band is None or dt < t_band:
                t_band, lam_band, used_fb = dt, np.asarray(lam_j), bool(fb)
        band_ok = True
    except Exception as e:
        band_ok, band_error = False, f"{type(e).__name__}: {e}"

    return {
        "label": label,
        "n": n,
        "cond": float(np.linalg.cond(Jn)),
        "ref_floor_adj": ref_floor,
        "ref_ok": ref_ok,
        **({} if ref_ok else {"ref_error": ref_error}),
        "dense": {
            "ok": dense_ok,
            **({} if dense_ok else {"error": dense_error}),
            **(
                {
                    "wall_s": t_dense,
                    "backward": bwd_res(lam_dense),
                    "forward_err": fwd_err(lam_dense),
                }
                if dense_ok
                else {}
            ),
        },
        "banded_equil": {
            "ok": band_ok,
            **({} if band_ok else {"error": band_error}),
            **(
                {
                    "wall_s": t_band,
                    "backward": bwd_res(lam_band),
                    "forward_err": fwd_err(lam_band),
                    "used_fallback": used_fb,
                }
                if band_ok
                else {}
            ),
        },
        "speedup_dense_over_banded": (
            (t_dense / t_band)
            if (t_dense is not None and t_band is not None and t_band > 0)
            else None
        ),
    }


def _device_at_bias_light(dev_fn, v, n_steps=6):
    """Cheaper Jacobian extraction than solver_causality.device_at_bias
    (n_steps=6 sweep instead of 11) for the N-scaling arm."""
    import jax

    from driftjax.numerics.residual import F_jacobian
    from driftjax.science.contacts import boundary_bias

    jax.config.update("jax_enable_x64", True)
    dev = dev_fn()
    sol = dj.simulate(dev, dj.Sweep(vmax=1.0, n_steps=n_steps))
    vv = np.asarray(sol.voltages)
    idx = int(np.argmin(np.abs(vv - v)))
    J = np.asarray(
        F_jacobian(sol.cell, boundary_bias(sol.cell, sol.voltages[idx]), sol.potentials[idx])
    )
    return J


def main():
    import jax

    jax.config.update("jax_enable_x64", True)
    out = {"environment": {"dtype": "float64", "x64": True}, "cases": {}}
    for label, fn, v in (
        ("homojunction_N100", _homo, 0.5),
        ("heterojunction_N100", lambda: ex2_device(n_points=100), 0.5),
        ("perovskite3_N15", _perovskite3, 0.25),
    ):
        J = sc_mod.device_at_bias(fn, v)
        g = np.ones(J.shape[0])
        t0 = time.perf_counter()
        out["cases"][label] = bench_case(J, g, label)
        out["cases"][label]["total_wall_s"] = round(time.perf_counter() - t0, 1)
        c = out["cases"][label]
        d, b = c["dense"], c["banded_equil"]
        print(
            f"{label} cond={c['cond']:.2e} "
            f"dense_ok={d['ok']} "
            + (
                f"dense_t={d['wall_s'] * 1e3:.2f}ms fwd={d['forward_err']:.2e} | "
                if d["ok"]
                else f"dense_err={d.get('error')} | "
            )
            + f"band_ok={b['ok']} "
            + (
                f"band_t={b['wall_s'] * 1e3:.2f}ms fwd={b['forward_err']} fb={b['used_fallback']}"
                if b["ok"]
                else f"band_err={b.get('error')}"
            ),
            flush=True,
        )
    # N-scaling arm: where does O(N^3) dense lose to the banded transpose?
    # (lighter n_steps=6 extraction; accuracy already established above,
    # here only wall times + fallback flags are recorded).
    out["scaling"] = {}
    for label, fn, v, _nn in (
        ("homojunction_N200", lambda: _homo(n_points=200), 0.5, 200),
        ("homojunction_N400", lambda: _homo(n_points=400), 0.5, 400),
        ("heterojunction_N200", lambda: ex2_device(n_points=200), 0.5, 200),
    ):
        t0 = time.perf_counter()
        Jn = _device_at_bias_light(fn, v)
        gn = np.ones(Jn.shape[0])
        td, dok = None, True
        try:
            for _ in range(4):
                t1 = time.perf_counter()
                np.linalg.solve(Jn.T, gn)
                dt = time.perf_counter() - t1
                td = dt if td is None else min(td, dt)
        except Exception as e:
            dok, derr = False, f"{type(e).__name__}"
        tb, fb, bok = None, None, True
        try:
            for _ in range(4):
                t1 = time.perf_counter()
                lam_j, fbj = adjoint_banded_solve(jnp.asarray(Jn), jnp.asarray(gn), tol=1e-8)
                dt = time.perf_counter() - t1
                if tb is None or dt < tb:
                    tb, fb = dt, bool(fbj)
        except Exception as e:
            bok, berr = False, f"{type(e).__name__}"
        rec = {
            "n": int(Jn.shape[0]),
            "cond": float(np.linalg.cond(Jn)),
            "total_wall_s": round(time.perf_counter() - t0, 1),
            "dense": {"ok": dok, **({"wall_s": td} if dok else {"error": derr})},
            "banded_equil": {
                "ok": bok,
                **({"wall_s": tb, "used_fallback": fb} if bok else {"error": berr}),
            },
        }
        if dok and bok and tb and tb > 0:
            rec["speedup_dense_over_banded"] = td / tb
        out["scaling"][label] = rec
        print(f"{label} cond={rec['cond']:.2e} dense={rec['dense']} banded={rec['banded_equil']}", flush=True)
    RECORD.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {RECORD}")


if __name__ == "__main__":
    main()
