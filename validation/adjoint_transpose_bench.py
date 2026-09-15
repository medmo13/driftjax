"""Dense vs equilibrated-banded transpose adjoint benchmark (Steps 6/12/13).

Phase-A decomposed timing (per the benchmark critique): each stage of the
banded backend is timed separately — dense-to-block extraction, row
equilibration, LAPACK band packing, the raw SciPy dgbsv kernel (prebuilt
band storage, no JAX), the full JAX wrapper, the blocks-direct API (no
dense extraction), and the residual check — against bare dense NumPy.
Timings are median+IQR over reps (warmup discarded), never min-only.

Devices: homojunction N=100, CdS/CdTe heterojunction N=100 (V=0.5 V),
3-layer perovskite N=15 (V=0.25 V; rank-deficient — both paths report
status FAILED rather than a zeros solution masquerading as a result).

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
from driftjax.numerics.banded_solve import (
    _blocks_to_lapack_banded,
    adjoint_banded_solve,
    adjoint_banded_solve_blocks,
    extract_blocks,
)
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


def _stats(ts):
    """Median + IQR over reps (warmup already discarded by caller)."""
    a = np.asarray(ts, dtype=float)
    return {
        "median_s": float(np.median(a)),
        "iqr_s": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "n": int(a.size),
        "all_s": [float(x) for x in a],
    }


def _time_fn(fn, reps=9):
    fn()  # warmup, discarded
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return _stats(ts)


def bench_case(Jn, g, label, reps=9):
    from scipy.linalg import solve_banded as _sb

    n = Jn.shape[0]
    gn = np.asarray(g, dtype=float)
    JeT, DrT = sc_mod.equilibrate(Jn.T)
    try:
        lam_ref, ref_floor = sc_mod.lu_longdouble(JeT, DrT @ gn)
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
        return float(np.linalg.norm(Jn.T @ lam - gn) / (np.linalg.norm(gn) + 1e-30))

    rec = {
        "label": label,
        "n": n,
        "cond": float(np.linalg.cond(Jn)),
        "ref_floor_adj": ref_floor,
        "ref_ok": ref_ok,
        **({} if ref_ok else {"ref_error": ref_error}),
    }

    # --- dense NumPy (bare kernel; the production CPU equivalent) ---
    try:
        lam_dense = np.linalg.solve(Jn.T, gn)
        rec["dense"] = {
            "status": "OK",
            "timing": _time_fn(lambda: np.linalg.solve(Jn.T, gn), reps),
            "backward": bwd_res(lam_dense),
            "forward_err": fwd_err(lam_dense),
        }
    except Exception as e:
        rec["dense"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}

    # --- Phase-A decomposition of the banded backend (numpy blocks) ---
    A_np, B_np, C_np = (np.asarray(t) for t in extract_blocks(jnp.asarray(Jn)))
    decomp = {}
    decomp["extract_dense_to_blocks"] = _time_fn(lambda: extract_blocks(jnp.asarray(Jn)), reps)
    s = np.max(np.abs(Jn), axis=1)
    s[~np.isfinite(s) | (s == 0)] = 1.0
    decomp["equilibrate_dense"] = _time_fn(lambda: np.diag(1.0 / s), reps)
    # Prebuilt band storage of the TRANSPOSE, then the RAW LAPACK kernel
    # alone (no JAX, no packing, no residual): the intrinsic dgbsv number.
    At = A_np.transpose(0, 2, 1)
    Ct = C_np.transpose(0, 2, 1)
    Bt = B_np.transpose(0, 2, 1)
    ab_t = np.asarray(
        _blocks_to_lapack_banded(jnp.asarray(At), jnp.asarray(Ct), jnp.asarray(Bt))[0]
    )
    decomp["pack_bands_jax"] = _time_fn(
        lambda: _blocks_to_lapack_banded(jnp.asarray(At), jnp.asarray(Ct), jnp.asarray(Bt)), reps
    )
    try:
        lam_raw = _sb((5, 5), ab_t, gn)
        decomp["raw_lapack_dgbsv"] = _time_fn(lambda: _sb((5, 5), ab_t, gn), reps)
        decomp["raw_lapack_backward"] = bwd_res(lam_raw)
        decomp["raw_lapack_forward_err"] = fwd_err(lam_raw)
    except Exception as e:
        decomp["raw_lapack_dgbsv"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
    if rec["dense"].get("status") == "OK":
        lam_d = np.linalg.solve(Jn.T, gn)
        decomp["residual_dense_check"] = _time_fn(lambda: Jn.T @ lam_d - gn, reps)
    else:
        decomp["residual_dense_check"] = None
    rec["decomposition"] = decomp

    # --- full JAX wrapper (legacy dense-J API) ---
    try:
        lam_j, fb = adjoint_banded_solve(jnp.asarray(Jn), jnp.asarray(gn), tol=1e-8)
        lam_j = np.asarray(lam_j)
        fb = bool(fb)
        t_wrap = _time_fn(
            lambda: adjoint_banded_solve(jnp.asarray(Jn), jnp.asarray(gn), tol=1e-8), reps
        )
        rec["banded_wrapper"] = {
            "status": "FAILED" if fb else "OK",
            "timing": t_wrap,
            "used_fallback": fb,
            **(
                {"backward": bwd_res(lam_j), "forward_err": fwd_err(lam_j)}
                if not fb
                else {"note": "fallback zeros are a failure marker, not a solution"}
            ),
        }
    except Exception as e:
        rec["banded_wrapper"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}

    # --- Phase-B blocks-direct API (no dense extraction) ---
    try:
        lam_b, fbb = adjoint_banded_solve_blocks(
            jnp.asarray(A_np), jnp.asarray(B_np), jnp.asarray(C_np), jnp.asarray(gn), tol=1e-8
        )
        lam_b = np.asarray(lam_b)
        fbb = bool(fbb)
        t_blocks = _time_fn(
            lambda: adjoint_banded_solve_blocks(
                jnp.asarray(A_np),
                jnp.asarray(B_np),
                jnp.asarray(C_np),
                jnp.asarray(gn),
                tol=1e-8,
            ),
            reps,
        )
        rec["banded_blocks_direct"] = {
            "status": "FAILED" if fbb else "OK",
            "timing": t_blocks,
            "used_fallback": fbb,
            **(
                {"backward": bwd_res(lam_b), "forward_err": fwd_err(lam_b)}
                if not fbb
                else {"note": "fallback zeros are a failure marker, not a solution"}
            ),
        }
    except Exception as e:
        rec["banded_blocks_direct"] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"}
    return rec


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


def _summarize_case(c):
    d, w, b = c["dense"], c["banded_wrapper"], c["banded_blocks_direct"]
    parts = [f"{c['label']} cond={c['cond']:.2e}"]
    parts.append(
        f"dense[{d.get('status')}]="
        + (
            f"{d['timing']['median_s'] * 1e3:.2f}ms fwd={d['forward_err']:.2e}"
            if d.get("status") == "OK"
            else f"{d.get('error')}"
        )
    )
    for name, r in (("wrap", w), ("blocks", b)):
        parts.append(
            f"{name}[{r.get('status')}]="
            + (
                f"{r['timing']['median_s'] * 1e3:.2f}ms fwd={r['forward_err']} fb={r['used_fallback']}"
                if r.get("status") == "OK"
                else f"{r.get('error', r.get('note'))}"
            )
        )
    return " ".join(parts)


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
        print(_summarize_case(out["cases"][label]), flush=True)
    # N-scaling arm (lighter extraction; wall times + fallback flags).
    out["scaling"] = {}
    for label, fn, v in (
        ("homojunction_N200", lambda: _homo(n_points=200), 0.5),
        ("homojunction_N400", lambda: _homo(n_points=400), 0.5),
        ("heterojunction_N200", lambda: ex2_device(n_points=200), 0.5),
    ):
        t0 = time.perf_counter()
        Jn = _device_at_bias_light(fn, v)
        gn = np.ones(Jn.shape[0])
        A_np, B_np, C_np = (np.asarray(t) for t in extract_blocks(jnp.asarray(Jn)))
        try:
            td = _time_fn(lambda Jn=Jn, gn=gn: np.linalg.solve(Jn.T, gn), 5)
            dense_rec = {"status": "OK", "timing": td}
        except Exception as e:
            dense_rec = {"status": "FAILED", "error": f"{type(e).__name__}"}
        try:
            lam_j, fbj = adjoint_banded_solve(jnp.asarray(Jn), jnp.asarray(gn), tol=1e-8)
            tw = _time_fn(
                lambda Jn=Jn, gn=gn: adjoint_banded_solve(
                    jnp.asarray(Jn), jnp.asarray(gn), tol=1e-8
                ),
                5,
            )
            wrap_rec = {"status": "FAILED" if bool(fbj) else "OK", "timing": tw}
        except Exception as e:
            wrap_rec = {"status": "FAILED", "error": f"{type(e).__name__}"}
        try:
            lam_b, fbb = adjoint_banded_solve_blocks(
                jnp.asarray(A_np), jnp.asarray(B_np), jnp.asarray(C_np), jnp.asarray(gn), tol=1e-8
            )
            tb = _time_fn(
                lambda A_np=A_np, B_np=B_np, C_np=C_np, gn=gn: adjoint_banded_solve_blocks(
                    jnp.asarray(A_np),
                    jnp.asarray(B_np),
                    jnp.asarray(C_np),
                    jnp.asarray(gn),
                    tol=1e-8,
                ),
                5,
            )
            blocks_rec = {
                "status": "FAILED" if bool(fbb) else "OK",
                "timing": tb,
                "used_fallback": bool(fbb),
            }
        except Exception as e:
            blocks_rec = {"status": "FAILED", "error": f"{type(e).__name__}"}
        out["scaling"][label] = {
            "n": int(Jn.shape[0]),
            "cond": float(np.linalg.cond(Jn)),
            "total_wall_s": round(time.perf_counter() - t0, 1),
            "dense": dense_rec,
            "banded_wrapper": wrap_rec,
            "banded_blocks_direct": blocks_rec,
        }
        print(
            f"{label} cond={out['scaling'][label]['cond']:.2e} "
            f"dense={dense_rec.get('timing', {}).get('median_s')} "
            f"wrap={wrap_rec.get('timing', {}).get('median_s')} "
            f"blocks={blocks_rec.get('timing', {}).get('median_s')}",
            flush=True,
        )
    RECORD.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {RECORD}")


if __name__ == "__main__":
    main()
