"""SrJSu scaling diagnostic: which scaling fixes primal AND transpose? (SrJSu gate)

Tests the four systems J, SrJ, JSu, SrJSu on saved device Jacobians, where
Sr = diag(1/rowmax(J)) and Su = diag(1/colmax(J)) are diagonal
equilibrations. For each system measures kappa_2 plus, via dense LU and
pivoted banded solve, backward residual and forward error against the
longdouble reference -- for BOTH the primal (Jx=b) and transpose (J^Tl=g)
problems.

Gate verdict sought: if some scaling collapses kappa from 1e45 toward
tractable AND restores banded accuracy, the JAX-native solver spec gets
simpler; if nothing rescues the rank-deficient case, that negative result
is published and the native spec must include a robust pivot strategy.

Writes docs/paper/records/scaling_2x2.json.
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
from driftjax.numerics.banded_solve import banded_solve, extract_blocks
from examples.support import ex2_device

RECORD = Path(__file__).resolve().parent.parent / "docs" / "paper" / "records" / "scaling_2x2.json"


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


def _scales(Jn):
    """Sr (row) and Su (column) equilibration diagonals from dense J."""
    r = np.max(np.abs(Jn), axis=1)
    r[~np.isfinite(r) | (r == 0)] = 1.0
    c = np.max(np.abs(Jn), axis=0)
    c[~np.isfinite(c) | (c == 0)] = 1.0
    return np.diag(1.0 / r), np.diag(1.0 / c)


def _banded_of(M, rhs):
    n = M.shape[0] // 3
    A, B, C = (np.asarray(t) for t in extract_blocks(jnp.asarray(M)))
    return np.asarray(
        banded_solve(jnp.asarray(A), jnp.asarray(B), jnp.asarray(C), jnp.asarray(rhs).reshape(n, 3))
    ).reshape(-1)


def _banded_t_of(M, rhs):
    n = M.shape[0] // 3
    A, B, C = (np.asarray(t) for t in extract_blocks(jnp.asarray(M)))
    At, Ct, Bt = A.transpose(0, 2, 1), C.transpose(0, 2, 1), B.transpose(0, 2, 1)
    from driftjax.numerics.banded_solve import banded_solve as _bs

    return np.asarray(
        _bs(jnp.asarray(At), jnp.asarray(Ct), jnp.asarray(Bt), jnp.asarray(rhs).reshape(n, 3))
    ).reshape(-1)


def run_2x2(Jn, g, label):
    n = Jn.shape[0]
    Sr, Su = _scales(Jn)
    systems = {
        "J": Jn,
        "SrJ": Sr @ Jn,
        "JSu": Jn @ Su,
        "SrJSu": Sr @ Jn @ Su,
    }
    # Reference solutions (scaling-independent): primal x via equilibrated
    # longdouble; transpose lam likewise. May fail on singular systems.
    try:
        Je, Dr = sc_mod.equilibrate(Jn)
        x_ref, fwd_floor = sc_mod.lu_longdouble(Je, Dr @ gn(g))
        x_ref = np.asarray(x_ref, dtype=float)
        ref_ok = True
    except Exception as e:
        x_ref, ref_ok = None, False
        _, ref_err = None, f"{type(e).__name__}: {e}"
    try:
        JeT, DrT = sc_mod.equilibrate(Jn.T)
        lam_ref, adj_floor = sc_mod.lu_longdouble(JeT, DrT @ gn(g))
        lam_ref = np.asarray(lam_ref, dtype=float)
        ref_adj_ok = True
    except Exception as e:
        lam_ref, ref_adj_ok = None, False
        _, ref_adj_err = None, f"{type(e).__name__}: {e}"

    rec = {"label": label, "n": n, "ref_ok": ref_ok, "ref_adj_ok": ref_adj_ok}
    if not ref_ok:
        rec["ref_error"] = ref_err
    if not ref_adj_ok:
        rec["ref_adj_error"] = ref_adj_err
    arms = {}
    for name, M in systems.items():
        # Primal: Mx=y solved directly (y=b for J; unscaling per arm below).
        # To keep every arm solving for the SAME x, use y = M @ x_ref-free
        # formulation: solve M z = rhs with rhs = M @ x_probe? No — honest
        # approach: solve the scaled systems for their own unknowns and
        # unscale to x, then compare against x_ref.
        arm = {"kappa": float(np.linalg.cond(M))}
        # Dense primal.
        try:
            if name == "J":
                xd = np.linalg.solve(M, gn(g))
            elif name == "SrJ":
                xd = np.linalg.solve(M, Sr @ gn(g))
            elif name == "JSu":
                yd = np.linalg.solve(M, gn(g))
                xd = Su @ yd
            else:
                yd = np.linalg.solve(M, Sr @ gn(g))
                xd = Su @ yd
            arm["dense_backward"] = float(
                np.linalg.norm(Jn @ xd - gn(g)) / (np.linalg.norm(gn(g)) + 1e-30)
            )
            arm["dense_forward_err"] = (
                float(np.linalg.norm(xd - x_ref) / (np.linalg.norm(x_ref) + 1e-30))
                if ref_ok
                else None
            )
        except Exception as e:
            arm["dense_status"] = f"FAILED: {type(e).__name__}"
        # Banded primal (blocks of the SCALED matrix; unscale as above).
        try:
            if name == "J":
                xb = _banded_of(M, gn(g))
            elif name == "SrJ":
                xb = _banded_of(M, Sr @ gn(g))
            elif name == "JSu":
                xb = Su @ _banded_of(M, gn(g))
            else:
                xb = Su @ _banded_of(M, Sr @ gn(g))
            arm["banded_backward"] = float(
                np.linalg.norm(Jn @ xb - gn(g)) / (np.linalg.norm(gn(g)) + 1e-30)
            )
            arm["banded_forward_err"] = (
                float(np.linalg.norm(xb - x_ref) / (np.linalg.norm(x_ref) + 1e-30))
                if ref_ok
                else None
            )
        except Exception as e:
            arm["banded_status"] = f"FAILED: {type(e).__name__}"
        # Transpose arms: transform the SYSTEM consistently, not just the
        # matrix (an earlier revision used rhs Sr g for the SrJ arm, which
        # targets the wrong system — caught because its forward error came
        # out ~1e9 instead of ~1e-13). See arm table below for derivations.
        # Transpose arms (derivations with Sr, Su nonsingular diagonal,
        # hence symmetric). Target in all lam-comparable arms: J^T lam = g.
        #   J:     J^T lam = g             -> lam = w,       c = g
        #   SrJ:   (SrJ)^T mu = J^TSr mu = g -> lam = Sr mu, c = g
        #   JSu:   (JSu)^T z = Su J^T z = g  -> J^T z = Su^-1 g: DIFFERENT
        #          rhs, so residual-vs-own-rhs only, no lam comparison.
        #   SrJSu: (SrJSu)^T w = Su J^TSr w = Su g -> lam = Sr w, c = Su g
        #          (left-multiply by Su^-1: J^TSr w = g).
        if name == "J":
            ct, lam_of, compares_lam = gn(g), lambda w: w, True
        elif name == "SrJ":
            ct, lam_of, compares_lam = gn(g), lambda w: Sr @ w, True
        elif name == "JSu":
            ct, lam_of, compares_lam = gn(g), lambda w: w, False
        else:
            ct, lam_of, compares_lam = Su @ gn(g), lambda w: Sr @ w, True
        try:
            wb = _banded_t_of(M, ct)
            lam_b = lam_of(wb)
            arm["transpose_banded_backward_vs_own_rhs"] = float(
                np.linalg.norm(M.T @ wb - ct) / (np.linalg.norm(ct) + 1e-30)
            )
            if compares_lam and ref_adj_ok:
                arm["transpose_banded_forward_err_vs_lam"] = float(
                    np.linalg.norm(lam_b - lam_ref) / (np.linalg.norm(lam_ref) + 1e-30)
                )
        except Exception as e:
            arm["transpose_banded_status"] = f"FAILED: {type(e).__name__}"
        arms[name] = arm
    rec["arms"] = arms
    return rec


def gn(g):
    return np.asarray(g, dtype=float)


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
        out["cases"][label] = run_2x2(J, g, label)
        out["cases"][label]["total_wall_s"] = round(time.perf_counter() - t0, 1)
        c = out["cases"][label]
        print(f"=== {label} ===", flush=True)
        for an, a in c["arms"].items():
            print(
                f"  {an}: kappa={a.get('kappa', float('nan')):.2e} "
                f"dense_bwd={a.get('dense_backward')} banded_bwd={a.get('banded_backward')} "
                f"t_banded_own={a.get('transpose_banded_backward_vs_own_rhs')}",
                flush=True,
            )
    RECORD.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {RECORD}")


if __name__ == "__main__":
    main()
