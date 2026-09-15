"""Four-way solver causality + longdouble reference (referee Obs 5, Tier 1D).

Question: is row-scaling imbalance the cause of structured-solve failure,
or one amplifier among several? Four arms on the SAME saved Jacobian/RHS:

  1. BT-raw:      historical unpivoted block-Thomas (reimplemented inline,
                  clearly marked; deleted from production in v0.1.17)
  2. BT-equil:    unpivoted block-Thomas on row-equilibrated system
  3. banded-raw:  pivoted LAPACK dgbsv (production forward default)
  4. banded-equil: dgbsv on row-equilibrated system
  (+ dense LU control)

Reference: hand-rolled partial-pivot LU in np.longdouble (64-bit
mantissa; numpy/scipy linalg refuse float128, and no extra dependency
is introduced). Forward error is measured against this reference, not
just the backward residual.

Devices: homojunction + CdS/CdTe heterojunction, N=100, V=0.5 V bias.
Writes JSON to docs/paper/records/solver_causality.json.
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
from driftjax.numerics.banded_solve import banded_solve, extract_blocks
from driftjax.numerics.residual import F_jacobian
from driftjax.science.contacts import boundary_bias
from examples.support import ex2_device

DET_TOL = 1e-30


def _inv3(A):
    a, b, c = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d_, e, f = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g, h, i_ = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    det = a * (e * i_ - f * h) - b * (d_ * i_ - f * g) + c * (d_ * h - e * g)
    det_safe = np.where(np.abs(det) < DET_TOL, np.where(det < 0.0, -DET_TOL, DET_TOL), det)
    invdet = 1.0 / det_safe
    return (
        np.stack(
            [
                (e * i_ - f * h),
                (c * h - b * i_),
                (b * f - c * e),
                (f * g - d_ * i_),
                (a * i_ - c * g),
                (c * d_ - a * f),
                (d_ * h - e * g),
                (b * g - a * h),
                (a * e - b * d_),
            ],
            axis=-1,
        ).reshape(A.shape)
        * invdet[..., None, None]
    )


def bt_solve_numpy(A, B, C, b):
    """Historical unpivoted block-Thomas in float64 numpy (experiment only)."""
    n = A.shape[0]
    Ap, bp = A.copy(), b.copy()
    for i in range(1, n):
        W = C[i - 1] @ _inv3(Ap[i - 1])
        Ap[i] = A[i] - W @ B[i - 1]
        bp[i] = b[i] - W @ bp[i - 1]
    Ap_inv = _inv3(Ap)
    x = np.zeros_like(b)
    x[n - 1] = (Ap_inv[n - 1] @ bp[n - 1][..., None])[..., 0]
    for j in range(n - 1):
        i = n - 2 - j
        x[i] = (Ap_inv[i] @ (bp[i] - B[i] @ x[i + 1])[..., None])[..., 0]
    return x


def lu_longdouble(A, b, n_refine=6):
    """High-precision reference via longdouble-residual iterative refinement.

    Direct hand-rolled longdouble LU was tried and rejected: residuals
    plateaued at ~1e-16 on test systems. Instead, a float64 LU
    factorisation is refined with residuals evaluated in np.longdouble
    (verified true 80-bit matvecs via a cancellation probe). The floor is
    ~cond*u_LD (cancellation in the longdouble residual amplified by the
    float64 correction solve); callers pass equilibrated systems
    (cond ~ 1e4) for a ~1e-15 reference, ~100x below the float64 residuals
    under study. Self-checks and reports the achieved floor.
    """
    Af = np.array(A, dtype=float)
    AL = np.array(A, dtype=np.longdouble)
    bL = np.array(b, dtype=np.longdouble)
    x = np.array(np.linalg.solve(Af, np.asarray(b, dtype=float)), dtype=np.longdouble)
    for _ in range(n_refine):
        r = bL - AL @ x
        d = np.linalg.solve(Af, np.asarray(r, dtype=float))
        x = x + np.array(d, dtype=np.longdouble)
    # RELATIVE residual: equilibrated right-hand sides carry norms ~1e26
    # from the Dr scaling, so absolute residuals are meaningless here.
    denom = float(np.max(np.abs(bL)))
    r = np.max(np.abs(AL @ x - bL)) / (denom + 1e-300)
    assert float(r) < 1e-16, f"longdouble reference failed: rel-resid {float(r):.2e}"
    return np.asarray(x, dtype=float), float(r)


def equilibrate(J):
    """Row equilibration. Returns (Je, Dr) with Je = Dr @ J, Dr = diag(1/rowmax).

    Forward: Je x = Dr b has the SAME solution as J x = b.
    Adjoint: solving Je^T y = b gives y with lam = Dr @ y solving J^T lam = b.
    """
    Jn = np.asarray(J, dtype=float)
    s = np.max(np.abs(Jn), axis=1)
    s[~np.isfinite(s) | (s == 0)] = 1.0
    Dr = np.diag(1.0 / s)
    return Dr @ Jn, Dr


def _banded_t_of(M, rhs, unscale):
    """Banded transpose solve on dense M with diagonal unscaling of y."""
    from driftjax.numerics.banded_solve import banded_transpose as _bt2

    AB, BB, CB = (np.asarray(t) for t in extract_blocks(jnp.asarray(M)))
    y = np.asarray(_bt2(jnp.asarray(AB), jnp.asarray(BB), jnp.asarray(CB),
                        jnp.asarray(rhs)))
    return unscale * y


def run_case(Jn, g, label):
    n = Jn.shape[0] // 3
    A, B, C = (np.asarray(t) for t in extract_blocks(jnp.asarray(Jn)))
    bn = np.asarray(g)
    Je, Dr = equilibrate(Jn)
    # Reference on equilibrated systems (tamed LU growth). Forward:
    # Je x = Dr b has the same solution as J x = b. Adjoint: the
    # transpose needs its OWN equilibration (transposing unbalances
    # rows again): JeT y = DrT b with JeT = DrT J^T gives y = lam.
    x_ref, ref_floor_fwd = lu_longdouble(Je, Dr @ bn)
    JeT, DrT = equilibrate(Jn.T)
    lam_ref, ref_floor_adj = lu_longdouble(JeT, DrT @ bn)
    rec_floor = {"fwd": ref_floor_fwd, "adj": ref_floor_adj}
    Ae, Be, Ce = (np.asarray(t) for t in extract_blocks(jnp.asarray(Je)))
    be = Dr @ bn  # scaled rhs: Je x = be <=> J x = bn

    def metrics(x):
        x = np.asarray(x, dtype=float)
        backward = np.linalg.norm(Jn @ x - bn) / (np.linalg.norm(bn) + 1e-30)
        forward = np.linalg.norm(x - x_ref) / (np.linalg.norm(x_ref) + 1e-30)
        return backward, forward

    def metrics_t(lam):
        lam = np.asarray(lam, dtype=float)
        backward = np.linalg.norm(Jn.T @ lam - bn) / (np.linalg.norm(bn) + 1e-30)
        forward = np.linalg.norm(lam - lam_ref) / (np.linalg.norm(lam_ref) + 1e-30)
        return backward, forward

    rec = {"label": label, "cond": float(np.linalg.cond(Jn)), "ref_floor": rec_floor}
    rec["fwd_bt_raw"] = metrics(bt_solve_numpy(A, B, C, bn.reshape(n, 3)).reshape(-1))
    rec["fwd_bt_equil"] = metrics(bt_solve_numpy(Ae, Be, Ce, be.reshape(n, 3)).reshape(-1))
    rec["fwd_banded_raw"] = metrics(
        np.asarray(
            banded_solve(
                jnp.asarray(A), jnp.asarray(B), jnp.asarray(C), jnp.asarray(bn).reshape(n, 3)
            )
        ).reshape(-1)
    )
    rec["fwd_banded_equil"] = metrics(
        np.asarray(
            banded_solve(
                jnp.asarray(Ae), jnp.asarray(Be), jnp.asarray(Ce), jnp.asarray(be).reshape(n, 3)
            )
        ).reshape(-1)
    )
    rec["fwd_dense"] = metrics(np.linalg.solve(Jn, bn))
    # Column and two-sided scaling (diagonal scalings preserve the block
    # pattern, so the banded solve applies; unscaling derived in comments).
    # Forward: J Dc y = b -> x = Dc y; Dr J Dc y = Dr b -> x = Dc y.
    # Adjoint: (J Dc)^T z = Dc g -> z = lam directly;
    #   (Dr J Dc)^T z = Dc g -> lam = Dr z.
    sc = np.max(np.abs(Jn), axis=0)
    sc[~np.isfinite(sc) | (sc == 0)] = 1.0
    Dc = np.diag(1.0 / sc)

    def _banded_of(M, rhs):
        AB, BB, CB = (np.asarray(t) for t in extract_blocks(jnp.asarray(M)))
        return np.asarray(
            banded_solve(jnp.asarray(AB), jnp.asarray(BB), jnp.asarray(CB),
                         jnp.asarray(rhs).reshape(n, 3))
        ).reshape(-1)

    rec["fwd_banded_col"] = metrics(Dc @ _banded_of(Jn @ Dc, bn))
    rec["fwd_banded_both"] = metrics(Dc @ _banded_of(Dr @ Jn @ Dc, Dr @ bn))
    rec["adj_banded_col"] = metrics_t(_banded_t_of(Jn @ Dc, Dc @ bn, np.ones(n * 3)))
    rec["adj_banded_both"] = metrics_t(_banded_t_of(Dr @ Jn @ Dc, Dc @ bn, Dr.diagonal()))
    # adjoint arms via transposed blocks
    At, Ct, Bt = A.transpose(0, 2, 1), C.transpose(0, 2, 1), B.transpose(0, 2, 1)
    rec["adj_bt_raw"] = metrics_t(bt_solve_numpy(At, Ct, Bt, bn.reshape(n, 3)).reshape(-1))
    Ate, Cte, Bte = Ae.transpose(0, 2, 1), Ce.transpose(0, 2, 1), Be.transpose(0, 2, 1)
    # Scaled adjoint: Je^T y = bn, then lam = Dr @ y solves J^T lam = bn.
    rec["adj_bt_equil"] = metrics_t(
        Dr @ bt_solve_numpy(Ate, Cte, Bte, bn.reshape(n, 3)).reshape(-1)
    )
    from driftjax.numerics.banded_solve import banded_transpose

    rec["adj_banded_raw"] = metrics_t(
        np.asarray(
            banded_transpose(jnp.asarray(A), jnp.asarray(B), jnp.asarray(C), jnp.asarray(bn))
        )
    )
    rec["adj_banded_equil"] = metrics_t(
        Dr
        @ np.asarray(
            banded_transpose(jnp.asarray(Ae), jnp.asarray(Be), jnp.asarray(Ce), jnp.asarray(bn))
        )
    )
    rec["adj_dense"] = metrics_t(np.linalg.solve(Jn.T, bn))
    return rec


def device_at_bias(dev_fn, v):
    import jax

    jax.config.update("jax_enable_x64", True)
    dev = dev_fn()
    sol = dj.simulate(dev, dj.Sweep(vmax=1.0, n_steps=11))
    vv = np.asarray(sol.voltages)
    idx = int(np.argmin(np.abs(vv - v)))
    J = np.asarray(
        F_jacobian(sol.cell, boundary_bias(sol.cell, sol.voltages[idx]), sol.potentials[idx])
    )
    return J


def main():
    import jax

    jax.config.update("jax_enable_x64", True)

    def homo():
        return dj.Device(
            n_points=100,
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

    out = {"arms": ["backward_resid", "forward_err_vs_longdouble"]}
    for label, fn in (
        ("homojunction_N100", homo),
        ("heterojunction_N100", lambda: ex2_device(n_points=100)),
    ):
        J = device_at_bias(fn, 0.5)
        g = np.ones(J.shape[0])
        t0 = time.perf_counter()
        out[label] = run_case(J, g, label)
        print(
            f"{label} cond={out[label]['cond']:.2e}",
            f"wall={time.perf_counter() - t0:.0f}s",
            flush=True,
        )
    dest = (
        Path(__file__).resolve().parent.parent
        / "docs"
        / "paper"
        / "records"
        / "solver_causality.json"
    )
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
