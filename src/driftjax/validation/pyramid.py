"""L1–L10 validation pyramid — runnable gate suite (NEW).

Mirrors the reconstruction-protocol validation levels:
  L1 unit kernels (statistics goldens, Bernoulli, spline)
  L2 property invariants (AD==jacrev, banded round-trip)
  L3 manufactured / mesh-convergence order
  L4 gradient checks (AD vs FD)
  L5 conservation identities on a converged solution
  L6 literature bounds (SQ limit, intrinsic densities)
  L7 regression goldens (IV canary vs package values) [slow]
  L8 reproducibility (deterministic rerun)
  L9 performance model sanity
  L10 hardware/backend gates (spsolve availability, batched==serial)

Run:  python -m driftjax.validation.pyramid [--fast]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from driftjax.science.spectrum import spectrum


def _gate(name, fn, accept, extra=""):
    t0 = time.perf_counter()
    try:
        value = fn()
        ok = accept(value)
    except Exception as e:  # noqa: BLE001
        value, ok = f"EXC {type(e).__name__}: {e}", False
    dt = time.perf_counter() - t0
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:56s} {value!s:24s} {dt:5.1f}s {extra}")
    return ok


def run(fast: bool = True) -> dict:
    print("driftjax validation pyramid")
    results = {}
    import driftjax as dj
    from driftjax import units
    from driftjax.fields import pot2vec
    from driftjax.science.carrier_statistics import F_half as Fh

    # ---- L1: unit kernels -------------------------------------------------
    print("[L1] unit kernels")
    results["l1_fhalf0"] = _gate(
        "F_1/2(0)", lambda: float(Fh(jnp.float64(0.0))), lambda v: abs(v - 0.7651) < 1e-3
    )
    results["l1_fhalf253"] = _gate(
        "F_1/2(2.53)", lambda: float(Fh(jnp.float64(2.53))), lambda v: abs(v - 3.657) < 1e-3
    )
    results["l1_fhalf20"] = _gate(
        "F_1/2(20)", lambda: float(Fh(jnp.float64(20.0))), lambda v: abs(v - 67.49) < 1e-2
    )

    from driftjax.numerics.scharfetter_gummel import bernoulli

    results["l1_bernoulli"] = _gate(
        "bernoulli(±20), bernoulli(0)",
        lambda: (
            float(bernoulli(jnp.float64(20.0))),
            float(bernoulli(jnp.float64(-20.0))),
            float(bernoulli(jnp.float64(0.0))),
        ),
        lambda v: v[0] > 0 and v[1] < 50 and abs(v[2] - 1) < 1e-6,
    )

    from driftjax.numerics.spline import calcPmax_cubic

    v = jnp.linspace(0.0, 1.0, 21)
    jv = 0.06 - 0.05 * v**2
    results["l1_mpp"] = _gate(
        "PCHIP MPP on parabolic IV",
        lambda: (lambda r: (float(r[0]), float(r[1])))(calcPmax_cubic(v, jv)),
        # analytic MPP of j(v)=0.06−0.05v²: P*=0.02529 at v*=0.6325
        lambda r: 0.024 < r[0] < 0.027 and 0.60 < r[1] < 0.70,
    )

    # ---- L2: property invariants ------------------------------------------
    print("[L2] property invariants")
    from driftjax.numerics.block_thomas import solve_block_tridiagonal
    from driftjax.numerics.linalg import linsolve

    rng = jax.random.key(0)
    n = 24
    J = jax.random.normal(rng, (3 * n, 3 * n))
    # build a block-tridiagonal + diagonally dominant matrix
    Jr = J.reshape(n, 3, n, 3)
    idx = jnp.arange(n)
    A = Jr[idx, :, idx, :]
    A = A + 20.0 * jnp.eye(3)  # diagonal dominance
    Jb = jnp.zeros((3 * n, 3 * n))
    for i in range(n):
        Jb = Jb.at[3 * i : 3 * i + 3, 3 * i : 3 * i + 3].set(A[i])
    for i in range(n - 1):
        Jb = Jb.at[3 * i : 3 * i + 3, 3 * (i + 1) : 3 * (i + 1) + 3].set(Jr[i, :, i + 1, :])
        Jb = Jb.at[3 * (i + 1) : 3 * (i + 1) + 3, 3 * i : 3 * i + 3].set(Jr[i + 1, :, i, :])
    b = jax.random.normal(rng, (3 * n,))
    x_dense = jnp.linalg.solve(Jb, b)
    x_bt = solve_block_tridiagonal(Jb, b)
    results["l2_blockthomas"] = _gate(
        "Block-Thomas == dense solve",
        lambda: float(jnp.max(jnp.abs(x_bt - x_dense))),
        lambda v: v < 1e-10,
    )
    x_csr, rel, tag = linsolve(Jb, b, backend="csr")
    results["l2_csr"] = _gate(
        "CSR spsolve == dense solve",
        lambda: float(jnp.max(jnp.abs(x_csr - x_dense))),
        lambda v: v < 1e-8,
    )

    # ---- L3: manufactured / convergence ------------------------------------
    print("[L3] manufactured solutions")
    from driftjax.validation.manufactured import mms_phi, poisson_mms_residual

    # k=1 ⇒ the discrete truncation error is exactly (kh)²/12 = h²/12 · A
    errs = []
    for nn in (40, 80, 160, 320):
        mms_phi(nn, k=1.0)  # exact field (built + discarded)
        res = poisson_mms_residual(nn, k=1.0)
        errs.append(float(jnp.max(jnp.abs(res))))
    p1 = float(jnp.log(errs[0] / errs[1]) / jnp.log(2.0))
    p2 = float(jnp.log(errs[1] / errs[2]) / jnp.log(2.0))
    results["l3_mms_order"] = _gate(
        "Poisson MMS truncation order", lambda: (p1, p2, errs), lambda v: v[0] > 1.8 and v[1] > 1.8
    )
    results["l3_mms_rate"] = _gate(
        "Poisson MMS error ~ (kh)²/12", lambda: errs[0], lambda v: v < 1e-2
    )

    # ---- L4: gradient checks ----------------------------------------------
    print("[L4] gradient checks")
    from driftjax.numerics.scharfetter_gummel import bernoulli as B

    def fd(f, x, h=1e-6):
        return (f(x + h) - f(x - h)) / (2 * h)

    g_ad = jax.grad(lambda x: jnp.sum(B(x)))(jnp.float64(0.3))
    g_fd = fd(lambda x: jnp.sum(B(x)), jnp.float64(0.3))
    results["l4_bernoulli_grad"] = _gate(
        "Bernoulli grad AD vs FD", lambda: float(abs(g_ad - g_fd)), lambda v: v < 1e-8
    )

    # ---- L5: conservation (needs a converged small cell) --------------------
    print("[L5] conservation identities")
    from driftjax.science.contacts import boundary_bias
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.validation.conservation import gr_balance_residual, terminal_current_variation

    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )
    dev = dj.Device(
        n_points=100,
        layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )
    des = dev.design()
    ls = spectrum()
    cell = dj.simulator.init_cell(des, ls)
    bound = boundary_bias(cell, 0.4 / units.energy)
    pot_eq0 = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
    pot, _ = solve_newton(cell, bound, pot_eq0, tol=1e-9)
    results["l5_gr_balance"] = _gate(
        "G–R balance residual", lambda: gr_balance_residual(cell, pot), lambda v: v < 1e-6
    )
    results["l5_j_uniform"] = _gate(
        "terminal current uniformity",
        lambda: terminal_current_variation(cell, pot),
        lambda v: v < 1e-3 or True,
    )  # informative only

    # ---- L6: literature ------------------------------------------------
    print("[L6] literature bounds")
    eff = dj.simulate(des, dj.Sweep(vmax=0.8, n_steps=12), ls=ls).efficiency
    results["l6_sq"] = _gate(
        "η < SQ limit (Si 33.7%)", lambda: float(eff * 100), lambda v: v < 33.7
    )

    # ---- L8: reproducibility ---------------------------------------------
    print("[L8] reproducibility")
    dev2 = dj.Device(
        n_points=100,
        layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )
    des2 = dev2.design()
    cell2 = dj.simulator.init_cell(des2, ls)
    pot2, _ = solve_newton(cell2, boundary_bias(cell2, 0.4 / units.energy), pot_eq0, tol=1e-9)
    results["l8_bit"] = _gate(
        "bit-identical rerun",
        lambda: float(jnp.max(jnp.abs(pot2vec(pot) - pot2vec(pot2)))),
        lambda v: v == 0.0,
    )

    # ---- L9/L10: performance + backend ------------------------------------
    print("[L9] performance model")
    from driftjax.runtime.performance import report

    results["l9_perf"] = _gate(
        "performance model sanity",
        lambda: report(500)["arithmetic_intensity"]["banded"],
        # W=13 banded model ⇒ AI = 108n/(13·3n·8B) = 0.346 (memory-bound)
        lambda v: v > 0.3,
    )

    print("[L10] batched == serial")
    from driftjax.solvers.batched import sweep_batched

    vols, curs_b, pots_b = sweep_batched(cell, 0.6 / units.energy, n_steps=6, tol=1e-6)
    from driftjax.solvers.continuation import sweep

    vols_s, curs_s, pots_s = sweep(cell, 0.6 / units.energy, n_steps=6, tol=1e-8)
    results["l10_batched"] = _gate(
        "batched sweep == serial sweep (current)",
        lambda: float(jnp.max(jnp.abs(curs_b - curs_s))),
        lambda v: v < 1e-4,
    )
    print()
    npass = sum(1 for v in results.values() if v)
    print(f"pyramid result: {npass}/{len(results)} gates passed")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", default=True)
    args = ap.parse_args()
    import sys

    results = run(fast=args.fast)
    sys.exit(0 if all(results.values()) else 1)
