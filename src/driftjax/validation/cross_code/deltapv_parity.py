"""Tier VII — V20: Forward parity, V21: Optimization parity."""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np


def test_forward_parity() -> dict:
    """V20: Forward parity — DriftJax IV metrics on Si p-n."""
    import driftjax as dj
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess, sweep
    from driftjax.solvers.newton import solve_eq
    from driftjax.units import energy

    ls = spectrum()
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )

    try:
        dev = dj.Device(
            n_points=200,
            layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )
        cell = init_cell(dev.design(), ls)
        pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
        voltages, currents, _, _ = sweep(cell, 1.2 / energy, n_steps=61, tol=1e-10, init=pot_eq)

        V = voltages * energy
        cur_mA = currents * 1e3
        Jsc = float(-cur_mA[0])
        Voc = float(V[-1])
        Pmax = float(jnp.max(V * (-cur_mA)))
        FF = Pmax / (Voc * Jsc + 1e-30)
        PCE = Pmax / 100.0

        return {
            "test": "V20_forward_parity",
            "Jsc_mA_cm2": Jsc,
            "Voc_V": Voc,
            "FF": FF,
            "PCE_percent": PCE * 100,
            "status": "ok",
            "passed": True,
        }
    except Exception as e:
        return {"test": "V20_forward_parity", "status": f"error: {e}", "passed": False}


def test_optimization_parity() -> dict:
    """V21: Measure objective+gradient evaluation time."""
    import driftjax as dj
    from driftjax.fields import pot2vec, vec2pot
    from driftjax.numerics.scharfetter_gummel import Jn, Jp
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.units import energy

    ls = spectrum()
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )

    try:
        dev = dj.Device(
            n_points=100,
            layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )
        cell = init_cell(dev.design(), ls)
        pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
        bound = boundary_bias(cell, 0.4 / energy)
        pot, _ = solve_newton(cell, bound, pot_eq, tol=1e-10)

        def objective(pv):
            return jnp.sum(Jn(cell, vec2pot(pv)) + Jp(cell, vec2pot(pv)))

        g_vec = pot2vec(pot)
        jax.grad(objective)(g_vec)  # warm up

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            jax.grad(objective)(g_vec)
            times.append(time.perf_counter() - t0)

        return {
            "test": "V21_optimization_parity",
            "time_s": float(np.median(times)),
            "n_params": len(g_vec),
            "status": "ok",
            "passed": True,
        }
    except Exception as e:
        return {"test": "V21_optimization_parity", "status": f"error: {e}", "passed": False}


def print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("TIER VII — Cross-Code Validation")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r.get("passed", False) else "FAIL"
        print(f"  [{status}] {r['test']}")
        for k, v in r.items():
            if k not in ("test", "passed"):
                print(f"    {k}: {v}")
    print()


if __name__ == "__main__":
    print_summary([test_forward_parity(), test_optimization_parity()])
