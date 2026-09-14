"""Tier VI — Forward solver scaling measurement."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def measure_forward_scaling(
    mesh_sizes: list[int] | None = None,
    n_warmup: int = 2,
    n_repeats: int = 3,
    bias_voltage: float = 0.4,
) -> dict:
    """Measure forward solve time vs mesh size."""
    import driftjax as dj
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.units import energy

    if mesh_sizes is None:
        mesh_sizes = [125, 250, 500, 1000, 2000]

    ls = spectrum()
    mat = dj.material(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19,
                       mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4)

    measurements: list[dict[str, Any]] = []
    for N in mesh_sizes:
        try:
            dev = dj.Device(
                n_points=N,
                layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
                Snl=1e7, Snr=0, Spl=0, Spr=1e7,
            )
            cell = init_cell(dev.design(), ls)
            bound = boundary_bias(cell, bias_voltage / energy)
            pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)

            for _ in range(n_warmup):
                solve_newton(cell, bound, pot_eq, tol=1e-10)

            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                solve_newton(cell, bound, pot_eq, tol=1e-10)
                times.append(time.perf_counter() - t0)

            t_median = float(np.median(times))
            measurements.append({"N": N, "time_s": t_median, "status": "ok"})
        except Exception as e:
            measurements.append({"N": N, "status": f"error: {e}"})

    valid = [m for m in measurements if m["status"] == "ok"]
    if len(valid) >= 2:
        log_N = np.log([float(m["N"]) for m in valid])
        log_T = np.log([float(m["time_s"]) for m in valid])
        coeffs = np.polyfit(log_N, log_T, 1)
        p = float(coeffs[0])
        a = float(np.exp(coeffs[1]))
        ss_res = np.sum((log_T - np.polyval(coeffs, log_N))**2)
        ss_tot = np.sum((log_T - np.mean(log_T))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        p, a, r2 = None, None, None

    return {"measurements": measurements, "fit": {"model": "T = a * N^p", "a_s": a, "p": p, "r_squared": r2, "n_points_used": len(valid)}}


def save_record(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def print_summary(result: dict) -> None:
    fit = result["fit"]
    print(f"\n{'='*60}")
    print("FORWARD SOLVER SCALING")
    print(f"{'='*60}")
    if fit["p"] is not None:
        print(f"  T = {fit['a_s']:.4e} * N^{fit['p']:.3f}  (R²={fit['r_squared']:.4f})")
    print(f"\n{'N':>8s} {'T (s)':>10s}")
    for m in result["measurements"]:
        if m["status"] == "ok":
            print(f"  {m['N']:>6d} {m['time_s']:>10.4f}")
    print()


if __name__ == "__main__":
    result = measure_forward_scaling()
    print_summary(result)
