"""Tier VI — V17: Memory scaling, V18: Adjoint scaling, V19: Cold vs warm JIT."""

from __future__ import annotations

import gc
import time

import jax
import jax.numpy as jnp
import numpy as np


def measure_memory_scaling(mesh_sizes: list[int] | None = None) -> dict:
    """V17: Theoretical M(N) for dense vs block-tridiagonal."""
    if mesh_sizes is None:
        mesh_sizes = [125, 250, 500, 1000]
    results = []
    for N in mesh_sizes:
        dense_bytes = (3 * N) ** 2 * 8
        banded_bytes = (27 * N - 18) * 8
        results.append(
            {
                "N": N,
                "dense_MB": dense_bytes / (1024**2),
                "banded_MB": banded_bytes / (1024**2),
                "ratio": dense_bytes / banded_bytes,
            }
        )
    return {"test": "V17_memory_scaling", "measurements": results}


def measure_adjoint_scaling(mesh_sizes: list[int] | None = None) -> dict:
    """V18: Measure T_adjoint(N) for dense solve."""
    import driftjax as dj
    from driftjax.fields import pot2vec, vec2pot
    from driftjax.numerics.residual import F_jacobian
    from driftjax.numerics.scharfetter_gummel import Jn, Jp
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.units import energy

    if mesh_sizes is None:
        mesh_sizes = [125, 250, 500, 1000, 2000]

    ls = spectrum()
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )

    measurements = []
    for N in mesh_sizes:
        try:
            dev = dj.Device(
                n_points=N,
                layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
                Snl=1e7,
                Snr=0,
                Spl=0,
                Spr=1e7,
            )
            cell = init_cell(dev.design(), ls)
            bound = boundary_bias(cell, 0.4 / energy)
            pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
            pot, _ = solve_newton(cell, bound, pot_eq, tol=1e-10)

            J_dense = F_jacobian(cell, bound, pot)

            def objective(pv, cell=cell):
                return jnp.sum(Jn(cell, vec2pot(pv)) + Jp(cell, vec2pot(pv)))

            g_obj = jax.grad(objective)(pot2vec(pot))

            times_dense = []
            for _ in range(3):
                t0 = time.perf_counter()
                jnp.linalg.solve(J_dense.T, g_obj)
                times_dense.append(time.perf_counter() - t0)
            t_dense = float(np.median(times_dense))

            measurements.append({"N": N, "t_dense_s": t_dense, "status": "ok"})
        except Exception as e:
            measurements.append({"N": N, "status": f"error: {e}"})

    return {"test": "V18_adjoint_scaling", "measurements": measurements}


def measure_jit_cold_warm() -> dict:
    """V19: Separate T_compile from T_warm."""
    import driftjax as dj
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_newton
    from driftjax.units import energy

    ls = spectrum()
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )
    dev = dj.Device(
        n_points=100, layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)], Snl=1e7, Snr=0, Spl=0, Spr=1e7
    )
    cell = init_cell(dev.design(), ls)
    bound = boundary_bias(cell, 0.4 / energy)
    pot_eq = equilibrium_guess(cell)

    gc.collect()
    t0 = time.perf_counter()
    solve_newton(cell, bound, pot_eq, tol=1e-10)
    t_cold = time.perf_counter() - t0

    times_warm = []
    for _ in range(5):
        t0 = time.perf_counter()
        solve_newton(cell, bound, pot_eq, tol=1e-10)
        times_warm.append(time.perf_counter() - t0)

    t_warm_median = float(np.median(times_warm))
    return {
        "test": "V19_jit_cold_warm",
        "t_cold_s": t_cold,
        "t_warm_median_s": t_warm_median,
        "compilation_overhead_s": t_cold - t_warm_median,
        "overhead_ratio": t_cold / (t_warm_median + 1e-30),
    }


def print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("TIER VI — Performance Benchmarks")
    print(f"{'=' * 60}")
    for r in results:
        print(f"\n  {r['test']}:")
        if r["test"] == "V17_memory_scaling":
            print(f"    {'N':>6s} {'Dense MB':>10s} {'Banded MB':>10s} {'Ratio':>8s}")
            for m in r["measurements"]:
                print(
                    f"    {m['N']:>6d} {m['dense_MB']:>10.2f} {m['banded_MB']:>10.4f} {m['ratio']:>8.1f}x"
                )
        elif r["test"] == "V18_adjoint_scaling":
            print(f"    {'N':>6s} {'Dense (s)':>10s}")
            for m in r["measurements"]:
                if m.get("status") == "ok":
                    print(f"    {m['N']:>6d} {m['t_dense_s']:>10.4f}")
        elif r["test"] == "V19_jit_cold_warm":
            print(
                f"    Cold: {r['t_cold_s']:.3f}s  Warm: {r['t_warm_median_s']:.3f}s  Overhead: {r['overhead_ratio']:.1f}x"
            )
    print()


if __name__ == "__main__":
    print_summary([measure_memory_scaling(), measure_adjoint_scaling(), measure_jit_cold_warm()])
