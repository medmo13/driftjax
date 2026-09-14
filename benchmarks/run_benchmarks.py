#!/usr/bin/env python
"""Benchmarking harness for DriftJax performance tracking.

Usage:
    python benchmarks/run_benchmarks.py              # Run all benchmarks
    python benchmarks/run_benchmarks.py --json       # Output as JSON
    python benchmarks/run_benchmarks.py --compare prev.json  # Compare with previous

Tracks:
    - Newton solve time (warm) for N=50, 100, 200, 400
    - Simulate time (warm) for N=100 with 10 and 20 bias points
    - Gradient time (warm) for N=100 with 10 bias points
    - Residual + Jacobian assembly time
    - Block-Thomas solve time
    - Memory usage
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp

# Ensure float64
jax.config.update("jax_enable_x64", True)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import driftjax as dj
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton


@dataclass
class BenchmarkResult:
    name: str
    value: float
    unit: str
    meta: dict = field(default_factory=dict)


@dataclass
class BenchmarkSuite:
    timestamp: str
    python: str
    jax: str
    platform: str
    results: list[BenchmarkResult] = field(default_factory=list)


def get_system_info() -> dict:
    import platform

    return {
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "platform": f"{platform.system()} {platform.release()}",
    }


def make_device(N: int) -> dj.Device:
    mat = dj.material(
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
    )
    return dj.Device(
        n_points=N,
        layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def prepare_cell(dev: dj.Device):
    ls = spectrum(normalize=False)
    cell = dj.simulator.init_cell(dev.design(), ls, alpha_mode="beer-lambert")
    bound_eq_ = boundary_eq(cell)
    pot0 = solve_eq(cell, bound_eq_, equilibrium_guess(cell).phi)
    bound = boundary_bias(cell, 0.5 / dj.units.energy)
    return cell, bound, pot0, ls


def bench_newton_warm(N: int, n_warmup: int = 1, n_measure: int = 10) -> BenchmarkResult:
    dev = make_device(N)
    cell, bound, pot0, ls = prepare_cell(dev)
    # Warm up
    for _ in range(n_warmup):
        solve_newton(cell, bound, pot0, tol=1e-10, max_steps=30)
    # Measure
    t0 = time.perf_counter()
    for _ in range(n_measure):
        solve_newton(cell, bound, pot0, tol=1e-10, max_steps=30)
    dt = (time.perf_counter() - t0) / n_measure
    return BenchmarkResult(
        name=f"newton_warm_n{N}",
        value=round(dt * 1000, 2),
        unit="ms",
        meta={"N": N, "n_measure": n_measure},
    )


def bench_simulate_warm(
    N: int, n_steps: int, n_warmup: int = 1, n_measure: int = 5
) -> BenchmarkResult:
    dev = make_device(N)
    ls = spectrum(normalize=False)
    # Warm up
    for _ in range(n_warmup):
        dj.simulate(
            dev,
            dj.Sweep(vmax=0.8, n_steps=n_steps),
            optics=dj.BeerLambert(alpha_mode="beer-lambert"),
            ls=ls,
        )
    # Measure
    t0 = time.perf_counter()
    for _ in range(n_measure):
        dj.simulate(
            dev,
            dj.Sweep(vmax=0.8, n_steps=n_steps),
            optics=dj.BeerLambert(alpha_mode="beer-lambert"),
            ls=ls,
        )
    dt = (time.perf_counter() - t0) / n_measure
    return BenchmarkResult(
        name=f"simulate_warm_n{N}_steps{n_steps}",
        value=round(dt * 1000, 2),
        unit="ms",
        meta={"N": N, "n_steps": n_steps},
    )


def bench_gradient_warm(
    N: int, n_steps: int, n_warmup: int = 1, n_measure: int = 3
) -> BenchmarkResult:
    dev = make_device(N)
    ls = spectrum(normalize=False)

    def eff_scalar(d):
        r = dj.simulate(
            d,
            dj.Sweep(vmax=0.8, n_steps=n_steps),
            optics=dj.BeerLambert(alpha_mode="beer-lambert"),
            ls=ls,
        )
        return jnp.asarray(r.efficiency)

    # Warm up
    for _ in range(n_warmup):
        g = jax.grad(eff_scalar)(dev)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, g
        )
    # Measure
    t0 = time.perf_counter()
    for _ in range(n_measure):
        g = jax.grad(eff_scalar)(dev)
        jax.tree_util.tree_map(
            lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else None, g
        )
    dt = (time.perf_counter() - t0) / n_measure
    return BenchmarkResult(
        name=f"gradient_warm_n{N}_steps{n_steps}",
        value=round(dt * 1000, 2),
        unit="ms",
        meta={"N": N, "n_steps": n_steps},
    )


def bench_residual_jacobian(N: int, n_warmup: int = 3, n_measure: int = 20) -> dict:
    dev = make_device(N)
    cell, bound, pot0, ls = prepare_cell(dev)
    from driftjax.numerics.analytic_jacobian import banded_jacobian
    from driftjax.numerics.residual import comp_F, comp_F_precomputed

    pot, _ = solve_newton(cell, bound, pot0, tol=1e-10, max_steps=30)

    # Warm up
    for _ in range(n_warmup):
        F = comp_F(cell, bound, pot)
        A, B, C = banded_jacobian(cell, bound, pot)

    # comp_F timing
    t0 = time.perf_counter()
    for _ in range(n_measure):
        F = comp_F(cell, bound, pot)
        if hasattr(F, "block_until_ready"):
            F.block_until_ready()
    dt_comp_f = (time.perf_counter() - t0) / n_measure

    # comp_F_precomputed timing
    t0 = time.perf_counter()
    for _ in range(n_measure):
        F2, n_v, p_v, ni_v = comp_F_precomputed(cell, bound, pot)
        if hasattr(F2, "block_until_ready"):
            F2.block_until_ready()
    dt_pre = (time.perf_counter() - t0) / n_measure

    # banded_jacobian timing
    t0 = time.perf_counter()
    for _ in range(n_measure):
        A, B, C = banded_jacobian(cell, bound, pot)
        A.block_until_ready()
        B.block_until_ready()
        C.block_until_ready()
    dt_bj = (time.perf_counter() - t0) / n_measure

    # Fused pair timing
    t0 = time.perf_counter()
    for _ in range(n_measure):
        F2, n_v, p_v, ni_v = comp_F_precomputed(cell, bound, pot)
        A, B, C = banded_jacobian(cell, bound, pot, n_v=n_v, p_v=p_v, ni_v=ni_v)
        A.block_until_ready()
        B.block_until_ready()
        C.block_until_ready()
    dt_fused = (time.perf_counter() - t0) / n_measure

    return {
        "comp_F": BenchmarkResult(
            name=f"comp_F_n{N}", value=round(dt_comp_f * 1000, 3), unit="ms", meta={"N": N}
        ),
        "comp_F_precomputed": BenchmarkResult(
            name=f"comp_F_precomputed_n{N}", value=round(dt_pre * 1000, 3), unit="ms", meta={"N": N}
        ),
        "banded_jacobian": BenchmarkResult(
            name=f"banded_jacobian_n{N}", value=round(dt_bj * 1000, 1), unit="ms", meta={"N": N}
        ),
        "fused_pair": BenchmarkResult(
            name=f"fused_pair_n{N}", value=round(dt_fused * 1000, 1), unit="ms", meta={"N": N}
        ),
    }


def bench_memory() -> BenchmarkResult:
    with open("/proc/self/status") as f:
        proc_status = f.read()
    rss_line = [line for line in proc_status.split("\n") if line.startswith("VmRSS")]
    rss_kb = int(rss_line[0].split()[1]) if rss_line else 0
    return BenchmarkResult(name="peak_rss", value=rss_kb, unit="KB")


def run_all_benchmarks() -> BenchmarkSuite:
    info = get_system_info()
    suite = BenchmarkSuite(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        python=info["python"],
        jax=info["jax"],
        platform=info["platform"],
    )

    print("Running benchmarks...")

    # Newton scaling
    for N in [50, 100, 200, 400]:
        r = bench_newton_warm(N)
        suite.results.append(r)
        print(f"  {r.name}: {r.value} {r.unit}")

    # Simulate warm
    for N, steps in [(100, 10), (100, 20)]:
        r = bench_simulate_warm(N, steps)
        suite.results.append(r)
        print(f"  {r.name}: {r.value} {r.unit}")

    # Gradient warm
    r = bench_gradient_warm(100, 10)
    suite.results.append(r)
    print(f"  {r.name}: {r.value} {r.unit}")

    # Residual + Jacobian
    rj = bench_residual_jacobian(100)
    for r in rj.values():
        suite.results.append(r)
        print(f"  {r.name}: {r.value} {r.unit}")

    # Memory
    r = bench_memory()
    suite.results.append(r)
    print(f"  {r.name}: {r.value} {r.unit}")

    return suite


def compare_benchmarks(current: BenchmarkSuite, previous: BenchmarkSuite) -> list[dict]:
    """Compare two benchmark suites and return regression/improvement report."""
    prev_map = {r.name: r for r in previous.results}
    report = []
    for r in current.results:
        if r.name in prev_map:
            prev = prev_map[r.name]
            if prev.value > 0:
                pct_change = (r.value - prev.value) / prev.value * 100
                report.append(
                    {
                        "name": r.name,
                        "current": r.value,
                        "previous": prev.value,
                        "change_pct": round(pct_change, 1),
                        "regression": pct_change > 5,
                    }
                )
    return report


def main():
    parser = argparse.ArgumentParser(description="DriftJax benchmarks")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--compare", type=str, help="Compare with previous JSON file")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    args = parser.parse_args()

    suite = run_all_benchmarks()

    if args.compare:
        with open(args.compare) as f:
            prev = BenchmarkSuite(**json.load(f))
        report = compare_benchmarks(suite, prev)
        regressions = [r for r in report if r["regression"]]
        if regressions:
            print(f"\n⚠ {len(regressions)} regressions detected:")
            for r in regressions:
                print(
                    f"  {r['name']}: {r['previous']:.1f} → {r['current']:.1f} ({r['change_pct']:+.1f}%)"
                )
        else:
            print("\n✓ No regressions detected.")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(asdict(suite), f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.json:
        print(json.dumps(asdict(suite), indent=2))


if __name__ == "__main__":
    main()
