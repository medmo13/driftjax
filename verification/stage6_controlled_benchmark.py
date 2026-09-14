"""Stage 6 — Controlled Performance Benchmarks for DriftJax v0.1.15.

Reduced-memory variant: fewer grid sizes, fewer repetitions.
Run:  PYTHONPATH=src /home/med/Desktop/final/venv_latest/bin/python verification/stage6_controlled_benchmark.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from typing import NamedTuple

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import driftjax as dj
from driftjax.fields import pot2vec, vec2pot
from driftjax.numerics.block_thomas import block_thomas_solve
from driftjax.numerics.residual import F_jacobian, comp_F
from driftjax.solvers.continuation import total_current
from driftjax.science.contacts import boundary_eq, boundary_bias
from driftjax.simulator import init_cell
from driftjax.units import thermal_scales

SI = dj.material(
    Eg=1.12, Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
    mn=1350.0, mp=450.0, tn=1e-5, tp=1e-5, A=2e4,
)
W_SIM = 2e-4


def make_homo(N: int):
    return dj.Device(
        n_points=N,
        layers=[(W_SIM / 2, SI, 1e16), (W_SIM / 2, SI, -1e16)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )


def divider(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# 1. SOLVER-LEVEL BENCHMARK
# ---------------------------------------------------------------------------

def benchmark_solver_level():
    divider("1. SOLVER-LEVEL BENCHMARK (N=500, Block-Thomas vs Dense LU)")
    N = 500
    dev = make_homo(N)
    ls = dj.science.spectrum.spectrum(normalize=False)

    # Build a converged forward state
    print("  Running forward solve to get converged state...")
    sys.stdout.flush()
    sol = dj.simulate(dev, dj.Sweep(vmax=0.8, n_steps=5), optics=dj.BeerLambert(), progress=False)
    jax.block_until_ready(sol)
    cell = init_cell(dev.design(), ls, alpha_mode="beer-lambert", statistics="boltzmann",
                     optics=dj.BeerLambert(), fused=True)
    sc = thermal_scales(dev.design().T)
    pot = sol.potentials[-1]
    bound = boundary_bias(cell, 0.7 / sc["energy"])
    F = comp_F(cell, bound, pot)
    J = F_jacobian(cell, bound, pot)

    from driftjax.numerics.analytic_jacobian import banded_jacobian as _bj
    A, B, C = _bj(cell, bound, pot)
    rhs = -F.reshape(N, 3)
    rhs_flat = -F

    # Block-Thomas (10 warm runs)
    def bt_fn():
        return block_thomas_solve(A, B, C, rhs)
    bt_times = []
    bt_fn()  # warm
    for _ in range(10):
        t0 = time.perf_counter()
        r = bt_fn()
        jax.block_until_ready(r)
        bt_times.append((time.perf_counter() - t0) * 1000.0)

    # Dense LU (10 warm runs)
    def dense_fn():
        return jnp.linalg.solve(J, rhs_flat)
    dense_times = []
    dense_fn()  # warm
    for _ in range(10):
        t0 = time.perf_counter()
        r = dense_fn()
        jax.block_until_ready(r)
        dense_times.append((time.perf_counter() - t0) * 1000.0)

    bt_arr = jnp.array(bt_times)
    dn_arr = jnp.array(dense_times)
    bt_mean, bt_std = float(jnp.mean(bt_arr)), float(jnp.std(bt_arr))
    dn_mean, dn_std = float(jnp.mean(dn_arr)), float(jnp.std(dn_arr))
    speedup = dn_mean / bt_mean if bt_mean > 0 else float("inf")

    # Correctness check
    x_bt = block_thomas_solve(A, B, C, rhs).reshape(-1)
    x_dn = jnp.linalg.solve(J, rhs_flat)
    agree = float(jnp.max(jnp.abs(x_bt - x_dn)))
    res_bt = float(jnp.linalg.norm(J @ x_bt + F)) / (float(jnp.linalg.norm(F)) + 1e-30)

    print(f"\n  Block-Thomas : {bt_mean:8.2f} ± {bt_std:.2f} ms")
    print(f"  Dense LU     : {dn_mean:8.2f} ± {dn_std:.2f} ms")
    print(f"  Speedup      : {speedup:.1f}×")
    print(f"  Max |x_bt - x_dense|: {agree:.2e}")
    print(f"  BT linear residual : {res_bt:.2e}")
    return bt_mean, bt_std, dn_mean, dn_std, speedup


# ---------------------------------------------------------------------------
# 2. FORWARD SOLVE BENCHMARK
# ---------------------------------------------------------------------------

def benchmark_forward():
    divider("2. FORWARD SOLVE BENCHMARK (Sweep, vmax=1.1, n_steps=61)")
    grid_sizes = [125, 250, 500]
    results = {}
    for N in grid_sizes:
        print(f"\n  --- N = {N} ---")
        sys.stdout.flush()
        dev = make_homo(N)
        protocol = dj.Sweep(vmax=1.1, n_steps=61)
        solver = dj.Newton(fused=True)

        times = []
        last_eff = None
        for run_i in range(3):
            t0 = time.perf_counter()
            sol = dj.simulate(dev, protocol, solver=solver, optics=dj.BeerLambert(), progress=False)
            jax.block_until_ready(sol)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            times.append(elapsed_ms)
            last_eff = float(sol.eff)
            print(f"    Run {run_i+1}: {elapsed_ms:.1f} ms | eff={last_eff:.4f}")
            sys.stdout.flush()

        median_ms = float(jnp.median(jnp.array(times)))
        results[N] = {"median_ms": median_ms, "eff": last_eff}

    print("\n  Summary:")
    print(f"  {'N':>6} {'Median(ms)':>12} {'Efficiency':>12}")
    for N in grid_sizes:
        print(f"  {N:>6} {results[N]['median_ms']:>12.1f} {results[N]['eff']:>12.6f}")
    return results


# ---------------------------------------------------------------------------
# 3. ADJOINT GRADIENT BENCHMARK
# ---------------------------------------------------------------------------

def benchmark_adjoint_gradient():
    divider("3. ADJOINT GRADIENT BENCHMARK (N=500)")
    N = 500
    dev = make_homo(N)
    protocol = dj.Sweep(vmax=1.1, n_steps=61)
    solver = dj.Newton(fused=True)
    optics = dj.BeerLambert()

    # Use the native DeviceDesign parameterization for gradient
    def objective(d):
        return dj.simulate(d, protocol, solver=solver, optics=optics, progress=False).eff

    # Compile + first run
    print("  Compiling gradient (first call)...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    g = jax.grad(objective)(dev)
    jax.block_until_ready(g)
    compile_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  First call: {compile_ms:.1f} ms")

    # 5 warm runs
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        g = jax.grad(objective)(dev)
        jax.block_until_ready(g)
        times.append((time.perf_counter() - t0) * 1000.0)

    grad_mean = float(jnp.mean(jnp.array(times)))
    grad_std = float(jnp.std(jnp.array(times)))

    # Get gradient w.r.t. design's material fields (a representative subset)
    # Design is a DeviceDesign pytree; we report the norm of the full gradient
    grad_norm = float(jax.tree_util.tree_reduce(lambda a, b: a + jnp.sum(b**2), g, 0.0))**0.5
    print(f"\n  Gradient timing (5 warm runs):")
    print(f"    Mean ± std: {grad_mean:.1f} ± {grad_std:.1f} ms")
    print(f"    Gradient L2 norm: {grad_norm:.4e}")

    # Finite-difference: use scalar parameterisation
    print("\n  Finite-difference comparison (h=1e-5, central)...")
    sys.stdout.flush()
    def build_from_params(Eg, Chi, eps_r, Nc, Nv, mp, t1, t2, Nd, Na):
        m = dj.material(Eg=Eg, Chi=Chi, eps=eps_r, Nc=Nc, Nv=Nv, mn=1350.0, mp=mp, tn=1e-5, tp=1e-5, A=2e4)
        return dj.Device(
            n_points=N, layers=[(t1, m, Nd), (t2, m, -Na)],
            Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
        )

    def pce_scalar(params):
        Eg, Chi, eps_r, Nc, Nv, mp, t1, t2, Nd, Na = params
        d = build_from_params(Eg, Chi, eps_r, Nc, Nv, mp, t1, t2, Nd, Na)
        return dj.simulate(d, protocol, solver=solver, optics=optics, progress=False).eff

    x0 = jnp.array([1.12, 4.05, 11.7, 2.8e19, 1.04e19, 450.0, 1e-4, 1e-4, 1e16, 1e16])
    param_names = ["Eg", "Chi", "eps", "Nc", "Nv", "mp", "t1", "t2", "Nd", "Na"]

    # AD gradient
    grad_fn = jax.grad(pce_scalar)
    g_ad = grad_fn(x0)
    jax.block_until_ready(g_ad)

    # Central FD
    eps_fd = 1e-5
    print(f"  Computing FD gradients (10 parameters × 2 evaluations each)...")
    sys.stdout.flush()
    g_fd = jnp.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.at[i].add(eps_fd)
        xm = x0.at[i].add(-eps_fd)
        fp = float(pce_scalar(xp))
        fm = float(pce_scalar(xm))
        g_fd = g_fd.at[i].set((fp - fm) / (2 * eps_fd))
        print(f"    FD {param_names[i]}: AD={float(g_ad[i]):.4e}, FD={float(g_fd[i]):.4e}")
        sys.stdout.flush()

    print(f"\n  {'Param':>6} {'AD':>14} {'FD':>14} {'RelErr':>10}")
    print(f"  {'-'*48}")
    max_rel = 0.0
    for i, name in enumerate(param_names):
        ad_val = float(g_ad[i])
        fd_val = float(g_fd[i])
        mag = max(abs(ad_val), abs(fd_val))
        rel = abs(ad_val - fd_val) / mag if mag > 1e-15 else 0.0
        max_rel = max(max_rel, rel)
        print(f"  {name:>6} {ad_val:>14.4e} {fd_val:>14.4e} {rel:>10.2e}")
    print(f"\n  Max relative error (significant params): {max_rel:.2e}")

    return {
        "grad_mean_ms": grad_mean, "grad_std_ms": grad_std, "compile_ms": compile_ms,
        "max_rel_err_fd": max_rel,
    }


# ---------------------------------------------------------------------------
# 4. MEMORY ESTIMATE
# ---------------------------------------------------------------------------

def benchmark_memory():
    divider("4. MEMORY ESTIMATE (N=500)")
    N = 500
    n_dof = 3 * N
    bp = 8  # bytes per float64

    dense_bytes = n_dof * n_dof * bp
    block_bytes = 3 * N * 9 * bp
    ratio = dense_bytes / block_bytes

    print(f"  N = {N}, DOFs = {n_dof}")
    print(f"  Dense Jacobian: 9×N²×8 = {dense_bytes:>12,} bytes ({dense_bytes/1e6:.1f} MB)")
    print(f"  Block storage:  3×N×9×8 = {block_bytes:>12,} bytes ({block_bytes/1e3:.0f} KB)")
    print(f"  Ratio:          {ratio:.0f}×")
    return dense_bytes, block_bytes, ratio


# ---------------------------------------------------------------------------
# 5. COMPILATION TIME
# ---------------------------------------------------------------------------

def benchmark_compilation():
    divider("5. COMPILATION TIME (N=500)")
    N = 500
    dev = make_homo(N)
    protocol = dj.Sweep(vmax=1.1, n_steps=61)
    solver = dj.Newton(fused=True)
    optics = dj.BeerLambert()

    gc.collect()
    jax.clear_caches()

    print("  First call (JIT compile + run)...")
    sys.stdout.flush()
    t0 = time.perf_counter()
    sol1 = dj.simulate(dev, protocol, solver=solver, optics=optics, progress=False)
    jax.block_until_ready(sol1)
    first_ms = (time.perf_counter() - t0) * 1000.0

    print("  Warm calls (cached)...")
    sys.stdout.flush()
    times_warm = []
    for _ in range(3):
        t0 = time.perf_counter()
        sol = dj.simulate(dev, protocol, solver=solver, optics=optics, progress=False)
        jax.block_until_ready(sol)
        times_warm.append((time.perf_counter() - t0) * 1000.0)
    warm_arr = jnp.array(times_warm)
    warm_mean = float(jnp.mean(warm_arr))
    warm_std = float(jnp.std(warm_arr))

    print(f"\n  First call (compile+run): {first_ms:.1f} ms")
    print(f"  Warm calls (mean ± std) : {warm_mean:.1f} ± {warm_std:.1f} ms")
    print(f"  Compile overhead        : {first_ms / warm_mean:.1f}×")
    return first_ms, warm_mean


# ---------------------------------------------------------------------------
# 6. SCALING STUDY
# ---------------------------------------------------------------------------

def benchmark_scaling():
    divider("6. SCALING STUDY (Forward solve time vs N)")
    grid_sizes = [100, 200, 400, 800]
    times_dict = {}

    for N in grid_sizes:
        print(f"\n  --- N = {N} ---")
        sys.stdout.flush()
        dev = make_homo(N)
        protocol = dj.Sweep(vmax=1.1, n_steps=61)
        solver = dj.Newton(fused=True)
        optics = dj.BeerLambert()

        run_times = []
        for run_i in range(3):
            t0 = time.perf_counter()
            sol = dj.simulate(dev, protocol, solver=solver, optics=optics, progress=False)
            jax.block_until_ready(sol)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            run_times.append(elapsed_ms)
            print(f"    Run {run_i+1}: {elapsed_ms:.1f} ms | eff={float(sol.eff):.4f}")
            sys.stdout.flush()

        median_ms = float(jnp.median(jnp.array(run_times)))
        times_dict[N] = median_ms

    # Log-log fit: log(t) = alpha * log(N) + c
    ns = jnp.array(sorted(times_dict.keys()), dtype=jnp.float64)
    ts = jnp.array([times_dict[n] for n in sorted(times_dict.keys())], dtype=jnp.float64)
    log_n = jnp.log(ns)
    log_t = jnp.log(ts)
    alpha = float(jnp.sum((log_n - jnp.mean(log_n)) * (log_t - jnp.mean(log_t))) / jnp.var(log_n))
    c = float(jnp.mean(log_t) - alpha * jnp.mean(log_n))
    r2 = float(1 - jnp.sum((log_t - (alpha * log_n + c))**2) / jnp.sum((log_t - jnp.mean(log_t))**2))

    t_ref = times_dict[400]
    print(f"\n  Scaling summary:")
    print(f"  {'N':>6} {'Time(ms)':>10} {'T/T400':>10}")
    for N in sorted(times_dict.keys()):
        ratio = times_dict[N] / t_ref
        print(f"  {N:>6} {times_dict[N]:>10.1f} {ratio:>10.2f}")
    print(f"\n  Fitted exponent α = {alpha:.2f} (R² = {r2:.4f})")
    print(f"  Complexity: O(N^{alpha:.2f})")
    return times_dict, alpha, r2


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  DriftJax v0.1.15 — Stage 6 Controlled Performance Benchmarks")
    print("=" * 70)
    print(f"  JAX version : {jax.__version__}")
    print(f"  Platform    : {jax.default_backend()}")
    print(f"  float64     : True")
    sys.stdout.flush()

    # Tiny warm-up
    jnp.sum(jnp.ones(1))
    jax.block_until_ready(jnp.ones(1))

    try:
        r1 = benchmark_solver_level()
    except Exception as e:
        print(f"  SOLVER FAILED: {e}")
        import traceback; traceback.print_exc()
        r1 = None

    try:
        r2 = benchmark_forward()
    except Exception as e:
        print(f"  FORWARD FAILED: {e}")
        import traceback; traceback.print_exc()
        r2 = None

    try:
        r3 = benchmark_adjoint_gradient()
    except Exception as e:
        print(f"  ADJOINT FAILED: {e}")
        import traceback; traceback.print_exc()
        r3 = None

    try:
        r4 = benchmark_memory()
    except Exception as e:
        print(f"  MEMORY FAILED: {e}")
        import traceback; traceback.print_exc()
        r4 = None

    try:
        r5 = benchmark_compilation()
    except Exception as e:
        print(f"  COMPILATION FAILED: {e}")
        import traceback; traceback.print_exc()
        r5 = None

    try:
        r6 = benchmark_scaling()
    except Exception as e:
        print(f"  SCALING FAILED: {e}")
        import traceback; traceback.print_exc()
        r6 = None

    # Summary
    divider("ALL RESULTS")
    if r1:
        bt_m, bt_s, dn_m, dn_s, sp = r1
        print(f"  1. SOLVER: Block-Thomas {bt_m:.1f}±{bt_s:.1f}ms vs Dense {dn_m:.1f}±{dn_s:.1f}ms → {sp:.1f}× speedup")
    if r2:
        print(f"  2. FORWARD: ", end="")
        for N, v in r2.items():
            print(f"N={N}:{v['median_ms']:.0f}ms ", end="")
        print()
    if r3:
        print(f"  3. ADJOINT: {r3['grad_mean_ms']:.0f}±{r3['grad_std_ms']:.0f}ms, compile={r3['compile_ms']:.0f}ms, FD-err={r3['max_rel_err_fd']:.2e}")
    if r4:
        d, b, ratio = r4
        print(f"  4. MEMORY: dense={d/1e6:.1f}MB, block={b/1e3:.0f}KB, ratio={ratio:.0f}×")
    if r5:
        f, w = r5
        print(f"  5. COMPILATION: first={f:.0f}ms, warm={w:.0f}ms ({f/w:.1f}× overhead)")
    if r6:
        td, alpha, r2v = r6
        print(f"  6. SCALING: O(N^{alpha:.2f}), R²={r2v:.4f}")

    print(f"\n{'=' * 70}")
    print("  ALL BENCHMARKS COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
