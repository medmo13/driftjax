# Performance — Fused Kernels, JIT, Profiling

DriftJax is designed for high performance through JAX's XLA compiler. This document covers the key optimization strategies and how to profile the code.

## JIT Compilation

All hot-path functions are decorated with `@jax.jit`, which compiles them to optimized XLA programs:

- `comp_F` / `fused_residual` — residual assembly
- `banded_jacobian` / `fused_jacobian_banded` — Jacobian assembly
- `banded_solve` — linear solver (LAPACK dgbsv via host callback)
- `_step_newton_impl` — full Newton step (residual + Jacobian + solve)

The first call triggers compilation (typically 1–5 seconds); subsequent calls reuse the compiled program.

### Static Arguments

Functions with data-dependent branching use `static_argnames` to mark arguments that affect the compiled code:

```python
@partial(jax.jit, static_argnames=("dense", "refinement", "analytic", "fused"))
def _step_newton_impl(cell, bound, x, dense, refinement, analytic, fused):
    ...
```

Changing a static argument triggers recompilation.

## Fused Kernels

The standard path calls `comp_F` and `banded_jacobian` separately, each computing carrier statistics (n, p, ni) independently. The **fused path** computes them once and shares:

### comp_F_precomputed

Computes n, p, ni once and passes them to the drift-diffusion and Poisson sub-routines:

```python
F, n_v, p_v, ni_v = comp_F_precomputed(cell, bound, pot)
```

**Savings**: 33% faster than `comp_F` (19.5ms vs 29.2ms for N=100).

### fused_residual_and_jacobian

Computes carrier stats once and passes them to both the residual and Jacobian:

```python
F, A, B, C = fused_residual_and_jacobian(cell, bound, pot)
```

**Savings**: 6–15% faster than separate calls.

### Why Not Rely on XLA CSE?

XLA's Common Subexpression Elimination (CSE) can theoretically share identical subgraphs. However:

1. The residual and Jacobian use **different code paths** for carrier stats (recombination.py vs analytic_jacobian.py)
2. Whether XLA CSE fires depends on the traced IR structure — not guaranteed
3. Explicit sharing is more reliable and portable

## Banded (dgbsv) vs Dense

| Operation | Banded (dgbsv) | Dense |
|-----------|-------------|-------|
| Jacobian assembly | O(9N) | O(9N²) |
| Linear solve | O(N·kl·ku), kl=ku=5 | O(N³) |
| Memory | O(27N) | O(9N²) |
| N=100 time | ~27ms | ~861ms |
| N=100 memory | 21 KB | 703 KB |

The analytic Jacobian + pivoted banded path is the default and should be used whenever possible. (Timings above predate the v0.1.17 solver switch; current controlled numbers are in the main article §6 and `docs/paper/records/bench_v017_dgbsv.json`.)

## O(N) Block Operations

### blockwise_matvec

Computes J·x using the block-tridiagonal structure without forming the dense matrix:

```python
from driftjax.numerics.analytic_jacobian import blockwise_residual
residual = blockwise_residual(A, B, C, F, p)  # J @ p + F
```

This is O(9N) — used for the Newton residual check instead of the O(9N²) `dense_from_blocks` approach.

## Memory Optimization

### PTC Diagonal

The PTC solver adds a diagonal mass matrix: `J + diag(C) / dt`. Instead of forming the dense diagonal matrix with `jnp.diag()` (O(N²) memory), we use indexed addition:

```python
L = J.at[idx, idx].add(C / dt)  # O(N) memory
```

### Transient Regularization

The transient solver adds Tikhonov regularization: `J + reg · I`. Instead of allocating `jnp.eye(n_dof)` (O(N²)), we use:

```python
J_t = J.at[idx, idx].add(reg)  # O(N) memory
```

## Profiling

### JAX Profiler

```python
import jax
with jax.profiler.trace("/tmp/jax_trace"):
    sol = dj.simulate(dev, dj.Sweep())
```

### Timing individual operations

```python
import time

# Warm up (compile)
sol0 = dj.simulate(dev, dj.Sweep())

# Measure
t0 = time.perf_counter()
for _ in range(10):
    sol = dj.simulate(dev, dj.Sweep())
dt = (time.perf_counter() - t0) / 10
print(f"Simulate: {dt*1000:.1f} ms")
```

### XLA Compilation Cache

JAX caches compiled XLA programs. The cache is per-process and invalidated when:
- Static arguments change
- Input shapes change
- The JAX version changes

To clear the cache: `jax.clear_caches()`.

## Performance Characteristics

### Scaling with N (nodes)

| N | Newton (warm) | Simulate (20 steps) | Gradient (10 steps) |
|---|---------------|---------------------|---------------------|
| 50 | ~19ms | ~38ms | ~600ms |
| 100 | ~24ms | ~55ms | ~815ms |
| 200 | ~25ms | ~70ms | ~1200ms |
| 400 | ~49ms | ~100ms | ~2400ms |

At N=500 with 61 bias points, the full objective+gradient evaluation takes 0.815 s — **75× faster than ∂PV** (measured on 1-core CPU, JIT warm).

### Scaling with N_bias (bias points)

The simulate time scales linearly with N_bias (each bias point is an independent Newton solve). The gradient time also scales linearly (adjoint solve per bias point).

### Bottleneck Analysis

For a typical N=100, 10-bias sweep:
- **Newton iterations**: ~3–5 per bias point
- **Residual + Jacobian**: ~36ms per iteration (fused)
- **Banded (dgbsv) solve**: ~2ms per iteration (plus host-callback round trip)
- **Total per bias**: ~55ms forward, ~82ms adjoint
- **Total simulate (warm)**: ~55ms
- **Total gradient (warm)**: ~815ms

### Adjoint Path

The adjoint backward pass (`_sweep_bwd`) computes `d η / d d` via IFT:
1. `analytic_g_x` — O(N) per-bias current sensitivity (no AD)
2. `banded_jacobian + dense_from_blocks` — forms dense J in ~29ms
   (was 293ms via `jacfwd(comp_F)` = **90% faster**)
3. `adjoint_dense_solve(J^T, g_x)` — dense `lax_lu_solve`
4. `analytic_per_bias` — O(N) cell cotangent (no AD)

Per-bias adjoint cost: ~82ms (dominated by dense Jacobian formation + solve).

The backward pass uses `@jax.checkpoint` on each per-bias adjoint evaluation, reducing memory by recomputing per-bias intermediates instead of storing all bias points simultaneously. This enables 61 bias points at N=500 on 7.6 GB RAM without OOM.

## Tips for Maximum Performance

1. **Use `fused=True`** when calling `solve_newton` directly
2. **Keep N reasonable** — N=100–200 is usually sufficient for <1% mesh error
3. **Minimize N_bias** — use coarser sweeps for exploration, fine sweeps for final results
4. **Warm up before timing** — first call includes compilation
5. **Use `jax.lax.while_loop`** for traced paths (automatic under `jax.grad`)
