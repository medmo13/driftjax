# Numerics — Discretization, Residual, Jacobian, Solvers

This document explains the numerical methods underlying DriftJax: how the continuous PDEs are discretized, how the residual is assembled, how the Jacobian is computed, and how the linear systems are solved.

## Finite-Volume Discretization

The Van Roosbroeck system is discretized on a **1-D non-uniform mesh** using the **finite-volume method**. Each control volume is a cell centered on a node, with faces halfway between adjacent nodes.

### Mesh

Given N nodes at positions x₀, x₁, …, x_{N-1}, the cell widths are:

```
d_i = x_{i+1} − x_i     (edge lengths)
ave_i = (d_{i-1} + d_i) / 2     (face-centered means)
```

**Code:** `numerics/mesh.py`, `fields.py:dgrid`

### Face-Centered Permittivity

The permittivity on each face is the **harmonic mean**:

```
ε_face = 2·ε_i·ε_{i+1} / (ε_i + ε_{i+1})
```

This is the only average that preserves the D = εE continuity condition across a material interface.

**Code:** `numerics/poisson.py:harmonic_ave_eps()`

## Residual Assembly

The full residual F(φ) is a 3N-dimensional vector assembled in the interleaved order `[φn₀, φp₀, φ₀, …]`.

### comp_F

```
comp_F(cell, bound, pot) → F ∈ R^{3N}
```

Assembly steps:
1. **Contact rows** (first and last 3 entries): boundary condition residuals
2. **Interior rows** (3(N−2) entries): drift-diffusion + Poisson at interior nodes

**Code:** `numerics/residual.py:comp_F()`

The interior blocks are:

```
F[3i]   = −R_i + G_i + (Jn_{i+1} − Jn_{i-1}) / ave_i       (electron continuity)
F[3i+1] =  R_i − G_i + (Jp_{i+1} − Jp_{i-1}) / ave_i       (hole continuity)
F[3i+2] =  (ε_face·∇φ)_i − (p_i − n_i + N_dop_i)           (Poisson)
```

### Precomputed Variant

`comp_F_precomputed` computes the carrier statistics (n, p, n_i) and recombination R **once**, then passes them to the drift-diffusion and Poisson sub-routines. This eliminates redundant carrier-stat evaluations (each n/p/ni call was previously made 6–8 times per residual evaluation).

**Code:** `numerics/residual.py:comp_F_precomputed()`

## Analytic Jacobian

The Jacobian J = ∂F/∂x is **block-tridiagonal** in 3×3 blocks because node i's equations only couple to nodes i−1, i, and i+1.

### Structure

```
J = | A₀  B₀                |
    | C₀  A₁  B₁            |
    |     C₁  A₂  B₂        |
    |         ·   ·   ·      |
    |             C_{N-2} A_{N-1} |
```

where A_k, B_k, C_k are 3×3 matrices:
- **A_k** = diagonal block (self-coupling)
- **B_k** = super-diagonal (coupling to node k+1)
- **C_k** = sub-diagonal (coupling to node k−1)

### Assembly

The Jacobian is assembled analytically — no automatic differentiation needed. The key components:

1. **Bernoulli edge derivatives** (`_banded_jacobian_fused`): all 8 edge derivatives (4 for Jn, 4 for Jp) computed in a single vectorized kernel.

2. **Recombination derivatives** (`_recomb_deriv`): partial derivatives of R with respect to φ_n, φ_p, φ, using Boltzmann algebra.

3. **Poisson derivatives**: the face-centered permittivity terms.

**Code:** `numerics/analytic_jacobian.py:banded_jacobian()`

### Precomputed Carrier Stats

`banded_jacobian` accepts optional `n_v`, `p_v`, `ni_v` arguments. When provided (from `comp_F_precomputed`), it skips re-evaluating the carrier statistics, sharing them with the residual computation.

**Code:** `numerics/analytic_jacobian.py:banded_jacobian(n_v=..., p_v=..., ni_v=...)`

## Pivoted Banded Solver

The block-tridiagonal system J·dx = −F (interleaved 3×3 blocks,
bandwidth kl=ku=5) is solved by LAPACK `dgbsv` with partial pivoting
(via `scipy.linalg.solve_banded` through a host callback). Flop cost is
**O(N·kl·ku)**, independent of the condition number. Pivoting is
essential: the former unpivoted block-Thomas elimination failed on
ill-conditioned heterojunctions while the banded solver stays at
residual ~1e-9 (see the main article, structured-solver diagnosis).

**Code:** `numerics/banded_solve.py:banded_solve()`

### Native JAX GE (opt-in)

For GPU-residency and free end-to-end autodiff of the linear solve,
`DRIFTJAX_NATIVE_BANDED=1` switches to the pure-JAX block Thomas
algorithm in `lax.scan` (`numerics/banded_ge.py:banded_ge_solve()`).
This uses 3×3 partial pivoting (`jnp.linalg.solve` at each block)
with O(N·bw²) complexity — asymptotically faster than the SVD
fallback and fully traceable inside XLA. Singularity is detected via
per-block determinant threshold.

**Code:** `numerics/banded_ge.py:banded_ge_solve()`

### Non-finite / singular guard

Near degenerate conditions (e.g., equilibrium Jacobian at flatband) the
blocks can be non-finite or singular. The solver checks the assembled
band matrix for finiteness and the linear residual against 1e-4, and
falls back to pivoted dense `JAX.linalg.solve` when either check trips.

**Code:** `numerics/banded_solve.py:banded_solve()`, `solvers/newton.py:_step_newton_impl()`

**Differentiability note:** the forward LAPACK path runs through a host
callback opaque to AD — forward Newton steps are not differentiable
through. The native GE path (`DRIFTJAX_NATIVE_BANDED=1`) is fully
differentiable. Gradients otherwise flow through the `custom_vjp`
implicit-adjoint path, which uses `adjoint_dense_solve` for the
transpose system.

## Dense Fallback

When the analytic Jacobian is unavailable (non-Boltzmann statistics, refinement mode) or when the banded residual check fails, the solver falls back to:

1. **Dense Jacobian** via `jax.jacfwd(comp_F)` — forward-mode AD of the flat residual
2. **Dense solve** via `jax.linalg.solve`

**Code:** `numerics/residual.py:F_jacobian()`, `solvers/newton.py:_linear_solve()`

The dense Jacobian is O(9N²) in both computation and memory, versus O(9N) for the analytic path. For N=100, this means 703 KB (dense) vs 21 KB (banded).

## O(N) Block-Matrix Operations

### blockwise_matvec

Computes J·x using the block-tridiagonal structure **without forming the dense matrix**:

```
out[k] = A[k] @ x[k] + B[k] @ x[k+1] + C[k-1] @ x[k-1]
```

This is O(9N) — used for the Newton residual check instead of the O(9N²) `dense_from_blocks` approach.

**Code:** `numerics/analytic_jacobian.py:blockwise_matvec()`, `blockwise_residual()`

## Mixed Precision

For ill-conditioned systems, DriftJax supports **FP32 iterative refinement**:

1. Factor the system in FP32 (faster, lower memory)
2. Compute the residual in FP64
3. Solve the correction in FP32
4. Update in FP64

This converges to FP64 accuracy while using FP32 arithmetic for the bulk of the work.

**Code:** `numerics/mixed_precision.py:solve_refined()`
