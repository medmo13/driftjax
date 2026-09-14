# Automatic Differentiation — IFT Adjoint

DriftJax's key feature is **exact gradients of any scalar output with respect to device design** — achieved through an implicit-function-theorem (IFT) adjoint implemented as a `custom_vjp`.

## Why Not Unroll Newton?

The naive approach to differentiating through a solver is to unroll the Newton loop and apply reverse-mode AD through every iteration. This fails for the DDP system because:

1. **Ill-conditioned Jacobians** near Voc make the unrolled gradient numerically unstable
2. **Memory cost** scales as O(N_iters × N_params), prohibitive for large meshes
3. **No convergence guarantee** for the adjoint system when the forward solve is near-degenerate

DriftJax uses the **IFT adjoint** instead — a single linear solve per bias point, regardless of how many Newton iterations were needed.

## The IFT Adjoint

Given the forward problem F(x, d) = 0 where x are the state variables and d are the design parameters, the adjoint equation is:

```
J^T · λ = g_x
```

where J = ∂F/∂x is the Jacobian at the converged state, g_x = ∂g/∂x is the gradient of the objective with respect to the state, and λ is the adjoint variable.

The design gradient is then:

```
g_d = −λ^T · F_d
```

where F_d = ∂F/∂d is the residual's sensitivity to the design.

### Key Properties

- **One linear solve per bias point** — regardless of Newton iterations
- **Exact to machine precision** — no truncation from unrolling
- **Memory O(N)** — only the adjoint vector λ, not the full Newton trajectory

## Implementation

### custom_vjp

The sweep function `_simulate_sweep` is wrapped in `jax.custom_vjp`:

```python
@jax.custom_vjp
def _simulate_sweep(design, solver, optics, protocol, progress, ls, statistics, init=None, fused=False):
    # Forward pass: solve at each bias point
    ...
```

The forward pass computes the IV curve. The backward pass solves the adjoint system.

**Code:** `simulate.py:_simulate_sweep()` (forward), `_sweep_bwd()` (backward)

### Backward Pass

The backward pass (`_sweep_bwd`) performs:

1. **Compute g_x** — the gradient of the objective with respect to potentials at each bias point
2. **Solve J^T · λ = g_x** — the adjoint system (dense or banded)
3. **Compute g_d** — the design gradient from the adjoint variable

The adjoint solve uses `jax.vmap` over bias points — a single trace, batched execution — to avoid per-bias re-tracing.

**Code:** `simulate.py:_sweep_bwd()`

### Dense-Only Adjoint

The adjoint solve uses pivoted dense LU unconditionally (`adjoint_dense_solve`). The banded transpose was removed after FD validation showed silent 10--30% gradient errors under ill-conditioning ($\kappa(J)\sim10^{14}$) that no cheap residual gate could certify.

**Code:** `adjoint/api.py:ImplicitAdjoint()`, `numerics/mixed_precision.py:adjoint_dense_solve()`

## gradient(simulate)

The complete gradient pipeline:

```python
g = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)
```

1. **Forward pass**: sweep → Newton at each bias → IV curve → efficiency
2. **Backward pass**: adjoint solve at each bias → design gradient

The gradient is **exact** (to machine precision) and **efficient** (O(N) per bias point with banded adjoint).

## Checkpointing

The backward pass uses `@jax.checkpoint` on each per-bias adjoint evaluation (`_sweep_bwd`), reducing memory by recomputing per-bias intermediates instead of storing all bias points simultaneously. This is enabled by default and enables 61 bias points at N=500 on 7.6 GB RAM without OOM.

**Code:** `simulate.py:_sweep_bwd()`

## vmap Safety

All DriftJax functions are `vmap`-safe — they can be batched over any leading dimension:

```python
# Batch over multiple devices
effs = jax.vmap(lambda d: dj.simulate(d).efficiency)(devices)
```

This is achieved by avoiding Python-level data-dependent control flow in all traced paths.

## Practical Considerations

### Gradient Accuracy

The IFT adjoint is exact to ~1e-10 relative accuracy. Full-pytree central FD verification achieves maximum relative discrepancy $5.56\times10^{-7}$ for 16 perovskite design parameters. For validation against finite differences:

```python
from driftjax.validation import check_grad_fd
check_grad_fd(dev, eps=1e-7)  # finite-difference vs adjoint comparison
```

### Compile Time

The first call to `jax.grad(simulate)` triggers XLA compilation of both the forward and backward passes. For N=200, this typically takes 5–10 seconds. Subsequent calls reuse the compiled program.

### Memory

Peak memory scales as O(N × N_bias) for the forward pass. The `@jax.checkpoint` on per-bias adjoint evaluations reduces the backward pass memory to O(N) per bias (recomputing intermediates instead of storing them).
