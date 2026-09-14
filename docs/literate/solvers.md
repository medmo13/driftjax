# Solvers — Newton, Continuation, PTC

DriftJax uses Newton's method to solve the nonlinear DDP system at each bias point, with several globalization strategies for robustness.

## Newton Solver

### Damped Newton

The basic update is:

```
J · dx = −F(x)
x_{new} = x + logdamp(dx)
```

where `logdamp` applies component-wise log damping to oversized steps:

```
logdamp(dx) = sign(dx) · log(1 + |dx| · 1.72)     if |dx| > 1
             dx                                      otherwise
```

This prevents the Newton step from overshooting in regions where the Jacobian is poorly conditioned.

**Code:** `solvers/newton.py:logdamp()`, `_step_newton_impl()`

### Convergence Criteria

The solver tracks two independent convergence metrics:

1. **Step norm**: `‖dx‖_∞ < tol` — the Newton update is small
2. **Residual gate**: `‖F‖_∞ < f_tol` — the residual is small (independent of Jacobian quality)

Both must be satisfied for certified convergence. The residual gate is critical for near-degenerate Jacobians where the step norm can be small even when the residual is large.

### Best-E iterate Tracking

The solver maintains the **lowest-residual iterate** seen. When Newton steps diverge (common near Voc on ill-conditioned devices), the best iterate is returned with a `stagnated=True` flag.

**Rebound detection**: if the residual at the final iterate is materially worse (>10×) than the best, the best is returned. This catches the "junk endgame" failure mode.

**Code:** `solvers/newton.py:_solve_newton_python()`, `_solve_newton_while()`

### Fused Residual + Jacobian

In the default path (`fused=False`), `comp_F` and `banded_jacobian` are called **separately**, each computing carrier statistics (n, p, ni) independently. With `fused=True`, `fused_residual_and_jacobian` computes carrier stats **once** and shares them between the residual and Jacobian.

**Performance**: The fused path saves ~6–15% per Newton iteration by eliminating redundant carrier-stat evaluations.

**Code:** `numerics/fused_kernels.py:fused_residual_and_jacobian()`

## Equilibrium Solver

The equilibrium solve (zero bias) is a **Poisson-only** problem: φ_n = φp = 0 are frozen, and only φ is solved.

### Thomas Algorithm

The equilibrium Jacobian is a scalar tridiagonal system (1 unknown per node), solvable by the standard Thomas algorithm in O(N).

**Code:** `solvers/newton.py:_eq_tridiag()`, `_thomas()`, `solve_eq()`

### Equilibrium Guess

For the bias sweep, the equilibrium solution is used as the starting point for the first bias. The `equilibrium_guess` function provides a smooth initial potential profile.

**Code:** `solvers/continuation.py:equilibrium_guess()`

## Bias Sweep (Continuation)

The bias sweep solves the coupled DDP system at N_bias voltage points from 0 to V_max.

### QFL Hot-Start

Each Newton solve is initialized from the **previous bias point's solution**, using the quasi-Fermi level (QFL) extrapolation to estimate the potential at the new bias. This dramatically reduces the number of Newton iterations needed per bias point.

**Code:** `solvers/continuation.py:sweep()`

### Linear Extrapolation

For the first bias point (or when the QFL hot-start fails), a linear extrapolation from the equilibrium solution is used.

**Code:** `solvers/continuation.py:linear_extrapolation()`

### find_voc

The open-circuit voltage is found by **root-finding** on J(V) = 0, using the sweep solution as a bracket.

**Code:** `solvers/continuation.py:find_voc()`

## Pseudo-Transient Continuation (PTC)

For hard-to-converge problems (degenerate doping, large band offsets), PTC replaces the Newton step with a pseudo-time relaxation:

```
(J + C/dt) · dx = −F
```

where C = |diag(J)| is the Bank–Rose mass matrix and dt grows over time.

### Why It Works

For M-matrix Jacobians (which the DDP operator is), J + C/dt remains an M-matrix for every dt. This guarantees the damped walk is a **descent direction** and cannot park on a spurious ‖F‖ local minimum.

### Implementation

The PTC solver:
1. Computes the full dense Jacobian J via `F_jacobian`
2. Adds the diagonal mass: `J + diag(C) / dt` (O(N) via indexed add)
3. Solves via dense or mixed-precision linear algebra
4. Accepts/rejects via Armijo on ‖F‖₂
5. Switches to Newton tail once ‖F‖₂ < switch_f

**Code:** `solvers/ptc.py:solve_ptc()`

## Line-Search Newton

The Armijo line-search variant is the **recommended globalization** for production use:

```
x_{new} = x + α · p
```

where α is chosen by backtracking to satisfy the Armijo condition:

```
‖F(x + αp)‖₂² ≤ ‖F(x)‖₂² · (1 − 2·c·α)
```

This is **globally convergent** to a critical point and **quadratic near the root**.

**Code:** `solvers/ptc.py:solve_newton_ls()`

## Globalization Strategy

The default `globalization="auto"` in `solve_newton`:

1. Try log-damped Newton (fast)
2. On failure, fall back to line-search Newton (robust)
3. As a last resort, fall back to dense solve

**Code:** `solvers/newton.py:solve_newton()` (lines 390–444)

## Transient Solver

The backward-Euler transient solver handles time-dependent problems:

```
C(V) · dV/dt + F(V) = 0
```

where C = diag(n, p, 0) per node is the carrier storage capacitance.

### Implementation

Each time step solves a modified Newton system using banded Jacobian + pivoted banded solve (LAPACK dgbsv):

```
(J + C/dt) · dx = −F_trans
```

with Tikhonov regularization and line search for robustness.

**Code:** `solvers/transient.py:solve_transient_step()`, `solve_transient()`

### AC Small-Signal Analysis

The AC admittance Y(ω) is computed by linearizing around a DC operating point:

```
Y(ω) = I_x^T · (jωC + J)^{-1} · F_v
```

At ω → 0, this reduces to the DC terminal sensitivity dI/dV by the IFT.

**Code:** `solvers/transient.py:ac_small_signal()`

## Nelder-Mead Optimizer

For problems where gradient-based line searches overshoot and oscillate (e.g., razor-thin V-shaped objective valleys), DriftJax provides a derivative-free Nelder-Mead simplex optimizer:

```python
from driftjax.optimize import nelder_mead

result = nelder_mead(
    objective,
    x0=initial_guess,
    maxiter=200,
)
```

The Nelder-Mead optimizer converges on valleys where SLSQP and L-BFGS-B fail, recovering exact device parameters to machine precision (e.g., MSE $5.1\times10^{-20}$ on example 13).

**Code:** `optimize/optimizers.py:nelder_mead()`
