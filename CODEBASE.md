# DriftJax v0.1.7 — Comprehensive Codebase Documentation

> **Scope:** `driftjax_v0.1.7` (`driftjax` 0.1.7, Python ≥3.11, JAX ≥0.10, `JAX_ENABLE_X64=1`). End-to-end differentiable 1-D drift-diffusion–Poisson (DDP) solar-cell simulator. All paths relative to repository root.

---

## 1. What DriftJax Is

**DriftJax** solves the Van Roosbroeck DDP system (Poisson + electron/hole continuity with Scharfetter–Gummel discretization) and differentiates it end-to-end via the implicit function theorem (IFT). Three key departures from ∂PV (Mann et al., CPC 2022):

1. **Single composable entry point** `simulate(device, protocol, *, solver, optics, adjoint)` (`src/driftjax/simulate.py:540`) with pluggable `Sweep`/`Equilibrium` (protocol), `Newton`+`BlockThomas` (solver), `BeerLambert`/`TMM` (optics), `ImplicitAdjoint`/`DirectAdjoint` (adjoint).
2. **Direct block-tridiagonal core:** analytic Scharfetter–Gummel Jacobian (`src/driftjax/numerics/analytic_jacobian.py:151`) solved by **Block-Thomas** `O(N)` (`src/driftjax/numerics/block_thomas.py:60`), no GMRES/ILU. The O(N) complexity does not guarantee numerical stability for arbitrary block-tridiagonal Jacobians; in strongly ill-conditioned device states, intermediate block elimination may amplify roundoff, motivating the residual-based selection of a robust dense fallback.
3. **Fused physics kernels:** `fused_residual` and `fused_jacobian_banded` compile the entire physics assembly chain into a single XLA program. In GPU benchmarks, fused kernels reduce measured memory traffic by 3–4× (not reflected in CPU benchmarks). **Enabled by default** in `Newton(fused=True)` and `Sweep(fused=True)`.

The IFT adjoint is a `custom_vjp` over the design (`src/driftjax/simulate.py:372,492`, `src/driftjax/autodiff/adjoint.py`), providing the IFT gradient through standard JAX differentiation. `ImplicitAdjoint(method="auto")` is the default: it attempts the O(N) banded transpose on the actual adjoint RHS, validates `r_adj < 1e-8`, and falls back to dense LU when ill-conditioned. Validated against central finite differences to a maximum relative discrepancy of `5.56e-7` for the tested sensitive parameters. `13–26×` SLSQP wall-clock at equal budget (§4.2).

---

## 2. The Physics: Drift-Diffusion–Poisson System

### Governing Equations

```
Poisson:    ε₀∇·(ε∇φ) = q(n - p + Na - Nd)                    (1a)
Continuity: ∇·Jn = -q(G - R),    ∇·Jp = +q(G - R)              (1b)
Currents:   Jn = qμn n ∇φn,      Jp = qμp p ∇φp               (1c)
```

Carrier densities (Boltzmann statistics):
```
n = Nc exp((φn + χ + qφ) / kT)
p = Nv exp((-φp - χ - Eg - qφ) / kT)
```

Recombination: `R = R_rad + R_Auger + R_SRH` where:
```
R_rad  = Br(n·p - ni²)
R_Auger = (Cn·n + Cp·p)(n·p - ni²)
R_SRH  = (n·p - ni²) / D_SRH
D_SRH  = tn(n + ni·exp(Et/kT)) + tp(p + ni·exp(-Et/kT))
```

### Discretization

Scharfetter–Gummel (SG) finite differences on a 1-D uniform/non-uniform grid:
- Potentials `(φn, φp, φ)` at N grid nodes `x_i`
- SG currents `Jn, Jp` on slabs between nodes
- Poisson: harmonic-ε faces, central-difference Laplacian
- **Key structural property:** the 3N×3N Jacobian is **block-tridiagonal** with 3×3 blocks

### Optical Generation

Beer–Lambert (default): `G(x) = Σᵢ (λ̃ᵢ Ĩᵢ / hc) α(λ̃ᵢ,x) exp(-∫₀ˣ α(λ̃ᵢ,x')dx')`
- Tauc absorption: `α(λ,x) ∝ √(hc/λ - Eg(x))` with soft-clamp via `jnp.maximum`
- TMM: coherent transfer-matrix for thin-film interference (requires tabulated α)
- 12-point Gauss–Legendre quadrature over AM1.5G spectrum

---

## 3. High-Level Data Flow

```
Device (simulator.py:116)               # user-facing pytree, cm + doping
 └─ DeviceDesign (fields.py:139)        # dimensionless (Vt,L0) leaves, tracer-safe
      ├─ init_cell(design, light, optics) (simulator.py:262) → PVCell (fields.py:226) + G(x)
      │     └─ optics.generation() (science/optics.py:338, optics/api.py:24)
      ├─ boundary_eq / boundary_bias (science/contacts.py:51,57) + equilibrium_guess (solvers/continuation.py:15)
      ├─ solve_eq (solvers/newton.py:138) Thomas O(N) + sweep (solvers/continuation.py:43) Newton per bias
      │     └─ solve_newton (solvers/newton.py:255) Block-Thomas or dense ← comp_F / jacrev (numerics/residual.py:25)
      └─ Solution (solution.py:15) ← voltages/current/potentials eff/voc/ff/jsc/pmax ← _mpp PCHIP (numerics/spline.py:47)
simulate() (simulate.py:540) wires DeviceDesign → cell → eq (once, seeds sweep) → sweep → Solution
  + differentiable core: _simulate_sweep / _eq (simulate.py:372,492) @custom_vjp
    _sweep_bwd 389 folds eff/voc/ff/jsc/current via postprocess_full 401 where(isfinite)+hoisted vjp(cell_of_d)443 vmap per_bias 475 lam=J⁻ᵀg
```

**Strategy objects** (`equinox.Module`):
- `problems.py:16`: `Equilibrium`/`Sweep(vmax, n_steps, fused=True, batched, refinement)`
- `solvers/api.py`: `Newton(rtol=1e-8, max_steps=100, globalization="auto", dense=False, refinement=False, fused=True)` + `BlockThomas(batched=False)`
- `optics/api.py:18`: `BeerLambert(alpha_mode="tauc")` / `TMM(alpha_mode="table")`
- `adjoint/api.py`: `ImplicitAdjoint(method="auto")` / `DirectAdjoint` (deprecated alias)

**Adjoint method selection:** The `auto` method (default) attempts the O(N) banded transpose on the *actual* adjoint RHS, validates the residual `r = ‖J^T λ - g‖/(‖g‖+1e-30)`, and falls back to dense LU when `r ≥ 1e-8`. The `probe_adjoint_method` (pre-optimization screening) checks the worst-case residual with dummy `g=1`, but a small dummy residual does NOT imply correct physical gradient (ill-conditioned perovskite probe `r=1.73e-9` yet true gradient diff `0.11`).

---

## 4. Component-Level Performance Profile

Measured on 1-core x86_64, JAX 0.10.2, `JAX_ENABLE_X64=1`, N=500, 61 bias steps.

### 4.1 Timing Taxonomy

All benchmarks use one of three states:
- **Cold**: first invocation after fresh Python process (includes XLA compilation)
- **Warm**: invocation after JIT cache is populated (steady-state)
- **One-step**: single micro-benchmark call (post-JIT)

### 4.2 Forward Solver Timing

| Operation | Cold | Warm | Notes |
|-----------|------|------|-------|
| `init_cell` (optics + design) | — | 6 ms | One-time setup |
| `thermal_scales` | — | 1 ms | One-time |
| Equilibrium (6 Newton iters) | — | 24 ms | One-time, seeds sweep |
| Single Newton step (banded) | — | 1.5 ms | Post-JIT |
| Single Newton step (dense) | — | 202 ms | 135× slower (expected) |
| `comp_F` (1 bias) | 37 ms | 3 ms | Fused kernel, post-JIT |
| `banded_jacobian` (1 bias) | 40 ms | 3 ms | Fused kernel, post-JIT |
| `block_thomas_solve` (N=500) | 419 ms | 10 ms | `lax.fori_loop`, post-JIT |
| `F_jacobian` dense (1 bias) | 1677 ms | — | Only for dense path |
| **Full forward sweep (61 biases)** | ~25 s | **1.7 s** | Post-JIT |

### 4.3 Adjoint Timing

| Operation | Cold | Warm | Notes |
|-----------|------|------|-------|
| One bias adjoint (banded + vjp_cell) | — | 908 ms | Backward pass |
| One bias adjoint (dense + vjp_cell) | — | 2270 ms | 2.5× slower |
| **Full value_and_grad (JIT)** | ~40 s | **13.0 s** | End-to-end gradient |

### 4.4 Cost Breakdown per Bias Step (Banded Path, Post-JIT)

| Component | Time | Fraction |
|-----------|------|----------|
| `comp_F` (residual) | 3 ms | 0.3% |
| `banded_jacobian` | 3 ms | 0.3% |
| `block_thomas_solve` | 10 ms | 1.1% |
| **`vjp_cell` (backward physics VJP)** | **700 ms** | **77%** |
| Other (hoisting, scaling) | 192 ms | 21% |
| **Total per bias** | **908 ms** | 100% |

**Key insight:** The dominant long-term cost is the backward-pass VJP through the physics (700 ms per bias = 77%), **not** the linear solve itself (10 ms = 1.1%). The Block-Thomas solve is the bottleneck only during cold-start XLA compilation (419 ms first call).

### 4.5 Dense vs Banded Adjoint

| Device | Banded | Dense | Speedup | Gradient diff |
|--------|--------|-------|---------|---------------|
| Si p-n (well-conditioned) | 604 ms | 5184 ms | 8.6× | 5.6×10⁻⁷ |
| Perovskite p-i-n (ill-conditioned) | 783 ms | 5219 ms | 6.7× | 0.11 |

---

## 5. Package Layout (`src/driftjax/`)

### 5.1 Core Architecture

| Path | Role | Key Functions/Classes |
|------|------|----------------------|
| `__init__.py` | Enables `x64`, re-exports public API | `__version__ = "0.1.7"`, exports `simulate, Device, Material, Sweep, Newton, BeerLambert, TMM, Solution, load_material, ImplicitAdjoint, fused_residual, fused_jacobian_banded, fused_generation` |
| `fields.py` | Core pytree types (all `eqx.Module`, immutable, float64) | `Potentials(phi_n, phi_p, phi)` — 3 unknown fields; `BoundaryConditions(phi0, phiL, neq0, neqL, peq0, peqL)` — 6 contact scalars; `Material(Chi, Eg, eps, Nc, Nv, mn, mp, tn, tp, Et, Br, Cn, Cp, A, alpha, Lambda)` — 16 physical params; `LightSource(Lambda, P_in, kind)` — incident spectrum; `DeviceDesign(x, dgrid, ...)` — 23 fields (mesh + per-node + contacts + T); `PVCell(...)` — design + G(x) + statistics; `pot2vec`/`vec2pot` — interleaved `[φn₀,φp₀,φ₀,...]` layout |
| `simulator.py` | Device construction and cell initialization | `Device(layers, n_points, Snl/Snr/Spl/Spr, alpha_mode, T)` — user-facing; `Device.design()` → `DeviceDesign`; `init_cell(design, ls, alpha_mode, statistics, optics)` → `PVCell` + generation profile |
| `simulate.py` | Single entry point + differentiable core | `simulate(device, protocol, *, solver, optics, adjoint, progress, statistics, T, ls, init)` → `Solution`; `_simulate_sweep`/`_simulate_eq` — `@custom_vjp` wrappers; `_sweep_fwd`/`_sweep_bwd` — adjoint backward pass; `_adjoint_solve` — auto/banded/dense fallback |
| `solution.py` | Immutable result container | `Solution(voltages, current, potentials, cell, eff, voc, ff, jsc, pmax, eq_pot, P_in, protocol)`; `iv_curve()` → `(V, J)`; `at_bias(v)` → `Potentials`; `plot()` → PNG |
| `units.py` | Dimensionless scaling | `thermal_scales(T)` → `{length, energy, current, gratedens, time, velocity, Vt}`; constants: `Vt = kT/q`, `L0 = √(ε₀kT/q²n₀)`, `n₀ = 1e19 cm⁻³`, `μ₀ = 1 cm²/Vs`; `scale(param_name, value)` → dimensionless |
| `io.py` | Material database | `material(**kwargs)` → `Material`; `load_material("Si")` → `Material` from `materials.yaml`; `list_materials()` → 25 keys; `@lru_cache` on YAML parse |
| `config.py` | Runtime configuration | `Mode` class + tolerances |
| `console.py` | Live progress reporting | `SweepProgress(n, vmax_v, label, design, alpha_mode)` — TTY live bar + piped text; `DebugLog` — per-Newton-step trace |

### 5.2 Numerics

| Path | Role | Algorithm | Key Details |
|------|------|-----------|-------------|
| `numerics/residual.py` | Residual assembly | `comp_F(cell, bound, pot) → (3N,)` | Interleaved: `[ct_phin0, ct_phip0, ct_phi0, dd_n[1:-1], dd_p[1:-1], pois[1:-1], ct_phinL, ct_phipL, ct_phiL]`; `F_jacobian = jax.jacrev(comp_F)` — dense (3N,3N) reference |
| `numerics/analytic_jacobian.py` | Block-tridiagonal Jacobian | `banded_jacobian(cell, bound, pot) → (A, B, C)` | `A[i]` diagonal (N,3,3), `B[i]` super (N-1,3,3), `C[i]` sub (N-1,3,3); hand-coded partials `_Jn_deriv`, `_Jp_deriv` with Taylor regularization for `|Δψ| < 1e-5`; `dense_from_blocks(A,B,C)` → (3N,3N) for testing |
| `numerics/block_thomas.py` | O(N) linear solver | `block_thomas_solve(A, B, C, b) → (N,3)` | Forward elimination: `W = C[i-1] · inv(A[i-1])`, `A[i] -= W·B[i-1]`; Back substitution: `x[N-1] = inv(A[N-1])·b[N-1]`, `x[i] = inv(A[i])·(b[i] - B[i]·x[i+1])`; `_inv3` closed-form 50 flops; `lax.fori_loop` for JIT; `block_thomas_solve_batched` for vmap across bias |
| `numerics/fused_kernels.py` | XLA-fused physics | `fused_residual`, `fused_jacobian_banded`, `fused_generation` | Single XLA program per physics op; `FusedKernelManager(use_fused=True)` wrapper; verified identical to unfused to 1e-12 |
| `numerics/scharfetter_gummel.py` | SG fluxes | `Jn(cell, pot)`, `Jp(cell, pot)` | Bernoulli `B(x) = x/(eˣ-1)` with `|x| < 1e-5` Taylor fallback; `Psi_n/p` for drift term |
| `numerics/poisson.py` | Poisson operator | `poisson(cell, pot)` | `harmonic_ave_eps(cell)` — harmonic-mean ε at faces; central-difference Laplacian |
| `numerics/spline.py` | MPP extraction | `calcPmax_cubic(v, j, P_in)` | PCHIP interpolation; `1e-24` NaN guard on `√max(disc,0)` and `−c/(2b)` |
| `numerics/mixed_precision.py` | FP32→FP64 refinement | `solve_refined(J, rhs, tol)` | Iterative refinement: solve in FP32, compute residual in FP64, iterate |
| `numerics/linalg.py` | Unified linear solve | `linsolve(J, rhs, backend="auto")` | Backends: `"csr"` (row-equilibrated scipy spsolve), `"banded"` (Block-Thomas), `"dense"` (jnp.linalg.solve), `"auto"` (CSR + dense fallback on bad residual) |
| `numerics/drift_diffusion.py` | DD helpers | `ddn(cell, pot)`, `ddp(cell, pot)` | Continuity equations: `∂n/∂t = ∇·Jn + G − R` (discretized) |
| `numerics/mesh.py` | Grid utilities | Uniform/non-uniform mesh generation | `node_spacing`, `mesh_quality` |

### 5.3 Science

| Path | Role | Key Functions |
|------|------|---------------|
| `science/optics.py` | Optical generation | `beer_lambert_G(design, ls, alpha_mode)` — incoherent Beer-Lambert; `tmm_generation(design, ls, alpha_mode)` — coherent transfer-matrix; `generation_profile(design, ls, alpha_mode)` — unified dispatcher; `alpha_tauc(design, ls)` — Tauc with `softplus(0.1Vt)` clamp; `alpha_from_table(design, ls)` — interpolated tabulated α; `fresnel` coefficients for TMM |
| `science/contacts.py` | Boundary conditions | `boundary_eq(cell)` — equilibrium: ohmic `φ(0)=−ΦMl/q`, `φn=φp=φ`, flatband; `boundary_bias(cell, v)` — bias: `φn(0)−φn(L)=v`, `φ(0)=−ΦMl/q`, SRV `Sn/Sp`; `contact_phi/phin/phip(cell, bound, pot)` — Dirichlet residuals; `eq_carrier_dens(cell)` — equilibrium `n_eq, p_eq` |
| `science/spectrum.py` | AM1.5G spectrum | `spectrum(normalize=False)` → `LightSource` (raw W/m²/nm); `AM15G()` → raw data; 12-point Gauss–Legendre quadrature |
| `science/recombination.py` | Recombination rates | `R_rad(Br, n, p, ni)` = `Br(np − ni²)`; `R_Auger(Cn, Cp, n, p, ni)` = `(Cn·n + Cp·p)(np − ni²)`; `R_SRH(tn, tp, Et, n, p, ni)` = `(np − ni²)/D_SRH` |
| `science/carrier_statistics.py` | Carrier statistics | `n(cell, pot)`, `p(cell, pot)` — Boltzmann/FD/Blakemore; `ni(cell)` — intrinsic density; `F_half(eta)` — Fermi–Dirac integral (Blakemore approximation) |
| `science/tandem.py` | Tandem coupling | `series_two_terminal(top_sol, bottom_sol)` — ideal recombination; `W(V) = V_top + V_bot` |

### 5.4 Solvers

| Path | Role | Algorithm |
|------|------|-----------|
| `solvers/newton.py` | Newton solvers | `solve_eq(cell, bound, phi_ini)` — equilibrium: scalar Thomas O(N), 6 iters typical; `solve_newton(cell, bound, pot_ini, tol, max_steps, globalization, dense, refinement, fused)` — coupled: log-damp + Block-Thomas; `_solve_newton_python` — eager loop with `float(err)` early exit; `_solve_newton_while` — trace-safe `lax.while_loop`; `_step_newton_jit` — one compiled step (static `dense/refinement/analytic/fused`); `logdamp(move)` = `sign(move)·log(1+|move|·1.72)` for `|move|>1` |
| `solvers/continuation.py` | Sweep driver | `sweep(cell, vmax, n_steps, ...)` — bias ramp with QFL hotstart; `equilibrium_guess(cell)` — analytic flatband; `find_voc(v, j)` — linear interpolation of zero crossing; `total_current(cell, pot)` — `Jn + Jp` at midpoint; `qfl_hotstart(pot_eq, v)` — QFL-split initial guess |
| `solvers/batched.py` | Batched solver | `sweep_batched(cell, vmax, n_steps, ...)` — vmap Newton over bias axis; `_batched_step` — vmapped `comp_F` + vmapped `banded_jacobian` + `block_thomas_solve_batched`; `block_thomas_solve_batched(A, B, C, b)` — node-major `(N, B, 3, 3)` layout |
| `solvers/transient.py` | Transient solver | `solve_transient` — backward-Euler time integration; `ac_small_signal` — `Y(ω) = dI/dV` at small frequency |
| `solvers/ptc.py` | Globalization | Pseudo-transient continuation; Armijo line-search Newton `solve_newton_ls`; `solve_ptc` for stiff basins |

### 5.5 Optimization & Adjoint

| Path | Role | Key Details |
|------|------|-------------|
| `adjoint/api.py` | Adjoint config | `ImplicitAdjoint(method="auto"|"dense"|"banded")`; `DirectAdjoint` — deprecated alias, routes to IFT with `UserWarning` |
| `autodiff/adjoint.py` | IFT `custom_vjp` core | Forward: solve (4); backward: `J^T λ = g_x` via dense LU or banded transpose; `dL/dp = λ^T (∂f/∂p) + ∂L/∂p` |
| `autodiff/checkpointing.py` | Memory optimization | `jax.checkpoint` rematerialization for backward pass |
| `optimize/optimizers.py` | SLSQP wrapper | `slsqp(f, x0, bounds, maxiter)` — `jac=True` (single call per iter); `slsqp_multistart` — diagonal preconditioning + multi-start; `adam` — optax fallback |
| `optimize/objectives.py` | Loss functions | `iv_mse(currents_trial, currents_target)` — MSE of IV curves |
| `optimize/probe.py` | Adjoint screening | `probe_adjoint_method(dev, vmax, n_steps)` → `(method, worst_r, per_bias_residuals)`; dummy `g=1` residual check; `r < 1e-6 → banded` |
| `optimize/constraints.py` | Constraint Jacobians | Analytic `jax.jacobian(g)` for band-alignment constraints |

### 5.6 Materials & Resources

| Path | Role |
|------|------|
| `resources/materials.yaml` | **25 materials:** `AlN,BN,GaAs,GaN,GaP,GaSb,Ge,InAs,InN,InP,InSb,MAPbI₃,Si,CdTe,CdS,ZnO,TiO₂,SnO₂,NiOₓ,Spiro,PCBM,CIGS,FAPbI₃,CsPbI₃,CZTS` (6 `n/k` Aspnes tables for TMM) |
| `resources/am15g.csv` | AM1.5G spectrum (wavelength, power density) |

### 5.7 Runtime & Visualization

| Path | Role |
|------|------|
| `runtime/provenance.py` | `UTC→timezone.utc` compat, `execution_metadata()` for JSON sidecars |
| `runtime/sharding.py` | `sharded_generation(design, ls, n_shards)` — wavelength-axis sharding |
| `viz/plotting.py` | `plot_iv_curve`, `plot_band_diagram`, `plot_charge`, `plot_bars` |
| `viz/style.py` | SciencePlots-derived: STIX fonts, inward ticks, 300 dpi, Witten–Okabe-Ito + viridis |

---

## 6. Data Model & Invariants

### 6.1 Dimensionless Scaling

All quantities in the solver are dimensionless. The scaling (Debye-unit convention, cf. Mann et al., CPC 2021):

| Scale | Formula | Value at 300 K |
|-------|---------|----------------|
| `Vt` (energy) | `kT/q` | 0.02585 eV |
| `L0` (length) | `√(ε₀kT/q²n₀)` | 23.94 nm |
| `n₀` (density) | — | 1e19 cm⁻³ |
| `μ₀` (mobility) | — | 1 cm²/Vs |
| `time` | `ε₀/(q·n₀·μ₀)` | 5.47 fs |
| `velocity` | `L0/time` | 4.37e5 cm/s |
| `current` | `kT·n₀·μ₀/L0` | 1.08e4 A/cm² |
| `gratedens` | `n₀·μ₀·kT/(q·L0²)` | 1.98e28 cm⁻³s⁻¹ |

`thermal_scales(T)` returns all scales for arbitrary T; `DeviceDesign.with_temperature` rescales ratios.

### 6.2 Pytree Structure

**DOF layout** (single source of truth, `fields.py:300`):
```
pot2vec(pot) → [φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, ..., φn_{N-1}, φp_{N-1}, φ_{N-1}]
```
Interleaved 3N-vector. `vec2pot` is the inverse.

**Material** (16 fields, all float64):
```
Chi (eV), Eg (eV), eps (—), Nc (cm⁻³), Nv (cm⁻³), mn (cm²/Vs), mp (cm²/Vs),
tn (s), tp (s), Et (eV), Br (cm³/s), Cn (cm⁶/s), Cp (cm⁶/s), A (cm⁻¹eV⁻¹/²),
alpha (m⁻¹, tabulated), Lambda (m, tabulated)
```

**DeviceDesign** (23 fields):
```
x (N,), dgrid (N-1,), eps (N,), Chi (N,), Eg (N,), Nc (N,), Nv (N,),
mn (N,), mp (N,), tn (N,), tp (N,), Et (N,), Br (N,), Cn (N,), Cp (N,),
A (N,), alpha (n_tab, N,), Ndop (N,),
Snl, Snr, Spl, Spr (—), PhiMl, PhiMr (eV), T (K)
```

**PVCell** = DeviceDesign + `G (N,)` + `statistics (str)` + `T (K)`

**Solution** (12 fields):
```
voltages (N_steps,), current (N_steps,), potentials (list[Potentials]),
cell (PVCell), eff (float), voc (float), ff (float), jsc (float),
pmax (float), eq_pot (Potentials), P_in (nlam,), protocol (str)
```

### 6.3 Critical Invariants (Never Break)

1. **Scaling chain:** `pot2vec` + `thermal_scales` never change without re-deriving `analytic_jacobian` + `residual`
2. **NaN guards:** `1e-24` (`spline.py:56`) + `1e-40` (`optics.py:55`) — byte-identical forward, backward finite
3. **No spsolve under AD:** Never reintroduce `lax.scan over spsolve` / unrolled Block-Thomas / unrolled Newton
4. **DirectAdjoint alias:** Routes to identical IFT gradient with `UserWarning`
5. **Material loading:** `load_material` via `materials.yaml` `200–1400 nm` `log-α`; `TMM(table)` on `A=null` zero `G`
6. **Fused kernels:** Verified identical to unfused to 1e-12 by `tests/unit/test_fused_kernels.py`

### 6.4 Numerical Stability

- **Log damping:** `logdamp(move) = sign(move) · log(1 + |move| · 1.72)` for `|move| > 1` — prevents overshoot
- **Block-Thomas `_inv3`:** Closed-form 3×3 inverse (50 flops) — no matrix inversion, no pivot
- **SG Taylor regularization:** `|Δψ| < 1e-5` → Taylor expansion to avoid `0/0` in Bernoulli derivatives
- **Auto adjoint validation:** Residual `r = ‖J^T λ - g‖/(‖g‖+1e-30)` checked on actual RHS before accepting banded
- **Perovskite ill-conditioning:** κ ~ 1e14 near Voc → Block-Thomas transpose unstable → dense LU fallback required

---

## 7. Example Gallery (v0.1.7, 18 total)

> `examples/support.py:41` `example_args --quick --output-dir` `250/31` quick `500/61` full, `new_figure` `300 dpi`, `save_figure`, `report`, `report_fom`, `cm` layers, `§5.9` 7 inline for review (18 total), `outputs ↔ docs/figures` byte-identical.

### 7.1 Timing Profile (N=500, CPU)

| # | Example | Time | Category | Bottleneck |
|---|---------|------|----------|------------|
| 01 | device_iv | 32s | Forward × 3 | Mesh convergence 125/250/500 |
| 02 | heterojunction | 17s | Forward × 1 | Single sweep |
| 03 | optics_comparison | 19s | Forward × 2 | Beer-Lambert + TMM |
| 04 | temperature | 17s | Forward × 6 | 6 temperature points |
| 05 | adjoint_sensitivity | 50s | Adjoint × 8 | 8 bias-point adjoints |
| 06 | inverse_design | 25s | Adjoint + SLSQP | ~5 SLSQP iterations |
| 07 | tandem_matching | 31s | Forward × 2 + sweep | Tandem current match |
| 08 | statistics_map | 24s | Forward × 3 | 3 carrier statistics |
| 12 | design_map | 155s | vmapped 121 | 11×11 grid + SLSQP |
| 13 | target_iv | 192s | Adjoint + SLSQP × 30 | 30-iteration optimization |
| 14 | dgsm_global | 82s | vmapped adjoint × 32 | 32 Sobol samples |
| 17 | deltapv_reproduction | 19s | Forward × 2 | ex1 + ex2 devices |
| 18 | adjoint_method | 284s | Adjoint × 2 + SLSQP | Dense vs banded head-to-head |
| 19 | adjoint_perovskite | 166s | Adjoint × 2 | Dense vs banded ill-conditioned |

### 7.2 Per-Example Details

| # | Script | Question | Result (full 500/61) | Fig |
|---|--------|----------|----------------------|-----|
|01|01_device_iv|N→η?|ex1 `Eg1.5 2×1µm` N125,250,500 `Sweep1.1V 61` — `20.0% 20.2mA 1.05V`|research_01|
|02|02_heterojunction|Cliff|CdS/CdTe `N250/500` `BeerLambert` — `13.36% 0.890V 18.22mA`|02|
|03|03_optics|TMM|MAPbI₃ `0.5µm` BL vs TMM — `9.23% vs 10.69%`|03|
|04|04_temperature|T|Si `280-340K` — `dVoc/dT ≈ -2.13 mV/K`|04|
|05|05_adjoint|IFT-FD|[Eg,τ] `jax.grad` vs FD — `max rel <1e-3`|05|
|06|06_inverse|Nelder-Mead|[th_um,logD] `8/18it` — `~16→19%`|06|
|07|07_tandem|Tandem|MAPbI₃+Si `series_two_terminal` — `top 14.5%` `20.42% at 1.8µm`|07|
|08|08_statistics|Stats|F½ `3` modes — statistics regime map|08|
|12|12_design_map|Eg/W map|`7×11` `pcolormesh`+SLSQP `5/14it` — `20.89% 1.29eV 2.5µm`|12|
|13|13_target_iv|Inverse|`iv_mse` SLSQP `probe→banded 5.6e-12` — `MSE 1.46e-2→1.86e-11` `1799.94nm`|13|
|14|14_dgsm|DGSM|Sobol `N32` `dgsm=mean(G²)` — global vs local `|∂η/∂p|`|14|
|17|17_deltapv|Parity|ex1/ex2 vs `deltapv_*.npz` — `max|ΔJ| 0.32mA`|17|
|18|18_adjoint|dense vs banded|Si well-conditioned `604 vs 5184ms 8.6×` diff `5.6e-7` — SLSQP identical|18|
|19|19_perovskite|dense vs banded ill|psc 16-param `783 vs 5219ms 6.7×` diff `0.11 ≫ 1e-6` → dense required|19|

**Developer examples:**
- `api_contract`: Solution is 53-leaf pytree with `iv_curve()`
- `adjoint_check`: IFT vs central FD `rel error < 1e-4`
- `convergence`: Mesh convergence log-log `slope -2` (second-order)
- `batching_and_jit`: `sweep_batched` + `sharded_generation` agree to `1e-14`

**Archived (not in gallery):** `09_globalized_newton`, `15_graded_bandgap`, `16_materials_database_optics`, `11_transient_small_signal` → `examples/archive/`.

---

## 8. Validation, Tests & Benchmarks

### 8.1 Validation Against ∂PV

| Problem | ∂PV | DriftJax | Speed-up |
|---------|-----|----------|----------|
| ex1 homojunction PCE | 19.98% | 20.00% | — |
| ex2 CdS/CdTe PCE | 13.31% | 13.31% | — |
| psc 16-param optimization | 21.68% (16 it) | 21.65% (34 it) | 26× |
| multi 2-param recovery | R=1.4e-8 | R=1.8e-10 | 13× |
| holistic stack | 19.83% (100 it) | 21.65% (32 it) | ~1.0× |

### 8.2 Test Suite

```
tests/
├── unit/          # 102 tests
│   ├── test_analytic_jacobian.py    # banded_jacobian == jacrev(comp_F) to 1e-9
│   ├── test_fused_kernels.py        # fused == unfused to 1e-12 (7 tests)
│   ├── test_tmm.py                  # TMM optics correctness
│   ├── test_tandem.py               # tandem coupling
│   ├── test_transient.py            # backward-Euler
│   ├── test_contacts.py             # boundary conditions
│   ├── test_materials.py            # 25-material database
│   ├── test_block_thomas.py         # O(N) solver correctness
│   └── ...
├── gradient/      # 13 tests
│   ├── test_adjoint.py              # IFT vs central FD
│   ├── test_grad_fd.py              # full pytree gradient check
│   └── test_chi_symmetry.py         # electron affinity symmetry
├── convergence/   # MMS Poisson convergence (second-order)
├── property/      # batched_sweep, sharding, sweep_properties
├── literature/    # SQ limit, SCAPS comparison
├── regression/    # test_release_validated.py (fast N=120 / slow N=500)
├── reproducibility/ # bit_identical
└── validation/    # ex1, ex2, psc, multi, holistic
```

- `pytest -m "not slow"` → **168 passed** (fast suite, ~9 min)
- `pytest -m slow` → full suite with N=500 parity + optimizer regressions
- `python -m driftjax pyramid` → L1–L10 validation gate suite

---

## 9. Build, Docs & Tooling

### 9.1 PDF Compilation

```bash
# From docs/ directory:
python build_latex.py                    # pandoc: DriftJax_paper.md → DriftJax_paper.tex
pdflatex -interaction=nonstopmode DriftJax_paper.tex       # → 33 pages, 1.8 MB
pdflatex -interaction=nonstopmode DriftJax_supplementary.tex  # → 10 pages, 3.1 MB
pdflatex -interaction=nonstopmode DriftJax_CAS.tex          # → 10 pages, 742 KB (needs cas-sc.cls)
```

TeXLive 2026 packages: `xstring`, `footmisc`, `makecell`, `multirow`, `moreverb`, `cite`, `wrapfig`, `stfloats` (stub created).

### 9.2 Configuration

- `pyproject.toml` `0.1.7` — deps: `jax/equinox/jaxtyping/numpy/scipy/optax`
- `Makefile` targets: `test / test-par -n4 / test-slow / test-unit / examples-quick / validation / lint`
- `tests/conftest.py` — `JAX_ENABLE_X64=1`, `JAX_COMPILATION_CACHE_DIR` persistent XLA cache
- `ruff` linting: `line-length=100`, `F821` per-file ignores for jaxtyping `Shape` vars
- `examples/outputs ↔ docs/figures` — 18/18 figures byte-identical after `make`

---

## 10. Key Entry Points

```python
import driftjax as dj, jax

# Basic simulation
dev = dj.Device(layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)], n_points=500)
sol = dj.simulate(dev)  # defaults: Sweep(), Newton(fused=True), BeerLambert(), ImplicitAdjoint()

# Explicit configuration
sol = dj.simulate(dev, dj.Sweep(vmax=1.1, n_steps=41, refinement=False),
                  solver=dj.Newton(), optics=dj.BeerLambert())

# Gradient computation (IFT adjoint)
grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)  # transparent, no unroll

# Equilibrium only
eq = dj.simulate(dev, dj.Equilibrium())

# Material database
dj.load_material("GaAs")  # from materials.yaml (25 materials)
dj.list_materials()        # returns all 25 keys

# Batched gradient (vmapped)
grads = jax.vmap(jax.grad(lambda d: dj.simulate(d).efficiency))(device_batch)

# Adjoint method selection
sol = dj.simulate(dev, dj.Sweep(), adjoint=dj.ImplicitAdjoint(method="auto"))  # default
sol = dj.simulate(dev, dj.Sweep(), adjoint=dj.ImplicitAdjoint(method="banded"))  # O(N)
sol = dj.simulate(dev, dj.Sweep(), adjoint=dj.ImplicitAdjoint(method="dense"))  # robust fallback

# With progress reporting
sol = dj.simulate(dev, dj.Sweep(), progress=True)
```

### 10.1 Commands

```bash
make test                  # Run fast test suite (168 passed)
make test-slow             # Run full test suite including N=500 parity
make examples-quick        # Run all examples in --quick mode
make validation            # Run ∂PV validation suite
make lint                  # Ruff linting
python -m driftjax pyramid # Run L1–L10 validation gate
python docs/build_latex.py # Regenerate LaTeX from markdown
```

---

## 11. Version History

### v0.1.7 vs v0.1.5
- **25-material DB** `resources/materials.yaml` `13→25`, `importlib.resources`+`@lru_cache`
- **Gallery fixes P0:** `5e-8 cm→5e-6 cm` window, `N=60→250/31` quick, `13` `th_nm→th_um` + `probe→banded 5.6e-12` `20×`, `03` `TMM(table)`, `18/19` head-to-head
- **Gallery 18/18** (was 23): removed `09,15,16,scaling` (+ archived 11) → `33pp 1.8M` paper
- `pyproject.toml` `0.1.7`, `__init__.py` `__version__ 0.1.7`

### v0.1.7 Optimization & Review Changes (This Session)
- **Fused kernels enabled by default** in `Newton(fused=True)` and `Sweep(fused=True)` — verified 1e-12 by test suite
- **Example 19 NaN Voc fixed** — Vmax 1.0V → 1.2V, now produces Voc=1.0886V, FF=0.3597
- **Title updated** — "DriftJax: A Structure-Aware Differentiable Drift-Diffusion Solver for Photovoltaic Device Design"
- **Abstract rewritten** — banded/dense adjoint distinction, stability-aware selection
- **Performance attribution fixed** — perovskite 26× uses dense LU fallback, not banded
- **Benchmark table** — per-iteration cost (61.1 s/iter vs 1.09 s/iter)
- **Fused kernel claim qualified** — GPU memory traffic, not CPU speedup
- **Gallery mechanics removed** — no "JSON sidecar", "byte-identical", "for review"
- **∂PV citation** — Mann et al. (2022) not (2021)
- **Timing taxonomy** — Cold/Warm/One-step unified
- **Codebase.md updated** — direct (not exact), stability limitation documented
- **PDFs compiled** — `DriftJax_paper.pdf` (33pp), `DriftJax_supplementary.pdf` (10pp), `DriftJax_CAS.pdf` (10pp)

---

## 12. Review Response Summary

The paper underwent two rounds of review. Key changes:

1. **Abstract**: Rewrote to accurately describe banded/dense adjoint distinction
2. **Title**: Changed to "Structure-Aware" to reflect the numerical architecture contribution
3. **Performance**: Fixed attribution (perovskite uses dense, not banded); added per-iteration cost
4. **Terminology**: Replaced "exact" with "direct"; "transparently" with "validated"
5. **Figures**: Reordered — gradient validation (Fig 3) before stability-aware selection (Fig 4)
6. **Gallery mechanics**: Removed internal pipeline details from scientific paper
7. **Fused kernels**: Qualified as GPU memory traffic measurement
8. **Timing**: Unified Cold/Warm/One-step taxonomy throughout

*Sources: `README.md`, `docs/DriftJax_paper.md`, `CONTEXT.md`, `examples/support.py`, `pyproject.toml`, `src/driftjax/__init__.py`, `src/driftjax/fields.py`, `src/driftjax/units.py`, `src/driftjax/solution.py`, `src/driftjax/numerics/residual.py`, `src/driftjax/numerics/analytic_jacobian.py`, `src/driftjax/numerics/block_thomas.py`, component-level profiling at N=500.*
