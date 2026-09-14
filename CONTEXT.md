# DriftJax v0.1.7 — Code Context File

> Generated 2026-08-29 for `/home/med/Desktop/final/open8/driftjax_v0.1.7` (package `driftjax` 0.1.7).
> Purpose: a grounded map of what the codebase does, how it works, and where everything lives.

---

## 1. What DriftJax Is

**DriftJax** is an end-to-end **differentiable 1-D drift–diffusion–Poisson (DDP) solar-cell simulator written in JAX**. Given a layered semiconductor device description, it solves the Van Roosbroeck drift–diffusion + Poisson system to produce a J–V curve and figures of merit (PCE/efficiency, Voc, Jsc, FF), and — the key feature — exposes **exact gradients of any scalar output with respect to device design** through `jax.grad(simulate)` via an implicit-function-theorem (IFT) adjoint implemented as a `custom_vjp` (no manual adjoint code required).

The library also supports **inverse design**: maximizing efficiency or fitting material parameters via gradient-based optimizers (SLSQP, multi-start LBFGS, Adam).

Validated against the ∂PV reference implementation (Mann et al., CPC 2022) to <0.2% PCE at N=500, with 13–26× SLSQP wall-clock speedup at equal budget.

**Hard constraint:** carrier densities ~1e19 cm⁻³ and Nc·Nv ~ 1e38 **overflow float32**, so 64-bit mode is force-enabled at import (`src/driftjax/__init__.py:18` — `jax.config.update("jax_enable_x64", True)`). Everything must run in float64; set `JAX_ENABLE_X64=1` in CI and scripts.

---

## 2. How It Works (Execution Flow)

```
User API:
  Device(layers=[(th, mat, dop), ...], n_points=500, Snl=1e7, ...)
       │
       ▼
  simulate(device, protocol=Sweep(), solver=Newton(), optics=BeerLambert(), adjoint=ImplicitAdjoint())
       │
       ▼
  ┌─ Device → DeviceDesign (fields.py)           # dimensionless per-node arrays (Vt, L0 units)
  │   ├─ init_cell(design, ls, optics) → PVCell + generation G(x)
  │   │     └─ optics.generation() — BeerLambert or TMM (science/optics.py)
  │   ├─ solve_eq (solvers/newton.py)             # equilibrium Newton solve (Poisson-only)
  │   │     └─ solve_newton: residual comp_F (numerics/residual.py)
  │   │          + ANALYTIC block-tridiagonal Jacobian (numerics/analytic_jacobian.py)
  │   │          + Block-Thomas O(N) solve (numerics/block_thomas.py) or dense fallback
  │   ├─ sweep over bias (solvers/continuation.py) # Newton per bias point
  │   └─ Solution (solution.py)                   # voltages/current/potentials + eff/voc/ff/jsc/pmax
  │        └─ _mpp via PCHIP spline (numerics/spline.py) for max-power-point extraction
  └─ simulate() (simulate.py:540) wires: Device → DeviceDesign → cell → equilibrium (seeds sweep) → bias sweep → Solution
```

### Differentiability

`_simulate_eq` / `_simulate_sweep` are `@custom_vjp` functions (`simulate.py`). The backward pass (`_sweep_bwd`) solves the adjoint system `J^T λ = g_x` (via dense or banded Block-Thomas with auto-fallback) and propagates `λ` through a VJP of the residual w.r.t. the design — so `jax.grad(lambda d: simulate(d).efficiency)(dev)` returns exact design gradients without unrolling Newton iterations. `DirectAdjoint` is only an alias for `ImplicitAdjoint` (a warning is emitted; unrolled Newton differentiation is deliberately not supported — it is unstable on the ill-conditioned DDP Jacobian).

The backward pass uses `jax.vmap` over bias points (single trace, batched) to avoid per-bias re-tracing, which makes `jax.grad(simulate)` compile in seconds instead of minutes.

---

## 3. Physics

- **Van Roosbroeck system:** drift–diffusion + Poisson with Scharfetter–Gummel discretization (`numerics/scharfetter_gummel.py`, `numerics/drift_diffusion.py`, `numerics/poisson.py`)
- **Recombination:** SRH + radiative + Auger (`science/recombination.py`)
- **Carrier statistics:** Boltzmann / Fermi–Dirac (Nilsson) / Blakemore (`science/carrier_statistics.py`)
- **Optics:** Beer–Lambert (Tauc α or tabulated α) + coherent Transfer-Matrix Method (TMM) (`science/optics.py`, `optics/api.py`)
- **Contacts:** surface-recombination-velocity (SRV) BCs with flatband work-function control (`science/contacts.py`)
- **Spectrum:** AM1.5G (`science/spectrum.py`)
- **Tandem:** series two-terminal tandem cells (`science/tandem.py`)

### Units Convention

All internal quantities are dimensionless (energies in units of thermal voltage Vt, densities in 1e19 cm⁻³, lengths in Debye length L0); physical units are applied only at the boundary of the library (`units.py`). The scaling is temperature-dependent via `thermal_scales(T)`.

### DOF Layout (Single Source of Truth)

The unknown field is a single interleaved vector: `[φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, ...]`, defined only in `pot2vec`/`vec2pot` (`fields.py`). This layout is critical — changing it requires re-deriving `analytic_jacobian` and `residual`.

---

## 4. Package Layout (`src/driftjax/`)

| Path | Role |
|------|------|
| `__init__.py` | Public API surface + forces float64. Exports `Device`, `simulate`, `Sweep`, `Newton`, `BeerLambert`, `TMM`, `Solution`, plotting, etc. |
| `fields.py` | Core eqx PyTrees: `Potentials`, `BoundaryConditions`, `Material`, `PVCell`, `DeviceDesign`, `LightSource`, `pot2vec`/`vec2pot`. |
| `simulator.py` | `Device` (user device definition: layers/thickness/doping/SRV/work-function/T), `_make_design`, `init_cell`. |
| `simulate.py` | The single composable entry point `simulate(device, protocol, *, solver, optics, adjoint, progress, statistics, T, ls, init)` + the `custom_vjp` differentiable core (IFT adjoint). |
| `solution.py` | `Solution` PyTree: IV curve, potentials per bias, eff/voc/ff/jsc/pmax, `at_bias()`, `plot()`. |
| `problems.py` | Protocol strategies: `Equilibrium`, `Sweep(vmax, n_steps)`. |
| `units.py` | Physical constants + `thermal_scales(T)` dimensionless scaling. |
| `io.py` | Material database: `material()` + `load_material()` from `resources/materials.yaml` (25 materials) via `importlib.resources` + `@lru_cache`. |
| `config.py` | `Mode` descriptor (statistics/optics/solver axes) + numeric tolerances + env overrides. |
| `console.py` | Opt-in live progress reporting (`SweepProgress`, `DebugLog`); never affects numerics. |

### `numerics/` — The Math Engine

| File | Role |
|------|------|
| `analytic_jacobian.py` | `banded_jacobian(A, B, C)` — hand-coded 3×3 block-tridiagonal Jacobian of `comp_F`. |
| `block_thomas.py` | `block_thomas_solve(A, B, C, b)` — exact O(N) direct solver for block-tridiagonal systems. |
| `residual.py` | `comp_F(cell, bound, pot)` — full 3N-dimensional residual assembly. `F_jacobian` via `jax.jacrev` (reference). |
| `scharfetter_gummel.py` | `Jn`, `Jp` Scharfetter–Gummel fluxes via Bernoulli functions, `Psi_n`/`Psi_p`. |
| `drift_diffusion.py` | DDP transport equation assembly (`ddn`, `ddp`). |
| `poisson.py` | `poisson()` equation + `harmonic_ave_eps` for face-centered permittivity. |
| `mesh.py` | 1-D grid utilities. |
| `spline.py` | PCHIP spline for max-power-point extraction (`calcPmax_cubic`), `1e-24` NaN guard. |
| `fused_kernels.py` | `FusedKernelManager` for JIT-fused residual/generation/Jacobian computation. |
| `mixed_precision.py` | FP32/BF16 → FP64 refinement linear solve path. |
| `linalg.py` | Small dense linear-algebra helpers + `linsolve` (CSR fallback). |

### `science/` — Physics Models

| File | Role |
|------|------|
| `optics.py` | `beer_lambert_G`, `alpha_tauc` (softplus-smoothed), `alpha_from_table`, `fresnel_generation`, `tmm_generation`, `generation_profile` dispatch. |
| `contacts.py` | `boundary_eq`, `boundary_bias`, `contact_phin`/`contact_phip`/`contact_phi` (SRV + Dirichlet BCs). |
| `recombination.py` | `R_rad`, `R_Auger`, `R_SRH`. |
| `carrier_statistics.py` | Boltzmann / Fermi–Dirac / Blakemore `n(pot)`, `p(pot)`. |
| `spectrum.py` | AM1.5G `spectrum()` + raw `AM15G()`. |
| `tandem.py` | `series_two_terminal` two-terminal tandem cell. |

### `solvers/` — Newton + Continuation

| File | Role |
|------|------|
| `newton.py` | `solve_eq` (Poisson-only Thomas), `solve_newton` (coupled Block-Thomas or dense), log-damping, globalization (line-search / PTC fallback). |
| `continuation.py` | `sweep` (bias sweep with QFL hot-start), `equilibrium_guess`, `find_voc`, `total_current`, batched variant. |
| `batched.py` | `block_thomas_solve_batched` — vmap over bias axis. |
| `transient.py` | Backward-Euler transient solver + `ac_small_signal`. |
| `ptc.py` | Pseudo-transient continuation. |

### `autodiff/` — Adjoint Engine

| File | Role |
|------|------|
| `adjoint.py` | IFT `custom_vjp` core — the backward pass solves `J^T λ = g_x`. |
| `checkpointing.py` | `jax.checkpoint` rematerialization for gradient memory reduction. |

### `adjoint/` — Strategy API

| File | Role |
|------|------|
| `api.py` | `ImplicitAdjoint(method="dense"|"banded")` / `DirectAdjoint` (alias). |

### `optics/` — Strategy API

| File | Role |
|------|------|
| `api.py` | `BeerLambert(alpha_mode)` / `TMM(table)`. |

### `optimize/` — Inverse Design

| File | Role |
|------|------|
| `optimizers.py` | SLSQP, multi-start, Adam. |
| `objectives.py` | Objective functions for optimization. |
| `constraints.py` | Bound and inequality constraints. |
| `probe.py` | `probe_adjoint_method` — picks banded vs dense adjoint by worst residual (r < 1e-6 → banded). |

### `runtime/`

| File | Role |
|------|------|
| `performance.py` | Timing utilities. |
| `sharding.py` | `sharded_generation` for parallel optical generation. |
| `provenance.py` | Lineage/reproducibility registration. |

### `validation/` — In-Library Validation Pyramid

| File | Role |
|------|------|
| `pyramid.py` | L1–L10 gate suite (`python -m driftjax pyramid`). |
| `analytic.py` | Analytic reference solutions. |
| `manufactured.py` | Method of Manufactured Solutions (MMS) for Poisson. |
| `conservation.py` | Current/charge conservation identities. |
| `convergence.py` | Mesh convergence analysis. |

### `viz/` — Publication Plotting

| File | Role |
|------|------|
| `plotting.py` | `plot_iv_curve`, `plot_band_diagram`, `plot_charge`, `plot_bars`. |
| `style.py` | 300 dpi publication theme. |
| `figs.py` | Figure utilities. |
| `io.py` | Figure I/O. |

### `resources/`

| File | Role |
|------|------|
| `materials.yaml` | **Single validated DB** with 25 materials (Si, GaAs, CdTe, MAPbI₃, CIGS, organics, etc.). 6 materials have tabulated n/k (Aspnes tables) for TMM. Loaded via `importlib.resources` + `@lru_cache`. |

---

## 5. Key Entry Points

```python
import driftjax as dj
import jax

# Load a material
Si = dj.load_material("Si")  # or dj.material(Eg=1.12, Chi=4.05, ...)

# Define a device
dev = dj.Device(layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)],
                n_points=500, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)

# Simulate
sol = dj.simulate(dev, dj.Sweep())           # AM1.5G by default; jit/grad/vmap-safe
v, j = sol.voltages, sol.current             # V (volts), j (A/cm^2)
eff, voc, jsc, ff = sol.efficiency, sol.voc, sol.jsc, sol.ff

# Exact IFT design gradient
grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)

# Composable strategies:
sol = dj.simulate(dev, protocol=dj.Sweep() | dj.Equilibrium(),
                 solver=dj.Newton(), optics=dj.BeerLambert() | dj.TMM(),
                 adjoint=dj.ImplicitAdjoint())

# Progress reporting
sol = dj.simulate(dev, progress=True)  # live terminal progress

# Equilibrium only
eq = dj.simulate(dev, dj.Equilibrium())

# Inverse design
result = dj.optimize.maximize_efficiency(dev, bounds=...)
```

---

## 6. Tests, Validation, Examples

### Tests (`tests/`)

Organized as a validation pyramid:

| Directory | Coverage |
|-----------|----------|
| `unit/` | TMM, materials, contacts, block-Thomas, fused kernels, mixed precision, transient, checkpointing, public API, spline, spectrum, temperature, statistics, analytic Jacobian, Bernoulli, tandem, units, provenance, console. |
| `gradient/` | Adjoint vs FD, chi symmetry, inverse-design gradients, ab-adjoint. |
| `conservation/` | Current/charge identity tests. |
| `convergence/` | MMS Poisson, spline mesh convergence. |
| `property/` | Hypothesis invariants, batched sweep, sharding, sweep properties, fused simulate. |
| `literature/` | Shockley–Queisser limit, SCAPS CdTe benchmark, ideal diode, experimental data. |
| `regression/` | `test_release_validated.py` (the release gate; fast structural + slow N=500 parity), `test_iv_reference`, `test_files_pad_mechanisms`. |
| `reproducibility/` | Bit-identical rerun tests. |
| `validation/` | Pyramid gate tests. |

### Validation (`validation/`)

Reference forward validations at N=500: ex1 np-junction (19.98% vs ∂PV 20.00%), ex2 heterojunction (13.31% vs 13.31%), PSC (21.65% vs 21.68%). Gradient-vs-FD checks, adjoint timing, numerics notes.

### Examples (`examples/`)

17 total (15 research + 2 developer), each accepting `--quick` and `--output-dir`, writing one figure + reproducibility JSON + `EXAMPLE_RESULT` summary line.

| # | Script | Question |
|---|--------|----------|
| 01 | `01_device_iv.py` | Mesh convergence: N→η |
| 02 | `02_heterojunction_physics.py` | CdS/CdTe cliff heterojunction |
| 03 | `03_optical_model_comparison.py` | Beer-Lambert vs TMM |
| 04 | `04_temperature_recombination.py` | Temperature dependence |
| 06 | `06_constrained_inverse_design.py` | Bounded optimization |
| 07 | `07_tandem_current_matching.py` | Two-terminal tandem |
| 08 | `08_statistics_regime_map.py` | Carrier statistics sensitivity |
| 09 | `09_srv_contacts.py` | Surface recombination velocity |
| 10 | `10_generation_profile.py` | Generation profile analysis |
| 11 | `11_transient_small_signal.py` | Transient small-signal response |
| 12 | `12_material_thickness_design_map.py` | Eg/width design map + SLSQP |
| 13 | `13_target_iv_structure.py` | Inverse design: fit target IV |
| 14 | `14_dgsm_global.py` | Global sensitivity analysis |
| 15 | `15_deltapv_crosscode.py` | ∂PV crosscode parity |
| 16 | `16_graded_doping.py` | Graded doping profiles |

Developer: `developer_adjoint_check`, `developer_batching_and_jit`.

---

## 7. Build, Tooling, Commands

### Dependencies

- Python ≥ 3.11; JAX ≥ 0.10, jaxlib ≥ 0.10, equinox ≥ 0.13, jaxtyping ≥ 0.3, numpy ≥ 1.26, scipy ≥ 1.13, optax ≥ 0.2
- Optional: optimistix/lineax (advanced solvers), matplotlib (viz), pytest/hypothesis/ruff/mypy/mkdocs (dev)

### Install

```bash
pip install -e .                  # library only
pip install -e ".[dev,viz]"       # + tests, lint, docs, plotting
```

### Makefile Commands

All wrap `JAX_ENABLE_X64=1` + persistent XLA compile cache:

| Command | What |
|---------|------|
| `make test` | Fast suite, serial (`-m "not slow"`) |
| `make test-par` | 4 xdist workers (needs ~8 GB RAM) |
| `make test-slow` | Full suite incl. N=500 parity + optimizer regressions |
| `make test-unit` | Unit layer only |
| `make lint` | Ruff (line-length 100) |
| `make examples-quick` | Smoke-run the whole example gallery |
| `make validation` | N=500 reference runs |

### CI/CD

- `.github/workflows/ci.yml`
- Ruff lint with jaxtyping F722/F821 per-file ignores
- mypy scoped to `src/driftjax`
- Coverage fail-under 60%

### Documentation

- `docs/DriftJax_paper.md` — full methods paper (CPC style, 33 pages)
- mkdocs-material for local docs
- `CITATION.cff` + BibTeX in README

---

## 8. Invariants (Do-Not-Break List)

1. **float64 everywhere** — `jax_enable_x64` is set at import; never remove the import-time config update.
2. **DOF ordering** — the interleaved `[φn, φp, φ]` layout is defined ONLY in `pot2vec`/`vec2pot` (`fields.py`); changing it requires re-deriving `analytic_jacobian` and `residual`.
3. **Analytic Jacobian + Block-Thomas** — never reintroduce autodiff-Jacobian Newton or `lax.scan over spsolve` / unrolled solves in the hot path (`solvers/newton.py`).
4. **NaN guards** — the `1e-24` (spline) / `1e-40` (optics) guards in MPP/Voc extraction keep the forward pass byte-identical; don't perturb them.
5. **`DirectAdjoint` stays an alias** for `ImplicitAdjoint` with a loud warning (`simulate.py`); unrolled-Newton differentiation is intentionally unsupported.
6. **Materials DB is a single YAML** (`resources/materials.yaml`) loaded via `importlib.resources` + `lru_cache`; TMM with `A=null` materials yields zero generation by design.
7. **Progress reporting must never affect numerics** (`simulate.py` progress hooks).

---

## 9. Version Deltas (v0.1.7 vs v0.1.6)

- **25-material DB** in `resources/materials.yaml` (expanded from prior versions), loaded via `importlib.resources` + `@lru_cache`.
- **Gallery 18/18** — removed archived scripts, consolidated to 18 research/developer examples.
- **Paper 33 pages** with 7 inline figures in §5.9.
- `pyproject.toml` version 0.1.7, `__init__.py` `__version__ = "0.1.7"`.

---

*Sources: `README.md`, `pyproject.toml`, `Makefile`, `src/driftjax/*` (verified against source 2026-08-29), `docs/DriftJax_paper.md`, `CHANGELOG.md`, `examples/`, `validation/`, `tests/`, `CITATION.cff`.*
