# DriftJax — Codebase Context (v0.1.10)

> Generated for the checkout at `/home/med/Desktop/final/open8/driftjax_v0.1.10`.
> Purpose: a grounded, source-verified map of **what this code does** and **how it works**.
>
> **Version note.** The folder and the git history are `v0.1.10` (latest commit:
> "DriftJax v0.1.10: Final manuscript with all 31 review points addressed", plus a
> scientific validation-pyramid restructure and a repair commit `c0e0342`). The
> in-tree `__version__` string (`src/driftjax/__init__.py:76`) and `pyproject.toml`
> still read `0.1.7` — a known, deliberately-inert version-string lag; treat `0.1.10`
> as the real release. If you are diffing against the older `CONTEXT.md` (v0.1.7),
> the biggest changes since then are the **Tier I–VII validation pyramid + CLI**
> (`python -m driftjax`) and the **validation test restructure**.

---

## 1. What DriftJax Is

**DriftJax** is an end-to-end **differentiable 1-D drift–diffusion–Poisson (DDP)
photovoltaic solar-cell simulator written in JAX** (`jax>=0.10`, `equinox`,
`jaxtyping`). Given a layered semiconductor device, it:

1. Solves the coupled **Van Roosbroeck** system (electron/hole drift–diffusion +
   Poisson) with Scharfetter–Gummel finite-volume discretisation to produce the
   **J–V curve** and figures of merit (**PCE/η, V<sub>oc</sub>, J<sub>sc</sub>, FF, MPP**).
2. Exposes **exact gradients** of any scalar output (e.g. `simulate(dev).efficiency`)
   w.r.t. every device-design parameter via plain `jax.grad(simulate)`, using an
   **implicit-function-theorem (IFT) adjoint** implemented as a `custom_vjp` —
   no manual adjoint code, no unrolling of the Newton iterations.
3. Supports **inverse design / optimisation**: maximum-PCE design, curve-fit material
   recovery (`optimize` module) via SLSQP / multi-start LBFGS / Adam.

It is validated against the archived ∂PV reference simulator (Mann et al., CPC 2022) to
`<0.2%` PCE at N=500 and shows 13–26× optimizer speedups at equal budget.

**Hard precision constraint.** Carrier densities ~`1e19 cm⁻³` and `Nc·Nv ~ 1e38`
**overflow float32**. So 64-bit floats are force-enabled at import
(`src/driftjax/__init__.py:18` → `jax.config.update("jax_enable_x64", True)`) and every
solve/gradient must run in **float64**; set `JAX_ENABLE_X64=1` in CI/notebooks.

### Public API surface (`src/driftjax/__init__.py`)
```python
import driftjax as dj

Si = dj.material(Eg=1.12, Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
                 mn=1400, mp=450, tn=1e-6, tp=1e-6, A=1e4)
dev = dj.Device(layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)],
                n_points=500, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)

sol = dj.simulate(dev, dj.Sweep())              # -> Solution (jit/grad/vmap-safe)
eff = dj.simulate(dev).efficiency               # power conversion efficiency
grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)  # transparent IFT gradient
```
Composable strategies are pluggable objects (Diffrax/Optimistix style):
`simulate(dev, protocol=Sweep(), solver=Newton(), optics=BeerLambert()|TMM(), adjoint=ImplicitAdjoint())`.

Top-level names: `Device, Material, material, load_material, simulate, Solution`,
`Sweep, Equilibrium, Newton, BlockThomas, BeerLambert, TMM, ImplicitAdjoint, DirectAdjoint`,
`AM15G, series_two_terminal, register_lineage, report_lineage, sharded_generation`,
`optimize, config, io, simulator`, `plot_band_diagram / plot_charge / plot_iv_curve / plot_bars`,
`Vt, current, energy`, plus the fused-kernel names (`fused_residual`, `fused_generation`,
`fused_jacobian_banded`, `fused_bernoulli_current`, `FusedKernelManager`).

---

## 2. Directory Layout (what lives where)

```
src/driftjax/
├── __init__.py          public API + x64 enable
├── __main__.py          CLI: python -m driftjax [info|simulate|pyramid|verify|validate|benchmark|release]
├── simulate.py          single composable entry point (custom_vjp forward/backward)
├── simulator.py         Device → design → relax/sweep/solve; temperature_sweep, spectral_sensitivity
├── solution.py          Solution (equinox.Module PyTree) + convenience props
├── problems.py          AbstractProblem: Equilibrium, Sweep(vmax, n_steps, fused, batched,...)
├── config.py            Mode / DEFAULT_MODE / mode_info
├── console.py           terminal progress reporter (SweepProgress, DebugLog, summary box)
├── fields.py            core types: Potentials, BoundaryConditions, Material, DeviceDesign, PVCell,
│                       LightSource, pot2vec/vec2pot (single DOF-layout source of truth)
├── io.py                load_material, material() from resources/materials.yaml (lru_cache)
├── units.py             Vt/current/energy units + thermal_scales(T)
├── numerics/            physics discretisation & linear algebra
│   ├── scharfetter_gummel.py  stable Bernoulli kernel (custom_jvp) + Jn/Jp
│   ├── drift_diffusion.py     electron/hole continuity residuals (FVM)
│   ├── poisson.py             Poisson residual (harmonic-mean ε)
│   ├── residual.py            comp_F (interleaved 3n residual), F_jacobian (jacrev), F_eq
│   ├── analytic_jacobian.py   hand-coded block-tridiagonal (A,B,C) Jacobian (fast backend)
│   ├── block_thomas.py        exact O(N) block-tridiagonal solve
│   ├── linalg.py              linear-solve dispatcher (csr/dense/auto)
│   ├── mixed_precision.py     FP32-core + FP64-refinement solve (solve_refined)
│   ├── fused_kernels.py       fused_residual / fused_generation / fused_jacobian_banded / bernoulli
│   ├── spline.py              PCHIP max-power-point extraction (calcPmax_cubic)
│   ├── mesh.py                built-in mesh
│   └── analytic_jacobian.py
├── solvers/
│   ├── api.py            AbstractLinearSolver, BlockThomas; AbstractSolver, Newton
│   ├── newton.py         damped Newton (equilibrium + coupled), trace-safe while_loop path
│   ├── continuation.py   bias warm-start (QFL hot-start + linear prediction), sweep, find_voc
│   ├── batched.py        opt-in vmapped bias sweep
│   ├── ptc.py            pseudo-transient continuation fallback
│   ├── transient.py      transient solver
│   └── logdens.py        log-density transform
├── adjoint/api.py        AbstractAdjoint: ImplicitAdjoint (method auto/dense/banded), DirectAdjoint alias
├── autodiff/             adjoint.py, checkpointing.py (+ __init__)
├── science/              contacts.py, carrier_statistics.py (Boltzmann/FD/Blakemore),
│                        recombination.py (SRH+Auger+radiative), optics.py (BeerLambert/TMM),
│                        spectrum.py (AM1.5G), tandem.py (series_two_terminal)
├── optics/api.py         AbstractOptics: BeerLambert, TMM
├── optimize/             objectives.py, constraints.py, optimizers.py (slsqp/slsqp_multistart/adam), probe.py
├── runtime/              provenance.py (lineage), sharding.py (sharded_generation), performance.py
├── validation/           scientific pyramid (see §6) + analytic.py, manufactured.py, conservation.py,
│                        convergence.py, pyramid.py, stability/, cross_code/ (deltapv_parity)
└── viz/                  plotting.py, style.py, optim.py, figs.py, io.py

resources/materials.yaml    25-material DB (Si, CdTe, perovskites, organics, ...)
tests/                      ~12 suites: unit, gradient, conservation, convergence, property,
                           literature, regression, reproducibility, validation, integration
examples/                   flagship/ + research/ (18 scripts) + developer/ + archives
validation/                 reference forward/gradient validation scripts + results
docs/                       DriftJax_paper.{md,tex,pdf}, DriftJax_supplementary, CAS build, figures
---

## 3. How It Works (Execution Flow)

```
User:  Device(layers=[(th,mat,dop),...], n_points=500, Snl=..., ...)
         │
         ▼  design()  (simulator.py / fields.py → DeviceDesign)
   dimensionless per-node arrays (Vt / L0 / 1e19 cm⁻³ units), layer bakes
         │
         ▼  simulate(device, protocol, solver, optics, adjoint, ...)  (simulate.py)
   ┌────────────────────────────────────────────────────────────────┐
   │  design → init_cell(design, ls, optics) → PVCell + generation G(x)│
   │       └─ optics.generation(): BeerLambert (Tauc α(λ)) or TMM      │
   │  solve_eq (newton.py) → equilibrium Poisson-only Newton solve     │
   │       └─ comp_F (residual.py)  +  analytic banded Jacobian        │
   │            (analytic_jacobian.py) + Block-Thomas O(N) solve       │
   │  bias sweep (continuation.py): Newton per bias point, QFL          │
   │       hot-start + linear extrapolation warm starts                │
   │  Solution (solution.py): voltages/current/potentials + η/Voc/FF/   │
   │       Jsc/pmax  (MPP via PCHIP spline, spline.py)                │
   └────────────────────────────────────────────────────────────────┘
```

### The three DOFs and the single layout invariant
Each mesh node carries three unknowns (`fields.py::Potentials`):
- `phi_n` electron quasi-Fermi potential,
- `phi_p` hole quasi-Fermi potential,
- `phi`   electrostatic potential.

All are dimensionless (energies in units of `Vt`, densities in `1e19 cm⁻³`, lengths in `L0`).
The flat state vector is the **interleaved** layout
`[φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, …]`, defined **only** in `pot2vec`/`vec2pot`
(`fields.py`). This ordering is a hard invariant: `residual.py`, `analytic_jacobian.py`,
`block_thomas.py`, and the adjoint all assume it. Changing it requires re-deriving all of them.

### Residual assembly `comp_F` (`numerics/residual.py`)
Assembles the 3n-dimensional residual `F(x)`:
- interior: electron continuity `ddn`, hole continuity `ddp` (from Scharfetter–Gummel
  currents `Jn`/`Jp` and total recombination), and Poisson `poisson` (harmonic-mean ε);
- boundary: contact residuals `contact_phin/phip/phi` from `science/contacts.py`.

`F_jacobian` is the **canonical reference** dense `(3n,3n)` Jacobian via `jax.jacrev(comp_F)`.
It is used only for verification/testing — the hot path uses the analytic backend.

### Discretisation (`numerics/`)
- **Scharfetter–Gummel** (`scharfetter_gummel.py`): finite-volume current
  discretisation using the Bernoulli function `B(z)=z/(e^z−1)`, evaluated stably
  (Taylor series near 0, clipped exponent, `custom_jvp` so the derivative is exact at
  `z≈0` — naive AD of the `where`-selected branch could emit `0/0`).
- **Poisson** (`poisson.py`): face permittivity is the **harmonic mean** `2ab/(a+b)`
  — the only average preserving `D=εE` continuity across a material jump (validated by
  MMS at order 2).
- **Carrier statistics** (`science/carrier_statistics.py`): Boltzmann (default),
  Fermi–Dirac, or Blakemore; `n`, `p`, `ni`, `EF_zero` (equilibrium guess).
### Newton solvers (`solvers/newton.py`)
Damped Newton. Two execution paths that converge to the same result:
- **Eager Python loop** when not tracing (fast, concrete `float()` early-exit);
- **`lax.while_loop`** (`_solve_newton_while`) when under `jit`/`grad`/`vmap` — one
  compiled body, runtime early-exit, trace-safe.

Linear solves route through `_linear_solve`: analytic Block-Thomas, CSR, dense, or
`mixed_precision.solve_refined` (FP32-core with FP64 refinement, with dense/CSR fallback).
A **log-damping** step (`logdamp`) tames oversized Newton moves; there is also a
globalization / PTC fallback path for hard cases.

### Bias sweep (`solvers/continuation.py`)
`0 → vmax` in `n_steps` bias points. Each step is warm-started with a quasi-Fermi-level
hot start (`qfl_hotstart`, ±V/2) then refined by linear extrapolation from the previous two
solutions, then Newton-converged. `find_voc` extracts open-circuit voltage by linear
interpolation of the current sign-change (NaN if Voc lies above the last bias — see the
v0.1.7 fix that raised perovskite `Vmax` to 1.2 V). An opt-in **batched** path
(`solvers/batched.py`) vmaps one Block-Thomas pass over the bias axis (Boltzmann, no warm
start). `Sweep.fused=True` and `Newton.fused=True` **by default** route physics assembly
through the fused XLA kernels.

### Differentiability — the IFT adjoint
`_simulate_eq` / `_simulate_sweep` in `simulate.py` are `@custom_vjp`. Backward pass
solves `λ` from `Jᵀλ = g` (the adjoint linear system) and propagates `λ` through the VJP of
the residual wrt design — so gradients are exact without unrolling Newton. This is
numerically vital: the DDP Jacobian is ill-conditioned (`κ≫1`), so unrolled-Newton
differentiation is unstable and is **deliberately not supported**; `DirectAdjoint` is an
explicit alias for `ImplicitAdjoint` (a loud `UserWarning` is emitted).

`ImplicitAdjoint.method` selects the adjoint linear solver:
- `"dense"`  — pivoted dense LU, robust, `O(N³)`;
- `"banded"` — transpose Block-Thomas on the analytic 3×3 blocks, `O(N)`;
- `"auto"` (default, recommended) — tries the `O(N)` transpose Block-Thomas on the actual
  adjoint RHS `g=∂L/∂u`, accepts when `r=‖Jᵀλ−g‖/(‖g‖+ε) < 1e-8` and `λ` finite, else falls
  back to dense LU. Probe heuristic: `driftjax.optimize.probe_adjoint_method`.

The backward pass uses `jax.vmap` over bias points (single trace, batched) so
`jax.grad(simulate)` compiles in seconds, not minutes.

---

## 4. Physics & Models

- **Van Roosbroeck system**: drift–diffusion for electrons/holes + Poisson
  `∇·(ε∇φ) = −(p − n + N_dop)`, finite-volume, Scharfetter–Gummel fluxes.
- **Contacts** (`science/contacts.py`): surface-recombination-velocity (SRV) boundary
  conditions with flatband work-function control (`PhiMl`/`PhiMr`; 0 = ohmic);
  `boundary_eq` for equilibrium, `boundary_bias(V)` for applied bias.
- **Optics** (`optics/api.py` + `science/optics.py`): `BeerLambert` (Tauc `α(λ)`) and
  coherent `TMM` (transfer-matrix); generation profile `G(x)` fed into continuity.
- **Spectrum** (`science/spectrum.py`): built-in ASTM G-173 AM1.5G (`AM15G()`,
  `spectrum()`); `LightSource(Lambda, P_in)`.
- **Tandem** (`science/tandem.py`): `series_two_terminal` — current-parametrised
  construction of the ideal two-terminal IV (current-matched plateau; validated, was
  rewritten after yielding non-monotonic MPP garbage).
- **Statistics**: Boltzmann / Fermi–Dirac / Blakemore (`statistics=` kwarg).
- **Temperature**: `T` override via `design.with_temperature(T)` (rescales energy/velocity/
  time/length; I5 review point); `temperature_sweep()` and `spectral_sensitivity()` helpers.
- **Recombination** (`science/recombination.py`): SRH + radiative (`Br`) + Auger (`Cn`,`Cp`).
- **Analytic Jacobian** (`analytic_jacobian.py`): hand-coded, fully vectorised
  3×3 block-tridiagonal blocks (A=diag, B=super, C=sub), pinned against `jacrev(comp_F)`
  by `tests/unit/test_analytic_jacobian.py` to `1e-9` on p-n, heterojunction, and biased states.
- **Block-Thomas** (`block_thomas.py`): exact `O(N)` block-tridiagonal solve (default).
- **MPP** (`spline.py`): cubic (PCHIP) max-power-point extraction with a `1e-24` NaN guard.
---

## 5. Materials, IO, Optimisation

### Materials (`resources/materials.yaml`, `io.py`)
Single **25-material YAML database** (Si, CdTe, perovskites, organics, …) loaded via
`importlib.resources` + `@lru_cache`. `material(name)` / `load_material(name)` return a
`Material`; `TMM` with `A=null` materials yields zero generation by design. Custom materials
are plain `Material(...)` constructors.

### Optimisation (`optimize/`)
- `objectives.py` — efficiency / curve-fit objectives.
- `constraints.py` — bounded design constraints.
- `optimizers.py` — `slsqp` (scipy wrapper, `jac=True`), `slsqp_multistart`, `adam`.
- `probe.py` — `probe_adjoint_method` conditioning heuristic.
Validated vs ∂PV to `<0.2%` PCE with 13–26× wall-clock speedup at equal budget.

---

## 6. Validation Science — `validation/` + CLI

The repo implements a **scientific validation pyramid** (restructured in 0.1.10; all 31
manuscript review points addressed) with seven tiers. The result is reachable two ways:

- **Tests**: `tests/validation/`, `tests/convergence/`, `tests/literature/`, `tests/gradient/`
  etc. Run via `make test` (fast) / `make test-slow` / `make test-par`.
- **CLI** (`src/driftjax/__main__.py`, `python -m driftjax …`):
  - `info` — environment/provenance record (`--json`).
  - `simulate` — canary Si n/p NP-junction run + live per-bias console bar
    (`--quick`, `--json`, `--no-progress`, `--debug-log` to an NDJSON trace).
  - `pyramid` — legacy L1–L10 pyramid.
  - `verify`  — Tier I–III: algebraic + physics verification (fast).
  - `validate` — Tier IV–VII: gradient (grad-vs-FD) + condition sweep + cross-code (∂PV
    parity) validation.
  - `benchmark` — Tier VI: forward-scaling measurement across mesh sizes.
  - `release` — full publication-quality suite (all gates + FD + condition + scaling), can
    dump JSON records to a directory.

Tiers address: I algebraic / manufactured (MMS), II physics limits (`physics/limits.py`),
III unit/invariant/conservation, IV gradient (`gradients/adjoint_vs_forward.py`,
`scalar_fd.py`), V stability (`stability/condition_sweep.py`, `solver_selection.py`),
VI performance scaling (`benchmarks/forward_scaling.py`, `scaling.py`), and
VII cross-code parity (`cross_code/deltapv_parity.py`) plus the ∂PV reproduction example
(`examples/research/17_deltapv_reproduction.py`, bundled reference IV data).

---

## 7. Build, Tooling, Commands

- **Dependencies**: Python ≥ 3.11; `jax>=0.10`, `jaxlib>=0.10`, `equinox>=0.13`,
  `jaxtyping>=0.3`, `numpy>=1.26`, `scipy>=1.13`, `optax>=0.2`. Optional: optimistix/lineax,
  matplotlib (viz), pytest/hypothesis/ruff/mypy/mkdocs (dev).
- **Install**: `pip install -e .` or `pip install -e ".[dev,viz]"`.
- **Makefile** (all wrap `JAX_ENABLE_X64=1` + XLA compile cache):
  `make test` (fast serial), `make test-par` (4 xdist workers), `make test-slow` (full suite),
  `make test-unit`, `make lint` (ruff, line-length 100), `make examples-quick` (smoke whole
  gallery), `make validation` (N=500 reference runs).
- **CI**: `.github/workflows/ci.yml`; ruff with jaxtyping F722/F821 per-file ignores; mypy
  scoped to `src/driftjax`; coverage fail-under 60%.
- **Docs**: `docs/DriftJax_paper.md` (+ `tex`/`pdf` via CAS format), mkdocs-material local
  docs, `CITATION.cff`, `CHANGELOG.md`, `CHANGES.md`.

---

## 8. Invariants (Do-Not-Break List)

1. **float64 everywhere** — `jax_enable_x64` is set at import; never remove it.
2. **DOF ordering** — interleaved `[φn,φp,φ]` layout is defined ONLY in `pot2vec`/`vec2pot`;
   a change requires re-deriving `residual`, `analytic_jacobian`, `block_thomas`.
3. **Hot path stays analytic-Block-Thomas** — never reintroduce autodiff-Jacobian Newton,
   `lax.scan` over `spsolve`, or unrolled solves in the Newton loop (`solvers/newton.py`).
   The reference `F_jacobian` (jacrev) is for verification only.
4. **NaN guards** — `1e-24` (spline MPP/Voc) and `1e-40` (optics) keep the forward run
   byte-identical; don't perturb.
5. **`DirectAdjoint` stays an alias** for `ImplicitAdjoint` with a loud warning;
   unrolled-Newton differentiation is intentionally unsupported.
6. **Materials DB is a single YAML** loaded via `importlib.resources` + `lru_cache`;
   TMM with `A=null` gives zero generation by design.
7. **Progress reporting must never affect numerics** (`simulate.py` hooks / `console.py`).
8. **`Sweep.fused` / `Newton.fused` default `True`** — the fused kernels must stay
   semantically identical to the unfused (verified to 1e-12 by the test suite).

---

## 9. Version delta quick-view (v0.1.7 → v0.1.10)

- **Scientific validation pyramid restructure** (Tier I–VII) + CLI commands
  (`verify` / `validate` / `benchmark` / `release`) across `validation/stability/`,
  `validation/benchmarks/`, `validation/gradients/`, `validation/cross_code/`.
- **Validation tests repaired** (commit `c0e0342`) for correctness (imports, BCs, material
  params).
- **Final manuscript** with all 31 review points addressed (`docs/DriftJax_paper.md/.tex/.pdf`,
  supplementary, CAS build).
- **Production defaults**: fused kernels on by default; adjoint `method="auto"`
  (stability-aware).
- Unchanged/inert: the internal `__version__` string is still `0.1.7` (not yet consolidated).

---

*Sources: `src/driftjax/**`, `pyproject.toml`, `README.md`, `Makefile`, CLI (`__main__.py`),
`CHANGES.md`, `CHANGELOG.md`, and git history (`c0e0342`, `4dd1738`, …); verified against
source 2026-08-30.*
```