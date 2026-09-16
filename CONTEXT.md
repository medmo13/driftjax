# DriftJax v0.1.17 — Context File

A differentiable 1-D drift–diffusion–Poisson (DDP) photovoltaic simulator in JAX.
Research code accompanying a Computer Physics Communications methods paper
(sources in `docs/paper/`).

Repo: `https://github.com/medmo13/driftjax` · License: MIT · Python >= 3.11,
JAX >= 0.10.2, **float64 mandatory** (`JAX_ENABLE_X64=1`, enabled on import).

---

## 1. What it does

Two capabilities:

1. **Forward simulation** — given a 1-D stacked device (layers, materials,
   doping, contacts) under a light spectrum, solve the semiconductor equations
   and return the IV curve + efficiency (`PCE`, `Voc`, `Jsc`, `FF`).
2. **Differentiation** — `jax.grad(dj.simulate)` yields exact design gradients
   (∂efficiency/∂thickness, ∂bandgap, ∂doping, …) via the **implicit-function-
   theorem (IFT) adjoint**, enabling gradient-based inverse design and
   sensitivity analysis.

Quick start:

```python
import driftjax as dj, jax
Si = dj.material(Eg=1.12, Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
                 mn=1400, mp=450, tn=1e-6, tp=1e-6, A=1e4)
dev = dj.Device(layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)],
                n_points=500, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)
sol  = dj.simulate(dev, dj.Sweep())               # -> Solution
grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)
```

Composable strategy objects (Diffrax/Optimistix/Lineax style):

```python
sol = dj.simulate(dev, dj.Sweep(), solver=dj.Newton(),
                  optics=dj.TMM(), adjoint=dj.ImplicitAdjoint())
```

### Verified running (this machine)
Interpreter: `/home/med/Desktop/final/venv_latest/bin/python`,
`PYTHONPATH=<repo>/src`, JAX 0.10.2. Si p-n homojunction, n=201:

```
PCE = 6.005 % | Voc = 0.4530 V | Jsc = 0.0165 A/cm2 | FF = 0.7211
gradient_status = CERTIFIED, max|F| = 9.32e-16     forward ~10 s (incl. compile)
```

Adjoint `d eff/dA` (Tauc prefactor) vs central FD:
`2.088832e-06` vs `2.088841e-06` → **rel. diff 4.7e-6**.

---

## 2. The physics (Van Roosbroeck system)

Three coupled nonlinear PDEs in 1-D, three unknown fields per node:

| Equation | Unknown | Form |
|---|---|---|
| Poisson | φ (electrostatic) | ∇·(ε∇φ) = −(p − n + N_dop) |
| electron continuity | φn (e⁻ quasi-Fermi) | ∂ₓJₙ = R − G |
| hole continuity | φp (h⁺ quasi-Fermi) | ∂ₓJₚ = −(R − G) |

Supporting physics, in `src/driftjax/science/`:

- **Carrier statistics** (`carrier_statistics.py`): Boltzmann (default /
  validated), Fermi–Dirac via F₁/₂ integral (128-pt Gauss–Legendre +
  Sommerfeld blend, ~1e-6), legacy Blakemore (inaccurate, provenance only).
- **Recombination** (`recombination.py`): SRH + radiative + Auger,
  all ∝ (np − ni²) (detailed balance).
- **Optics** (`optics.py`, `optics/api.py`): Beer–Lambert (Tauc α = A√(hν−Eg)
  or tabulated α), coherent **TMM**, Fresnel double-pass with rear reflector;
  AM1.5G spectrum (`spectrum.py`).
- **Contacts** (`contacts.py`): SRV (Robin) BCs; ohmic/Schottky via metal work
  function.
- **Materials** (`io.py` + `resources/materials.yaml`): 25-material database
  (Si, CdTe, perovskites, CIGS, GaAs, organics…) + custom materials.

**Nondimensionalization** (`units.py`): energy in Vt = kT/q, density in
1e19 cm⁻³, length in Debye length. Float64 is mandatory because carrier
densities ~1e19 and Nc·Nv ~1e38 overflow float32.

---

## 3. Solve pipeline

```
Device (PyTree) → DeviceDesign → PVCell (bake: mesh + G(x) from optics)
              → Newton solves → Solution
```

1. **Build** (`simulator.py`, `fields.py`). `Device` keeps thickness/doping as
   *dynamic PyTree leaves* (`jit`/`grad`/`vmap`-safe) and bakes a
   `DeviceDesign`: uniform mesh, per-node material/doping arrays in
   dimensionless units. `init_cell()` runs the optics model → generation
   profile `G(x)` → a `PVCell`.
2. **Equilibrium solve** (`solvers/newton.py:solve_eq`). Poisson-only
   (φn = φp = 0) with an exact O(n) **tridiagonal Thomas solve** on the
   analytic Poisson Jacobian, seeded from the Fermi-level guess.
3. **Bias sweep** (`solvers/continuation.py:sweep`). Bias 0 → Vmax, each point
   by damped Newton, **warm-started** from the previous solution (QFL
   hot-start, then linear extrapolation). Current = mean(Jn + Jp). Voc from
   sign-change + bisection refinement; MPP from cubic/PCHIP spline
   (`numerics/spline.py`); PCE = Pmax / P_in.
4. **Discretization** (`numerics/`):
   - **Scharfetter–Gummel fluxes** (`scharfetter_gummel.py`) with a stable
     **Bernoulli kernel** B(z) = z/(e^z − 1) (Taylor branch near 0) and custom
     JVPs protecting the derivative at z ≈ 0.
   - Finite-volume residuals (`poisson.py`, `drift_diffusion.py`,
     `residual.py`); harmonic-mean ε at faces (preserves D = εE across
     heterojunctions).
   - DOF layout invariant (defined only in `pot2vec`/`vec2pot`):
     **interleaved** `[φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, …]`.
5. **Newton + linear algebra**:
   - **Analytic block-tridiagonal Jacobian** (`analytic_jacobian.py`): 3×3
     blocks, bandwidth kl = ku = 5; pinned vs `jax.jacfwd` to ~5e-17.
   - **Pivoted banded LAPACK `dgbsv`** (`banded_solve.py`, O(N·bw²) = O(121N))
     via `scipy.linalg.solve_banded` through `jax.pure_callback`. Replaced the
     old unpivoted block-Thomas in v0.1.17 after a 5-device study showed Thomas
     failing on heterojunctions (residuals ~2–370 vs ~1e-9). Fallbacks:
     **truncated-SVD `lstsq`** on singular systems, then dense LU. Each step
     records which path ran (`fallback_used`, `gradient_status`).
   - Globalization: log-damping (default), Armijo line search,
     pseudo-transient continuation (`ptc.py`).
6. **Fast paths.** Fused `lax.scan` forward (`_forward_fused_scan`) compiles
   equilibrium + sweep into one XLA program, with a NaN guard that falls back
   to the serial sweep. Fused kernels (`numerics/fused_kernels.py`); batched
   vmapped banded solver (`solvers/batched.py`).

---

## 4. Gradients — the key innovation

Differentiating *through* Newton iterations is unstable for this
ill-conditioned Jacobian. Instead `simulate` is a `custom_vjp`
(`simulate.py:_sweep_bwd`, `autodiff/adjoint.py`) using the **implicit function
theorem**: for the converged root u*(θ) of F(u,θ) = 0,

```
du*/dθ = −J⁻¹ · ∂F/∂θ
```

Backward pass:

1. Form state Jacobian **J** (same analytic blocks) at the converged solution.
2. Solve the **adjoint system Jᵀλ = g** — pivoted **dense LU** by default.
   (An O(N) banded-transpose path was *removed* in v0.1.12: silent 10–30 %
   gradient errors on ill-conditioned devices that no cheap gate could
   certify. Banded adjoint is now opt-in: `DRIFTJAX_BANDED_ADJOINT=1`.)
3. Combine the **direct channel** (∂current/∂cell — optics/doping) with the
   **implicit channel** (∂F/∂cell)ᵀλ.
4. **Voc** treated specially: dVoc/dθ = −J_θ/J_V at the refined root, rather
   than differentiating the discrete sign-change scan.
5. Per-bias adjoint is **vmapped over the bias axis** (one traced graph) with
   `@jax.checkpoint` for memory; the design→cell map is traced once and pulled
   back once.

**Failure semantics are first-class** (`solution.py`): `converged`,
`max_residual`, `per_bias_residuals`, `fallback_used`, and a tri-state
`gradient_status` (`CERTIFIED` / `UNRELIABLE` / `UNVERIFIED`). The backward
pass re-measures max|F| and warns loudly if the primal was not converged — a
gradient is never returned silently on an unconverged state.

---

## 5. Verification & validation

Unusually well-verified research code — the point of the paper.

- **249 tests**: unit / gradient / conservation / convergence / property /
  literature / regression / reproducibility (`tests/`).
- **L1–L10 validation pyramid** (`validation/pyramid.py`): unit kernels →
  invariants → manufactured-solution mesh convergence → gradient-vs-FD →
  conservation identities → literature bounds → regression goldens →
  reproducibility → performance → backend gates.
- **Cross-code parity** vs reference **∂PV** (Mann et al., CPC) at N=500:
  Si homojunction PCE 19.89 % vs 20.00 % (0.11 pp); CdS/CdTe 13.31 % vs
  13.31 % (<0.01 pp). Also **SCAPS-1D** CdTe anchors and the
  **Shockley–Queisser limit**.
- Evidence archived in `docs/paper/records/` (analytic-vs-AD Jacobian 5e-17,
  current conservation <1e-17, Taylor tests with O(h) scaling, scaling
  break-even, solver causality).
- `examples/`: 15 research scripts + 4 tutorials + 2 developer checks; each
  writes a figure + reproducibility JSON sidecar and accepts `--quick`.

**Optimization** (`optimize/`): SLSQP with JAX gradients (non-finite-probe
penalization), multi-start L-BFGS-B with gradient preconditioning, optax Adam,
Nelder-Mead.

**Runtime:** provenance/lineage (`runtime/provenance.py`), sharding helpers,
live progress bar with per-bias solver info (`console.py`), CLI
(`python -m driftjax simulate|verify|validate|benchmark|release`), publication
plotting (`viz/`).

---

## 6. Practical notes

- Install `pip install -e ".[dev,viz]"`. Tests: `make test` (serial),
  `make test-slow` (full N=500 parity), `make examples-quick`,
  `make validation`.
- **Gotcha (hit & confirmed):** `Device` **bakes** its `DeviceDesign` at
  construction, so mutating `dev.Ls` (or any leaf) afterwards has no effect —
  it produced an exactly-zero gradient. To differentiate a geometric/material
  parameter, pass it as a **tracer leaf at construction** (build the
  `Material`/`Device` inside the traced function), as in the `d eff/dA`
  verification above.
- First call includes XLA compilation (~10 s forward, ~30 s first gradient);
  a persistent on-disk compilation cache is enabled by default
  (`DRIFTJAX_COMPILATION_CACHE`, `""` to disable).
- Env knobs: `DRIFTJAX_BANDED_ADJOINT=1`, `DRIFTJAX_NATIVE_BANDED=1`,
  `DRIFTJAX_OPTIMISTIX=1`, `DRIFTJAX_BANDED=1`.

---

## 7. Key source map (`src/driftjax/`)

| Path | Role |
|---|---|
| `__init__.py` | public API, x64 + compilation-cache setup |
| `simulate.py` | `simulate` entry point, `custom_vjp` fwd/bwd, fused scan |
| `simulator.py` | `Device`, `_make_design`, `init_cell`, MPP |
| `fields.py` | PyTrees: `Material`, `DeviceDesign`, `PVCell`, `Potentials`, `pot2vec`/`vec2pot` |
| `problems.py` | `Equilibrium`, `Sweep` protocols |
| `solution.py` | `Solution` + certification fields |
| `units.py` | constants, `thermal_scales`, scaling table |
| `science/` | carrier statistics, recombination, optics, contacts, spectrum, tandem |
| `numerics/` | poisson, drift_diffusion, scharfetter_gummel, residual, analytic_jacobian, banded_solve, spline, fused_kernels, mixed_precision |
| `solvers/` | newton, continuation (sweep), batched, ptc, transient |
| `autodiff/` | adjoint, checkpointing |
| `adjoint/api.py` | `ImplicitAdjoint`, `DirectAdjoint` |
| `optics/api.py` | `BeerLambert`, `TMM`, `Fresnel` |
| `solvers/api.py` | `Newton`, `BandedLapack` (alias `BlockThomas`) |
| `optimize/` | slsqp, multistart L-BFGS-B, Adam, Nelder-Mead |
| `validation/` | pyramid, analytic, manufactured, conservation, cross-code, benchmarks |
| `viz/` | publication plotting |
| `runtime/` | provenance, sharding, performance |
| `io.py`, `resources/materials.yaml` | materials database (25 materials) |

---

_Bottom line:_ DriftJax is a carefully validated, fully differentiable
reimplementation of the classic 1-D solar-cell DDP solve — Scharfetter–Gummel /
Bernoulli discretization, analytic block-tridiagonal Jacobian, and pivoted
banded LAPACK solve as the forward backbone, with an IFT adjoint that makes
`jax.grad(simulate)` correct and fast for photovoltaic inverse design.
