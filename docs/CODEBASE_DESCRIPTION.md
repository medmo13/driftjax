# DriftJax v0.1.15 — Codebase Description

## What It Is
DriftJax is an end-to-end differentiable 1-D drift-diffusion–Poisson (DDP) solar-cell simulator written in JAX. Given a layered semiconductor device description, it solves the Van Roosbroeck drift-diffusion + Poisson system to produce a J–V curve and figures of merit (PCE, Voc, Jsc, FF), with exact gradients of any scalar output with respect to device design through `jax.grad(simulate)` via an implicit-function-theorem (IFT) adjoint.

## Key Features
- **Scharfetter–Gummel discretization** on non-uniform grids
- **Analytic block-tridiagonal Jacobian** + O(N) Block-Thomas Newton solve
- **IFT adjoint** via `custom_vjp`, with pivoted dense LU transpose and `@jax.checkpoint`
- **Carrier statistics**: Boltzmann, Blakemore, exact Fermi–Dirac (128-pt Gauss–Legendre quadrature)
- **Optics**: Beer–Lambert and coherent Transfer-Matrix Method (TMM)
- **Recombination**: radiative, SRH, Auger
- **Contacts**: Ohmic and Schottky with surface recombination velocities
- **Tandem**: two-terminal series connection
- **Transient**: backward-Euler time stepping
- **AC admittance**: small-signal frequency-domain
- **Optimizers**: SLSQP, multi-start, Adam, Nelder-Mead (derivative-free)
- **25-material database** (Si, GaAs, CdTe, MAPbI₃, CIGS, organics, etc.)

## Package Layout
```
src/driftjax/
├── __init__.py          # Public API + float64 enforcement
├── fields.py            # Core PyTrees: Potentials, Material, PVCell, DeviceDesign
├── simulator.py         # Device definition, init_cell
├── simulate.py          # Main entry point + custom_vjp adjoint core
├── solution.py          # Solution PyTree: IV, eff, voc, ff, pmax
├── problems.py          # Equilibrium, Sweep protocols
├── units.py             # Physical constants, thermal_scales(T)
├── io.py                # Material database (materials.yaml)
├── config.py            # Mode descriptor, numeric tolerances
├── numerics/            # Block-Thomas, Jacobian, residual, spline, mixed_precision
├── science/             # carrier_stats, optics, contacts, recombination, spectrum, tandem
├── solvers/             # Newton, continuation, transient, batched, PTC
├── autodiff/            # adjoint (custom_vjp), checkpointing
├── adjoint/             # ImplicitAdjoint, DirectAdjoint API
├── optics/              # BeerLambert, TMM API
├── optimize/            # SLSQP, objectives, constraints
├── validation/          # Pyramid (L1-L10), manufactured solutions, cross-code
├── viz/                 # Publication plotting
└── runtime/             # performance, sharding, provenance
```

## Performance (N=500, 1-core CPU, JAX 0.10.2)
- Forward sweep: 0.26 s
- Gradient: ~0.8 s
- Transient (50 steps): 0.55 s (135× vs eager)
- SLSQP optimization: 37 s
- Cross-code vs ∂PV v0.0.5: 200–300× forward, 73× gradient

## Validation
- 31 tests (24 unit + 7 gradient)
- 10-level validation pyramid (L1–L10)
- 15 research examples
- Cross-code agreement with ∂PV at sub-percent level
- Second-order spatial convergence

## Citation
```bibtex
@software{driftjax2026,
  author    = {Benaissa, Mohammed and Noua, Abd Elouahab and Maouadj, Mohamed},
  title     = {DriftJax: A Structure-Aware Differentiable Drift-Diffusion Solver},
  version   = {0.1.15},
  year      = {2026},
  institution = {CRTSE, Algiers, Algeria}
}
```
