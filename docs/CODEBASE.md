# DriftJax Codebase Description

**Version**: 0.1.18  
**License**: MIT  
**Language**: Python ≥3.11 with JAX  
**Repository**: github.com/medmo13/driftjax

## Overview

DriftJax is a 1-D drift-diffusion–Poisson (DDP) solar-cell simulator built in JAX. It is end-to-end differentiable and uses a single-program XLA megakernel for the forward Newton solve. The key architectural feature is a native block-tridiagonal GE solver (pure JAX via lax.scan) that replaces LAPACK callbacks for the forward path, with automatic LAPACK fallback for ill-conditioned blocks.

---

## Project Structure

```
driftjax/
├── src/driftjax/                  # Main package source (~170KB)
├── tests/                         # Test suite (~210KB, 50+ test files)
├── docs/                          # Documentation and manuscript
├── .github/                       # CI workflows, issue templates
├── .gitignore
├── .pre-commit-config.yaml        # Ruff formatting/linting
├── pyproject.toml                 # Package config, deps, pytest
├── requirements.txt               # Core dependencies
├── requirements-pinned.txt        # Exact-pinned env for reproduction
├── CHANGELOG.md                   # Release history
├── CITATION.cff                   # Machine-readable citation
├── README.md                      # Project README
├── CONTRIBUTING.md                # Contribution guide
├── CODE_OF_CONDUCT.md             # Code of conduct
├── SECURITY.md                    # Security policy
├── LICENSE                        # MIT license
├── Makefile                       # Dev convenience targets
├── mkdocs.yml                     # Documentation site config
└── MANUSCRIPT_AUDIT.md            # Manuscript audit log
```

---

## Source Code Organization (`src/driftjax/`)

### Core API and Orchestration

| File | Size | Description |
|------|------|-------------|
| `__init__.py` | 3KB | Package init — public API exports, `__version__ = "0.1.18"` |
| `__main__.py` | 12KB | CLI entry point for device simulation |
| `simulate.py` | 56KB | Primary entry point — `simulate(device, protocol, *, solver, optics, adjoint, progress)`. Full workflow from design to IV curve. |
| `simulator.py` | 29KB | High-level simulator API: relax/sweep/solve |
| `fields.py` | 10KB | Core PyTree types: PVCell, Potentials, BoundaryConditions, Material (equinox-based, JIT-safe) |
| `solution.py` | 8KB | Solution container — IV curve, efficiency, per-bias potentials, provenance flags |
| `units.py` | 4KB | Physical constants and dimensionless scaling (Vt = kT/q, Debye length convention) |
| `config.py` | 3KB | Configuration management |
| `_util.py` | 2KB | Internal utilities |
| `problems.py` | 1KB | Standard test problems |
| `io.py` | 15KB | I/O utilities for materials, spectra, JSON sidecars |

### Solvers (`solvers/`)

| File | Size | Description |
|------|------|-------------|
| `newton.py` | 37KB | Damped Newton solver with single-program XLA megakernel backend. Trace-safe lax.while_loop, automatic LAPACK fallback, dense-LU retry |
| `ptc.py` | 22KB | Pseudo-transient continuation (PTC) and Armijo line-search |
| `batched.py` | 8KB | Batched/vmapped solver machinery for parallel bias sweeps |
| `transient.py` | 7KB | Transient analysis support |
| `api.py` | 1KB | Solver API definitions — BandedLapack, BlockThomas, Newton classes |

### Numerics (`numerics/`)

| File | Size | Description |
|------|------|-------------|
| `banded_ge.py` | 12KB | **Native GE solver** — Pure-JAX pivoted block-Thomas GE via lax.scan with row equilibration. Default forward backend. No host callbacks. |
| `banded_solve.py` | 17KB | Banded solve dispatch — routes to native GE (default, `DRIFTJAX_NATIVE_BANDED=1`) or scipy.linalg callback fallback (`DRIFTJAX_NATIVE_BANDED=0`) |
| `banded_native.py` | 1KB | Native banded solver wrapper |
| `analytic_jacobian.py` | 15KB | Analytical block-tridiagonal Jacobian for coupled DDP system |
| `analytic_adjoint.py` | 18KB | Analytic adjoint cell-cotangents — single-kernel backward solve |
| `scharfetter_gummel.py` | 5KB | SG current discretization with stable Bernoulli kernel B(z) = z/(e^z − 1) |
| `residual.py` | 3KB | Residual assembly F(x) for coupled DDP system |
| `poisson.py` | 1KB | Poisson equation components |
| `drift_diffusion.py` | 1KB | Drift-diffusion model |
| `fused_kernels.py` | 5KB | Fused physics kernels — single compiled program per op |
| `mixed_precision.py` | 8KB | Mixed-precision solver with iterative refinement |
| `mesh.py` | 0KB | Mesh utilities |
| `spline.py` | 7KB | Spline interpolation |
| `linalg.py` | 9KB | Sparse linear algebra utilities (sc splinalg wrappers) |
| `ffi/` | 4KB | Foreign function interface bindings |

### Science (`science/`)

| File | Size | Description |
|------|------|-------------|
| `contacts.py` | 3KB | Contact BC: ohmic/Schottky, SRV (Robin) conditions |
| `carrier_statistics.py` | 9KB | Boltzmann (default) and Fermi-Dirac statistics |
| `recombination.py` | 2KB | Radiative, Auger, SRH recombination models |
| `optics.py` | 15KB | Optical model — generation profile from spectra |
| `spectrum.py` | 8KB | Light source spectrum handling (AM1.5G, monochromatic) |
| `tandem.py` | 7KB | Tandem multi-junction cell modeling |

### Autodiff (`autodiff/`)

| File | Size | Description |
|------|------|-------------|
| `adjoint.py` | 4KB | Adjoint differentiation via implicit function theorem |
| `checkpointing.py` | 4KB | Gradient checkpointing (remat) for memory optimization |

### Optimization (`optimize/`)

| File | Size | Description |
|------|------|-------------|
| `optimizers.py` | 8KB | scipy-SLSQP with JAX gradients, Nelder-Mead, optax Adam |
| `objectives.py` | 1KB | Design objective functions (PCE, Jsc, Voc, FF) |
| `constraints.py` | 0KB | Design parameter constraints |

### Runtime (`runtime/`)

| File | Size | Description |
|------|------|-------------|
| `performance.py` | 2KB | Performance tracking and timing utilities |
| `provenance.py` | 5KB | Reproducibility provenance (env, commit, config fingerprint) |
| `sharding.py` | 3KB | XLA sharding strategies for multi-device |

### Validation (`validation/`)

| File | Size | Description |
|------|------|-------------|
| `analytic.py` | 11KB | Analytic test problems and solutions |
| `benchmarks/` | 4KB | Benchmark runners |
| `conservation.py` | 1KB | Conservation law checks |
| `cross_code/` | 4KB | deltapv parity tests (V20 forward, V21 optimization) |
| `grads.py` | 4KB | Gradient verification (FD vs adjoint) |
| `manufactured.py` | 2KB | Method of manufactured solutions (Poisson + cell-level) |
| `physics.py` | 4KB | Physics validation tests |
| `pyramid.py` | 8KB | Pyramid gate validation |
| `record.py` | 5KB | Validation record management |
| `stability/` | 4KB | Solver stability diagnostics |
| `verification/` | 4KB | Numerical verification tests |

### Visualization (`viz/`)

| File | Size | Description |
|------|------|-------------|
| `plotting.py` | 36KB | Plotting utilities (IV curves, band diagrams, design maps) |
| `style.py` | 9KB | Plotting style and theme |
| `io.py` | 1KB | Plot I/O utilities |
| `optim.py` | 2KB | Optimization visualization |
| `figsize.py` | 1KB | Figure sizing utilities |

### Resources
- `resources/materials.yaml` (38KB) — Material database with Si, CdTe, CIGS, perovskites

---

## Test Suite (`tests/`)

Organized by testing tier:

| Directory | Tests | Focus |
|-----------|-------|-------|
| `unit/` | 34 files | Unit tests: GE solver, megakernel backend, provenance, public API |
| `regression/` | 9 files | Regression tests: IV references, native GE fallback, perovskite |
| `gradient/` | 5 files | Gradient verification: adjoint vs FD, inverse design |
| `literature/` | 4 files | Literature validation: experimental, ideal diode, SCAPS, SQ limit |
| `property/` | 4 files | Property-based tests: fused simulation, invariants, sharding |
| `conservation/` | 1 file | Conservation law checks |
| `convergence/` | 2 files | Convergence analysis: MMS Poisson, spline |
| `reproducibility/` | 1 file | Bit-identical reproducibility |
| `validation/` | 2 files | External validation |
| `conftest.py` | — | Pytest fixtures — shared test configuration |

**Total: ~70 test files, 101+ passing tests**

---

## Documentation (`docs/`)

### Literate API Docs (`docs/literate/`)
| File | Description |
|------|-------------|
| `index.md` | Documentation index |
| `api.md` | API reference |
| `autodiff.md` | Automatic differentiation |
| `numerics.md` | Numerical methods |
| `optics.md` | Optical model |
| `performance.md` | Performance analysis |
| `physics.md` | Physics model |
| `solvers.md` | Solver documentation |

### Manuscript (`docs/paper/`)
| File | Description |
|------|-------------|
| `DriftJax_paper.tex` | Main article (LaTeX, v0.1.18) |
| `DriftJax_supplementary.tex` | Supplementary sections S1–S9 |
| `DriftJax_content.tex` | Article content body |
| `DriftJax_abstract.tex` | Common abstract |
| `DriftJax_shared.tex` | Shared notation and commands |
| `DriftJax_CAS.tex` | Elsevier CAS format version |
| `cas-refs.bib` | Bibliography (44 references incl. Selberherr, Gummel, DEVSIM) |
| `paper.template.tex` | LaTeX build template |
| `build_latex.py` | LaTeX build script |
| `BUILD_INSTRUCTIONS.md` | Compilation instructions |
| `verify_sources.py` | Source integrity verification |
| `records/` | Benchmark records (JSON + PNG figures) |
| `provenance_table.tex` | Figure/claim provenance tracking |

---

## Key Architectural Features

### 1. Single-Program XLA Megakernel
The entire forward Newton iteration — flux evaluation, block-Jacobian assembly, and banded GE — is expressed as a single trace-safe `lax.while_loop`. This enables end-to-end differentiation through JAX's `custom_vjp` implicit-adjoint path.

### 2. Native GE Default Backend
- **Default**: Pure-JAX native block-Thomas GE (`banded_ge.py`), no host callbacks
- **Fallback**: LAPACK `dgbsv` via scipy callback (opt-in, `DRIFTJAX_NATIVE_BANDED=0`)
- Automatic fallback to dense LU on singular blocks

### 3. Automatic Backend Switching
- Per-block determinant threshold detection
- Transparent dispatch to LAPACK `dgesv` on ill-conditioned blocks
- Provenance tracking via `fallback_used` flags

### 4. Tri-State Gradient Status
- `CERTIFIED`: converged, no fallback
- `UNRELIABLE`: unconverged or fallback used
- `UNVERIFIED`: audit impossible under tracing

### 5. Cross-Code Validation
Continuous parity testing against deltapv (∂PV):
- V20: Forward IV parity on Si p-n junction
- V21: Optimization gradient parity

---

## Running the Code

```bash
# Install (requires Python 3.11+)
pip install -r requirements-pinned.txt

# Run tests
PYTHONPATH=src JAX_ENABLE_X64=1 python -m pytest tests/ -x -q --timeout=120

# Run a simulation
python -m driftjax --device si_pn --simulate

# Build docs
mkdocs serve
```
