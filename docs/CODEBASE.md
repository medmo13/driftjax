# DriftJax Codebase Description

## Overview

DriftJax (v0.1.18) is a 1-D drift-diffusion–Poisson (DDP) solar-cell simulator built in JAX. It is end-to-end differentiable and uses a **single-program XLA megakernel** architecture for forward Newton steps with native block-tridiagonal GE.

**Repository**: `github.com/medmo13/driftjax`
**Version**: 0.1.18
**License**: MIT
**Language**: Python >=3.11 with JAX (64-bit enabled)

---

## Top-Level Project Structure

```
driftjax/
├── src/driftjax/          # Main package source code
├── tests/                 # Test suite (unit, regression, validation, etc.)
├── docs/                  # Documentation and manuscript sources
├── .github/               # CI workflows, issue templates
├── CITATION.cff           # Citation metadata (Zenodo DOI)
├── CHANGELOG.md           # Release history
├── LICENSE                # MIT license
├── README.md              # Project README with quickstart
├── CONTRIBUTING.md        # Contribution guidelines
├── CODE_OF_CONDUCT.md     # Code of conduct
├── SECURITY.md            # Security policy
├── .gitignore             # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks
├── pyproject.toml         # Package configuration, dependencies, entry points
├── requirements.txt       # Installation requirements
└── Makefile               # Development convenience targets
```

---

## Source Code (`src/driftjax/`)

The DriftJax package is organized into the following subpackages and modules:

### Core API and Orchestration

| Module | Description |
|--------|-------------|
| `__init__.py` | Package init — exports public API, version string |
| `__main__.py` | CLI entry point for running device simulations |
| `simulate.py` | **Primary entry point** — `simulate(device, protocol, *, solver, optics, adjoint, progress)`. Handles full workflow from design to IV curve. |
| `simulator.py` | High-level simulator API: `Device → relax / sweep / solve` |
| `fields.py` | Core pytree types — `PVCell`, `Potentials`, `BoundaryConditions`, `Material`. Frozen, JIT-safe dataclasses built on equinox. |
| `solution.py` | Immutable `Solution` PyTree carrying IV curve, efficiency, per-bias potentials, and provenance flags |
| `units.py` | Physical constants and dimensionless scaling (Debye-unit convention, Vt = kT/q) |
| `config.py` | Configuration management |
| `_util.py` | Internal utilities |
| `problems.py` | Standard test problems and references |
| `io.py` | I/O utilities for materials, spectra, JSON sidecars |

### Solvers (`solvers/`)

| Module | Description |
|--------|-------------|
| `solvers/newton.py` | **Damped Newton solver** with single-program XLA megakernel. Native block-tridiagonal GE inlined via `lax.while_loop`, automatic LAPACK fallback, dense-LU retry. |
| `solvers/ptc.py` | Pseudo-transient continuation (PTC) and Armijo line-search globalization |
| `solvers/batched.py` | Batched (vmapped) solver for parallel bias sweeps |
| `solvers/transient.py` | Transient analysis support |
| `solvers/api.py` | Solver API definitions — `BandedLapack`, `BlockThomas`, `Newton` classes |

### Numerics (`numerics/`)

| Module | Description |
|--------|-------------|
| `numerics/banded_ge.py` | **Native GE solver** — Pure-JAX pivoted banded GE via `lax.scan` (block Thomas algorithm). Default backend. No host callbacks. |
| `numerics/banded_solve.py` | Banded solve dispatch — routes to native GE or LAPACK fallback. `DRIFTJAX_NATIVE_BANDED` defaults to "1". |
| `numerics/banded_native.py` | Native banded solver wrapper |
| `numerics/analytic_jacobian.py` | Analytical block-tridiagonal Jacobian assembly for the coupled DDP system |
| `numerics/analytic_adjoint.py` | Analytic adjoint cell-cotangents — single-kernel backward, zero-AD per-bias |
| `numerics/scharfetter_gummel.py` | Scharfetter–Gummel current discretization with stable Bernoulli kernel B(z) = z/(e^z − 1) |
| `numerics/residual.py` | Residual assembly F(x) for the coupled DDP system |
| `numerics/poisson.py` | Poisson equation solver components |
| `numerics/drift_diffusion.py` | Drift-diffusion model |
| `numerics/fused_kernels.py` | Fused physics kernels — one compiled program per physics operation |
| `numerics/mixed_precision.py` | Mixed-precision solver with iterative refinement |
| `numerics/mesh.py` | Mesh utilities |
| `numerics/spline.py` | Spline interpolation |
| `numerics/linalg.py` | Sparse linear algebra utilities |
| `numerics/ffi/` | Foreign function interface bindings |

### Science (`science/`)

| Module | Description |
|--------|-------------|
| `science/contacts.py` | Contact boundary conditions: ohmic/Schottky φ, SRV (Robin) conditions |
| `science/carrier_statistics.py` | Carrier statistics: Boltzmann (default), Fermi-Dirac approximations |
| `science/recombination.py` | Recombination models (radiative, Auger, SRH) |
| `science/optics.py` | Optical model — generation profile from spectra |
| `science/spectrum.py` | Light source spectrum handling |
| `science/tandem.py` | Tandem multi-junction cell modeling |

### Autodiff (`autodiff/`)

| Module | Description |
|--------|-------------|
| `autodiff/adjoint.py` | Adjoint differentiation through Newton solve (implicit function theorem) |
| `autodiff/checkpointing.py` | Gradient checkpointing for memory/compute trade-off |

### Optimize (`optimize/`)

| Module | Description |
|--------|-------------|
| `optimize/optimizers.py` | scipy-SLSQP with JAX gradients, preconditioned multi-start, Nelder-Mead |
| `optimize/objectives.py` | Objective functions (efficiency, Jsc, Voc, FF, PCE) |
| `optimize/constraints.py` | Design constraints |

### Runtime (`runtime/`)

| Module | Description |
|--------|-------------|
| `runtime/performance.py` | Performance tracking and timing |
| `runtime/provenance.py` | Reproducibility provenance — env, commit, config fingerprint |
| `runtime/sharding.py` | XLA sharding strategies |

### Validation (`validation/`)

| Module | Description |
|--------|-------------|
| `validation/analytic.py` | Analytic test problems |
| `validation/benchmarks/` | Benchmark scripts |
| `validation/conservation.py` | Conservation law checks |
| `validation/convergence.py` | Convergence analysis |
| `validation/cross_code/` | Cross-code comparison with deltapv (V20 forward parity, V21 optimization parity) |
| `validation/grads.py` | Gradient checks vs finite differences |
| `validation/manufactured.py` | Method of manufactured solutions |
| `validation/physics.py` | Physics validation |
| `validation/pyramid.py` | Pyramid gate validation |
| `validation/record.py` | Validation record management |
| `validation/stability/` | Solver stability tests |
| `validation/verification/` | Numerical verification |

### Resources and Visualization

| Directory/Module | Description |
|----------------|-------------|
| `resources/materials.yaml` | Material database (Si, CdTe, CIGS, perovskites, etc.) |
| `viz/` | Visualization utilities — `plotting.py`, `style.py`, `io.py`, `optim.py` |
| `optics/` | Optical model components |

---

## Tests (`tests/`)

Test suite organized by tier:

```
tests/
├── conftest.py              # Pytest fixtures and configuration
├── helpers.py               # Test helper utilities
├── resources/               # Test fixtures and data
├── unit/                    # Unit tests (34 files)
│   ├── test_banded_ge.py    # Native GE solver tests
│   ├── test_megakernel_backend.py  # Single-program XLA tests
│   ├── test_native_ge_fallback.py  # LAPACK fallback tests
│   ├── test_provenance_lineage.py  # Provenance tracking
│   ├── test_public_api.py   # Public API surface
│   ├── test_units.py        # Unit system consistency
│   ├── test_bernoulli.py    # SG Bernoulli kernel
│   └── ...                  # 27 more unit tests
├── regression/             # Regression tests (9 files)
│   ├── test_iv_reference.py  # IV curve references
│   ├── test_native_ge_fallback.py  # Fallback behavior
│   ├── test_perovskite_pin.py  # Perovskite device model
│   ├── test_three_layer_stress.py  # Stress test for ill-conditioned systems
│   └── release_helpers.py    # Release validation helpers
├── gradient/               # Gradient/adjoint tests (5 files)
│   ├── test_adjoint_ab.py   # Adjoint A/B testing
│   ├── test_grad_fd.py      # Gradient vs finite differences
│   ├── test_hetero_builder_grad.py  # Heterojunction gradients
│   └── test_inverse_design_grad.py  # Inverse design gradients
├── literature/             # Literature validation (4 files)
│   ├── test_experimental.py # Experimental data comparison
│   ├── test_ideal_diode.py  # Ideal diode limits
│   ├── test_scaps_cdte.py   # SCAPS cross-validation
│   └── test_sq_limit.py     # SQ limit validation
├── convergence/            # Convergence tests (2 files)
├── conservation/           # Conservation law tests (1 file)
├── property/               # Property-based tests (4 files)
│   ├── test_fused_simulate.py  # Fused kernel properties
│   ├── test_invariants.py   # Physical invariants
│   └── test_sharding.py     # XLA sharding properties
├── reproducibility/        # Bit-identical reproducibility (1 file)
└── validation/             # External validation (2 files)
```

---

## Documentation (`docs/`)

```
docs/
├── index.md               # Landing page
├── literate/              # Literate API documentation
│   ├── api.md             # API reference
│   ├── autodiff.md        # Automatic differentiation
│   ├── index.md           # Documentation index
│   ├── numerics.md        # Numerical methods
│   ├── optics.md          # Optical model
│   ├── performance.md     # Performance analysis
│   ├── physics.md         # Physics model
│   └── solvers.md         # Solver documentation
├── paper/                 # Manuscript sources
│   ├── DriftJax_paper.tex      # Main article (LaTeX)
│   ├── DriftJax_supplementary.tex  # Supplementary sections S1-S9
│   ├── DriftJax_content.tex    # Main content body
│   ├── DriftJax_abstract.tex   # Common abstract
│   ├── DriftJax_shared.tex     # Shared notation/commands
│   ├── DriftJax_CAS.tex        # Elsevier CAS version
│   ├── cas-refs.bib           # Bibliography
│   ├── paper.template.tex     # Build template
│   ├── build_latex.py         # LaTeX build script
│   ├── BUILD_INSTRUCTIONS.md  # Compilation guide
│   ├── README.md             # Paper directory README
│   ├── verify_sources.py     # Source verification
│   └── records/              # Benchmark records and figures
│       ├── *.json            # Machine-readable benchmark records
│       └── *.png             # Figures (33 files)
└── scripts/               # Documentation scripts
    └── mesh_convergence.py  # Mesh convergence study
```

---

## Configuration Files

| File | Description |
|------|-------------|
| `pyproject.toml` | Package metadata, dependencies, pytest config, linting (ruff) |
| `requirements.txt` | Core installation requirements |
| `requirements-pinned.txt` | Pinned version requirements |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, formatting) |
| `Makefile` | Development convenience targets |
| `mkdocs.yml` | MkDocs documentation site configuration |

---

## Key Architectural Components

### 1. Single-Program XLA Megakernel
The entire forward Newton iteration — Scharfetter-Gummel flux evaluation, block-tridiagonal Jacobian assembly, and pivoted banded elimination — is expressed as a **single trace-safe lax.while_loop** with no host callbacks. This makes every forward step **end-to-end differentiable** through JAX's custom_vjp implicit-adjoint path.

### 2. Native GE Backend (`banded_ge.py`)
- Pure-JAX pivoted banded GE via lax.scan (block Thomas algorithm)
- Replaces O(N³) SVD native solve with O(N·bw) complexity
- Automatic row equilibration for ill-conditioned blocks
- No host callbacks, fully JIT/vmap compatible

### 3. Automatic LAPACK Fallback
- When native GE detects near-singular blocks (via per-block determinant thresholds)
- Transparently falls back to LAPACK dgesv within the same XLA program
- Provenance tracking via fallback_used flags
- Never materializes host arrays

### 4. Custom VJP Adjoint
- Implicit function theorem through the converged Newton state
- Analytic Jacobian + native GE transpose solve
- Tri-state gradient status: CERTIFIED / UNRELIABLE / UNVERIFIED

### 5. Cross-Code Validation
- Continuous parity testing against deltapv (DPV)
- V20: Forward parity (IV curve comparison on Si p-n junction)
- V21: Optimization parity (gradient evaluation timing and values)
- Both tests pass with machine-precision agreement on clean systems

---

## Development Commands

```bash
# Run tests
PYTHONPATH=<base>/src JAX_ENABLE_X64=1 python -m pytest tests/ -x -q --timeout=120

# Run a simulation
python -m driftjax --device si_pn --simulate

# Build docs
mkdocs serve

# Format code
pre-commit run --all-files
```
