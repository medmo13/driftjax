# DriftJax

[![CI](https://github.com/medmo13/driftjax/actions/workflows/ci.yml/badge.svg)](https://github.com/medmo13/driftjax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)](pyproject.toml)

A differentiable 1-D drift–diffusion–Poisson photovoltaic simulator in [JAX](https://github.com/jax-ml/jax).

DriftJax solves the Van Roosbroeck system with Scharfetter–Gummel
discretization, an analytically assembled block-tridiagonal Jacobian, and a
pivoted banded direct solver (LAPACK `dgbsv`). Implicit-function-theorem
design gradients flow through `jax.grad(simulate)` via a `custom_vjp`
adjoint — verified against finite differences on selected configurations
(see the paper); gradients are certified only when the returned
`Solution.converged` is True.

**Precision note:** carrier densities (~1e19 cm⁻³) and Nc·Nv (~1e38) overflow
float32. DriftJax enables JAX 64-bit mode on import; everything must run in
float64 (`JAX_ENABLE_X64=1`).

## Features

- **Physics** — Van Roosbroeck DDP; Boltzmann / Fermi–Dirac / Blakemore statistics; SRH + radiative + Auger recombination
- **Optics** — Beer–Lambert (Tauc α) and coherent Transfer-Matrix Method (TMM); AM1.5G spectrum
- **Materials** — 25-material database (Si, CdTe, perovskites, CIGS, organics) plus custom materials
- **Contacts** — Surface-recombination-velocity BCs with flatband work-function control
- **Solvers** — Newton with analytic block-tridiagonal Jacobian, O(N) Block-Thomas, `@jax.checkpoint` memory-efficient backward passes
- **Autodiff** — Reverse-mode IFT adjoint; `jax.jit` / `jax.vmap` compatible
- **Optimization** — SLSQP, multi-start L-BFGS-B, Adam, and derivative-free Nelder-Mead

## Installation

```bash
pip install driftjax                  # from PyPI
pip install -e .                      # editable (local development)
pip install -e ".[dev,viz]"           # + tests, lint, docs, plotting
```

Requires Python ≥ 3.11 and JAX ≥ 0.10. For GPU/TPU support, install
`jaxlib` for your platform first — see the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) —
then install DriftJax.

## Quick start

```python
import driftjax as dj
import jax

Si = dj.material(
    Eg=1.12,
    Chi=4.05,
    eps=11.7,
    Nc=2.8e19,
    Nv=1.04e19,
    mn=1400,
    mp=450,
    tn=1e-6,
    tp=1e-6,
    A=1e4,
)

dev = dj.Device(
    layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)],
    n_points=500,
    Snl=1e7,
    Snr=1e7,
    Spl=1e7,
    Spr=1e7,
)

sol = dj.simulate(dev, dj.Sweep())  # AM1.5G by default

v, j = sol.voltages, sol.current
eff, voc, jsc, ff = sol.efficiency, sol.voc, sol.jsc, sol.ff

grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)

dj.plot_iv_curve(v, j, path="iv.png")
dj.plot_band_diagram(dev, sol.potentials, path="bands.png")
```

Solver, optics, and adjoint strategies are keyword arguments with
solver-correct defaults:

```python
sol = dj.simulate(
    dev,
    dj.Sweep(),
    solver=dj.Newton(),
    optics=dj.BeerLambert(),
    adjoint=dj.ImplicitAdjoint(),
)
```

## Examples and validation

- `examples/` — 15 research scripts, 4 tutorials, 2 developer checks. Every
  script accepts `--quick` and `--output-dir`, writes one figure plus a
  reproducibility JSON sidecar, and prints a machine-parseable
  `EXAMPLE_RESULT` summary line.
- `validation/` — reference forward validations at N=500, gradient-vs-FD
  checks, and timing scripts.
- `tests/` — unit, gradient, conservation, convergence, property,
  literature, regression, and reproducibility suites.

```bash
git clone https://github.com/medmo13/driftjax.git
cd driftjax
pip install -e ".[dev,viz]"
make test            # fast suite (serial)
make test-par        # fast suite, 4 workers (needs ~8 GB RAM)
make test-slow       # full suite incl. N=500 parity gates
make examples-quick  # smoke-run the example gallery
make validation      # N=500 reference validations
```

Measured against the ∂PV reference (Mann et al., CPC 2022) at N=500:

| Device | DriftJax PCE | ∂PV PCE | Difference |
|--------|-------------|---------|------------|
| Si p-n homojunction | 19.89% | 20.00% | 0.11 pp |
| CdS/CdTe heterojunction | 13.31% | 13.31% | <0.01 pp |

## Documentation

- [Literate docs](docs/literate/index.md) — physics, numerics, solvers, autodiff, optics, performance, API reference (`make docs` or `mkdocs serve` to build)
- [Paper sources](docs/paper/) — methods paper + supplementary LaTeX for the CPC submission
- [Contributing](CONTRIBUTING.md) — workflow, code style, invariants

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{driftjax,
  title   = {DriftJax: A Differentiable 1-D Drift-Diffusion Solar-Cell Simulator in JAX},
  version = {0.1.16},
  year    = {2026},
  url     = {https://github.com/medmo13/driftjax}
}
```

A [`CITATION.cff`](CITATION.cff) is also provided.
