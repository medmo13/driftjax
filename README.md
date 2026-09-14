# driftjax

[![CI](https://github.com/driftjax/driftjax/actions/workflows/ci.yml/badge.svg)](https://github.com/driftjax/driftjax/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/driftjax.svg)](https://pypi.org/project/driftjax/)
[![Python versions](https://img.shields.io/pypi/pyversions/driftjax.svg)](https://pypi.org/project/driftjax/)
[![Paper](https://img.shields.io/badge/paper-CPC-blue.svg)](docs/DriftJax_paper.md)

**DriftJax** is a modern, end-to-end differentiable 1-D drift-diffusion-Poisson
photovoltaic-cell simulator written in [JAX](https://github.com/jax-ml/jax).
Gradients of the power-conversion efficiency (or any scalar figure of merit)
with respect to device-design parameters are obtained transparently through
`jax.grad(simulate)` via an implicit-function-theorem (IFT) adjoint implemented
as a `custom_vjp` — no manual adjoint code required.

> **Precision note.** The DD Jacobian involves carrier densities
> `~1e19 cm⁻³` and `Nc·Nv ~ 1e38`, which **overflow float32**. DriftJax enables
> JAX 64-bit floats on import (`jax.config.update("jax_enable_x64", True)`), and
> all solves and gradients must run in float64. Set `JAX_ENABLE_X64=1` in CI and
> notebooks.

## Features

- **Physics** — Van Roosbroeck system (drift-diffusion + Poisson) with
  Scharfetter-Gummel discretization; Boltzmann / Fermi-Dirac / Blakemore statistics.
- **Optics** — Beer-Lambert absorption (Tauc α(λ)) + coherent Transfer-Matrix Method (TMM).
- **Materials** — 25-material database (Si, CdTe, perovskites, organics, custom).
- **Contacts** — surface-recombination-velocity BCs with flatband work-function control.
- **Solvers** — Newton-Raphson with an *analytically assembled* block-tridiagonal
  Jacobian inverted by the exact O(N) Block-Thomas algorithm; fused, JIT-able bias sweep.
- **Autodiff** — reverse-mode IFT adjoint (`custom_vjp`) + forward-mode (FD/exact)
  design gradients; `jax.jit` / `jax.vmap` compatible.
- **Inverse design** — SLSQP, multi-start LBFGS, Adam, and Nelder-Mead (derivative-free)
  optimizers for PCE maximization and curve-fit material recovery.
- **Performance** — 75× faster adjoint vs ∂PV, O(N) Newton, fused carrier statistics,
  `@jax.checkpoint` for memory-constrained backward passes.

## Installation

```bash
pip install driftjax                  # from PyPI
pip install -e .                      # editable (local dev)
pip install -e ".[dev,viz]"           # + tests, lint, docs, plotting
```

Requires Python >= 3.11 and JAX >= 0.10.

## Quick Start

```python
import driftjax as dj
import jax

# A material from physical parameters
Si = dj.material(Eg=1.12, Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
                 mn=1400, mp=450, tn=1e-6, tp=1e-6, A=1e4)

# A Device: stacked (thickness_m, material, doping_cm^-3) layers on a grid
dev = dj.Device(layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)],
                n_points=500, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)

# Simulate the J-V sweep (AM1.5G is the default light source)
sol = dj.simulate(dev, dj.Sweep())          # -> Solution (jit/grad/vmap-safe)

v, j = sol.voltages, sol.current            # V (volts), j (A/cm^2)
eff = sol.efficiency                        # efficiency (fraction)
voc = sol.voc                               # open-circuit voltage (V)
jsc = sol.jsc                               # short-circuit current density (A/cm^2)
ff  = sol.ff                                # fill factor

# Transparent IFT gradient of the efficiency w.r.t. the whole design
grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)

# Plots
dj.plot_iv_curve(v, j, path="iv.png")
dj.plot_band_diagram(dev, sol.potentials, path="bands.png")
dj.plot_charge(dev, sol.potentials, path="charge.png")
```

Composable strategies:

```python
sol = dj.simulate(
    dev,
    protocol=dj.Sweep() | dj.Equilibrium(),
    solver=dj.Newton(),
    optics=dj.BeerLambert() | dj.TMM(),
    adjoint=dj.ImplicitAdjoint(),
)
```

## Development

```bash
git clone https://github.com/driftjax/driftjax.git
cd driftjax
pip install -e ".[dev,viz]"
pre-commit install          # optional: auto-lint on commit
make test                   # fast suite (~8 min)
make test-smoke             # sub-60s dev loop
make lint                   # ruff check
make format                 # auto-format
make typecheck              # mypy
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development workflow.

## Examples & Validation

- **Examples** — `examples/` contains 15 research + 2 developer scripts. Each
  accepts `--quick` and `--output-dir`, writes one figure + reproducibility JSON,
  and prints a machine-parseable `EXAMPLE_RESULT` summary line.
- **Validation** — `validation/` contains reference forward validations (N=500),
  gradient-vs-FD checks, adjoint timing, and the in-library L1-L10 gate suite
  (`python -m driftjax pyramid`).
- **Tests** — 204 fast tests + 64 smoke tests; 32 slow regression gates.

```bash
make test            # fast checks (serial, safe everywhere)
make test-par        # 4 workers (needs ~8 GB RAM)
make test-slow       # full suite incl. N=500 parity + optimizer regressions
make examples-quick  # smoke-run the whole example gallery
make validation      # N=500 reference forward validations
```

## Documentation

- **Paper** — [`docs/DriftJax_paper.md`](docs/DriftJax_paper.md): full methods,
  software architecture, validation pyramid, and references (CPC style).
- **Literate docs** — `docs/literate/`: physics, numerics, solvers, autodiff, optics,
  performance, and API reference.
- **Local docs** — `make docs` or `mkdocs serve`.

## External Validation

DriftJax is validated against the ∂PV reference implementation (Mann et al.,
CPC 2022) on two benchmark problems at N=500:

| Device | DriftJax PCE | ∂PV PCE | Difference |
|--------|-------------|---------|------------|
| Si p-n homojunction | 19.89% | 20.00% | 0.11 pp |
| CdS/CdTe heterojunction | 13.31% | 13.31% | <0.01 pp |

Objective+gradient is **75×** faster than ∂PV; total optimization is **26×** faster.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use DriftJax in research, please cite the software and the foundational
method papers. A ready-to-use `CITATION.cff` is provided; the BibTeX is:

```bibtex
@software{driftjax,
  title  = {DriftJax: A Differentiable 1-D Drift-Diffusion Solar-Cell Simulator in JAX},
  version = {0.1.15},
  year   = {2026},
  url    = {https://github.com/driftjax/driftjax}
}
```
