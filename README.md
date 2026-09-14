# DriftJax

A differentiable 1-D drift–diffusion–Poisson photovoltaic simulator in [JAX](https://github.com/jax-ml/jax).

## Overview

DriftJax solves the Van Roosbroeck drift–diffusion–Poisson system for
semiconductor devices using Scharfetter–Gummel discretization, an analytically
assembled block-tridiagonal Jacobian, and an O(N) Block-Thomas direct solver.
Exact design gradients are obtained via `jax.grad(simulate)` through an
implicit-function-theorem (IFT) adjoint implemented as `custom_vjp`.

**Key features:**

- **Physics:** Van Roosbroeck DDP, Boltzmann/Fermi–Dirac/Blakemore statistics, SRH + radiative + Auger recombination
- **Optics:** Beer–Lambert (Tauc α) and coherent Transfer-Matrix Method (TMM)
- **Materials:** 25-material database (Si, CdTe, perovskites, CIGS, organics, custom)
- **Contacts:** Surface-recombination-velocity BCs with flatband work-function control
- **Solvers:** Newton with analytic block-tridiagonal Jacobian, O(N) Block-Thomas, `@jax.checkpoint` for memory-efficient backward passes
- **Autodiff:** Reverse-mode IFT adjoint, forward-mode design gradients, `jax.jit`/`jax.vmap` compatible
- **Optimization:** SLSQP, multi-start L-BFGS-B, Adam, and Nelder-Mead for PCE maximization and curve-fit material recovery

**Precision note:** Carrier densities ~10^19 cm⁻³ and Nc·Nv ~ 10^38 overflow float32. DriftJax enables JAX 64-bit floats on import and requires float64 throughout.

## Installation

```bash
pip install driftjax                  # from PyPI
pip install -e .                      # editable (local development)
pip install -e ".[dev,viz]"           # + tests, lint, docs, plotting
```

Requires Python ≥ 3.11 and JAX ≥ 0.10.

## Quick Start

```python
import driftjax as dj
import jax

# Define a material
Si = dj.material(Eg=1.12, Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
                 mn=1400, mp=450, tn=1e-6, tp=1e-6, A=1e4)

# Define a device
dev = dj.Device(layers=[(1e-4, Si, 1e16), (1e-4, Si, -1e16)],
                n_points=500, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)

# Simulate (AM1.5G by default)
sol = dj.simulate(dev, dj.Sweep())

v, j = sol.voltages, sol.current    # V, A/cm²
eff, voc, jsc, ff = sol.efficiency, sol.voc, sol.jsc, sol.ff

# Exact IFT design gradient
grad = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)

# Plot
dj.plot_iv_curve(v, j, path="iv.png")
dj.plot_band_diagram(dev, sol.potentials, path="bands.png")
```

## Documentation

- **Paper:** `docs/DriftJax_content.tex` — full methods, verification, and applications
- **Supplementary:** `docs/DriftJax_supplementary.tex` — extended results and provenance

## Development

```bash
git clone https://github.com/driftjax/driftjax.git
cd driftjax
pip install -e ".[dev,viz]"
make test                   # fast suite
make test-par               # parallel (4 workers, ~8 GB RAM)
make lint                   # ruff check
make typecheck              # mypy
make examples-quick         # smoke-run all examples
```

## Validation

DriftJax is validated against the ∂PV reference implementation (Mann et al.,
CPC 2022) on two benchmark problems at N=500:

| Device | DriftJax PCE | ∂PV PCE | Difference |
|--------|-------------|---------|------------|
| Si p-n homojunction | 19.89% | 20.00% | 0.11 pp |
| CdS/CdTe heterojunction | 13.31% | 13.31% | <0.01 pp |

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{driftjax,
  title  = {DriftJax: A Differentiable 1-D Drift–Diffusion Solar-Cell Simulator in JAX},
  version = {0.1.16},
  year   = {2026},
  url    = {https://github.com/driftjax/driftjax}
}
```
