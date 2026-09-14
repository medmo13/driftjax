# DriftJax

A differentiable 1-D drift-diffusion-Poisson solar-cell simulator in JAX.

## Getting Started

```bash
pip install driftjax
```

See the [Installation](#installation) section in the README for full details.

## Contents

- [Physics](physics.md) — Van Roosbroeck system, carrier statistics, recombination
- [Numerics](numerics.md) — Scharfetter-Gummel discretization, block-tridiagonal Jacobian
- [Solvers](solvers.md) — Newton-Raphson, pivoted banded solve, bias continuation
- [Autodiff](autodiff.md) — IFT adjoint, custom_vjp, gradient verification
- [Optics](optics.md) — Beer-Lambert, TMM, AM1.5G spectrum
- [Performance](performance.md) — O(N) algorithms, fused kernels, memory optimization
- [API Reference](api.md) — Public interface documentation
