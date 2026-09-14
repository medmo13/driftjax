# DriftJax

A differentiable 1-D drift-diffusion-Poisson solar-cell simulator in JAX.

## Getting started

```bash
pip install driftjax
```

See the project [README](../README.md) for installation details, the quick-start
example, and validation results.

## Contents

- [Physics](literate/physics.md) — Van Roosbroeck system, carrier statistics, recombination
- [Numerics](literate/numerics.md) — Scharfetter-Gummel discretization, block-tridiagonal Jacobian
- [Solvers](literate/solvers.md) — Newton-Raphson, Block-Thomas, bias continuation
- [Autodiff](literate/autodiff.md) — IFT adjoint, custom_vjp, gradient verification
- [Optics](literate/optics.md) — Beer-Lambert, TMM, AM1.5G spectrum
- [Performance](literate/performance.md) — O(N) algorithms, fused kernels, memory optimization
- [API Reference](literate/api.md) — Public interface documentation

The methods paper sources live in [`paper/`](https://github.com/medmo13/driftjax/tree/main/docs/paper).
