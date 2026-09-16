# DriftJax validation

This directory contains reproducible, in-library validation workflows. The
release package is self-contained; archived external comparison artifacts are
not distributed and are not required to run the tests.

The public entry point is:

```python
from driftjax import BeerLambert, Newton, Sweep, simulate

solution = simulate(device, Sweep(vmax=1.1, n_steps=25), solver=Newton(), optics=BeerLambert())
print(solution.efficiency, solution.voc, solution.ff)
```

## Included checks

- `grad_vs_fd.py` compares the implicit adjoint with central finite
  differences over a parameterized two-layer junction.
- `timing.py` reports first-call compilation and warm-call execution costs.
- `transpose_banded_stability.py` records the numerical rationale for
  the stable transposed linear solve used by the adjoint (historical
  unpivoted-transpose instability plus v0.1.17 pivoted-banded port).
- `psc.py`, `multi.py`, and `holistic.py` are standalone inverse-design
  workflows for perovskite design, IV-curve material recovery, and constrained
  device optimization.

Run the fast release checks with:

```bash
JAX_ENABLE_X64=1 python -m pytest -m "not slow"
```

The complete release gate is
`tests/regression/test_release_validated.py`; it includes forward parity
checks, finite-difference gradient checks, optimizer convergence checks, and
warm-start differentiation checks.

- `perovskite_pin_control.py` regenerates
  `docs/paper/records/perovskite_p-i-n_audit.json` -- the structurally-correct
  p-i-n positive control that demonstrates the solver handles perovskite
  parameters when the device geometry is sound (counterpart to the 3-layer
  n-p-n stress test).
