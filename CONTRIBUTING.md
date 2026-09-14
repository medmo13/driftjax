# Contributing to DriftJax

Thank you for considering contributing to DriftJax! This document outlines the
development workflow, coding standards, and pull-request process.

## Getting Started

1. Fork the repository and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/driftjax.git
   cd driftjax
   ```

2. Create a virtual environment and install in editable mode:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,viz]"
   ```

3. Install pre-commit hooks (optional but recommended):

   ```bash
   pre-commit install
   ```

## Development Workflow

### Branching

- `main` / `master` is the release branch. Never push directly.
- Create a feature branch from `main`:

  ```bash
  git checkout -b fix/solver-stability main
  ```

- Use descriptive branch names: `fix/...`, `feat/...`, `docs/...`, `refactor/...`.

### Making Changes

1. Write code following the existing style (see [Code Style](#code-style)).
2. Add or update tests for any new functionality or bug fixes.
3. Run the fast test suite before committing:

   ```bash
   make test          # ~8 min
   make test-smoke    # ~60 s for quick iteration
   ```

4. Run lint and type checks:

   ```bash
   make lint
   make typecheck
   make format        # auto-fix formatting
   ```

### Commit Messages

Use clear, concise commit messages:

```
fix: resolve OOM in adjoint backward pass for 61 bias points

- Add @jax.checkpoint on per_bias in _sweep_bwd
- Reduces backward memory by recomputing per-bias intermediates
- Enables N=500 with 61 bias points on 7.6 GB RAM
```

Format: `<type>: <short summary>` where type is one of:
`feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `ci`, `chore`.

### Pull Requests

1. Push your branch and open a PR against `main`.
2. Fill in the PR description with:
   - What changed and why
   - How to test the change
   - Any breaking changes or migration notes
3. CI must pass (lint + tests). The slow test suite runs on push or when
   the `run-slow` label is added.
4. Request a review from at least one maintainer.
5. Squash-merge is preferred for clean history.

## Code Style

### Formatting

- **Line length:** 100 characters (ruff).
- **Formatter:** `ruff format`.
- **Linter:** `ruff check` with rules `E4, E7, E9, F, I, W, UP, B, SIM, C4`.

### Conventions

- **float64 everywhere.** Never remove the `jax_enable_x64` import-time config
  update in `src/driftjax/__init__.py`.
- **DOF ordering.** The interleaved `[φn, φp, φ]` layout is defined only in
  `pot2vec`/`vec2pot` (`fields.py`). Changing it requires re-deriving the
  analytic Jacobian and residual.
- **Analytic Jacobian + Block-Thomas.** Never reintroduce autodiff-Jacobian
  Newton or unrolled linear solves in the hot path.
- **NaN guards.** The `1e-24` (spline) and `1e-40` (optics) guards keep the
  forward pass byte-identical. Do not perturb them.
- **Progress reporting must never affect numerics.**

### Type Annotations

DriftJax uses [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for
tensor shape annotations. The ruff per-file-ignores suppress the F722/F821
false positives these create.

### Testing

- Tests are in `tests/` organized by layer: `unit/`, `gradient/`, `conservation/`,
  `convergence/`, `property/`, `literature/`, `regression/`, `reproducibility/`.
- Use `pytest` markers: `smoke` (<60s), `slow` (N=500 regression), `experimental`.
- Property-based tests use [Hypothesis](https://hypothesis.readthedocs.io/).

## Reporting Issues

Open a GitHub issue with:

- A minimal reproducer (code snippet or script).
- Python/JAX version and platform.
- Full error traceback.
- Expected vs actual behavior.

## License

By contributing, you agree that your contributions will be licensed under the
MIT License.
