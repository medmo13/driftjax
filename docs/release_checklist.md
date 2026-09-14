# DriftJax v0.1.14 release checklist

## Verified in this checkout

- [x] Editable installation in venv
- [x] Package metadata and runtime version report `0.1.14`
- [x] `ruff check src tests`
- [x] `ruff format --check src tests`
- [x] Fast suite: 204 tests pass (~8 min)
- [x] Smoke suite: 64 tests pass (~45 s)
- [x] Slow suite: 32 tests deselected by default
- [x] All 15 research examples pass
- [x] All 2 developer examples pass
- [x] All 14 validation scripts pass
- [x] Paper, CAS, and supplementary updated for v0.1.14
- [x] CHANGELOG updated with v0.1.14 release notes
- [x] CITATION.cff version 0.1.14, date 2026-09-13
- [x] README badges and version updated
- [x] GitHub repo setup: CONTRIBUTING, CODE_OF_CONDUCT, mkdocs.yml, dependabot

## Reproducibility commands

```bash
export JAX_ENABLE_X64=1
export MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR=/tmp/driftjax-xla-cache
export PYTHONPATH=src

# Lint + format
ruff check src tests
ruff format --check src tests

# Fast tests
python -m pytest -m "not slow" -q

# Smoke tests
python -m pytest -m "smoke" -q

# Slow tests (N=500 parity + optimizer regressions)
python -m pytest -m "slow" -q --timeout=1200

# Validation
python validation/ex1_np_junction.py
python validation/ex2_np_hetero.py
python validation/grad_vs_fd.py

# Build
python -m build
```

## Key v0.1.14 performance numbers (N=500, 1-core CPU, JIT warm)

| Metric | Value |
|--------|-------|
| Forward solve | 1.7 s |
| Objective + gradient | 0.815 s |
| Full gradient speedup vs ∂PV | 75× |
| SLSQP optimization | 37 s |
| Total optimization speedup vs ∂PV | 26× |
| Nelder-Mead inverse recovery | 98 s, MSE 5.1e-20 |

## Required before publication

- [ ] Add the final repository URL and immutable release tag.
- [ ] Mint and add the software DOI to `CITATION.cff` and the manuscript.
- [ ] Archive external-baseline source revisions, raw outputs, and hardware/
  software metadata before making parity or speed claims.
- [ ] Install the build backend in an environment with package-index access and
  run `python -m build --wheel --sdist`.
- [ ] Review all generated figures at final journal dimensions and color/grey
  scale accessibility.
