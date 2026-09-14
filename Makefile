# DriftJax — developer commands
#
# Tests run SERIAL by default (safest: JAX float64 solvers are memory-hungry).
# For a parallel CPU run use `make test-par` — capped at 4 workers on purpose:
# each worker is a full JAX process, and console tests spawn subprocesses on
# top. With ~8 GB RAM do not exceed -n4; higher caused system-wide stalls.

PYTEST ?= python -m pytest

# Common XLA environment: float64 + persistent compile cache.
JAX_ENV = JAX_ENABLE_X64=1 MPLBACKEND=Agg \
          JAX_COMPILATION_CACHE_DIR=$(HOME)/.cache/driftjax-xla-cache \
          JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0.2 \
          JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all \
          PYTHONPATH=src OMP_NUM_THREADS=1

.PHONY: help install test test-smoke test-par test-slow test-unit \
        lint format typecheck docs docs-serve clean build check all \
        validation examples-quick

## Show this help message
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/^## //' | column -t -s '	'

## Install in editable mode with dev + viz extras
install:
	pip install -e ".[dev,viz]"

## Smoke tier: sub-60s dev loop (no solves, no subprocesses, no hypothesis)
test-smoke:
	$(JAX_ENV) $(PYTEST) -m "smoke"

## Fast suite (serial, safe everywhere)
test:
	$(JAX_ENV) $(PYTEST) -m "not slow"

## Fast suite on 4 CPU cores (opt-in; needs >=8 GB free RAM)
test-par:
	@echo "WARNING: spawning 4 JAX workers (~2-3 GB peak). Ctrl-C if RAM is tight."
	$(JAX_ENV) $(PYTEST) -n4 --dist worksteal -m "not slow"

## Full suite incl. optimizer-convergence regressions (serial)
test-slow:
	$(JAX_ENV) $(PYTEST)

## Unit layer only (serial)
test-unit:
	$(JAX_ENV) $(PYTEST) tests/unit -m "not slow"

## Lint with ruff
lint:
	ruff check src tests

## Auto-format with ruff
format:
	ruff format src tests
	ruff check --fix src tests

## Type check with mypy
typecheck:
	mypy src/driftjax

## Build documentation locally
docs-serve:
	mkdocs serve

## Build documentation (static site)
docs:
	mkdocs build

## Remove build artifacts and caches
clean:
	rm -rf dist/ build/ *.egg-info .mypy_cache .ruff_cache .pytest_cache
	rm -rf src/*.egg-info src/driftjax/*.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

## Build sdist + wheel
build: clean
	python -m build

## Run lint + fast tests (CI pre-merge check)
check: lint test

## Run everything: lint + tests + typecheck + build
all: lint typecheck test build

## Reference forward validations (N=500; several minutes each)
validation:
	PYTHONPATH=src JAX_ENABLE_X64=1 python validation/ex1_np_junction.py
	PYTHONPATH=src JAX_ENABLE_X64=1 python validation/ex2_np_hetero.py
	PYTHONPATH=src JAX_ENABLE_X64=1 python validation/grad_vs_fd.py

## Smoke-run the whole example gallery
examples-quick:
	PYTHONPATH=src JAX_ENABLE_X64=1 MPLBACKEND=Agg python -m examples
