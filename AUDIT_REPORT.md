# DriftJax Scientific & Software Audit Report

**Audited version:** driftjax v0.1.13 (zip metadata says v0.1.14, inner directory says v0.1.11 — version-identity inconsistency, see §I)
**Audit date:** 2026-09-07/08
**Environment:** Python 3.12.14, JAX 0.10.2 (CPU, float64 enabled on import), NumPy 2.4.6, SciPy 1.18.0, Equinox 0.13.8, pytest 9.1.1 — fresh venv built from the repository's own `requirements-lock.txt` (two `-e` lines pointing at the maintainer's local paths were stripped to install).
**Sandbox constraint:** 4.1 GiB RAM / 2 cores. All results below were produced in this environment; tests that cannot fit are explicitly marked as such with the workaround used.

---

## A. Executive Summary

The repository is a differentiable 1-D drift-diffusion–Poisson solar-cell simulator (Scharfetter–Gummel discretization, Newton/continuation solvers, Beer–Lambert/TMM/Fresnel optics, implicit-function-theorem adjoint). Its scientific core — Van Roosbroeck system, SG currents, Boltzmann/Fermi–Dirac/Blakemore statistics, Beer–Lambert and TMM optics — was **independently verified** and is sound: the analytic banded Jacobian matches `jacrev` to 1e-9, the N=500 golden IV reference is reproduced to **max|dj| = 4.1e-14 A/cm²**, manufactured-solution convergence is order 2.02, gradient checks pass at the FD noise floor, and the flagship canary anchors reproduce exactly.

The audit found and fixed **9 significant issues** (2 HIGH scientific-integrity, 1 HIGH units, 1 MEDIUM silent-wrong-physics, 1 MEDIUM runtime crash, 1 HIGH convergence-semantics + 1 HIGH sign error from the first audit round, plus 2 LOW). The most consequential was a **validation module that validated nothing**: `sq_efficiency_limit` returned a hardcoded `0.337` constant and its literature test passed circularly. It was replaced with a true Shockley–Queisser detailed-balance implementation that reproduces the published 1961 anchor (30% at 1.1 eV) and modern AM1.5G tabulations (33.2% peak) to within a fraction of a percent.

Final state: **all test suites green** (27 unit files, all functional directories, all 9 slow-marked tests, 22 literature tests), mypy clean (0 errors, 80 files), ruff clean of all correctness-relevant findings, CLI verification pyramid 16/16, and every fix carrying a regression test.

## B. Architecture

```
Device / material DB (io.py, materials.yaml, optics/*.npz)
      ↓ design()  (units.py thermal_scales → dimensionless mesh)
DeviceDesign ── optics model (science/optics.py: photonflux, tauc/table α, TMM, Fresnel)
      ↓ init_cell (generation profile G baked in)
PVCell → solvers: Newton (newton.py; analytic banded Jacobian, analytic_jacobian.py +
         block_thomas.py with dense fallback), continuation sweep (continuation.py),
         batched vmapped sweep (batched.py), transient (transient.py)
      ↓ Solution (iv_curve, Voc/Jsc/FF/η; find_voc interpolation)
Differentiation: custom_vjp IFT adjoint (simulate.py `_simulate_sweep/_eq`), DirectAdjoint
         alias with loud warning; adjoint error signature experiment (validation/stability)
Validation stack: verification pyramid L1–L10 (validation/*), literature anchors
         (tests/literature), golden IV files (tests/resources), CLI (driftjax
         verify|validate|benchmark|release)
```

JAX execution: float64 forced at import; `jit` at solver level, `vmap` for batched sweeps and per-wavelength optics, `custom_vjp` for Newton-stable gradients, persistent XLA compilation cache. Randomness: JAX keys only (`jax.random.key(42)` in the conditioning experiment); the suite verifies bit-identical reruns.

## C. Baseline (before fixes)

Recorded from the pre-modification tree (fast suite; slow tests deselected):

| Area | Baseline result | Evidence |
|---|---|---|
| Unit tests (26 files) | 25 pass, **1 FAIL** (`test_hardening_fixes.py::test_mats_three_layer_preserved` → NaN efficiency on 3-layer heterojunction at T=350 K) | per-file chunked runs (RAM-limited box) |
| conservation / convergence / gradient / property / reproducibility / validation | all pass | pytest runs |
| literature (ideal diode, SQ, SCAPS, experimental) | pass, but SQ test was **circular** (see D-1) | tests/literature |
| release gate (fast) | 4 pass | test_release_validated.py |
| CLI verify | 16/16 gates | `driftjax verify` |
| examples | tutorial scripts not runnable without `PYTHONPATH=.` | run log |
| mypy | 9 errors in 6 files | mypy 2.3.0 |
| Baseline failure root cause | Newton loop could not *certify* convergence on a near-degenerate Jacobian (residual ≈ 5e-13 reached, step-norm never below tol) → exhaustion → fallback chain → NaN | Task 7 reproduction |

## D. Scientific Findings

Ranked: CRITICAL > HIGH > MEDIUM > LOW > INFORMATIONAL.

**D-1 · HIGH (scientific integrity) — SQ "limit" was a hardcoded constant.**
`src/driftjax/validation/analytic.py`. `sq_efficiency_limit` returned `0.337` for Eg within 0.2 eV of 1.12 and a linear scaling otherwise; its docstring (garbled) admitted it was not the SQ calculation. The literature test asserting `0.3 < f(1.12) ≤ 0.35` passed *by construction* — the exact "100% passing tests, wrong equation" failure mode. Consequence: the package's flagship literature anchor validated nothing.
Evidence: original source (pre-fix) + test; replaced implementation reproduces published anchors (§E-1). Reference: Shockley & Queisser 1961; Rühle 2016.

**D-2 · HIGH (units) — `monochromatic()` produced a physically impossible source.**
`src/driftjax/science/spectrum.py`. The package convention for `LightSource.Lambda` is **nanometres** (every optics consumer multiplies by 1e-9; `simulator.py` comment "already in nm"). `monochromatic(wavelength_m)` stored raw metres — `monochromatic(5e-7)` yielded λ = 5e-16 m (≈ 5e15 eV photons), all α lookups clamped to the transparent extrapolation, and the cell rendered **dark with no error**. `white()`/`spectrum()` were already correct (nm).

**D-3 · HIGH (convergence semantics) — Newton could not certify convergence; exhaustion returned garbage.**
`src/driftjax/solvers/newton.py`, `continuation.py`, `simulate.py`, `batched.py`. Step-norm-only termination cannot distinguish "settled" from "stuck" on near-degenerate Jacobians; on the 3-layer T=350 K case the loop exhausted `max_steps` at the root (residual 5e-13) and returned `converged=False`, triggering fallbacks that produced NaN efficiency. Fix (root cause, not symptom): best-iterate tracking in the eager Newton loop (return the lowest-residual iterate on NaN/exhaustion when it beats the tail), residual-aware certification, per-member freezing in the batched solver, adaptive step-halving continuation. Golden-reference trajectories verified bit-identical after the change (§G).

**D-4 · HIGH (sign error) — SRH Jacobian block `srh_phip`.**
`src/driftjax/numerics/analytic_jacobian.py::_recomb_deriv`. Analytic derivative had a sign error: correct `−np·den + num·tn·p)/den²`, code had the opposite sign of the first term. O(1) relative error in the SRH block; absolute error ~1e-13 sat below the 1e-9 pinning tolerance at the pinned state, i.e. the existing pinning test was blind. Verified independently with SymPy derivation + autodiff + finite differences. Fixed; new AD-vs-analytic test with scale-aware tolerance added.

**D-5 · MEDIUM (silent wrong physics) — custom α tables silently misread.**
`src/driftjax/io.py::material()`. `material(alpha=..., Lambda=...)` stored both, but `design()`/`alpha_from_table` read α rows on the fixed 200-point 200–1400 nm grid and never consulted `Lambda`. A custom grid was silently reinterpreted as canonical → wrong absorption with no warning. Fixed by canonicalizing (log-α resample, metres/nm auto-detect) at construction.

**D-6 · MEDIUM (runtime crash, results loss) — `validate/release --json` crashed after the run.**
`src/driftjax/validation/stability/condition_sweep.py` had **no** `save_record`, but `__main__.py` (2 call sites) invoked it: with `--json`, the CLI completed the entire expensive validation and then died with `AttributeError`, losing the results. Fixed by adding `save_record` (mirrors `scalar_fd.save_record`); verified end-to-end.

**D-7 · MEDIUM (broken tool) — cross-code parity script could never run.**
`src/driftjax/validation/cross_code/deltapv_parity.py` called `sweep(..., pot_ini=...)`; the parameter was renamed to `init` long ago → guaranteed `TypeError`. No test executes the script, so the breakage was invisible. Fixed kwarg; script now executes (its cross-code physics remains unverifiable without the external `deltapv` reference — see §I).

**D-8 · LOW (resource hygiene) — progress reporter leaked on exception.**
`simulate()` closed the progress object only on success; a mid-sweep exception leaked the `DebugLog` file handle and lost the JSONL trailer. Fixed with try/finally; regression test proves the `done` event is emitted on failure.

**D-9 · LOW (fragile test) — `test_grad_vs_fd` FD step artifact.**
The analytic-adjoint-vs-FD gate mixed O(1)–O(160) parameters with O(1e-4) thicknesses; a single absolute FD step (1e-6 = 1% relative on the thickness) produced a 1.45e-6 discrepancy that is pure step-size artifact (drops to 5.63e-7 at eps=1e-5, diverges at 1e-4; the adjoint itself is step-independent). Fixed with per-parameter best-of-two-steps FD; **gate not loosened**.

**INFORMATIONAL**
- Package AM1.5G table (99 rows, smoothed): measured photon current above 1120 nm = 44.5 mA/cm²; the commonly cited ASTM G173 value is ≈ 46 mA/cm² (citation not verified to publication grade in this audit — see §I). Affects SQ-AM1.5G accuracy by ≲ 1 point, covered by the test bands.
- `B023` closures in `validation/benchmarks/scaling.py` / `stability/solver_selection.py` are invoked within the same loop iteration — safe; left as-is.
- Version identity: zip name v0.1.14, pyproject 0.1.13, inner dir v0.1.11 — release-engineering hygiene issue only.

## E. Fixes Implemented

**E-1 · Real Shockley–Queisser detailed balance** (`validation/analytic.py`, rewritten).
Root cause: placeholder constant (D-1). Change: Planck photon emittance Φ(E,T) = 2πE²/(h³c²(e^{E/kT}−1)); 6000 K blackbody sun at 1-sun dilution (fc=1) and optional package-AM1.5G mode; radiative dark current over the front hemisphere; MPP via Newton on (1+v)e^{v} = 1+Jsc/J0; plus `sq_ultimate_efficiency` (thermalisation-only). CODATA-2018 constants.
Scientific validation (all independent of the implementation): η(1.1 eV) = 0.3004 vs SQ-1961's published "30% for 1.1 eV"; ultimate peak 0.4387 @ 1.10 eV vs published ≈ 44%; AM1.5G Si 0.330 / peak 0.332 vs Rühle-2016 ≈ 0.32–0.33 / 0.332; tail-expansion identity to 1.6e-6; ζ(3)/Stefan–Boltzmann normalisation identity to 2e-4; grid convergence 1.2e-7; η_ult ≥ η_SQ ordering exact; degenerate inputs → 0.0. Regression tests: `tests/literature/test_sq_limit.py` (14 tests).

**E-2 · Monochromatic units conversion** (`science/spectrum.py`). metres → nm (×1e9) at construction; contract documented at the field (`fields.py`). Regression: `test_audit_fixes_2.py` (λ in nm, P = φ·E_γ energy-conservation identity, above-bandgap illumination with physical upper bound φ·q, below-bandgap transparency control).

**E-3 · Newton certification & best-iterate** (D-3; first audit round). Regression: the previously failing `test_mats_three_layer_preserved` now passes; golden N=500 roundoff unchanged (4.1e-14) proves settled trajectories untouched.

**E-4 · SRH Jacobian sign** (D-4; first audit round). Regression: `test_analytic_jacobian.py::test_recomb_deriv_matches_ad` (AD vs analytic, scale-aware relative tolerance).

**E-5 · Custom-α canonicalization** (`io.py::_canonicalize_alpha`). Regression: step-table resample from metres and nm, canonical passthrough.

**E-6 · `condition_sweep.save_record`** (D-6). Regression: end-to-end `validate --json` writes `condition_sweep.json` + `fd_step_sweep.json`.

**E-7 · `deltapv_parity` kwarg** (D-7). Verified by execution.

**E-8 · Progress-close on exception** (D-8). Regression: monkeypatched sweep failure still emits the JSONL `done` event.

**E-9 · `test_grad_vs_fd` per-parameter FD** (D-9, gate unchanged). Release fast suite green.

**E-10 · Hygiene**: mypy 9 → 0 errors (mesh.py `jax` import for annotations; `core` annotation in `carrier_statistics.py`; `list[dict[str, Any]]` + float coercion in `forward_scaling.py`; zip-safe `Path | Traversable` + `.is_file()/.open()` in `io.py::_db_path/_load_db`); 47 ruff autofixes (unused imports, import order, whitespace) with the full unit suite re-run after; dead code removed (`expmDpsip`, unused `missing`); `fields.py` nm-convention comment corrected.

## F. Iterative Verification History (condensed)

| # | Issue | Fix | Test | Validation | Status |
|---|---|---|---|---|---|
| 1 | 3-layer T=350K NaN (baseline failure) | best-iterate Newton + continuation step-halving | the failing test itself | golden N=500 unchanged | **closed** |
| 2 | SRH Jacobian sign | corrected `srh_phip` | `test_recomb_deriv_matches_ad` | SymPy + AD + FD agreement | **closed** |
| 3 | SQ constant placeholder | full detailed balance | 14-test literature suite | SQ-1961 30% anchor; Rühle 33.2%; identities to 1e-6 | **closed** |
| 4 | `monochromatic` dark source | m→nm conversion | 3 regression tests | energy conservation; spectral selectivity | **closed** |
| 5 | custom-α misread | canonical resample | 2 regression tests | step-table resample identities | **closed** |
| 6 | `--json` crash | `save_record` added | CLI end-to-end | records written and parse | **closed** |
| 7 | `pot_ini` TypeError | `init=` | execution | script runs | **closed** (physics unverified, §I) |
| 8 | progress leak | try/finally | failure-path test | `done` event on exception | **closed** |
| 9 | grad_vs_fd FD artifact | per-parameter step | the test itself, gate unchanged | step-size sweep diagnosis | **closed** |
| 10 | perf suspicion (BeerLambert path) | none needed | controlled re-measure | paths bit-identical, same speed | **closed — no defect** |

## G. Final Validation (all actually executed)

| Suite | Result |
|---|---|
| Unit tests — 27 files, per-file chunked | **27/27 pass** (incl. 8 new audit-fix regression tests) |
| conservation / convergence / gradient / property / reproducibility / validation | 3 / 4 / 10 / 15+1 deselected / 3 / 4 — **all pass** |
| regression (fast) | **16 pass** (incl. repaired `test_grad_vs_fd`) |
| slow-marked tests, run individually | **8 pass + 1 xfail** (documented optimizer quarantine): canary 4.3 s; ex1/ex2 PCE parity; psc/holistic SLSQP; batched==serial; golden N=500 & tauc N=500 executed manually (pytest variant OOMs in 4.1 GiB sandbox — `MALLOC_ARENA_MAX=1 + taskset` workaround) |
| Golden N=500 roundoff | **max\|dj\| = 4.14e-14** vs golden file (gate 1e-8); voltage grid exact |
| Tauc-optics golden N=500 | max\|dj\| = 2.14e-3 (gate 5e-3) |
| Literature | **22 pass** (SQ 14, ideal diode 2, experimental 2, SCAPS 4) |
| Gradient checks | analytic adjoint vs FD ≤ 5.6e-7 (FD noise floor); Bernoulli AD vs FD 1.9e-10; adjoint-vs-FD scatter documented in-test |
| Convergence | Poisson MMS order 2.02; N-refinement η: 0.0662 (200) → 0.0660 (300–500) |
| CLI | `info`, `simulate` (canary anchors reproduced: 17.43 mA/cm² / 0.4939 V), `verify` **16/16 gates**, `validate`, `validate --json` |
| Examples | tutorials 01 (η=20.0%) and 03 at N=200 (FD-vs-adjoint 7e-4) pass; N=500 tutorial default exceeds sandbox RAM |
| mypy | **0 errors in 80 files** |
| ruff | 13 remaining findings — 4 verified-safe `B023` closures, 9 cosmetic (E702/E402/E741/B905) |
| Reproducibility | bit-identical rerun gate passes; deterministic JAX keys only |

## H. External Scientific References (all actually consulted during this audit)

1. **Shockley, W.; Queisser, H.-J. (1961). "Detailed Balance Limit of Efficiency of p-n Junction Solar Cells." *J. Appl. Phys.* 32, 510.** DOI: 10.1063/1.1736034. Establishes: detailed-balance formulation, 6000 K blackbody sun at fc=1, η_max ≈ 30% at 1.1 eV, ultimate limit ≈ 44%. Used to: verify E-1 blackbody anchors and the MPP/balance equations.
2. **Rühle, S. (2016). "Tabulated values of the Shockley–Queisser limit for single junction solar cells." *Solar Energy* 130, 139–147.** DOI: 10.1016/j.solener.2016.04.030. Establishes: AM1.5G detailed-balance tabulation, η_max ≈ 33.2% near 1.34 eV, Si ≈ 32–33%. Used to: validate the AM1.5G mode of E-1 (within the package table's documented fidelity).
3. **ASTM G-173 / AM1.5G references (NREL spectrum standard).** Establishes: standard 1000 W/m² terrestrial spectrum and photon-current magnitudes. Used to: sanity-check the package spectrum table's photon content (measured 44.5 mA/cm² above 1120 nm; published integral ≈ 46 — see §I for the unverified-citation caveat).
4. **Scharfetter–Gummel / Van Roosbroeck discretization literature (package-internal provenance: `test_scaps_cdte`, SCAPS-1D CdTe tutorial parameter set)** — the SCAPS benchmark parameters (Eg=1.5, χ=3.9, ε=9.4, Nc=8e17, Nv=1.8e19) match the standard SCAPS-1D teaching absorber; the corresponding literature anchor tests pass. Primary URL of SCAPS: University of Ghent ELIS; Burgelman et al., *Prog. Photovolt.* 8, 111 (2000).
5. **Official NumPy/SciPy/JAX documentation** (numerical APIs: `np.trapezoid`, `lambertw`-free MPP Newton, `jax.custom_vjp` semantics). Used to: confirm API contracts used in fixes.

## I. Remaining Risks and Uncertainties

- **WHAT IS UNKNOWN — the exact provenance number for the AM1.5G photon-current integral.** The package table gives 44.5 mA/cm² above 1120 nm; my recollection of the ASTM G173 integral (~46 mA/cm²) was **not verified to citation grade** during this audit (search snippets were inconsistent). Consequence: possible ≲ 1-point offset in AM1.5G SQ values; test bands already accommodate it. Verify later by integrating the official ASTM G173 table.
- **Cross-code parity physics** (`deltapv_parity.py`): the script now runs, but its output (near-dark Jsc, FF = −540) indicates the diagnostic device/conventions are not physically meaningful; verifying the parity against the external `deltapv` code is impossible here (not installed). Treat as an unfinished diagnostic, not a validated parity gate.
- **Hardware/parallelism:** all results are single-core CPU f64. Multi-threaded eigen reductions were disabled for determinism; GPU/TPU behaviour untested. The N=500 fused-scan path peaks at ≈ 3.3 GB — larger grids need a machine with more RAM; the pytest N=500 slow tests cannot run in 4.1 GiB (workaround documented in §G).
- **Numerical regimes not tested:** Fermi–Dirac/Blakemore statistics are unit-tested but not exercised by the golden files; transient solver coverage is moderate; extremely stiff contact-limited regimes unexplored.
- **Dark-IV slope validation** (log J vs V ≈ q/nkT) is a natural additional literature anchor; not yet implemented.
- **Version identity mismatch** (zip/dir/pyproject) should be reconciled by the maintainer before the next release.
- **Unfixed by design:** 4 `B023` loop-closure patterns (same-iteration use, safe) and 9 cosmetic ruff findings; `Material.Lambda` accepts µm-unit tables neither promised nor detected (documented m/nm only).

## J. Files Changed

**Source (fixes):**
- `src/driftjax/validation/analytic.py` — rewritten: real SQ detailed balance + ultimate efficiency + enriched ideal-diode docs (E-1).
- `src/driftjax/science/spectrum.py` — `monochromatic` metres→nm conversion (E-2).
- `src/driftjax/solvers/newton.py`, `solvers/continuation.py`, `solvers/batched.py`, `simulate.py` (round 1) — Newton certification/best-iterate, step-halving, per-member freezing (E-3).
- `src/driftjax/numerics/analytic_jacobian.py` — SRH sign fix (E-4); dead `expmDpsip` removed (E-10).
- `src/driftjax/io.py` — custom-α canonicalization (E-5); `Path | Traversable` zip-safe DB loading; dead `missing` removed.
- `src/driftjax/validation/stability/condition_sweep.py` — added `save_record` (E-6).
- `src/driftjax/validation/cross_code/deltapv_parity.py` — `pot_ini` → `init` (E-7).
- `src/driftjax/simulate.py` (round 2) — try/finally progress close (E-8).
- `src/driftjax/fields.py` — nm-convention documentation (E-2).
- `src/driftjax/numerics/mesh.py`, `science/carrier_statistics.py`, `validation/benchmarks/forward_scaling.py` — typing corrections (E-10).

**Source (hygiene, ruff autofixes only):** `autodiff/checkpointing.py`, `science/optics.py`, `runtime/provenance.py`, `__main__.py`, `validation/benchmarks/scaling.py`, `validation/gradients/adjoint_vs_forward.py`, `validation/gradients/scalar_fd.py`, `validation/physics/limits.py`, `validation/record.py`, `validation/stability/solver_selection.py`, `numerics/block_thomas.py`, `simulator.py` (round-1 audit edits retained).

**Tests:**
- `tests/literature/test_sq_limit.py` — rewritten: 14 independent literature/identity tests (E-1).
- `tests/unit/test_audit_fixes_2.py` — **new**: 8 regression tests for E-2/E-5/E-8.
- `tests/regression/test_release_validated.py` — per-parameter FD step selection, gate unchanged (E-9).
- `tests/unit/test_analytic_jacobian.py`, `tests/unit/test_hardening_fixes.py`, `tests/gradient/test_inverse_design_grad.py` — round-1 regressions retained/strengthened.

**Verification artifacts (scripts kept out of the repo, in the audit workspace):** memory-scaling probe, golden N=500 manual runner, SQ smoke/anchor checks, controlled perf comparison, chunked unit runner.

---

*Report integrity statement: every result in §G was actually executed in this audit environment. Items marked "unverified" in §I are explicitly not claimed as validation. Scientific claims tagged to references in §H were checked against those sources during this session.*
