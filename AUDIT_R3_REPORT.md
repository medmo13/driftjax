# DriftJax Third-Round Scientific & Software Audit Report (R3)

**Target:** `/home/med/Desktop/final/open14/driftjax_v0.1.13/driftjax_v0.1.11` (driftjax 0.1.13)
**Audit date:** 2026-09-11
**Environment:** Python 3.13.13, JAX/jaxlib 0.10.2 (CPU, float64 forced at import),
NumPy 2.4.6, SciPy 1.18.0, Equinox 0.13.8, pytest 9.1.1, ruff 0.16.2, mypy 2.3.0.
Venv: `/home/med/Desktop/final/venv_latest`. Box: 1 CPU / ~7 GiB RAM.
**Baseline:** the pristine copy at `/home/med/Desktop/final/open12/driftjax_v0.1.13_audited/driftjax_v0.1.11`
was byte-identical to the target before modification (verified with `diff -rq`);
it served as the pre-change reference. The prior rounds are recorded in
`AUDIT_REPORT.md` (2026-09-07/08); this report covers only the new R3 work.

---

## 1. Architecture Overview

DriftJax is an end-to-end differentiable 1-D drift–diffusion–Poisson (Van Roosbroeck)
solar-cell simulator in JAX. Data flow:

```
Device/layers + materials.yaml ──► DeviceDesign (dimensionless Vt/L0 units, units.py)
  ──► optics.generation (Beer–Lambert/TMM/Fresnel, science/optics.py; AM1.5G spectrum.py)
  ──► init_cell ──► PVCell (+G(x))
  ──► solve_eq (Poisson–Thomas) ──► sweep (Newton + Block-Thomas O(N), continuation.py)
  ──► Solution (IV, eff/voc/ff/jsc/pmax via PCHIP MPP, spline.py)
  ──► jax.grad(simulate) via custom_vjp IFT adjoint (simulate.py, autodiff/adjoint.py)
```

Numerically load-bearing modules: `numerics/residual.py` (3N residual assembly),
`numerics/analytic_jacobian.py` (hand-derived block-tridiagonal Jacobian),
`numerics/block_thomas.py` (unpivoted O(N) direct solve), `solvers/newton.py`,
`science/carrier_statistics.py` (Boltzmann / 128-pt GL + Sommerfeld FD / Blakemore),
`science/recombination.py` (SRH + radiative + Auger). External deps: jax, equinox,
jaxtyping, numpy, scipy, optax, PyYAML (+matplotlib/optimistix optional).

## 2. Baseline

| Area | Result (pre-change reference) |
|---|---|
| Prior audit state | All suites green per `AUDIT_REPORT.md` §G (27 unit files, gradient/conservation/convergence/property/literature/regression, CLI 16/16, mypy 0 errors) |
| R3 pre-fix demonstrations (this round) | Eq-adjoint cell-objective: AD dChi0 = **−50.0 / −8.96** vs FD **−2.4e-4** (boltzmann/fd); Fresnel Rb parameter provably unused (code read + post-fix sums differ); `white()` ΣP = 20000 W/m²; `_inv3` sign flip, linsolve 0/0-NaN, `UTC = UTC` NameError path confirmed by reading |
| R3 post-fix full runs | Unit 148 passed; property+reproducibility+validation+literature 44 passed; gradient+conservation+convergence 17 passed; regression fast 16 passed; batched slow 2 passed; CLI `verify` 16/16; tutorial_01 η = 20.0002% (ex1 anchor); mypy 0 errors / 80 files; ruff parity with baseline (all remaining findings pre-exist) |

No test was deleted, skipped, weakened, or de-quarantined. The one `slow`/`xfail`
optimizer quarantine from the prior round was left untouched (out-of-scope for 1 CPU).

## 3. Scientific and Mathematical Findings (R3)

**CRITICAL-1 · Dropped direct channel in the equilibrium adjoint (silent wrong gradients).**
`simulate.py::_eq_bwd` assumed the equilibrium "loss" is the potential itself and
returned `dL/dd = −λᵀdF/dd` only. Any objective with explicit cell dependence
(e.g. Σn(cell,pot)) lost its `dL/dcell·dcell/dd` channel: measured d(Σn)/dChi[0] =
−50.0 (boltzmann) / −8.96 (fd) vs finite-difference −2.4e-4 — wrong by ~10⁵×/10⁴×.
Same function also built its VJP cell without the forward `statistics`
(FD/Blakemore equilibrium gradients used the Boltzmann VJP cell).
Evidence: AD-vs-central-FD on a degenerate (5e20 cm⁻³) device, both statistics modes.

**HIGH-2 · Fresnel `rear_reflectance` silently ignored (wrong physics).**
`science/optics.py::_fresnel_per_lambda` had no `R_b` argument although its
docstring derives `G = αΦ₀(1−Rf)[e^{−τ} + R_b·e^{−2τL}·e^{+τ}]`; `fresnel_generation`
accepted `rear_reflectance` and dropped it, so every Fresnel call used a perfect
rear mirror. The existing `test_fresnel_front_reflection_loss` passed regardless
(thick-cell ratio sits in-band either way) — a test that could not detect the bug.

**MEDIUM-3 · SRH trap-level convention misdocumented.**
`fields.py` documented `Et` as "trap level above Ev (eV)", but the implementation
`n1 = ni·e^{Et}, p1 = ni·e^{−Et}` with the physical default Et=0 (midgap) is the
Ei-relative convention. Verified against SRH theory (Shockley–Read 1952; Kiel
SRH derivation: emission ∝ ni·exp(±(EDL−EMB)/kT); recombination peaks at midgap).
Math unchanged; docs corrected.

**MEDIUM-4 · `white()` was a ~20-sun source.** 200 W/m² × 100 points = 20000 W/m².
Renormalised to 1 sun (ΣP = 1000 W/m²). Only consumer was its own test.

**MEDIUM-5 · Batched-sweep tolerance/gradient inconsistencies.**
`solvers/batched.py`: (a) freeze gate `f_tol` hardcoded 1e-10, ignoring caller's
`tol` (tol=1e-12 "converged" without meeting it); (b) currents used raw
`mean(Jn+Jp)`, bypassing `total_current`'s custom JVP, so serial and batched
gradients differed by construction; (c) `solve_eq` called without `allow_trace`,
returning the unsolved guess as the batch warm start under jit/grad.

**MEDIUM-6 · `total_current` cell-channel JVP swallowed failures as 0.**
`except Exception: dI_cell = 0` silently dropped the entire optics/doping gradient
channel. Now propagates.

**MEDIUM-7 · `slsqp` discarded an explicit Jacobian** (`jac=jac is None`) and its
tuple/scalar handling broke under explicit-jac use. Now forwarded.

**LOW-8 · Linear-solver residual NaNs at exact roots.** `linsolve` dense/banded/auto
(and `newton._linear_solve` dense, `ptc._linear_solve` dense) divided by `‖rhs‖`
without a floor → 0/0 = NaN at converged roots; `ptc` additionally returned an
absolute residual where the contract/gate is relative; unknown backend strings
fell through to auto-CSR silently. All aligned (`+1e-30`, relative, ValueError).

**LOW-9 · Continuation edge cases.** `linear_extrapolation` 0/0 → NaN guess for
degenerate schedules; `find_voc`'s absolute 1e-300 floor fabricated huge Voc from
denormal (±1e-320) legs; convergence check tested `phi` finiteness only
(phi_n/phi_p NaNs passed as converged); step-halving recovery reported
`converged: False` (stats lacked `resid`).

**LOW-10 · `_inv3` sign-flipping guard.** `where(|det|<1e-30, +1e-30, det)` mapped
e.g. det=−2e-31 to +1e-30, inverting the block. Now sign-preserving.

**LOW-11 · `provenance.UTC` fallback (`UTC = UTC` → NameError on py<3.11)**,
material-name path traversal (`../../etc/x` escaped the resource dir),
`np.load` without `allow_pickle=False`, `at_bias` tracer-concretization without a
diagnostic, fused-path `ff` missing the serial NaN guard, Blakemore docstring
claiming "~1% for η≳1" (measured envelope: 60% at η=1, 67% at η=5 — the suite
already pins the honest envelope; formula kept as provenance).

**INFORMATIONAL.** Spectrum-table `ΣP` vs `∫P dλ`: the AM1.5G table entries are
bin-integrated powers (peak ~40 W/m² over ~30 nm bins), so the unweighted sum is
the correct normalisation and a `trapz`-based "correction" would be wrong; the
`spectrum.py` docstring wording (`ΣP·Δλ-weighted`) is misleading but the code is
right — left unchanged. The `_solve_newton_while` docstring over-claims
best-iterate tracking present only in the eager loop (docs-only gap, behavior
verified identical); noted for the maintainer.

## 4. Fixes Implemented

| # | Root cause → change | Justification / test |
|---|---|---|
| C1 | `_eq_bwd` ignored `g_sol.cell` and `statistics` → pull cell cotangent through `init_cell` VJP (`g_direct − λF`); mirror forward cell exactly | IFT total-derivative identity; `test_eq_adjoint_cell_objective_matches_fd[boltzmann,fd]`: AD vs central FD to 1e-4 (measured ~1e-9) |
| H2 | `_fresnel_per_lambda(..., rear_reflectance=0.9)` + forward the parameter | Documented double-pass equation; Rb-monotonicity + Rb=0 single-pass absorptance identity tests |
| M3 | `Et` docs → Ei-relative (0 = midgap) | SRH theory (n1·p1 = ni², peak at midgap); doc-only |
| M4 | `white()`: 1000/n_points per point | 1-sun convention; 1-sun sum test |
| M5 | `f_tol = min(tol, 1e-10)`; currents via vmapped `total_current`; `allow_trace=_is_tracer(cell)` | Caller tolerance honored; one shared custom JVP; trace-safe eq start; L10 pyramid gate + slow batched==serial |
| M6 | Remove silent `dI_cell = 0` fallback | Wrong gradient worse than exception |
| M7 | `jac=True if jac is None else jac` + scalar fun under explicit jac | SciPy contract; counting-jac test |
| L8 | `+1e-30` floors; relative ptc residuals; `ValueError` on unknown backend | Residual contract; zero-rhs + backend tests |
| L9 | Degenerate-extrapolation fallback; denormal-Voc guard; 3-field finite checks; step-halving `resid` | Honest convergence reporting; 3 regression tests |
| L10 | Sign-preserving det guard | Block-definiteness; `_inv3` sign test |
| L11 | `timezone.utc` fallback + noqa; `_safe_name` allowlist; `allow_pickle=False`; `at_bias` tracer error; fused `ff` NaN guard; Blakemore doc correction | Security/hygiene/parity; traversal, 1-sun, provenance tests |

## 5. Validation

- New: `tests/unit/test_audit_fixes_3.py` — 14 tests, all pass (each verified to
  fail pre-fix by code-path analysis and, for C1/H2, by direct pre-fix measurement).
- Post-fix suites: unit 148 ✓; property+reproducibility+validation+literature 44 ✓;
  gradient+conservation+convergence 17 ✓; regression fast 16 ✓; slow
  `test_batched_sweep` 2 ✓; CLI `verify` 16/16 ✓; tutorial_01 η=20.0002%,
  Voc=1.0504 V, Jsc=20.255 mA/cm², FF=0.8459 (ex1 anchor) ✓.
- Gradient checks: eq-adjoint vs central FD to ~1e-9 relative (degenerate device,
  both statistics); sweep-adjoint gates unchanged and green.
- Static analysis: mypy 0 errors / 80 files; ruff parity with baseline (remaining
  77 findings all pre-exist; 1 new UP017 false-positive on the intentional 3.10
  fallback suppressed with justification).
- Benchmarks/profiling: no performance-motivated changes (all fixes are O(1)
  reporting/forwarding corrections); forward trajectories untouched — the only
  primal-behavior changes are the documented physics corrections (Fresnel Rb,
  white 1-sun), which have no golden coverage by design.

## 6. External Scientific References

1. **Shockley, W.; Queisser, H.-J., J. Appl. Phys. 32, 510 (1961).** Detailed-balance
   limit (30% @ 1.1 eV blackbody; ~44% ultimate). Validates the (prior-round)
   `validation/analytic.py` SQ implementation used as the literature yardstick.
2. **Rühle, S., Solar Energy 130, 139 (2016).** AM1.5G SQ tabulation (33.2% peak).
   Validates the AM1.5G branch of the SQ yardstick.
3. **Scharfetter, D. L.; Gummel, H. K., IEEE Trans. Electron Devices 16, 64–77
   (1969).** SG discretization via Bernoulli function B(x) = x/(eˣ−1) under
   constant-current-between-nodes assumption. Validates `numerics/scharfetter_gummel.py`
   (Bernoulli + Taylor fallback + analytic derivatives re-verified this round).
4. **Shockley, W.; Read, W. T., Phys. Rev. 87, 835 (1952) + Hall, R. N., Phys. Rev.
   87, 387 (1952); Kiel Univ. SRH derivation (Föll script).** SRH rate with
   n1/p1 measured from the intrinsic level; recombination maximal at midgap.
   Validates `science/recombination.py` math and the corrected Et convention.
5. **Burgelman et al., Prog. Photovolt. 8, 111 (2000) / SCAPS-1D.** CdTe absorber
   parameter anchor used by `tests/literature/test_scaps_cdte.py` (green).
6. **Mann et al., Comput. Phys. Commun. 2022 (∂PV).** Reference
   implementation for ex1/ex2 forward parity (20.00%/13.31% anchors reproduced).
7. **JAX/NumPy/SciPy docs** (`custom_vjp`/`pure_callback`/`spsolve` semantics).
   Validates the IFT plumbing and the host-callback isolation in `linalg.py`.

## 7. Remaining Risks

- **Untested regimes:** Fermi–Dirac/Blakemore golden coverage absent (unit-tested
  only); transient solver coverage moderate; GPU/TPU untested (all results CPU f64);
  N=500 slow parity not re-run here (1-CPU box; forward code paths touched only by
  the two documented physics corrections, neither covered by goldens).
- **Known approximation debt:** unpivoted Block-Thomas without pivoting on
  κ~1e14 Jacobians (dense fallback is the mitigation); `_solve_newton_while`
  docstring over-claims best-iterate tracking; `Material.Lambda` non-m/nm tables
  neither promised nor detected; `examples/` `--quick` flag inconsistency with README.
- **Quarantined:** prior-round slow optimizer `xfail` left as-is.
- **Recommendation:** add a dark-IV slope anchor (log J vs V ≈ q/nkT) and
  reconcile the zip/dir/pyproject version-identity mismatch before release.
- **R3 follow-up (implemented 2026-09-11, same session):** the three "best
  things" were applied — (1) `science/spectrum.py` docstring now states the
  bin-integrated-powers verdict with the trapezoid anti-note; (2) public
  `simulate()` emits a `UserWarning` steering `statistics="blakemore"` users to
  `"exact"` (`simulate.py`); (3) superseded — see R3-followup-2 below.
- **R3 follow-up-2 (same session): best-iterate tracking ported into the
  traced loop.** Review feedback correctly noted the eager-only recovery was
  asymmetric for a JAX library whose hot path is compiled. `_solve_newton_while`
  now carries `(best_pot, best_resid, failed)` in the loop state with
  rebound/settled certification mirroring the eager branches one-for-one
  (NaN steps freeze + exit via rebound). Converged-in-step-norm solves return
  the final iterate exactly as before (bit-identical; measured while-vs-eager
  max diff 1.6e-16). Covered by 3 more tests (parity, forced-exhaustion
  settled-tail certification, docstring pin) — audit file now 19 tests.
  Re-validated: gradient+conservation+convergence+fused 19 passed, CLI 16/16,
  mypy 0 errors, ruff clean. **Validation boundary (honest):** the N=500 slow
  parity gates were not re-run (1-CPU box); converged solves are provably
  untouched (identical return path), but any golden relying on a
  previously-non-converged traced tail would change — re-run `make test-slow`
  on a bigger machine before release.
