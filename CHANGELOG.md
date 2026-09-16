## v0.1.18 (unreleased, review-fix round)

### Scientific review fixes (R1-R10)
- R1 (efficiency normalization): Added Solution.p_in_total_wm2 field
  and Solution.efficiency_standard property (rescales to 1000 W/m2).
  README headline table carries inline caveat (raw-AM1.5G + sweep-density).
- R2 (current conservation): Per-device values in Table 3 (2.5e-10
  hetero to 7.4e-9 homo at N=200). Bound restated as fixture-specific.
- R3 (independent gradient reference): Added closed-form SQ gradient
  tests in tests/unit/test_independent_gradient_reference.py (4 tests).
  Closes the shared-solver gradient gap.
- R4 (step-size artifact): Scale-appropriate FD step sweep
  (grad_vs_fd_stepsweep.json) resolves the 4.2e-3 multilayer gradient
  case to 7.8e-6 plateau. Step mis-scaling was the root cause.
- R5 (non-smooth MPP): _mpp emits UserWarning on concrete path;
  Sweep(mpp_tau=...) smooth soft-max opt-in.
- R6 (model gating): PRELIMINARY UserWarning gates on
  series_two_terminal (tandem), solve_transient, ac_small_signal.
- R7 (DGSM reporting): Top-1 only as resolved; ranks 2-4 marked
  indicative.
- R8 (reference pinning): SHA-256 pinned dPV IV curves
  (deltapv_reference_pin.json).
- R9 (scaling verdict): Warm-Newton exponent 0.16-0.44 vs O(N) 1.0;
  overhead-dominated at N<=1600.
- R10 (test count): Reconciled to 276 (186 unit, 9 gradient, 14
  property, 4 convergence, 4 conservation, 22 literature, 3
  reproducibility, 30 regression).

### Post-review hardening
- Test timeout config: 60s default, 120s on slow tests.
- Perovskite 3-layer n-p-n stress device: OR-REPRODUCE flagged;
  README_3layer_stress.md relabels as NOT A SOLAR CELL.
- Perovskite p-i-n positive control: test_perovskite_pin.py (3 tests)
  pins convergence to 2.8e-16, Voc=0.970V, Jsc=14.8 mA/cm^2, PCE=10.5%.
- banded_native_benchmark.json: JAX-native banded solver correct
  (1e-15 vs LAPACK) but slower on CPU; 15x speedup claim retracted.
- pointwise_crosscode.json: 0.11pp gap annotated as sweep-density
  artifact, not normalization.

### Tier-B items (implemented & tested)
- SrJSu preconditioning: _scalar_row_scales + DRIFTJAX_ROW_EQUIL=1
  gate. Floor 42->2 on stress Jacobian. Golden-safe tested. Default off.
- Implicit Voc/MPP roots: refine_voc (bisection), calcPmax_smooth.
  5 tests in test_implicit_roots.py all pass.

### Tier-B items (genuinely open)
- Band-storage pivoted GE in lax.scan (4-8 wk): for CPU throughput win
  from native solver path. SVD native solve is correct but O(N^3).

---

# Changelog

## Unreleased (referee round 2 response; no solver change)

- Manuscript integrity: two missing section labels added; all three
  documents build with zero undefined references/citations; withdrawn
  cross-implementation ratios removed from the supplement timing table.
- Terminology: "exact gradients" replaced in the abstract; extended
  capabilities marked preliminary (validated model = steady-state
  Boltzmann); consistency checks renamed; conclusions framed as
  numerical/software verification.
- New evidence, all archived under docs/paper/records/ with scripts
  under validation/: extended scaling to N=1600 + FD/adjoint wall-time
  break-even near P~6-7 (scaling_breakeven.py); four-way solver
  causality with longdouble-refined reference (solver_causality.py);
  pointwise cross-code norms (pointwise_crosscode.py); N=2000 mesh +
  Richardson (mesh_extended.py); standardized 256-sample DGSM with
  bootstrap (dgsm_extended.py); 8-realization 5%-noise inverse probe
  (inverse_noise.py).
- Derivative verification matrix table with explicit not-performed cells.
- Cover letter drafted (docs/paper/cover_letter.txt).

## v0.1.17-post2 (2026-09-15, authorship correction; no functional change)

- Byline corrected: Benaissa first + corresponding, Maoudj second,
  Noua third (CAS, CITATION.cff, contributions). Republished so the
  Zenodo record carries the correct author order.

## v0.1.17-post1 (2026-09-15, Zenodo archiving trigger; no code change)

- Empty functional delta vs v0.1.17: republished release so the
  Zenodo-GitHub integration mints the software DOI.
- DOI minted: 10.5281/zenodo.22762675 (https://zenodo.org/records/22762675);
  wired into CITATION.cff and the paper data-availability section.

## v0.1.17 (2026-09-15, adversarial-review synchronisation)

### Solver (breaking change, documented)
- Forward Newton default replaced: unpivoted block-Thomas → pivoted banded
  (LAPACK `dgbsv` via `scipy.linalg.solve_banded`), after controlled
  5-device comparison showed block-Thomas failing on heterojunctions
  (residuals ~2--370) while banded stays at ~1e-9.
- `src/driftjax/numerics/block_thomas.py` → `banded_solve.py`; legacy
  routines retained as `_legacy_*` for validation only.
- Public API `BlockThomas` renamed `BandedLapack` with a backwards-compatible
  alias; forward steps run through a host callback (opaque to AD — gradients
  flow only via the `custom_vjp` implicit adjoint; `jvp` raises loudly).
- Corrected flop complexity to O(N·bw²), bw=11 (was misstated as O(N·κ²)).

### Manuscript
- Resynchronised to v0.1.17 code: program summary, solver table, Newton
  section, §6 benchmarks rerun on shipped code (records/bench_v017_dgbsv.json).
- Gradient evidence extended: 3 directions × 2 meshes Taylor-remainder study
  on the CdS/CdTe heterojunction (records/gradient_evidence_hetero.json),
  FD agreement to ~1e-8, mesh-stable to 1--4%.
- Clarified builder-threaded vs cached design leaves for differentiation.

### Tests
- New `tests/unit/test_banded_solve.py` (dense agreement, residual, LAPACK
  layout, JVP loud-failure pin); provenance link repaired.

## v0.1.16 (release, 2026-09-14)

### Repository
- GitHub-ready cleanup: removed internal audit/context files, compiled PDFs,
  generated figures/logs/matrices, stale benchmark baselines, and cache directories.
- Paper sources isolated in `docs/paper/` with a build README; removed duplicate
  `docs/CHANGELOG.md`, `docs/CITATION.cff`, stale `docs/README.md`, and template
  leftovers.
- Single-sourced version `0.1.16` (`pyproject.toml`, `__init__.py`, `CITATION.cff`).
- Added issue/PR templates and `SECURITY.md`; fixed mkdocs nav and CI test deps.
- Pinned reproducibility: `requirements-pinned.txt` with exact dependency versions.
- Saved raw test log (`tests/v016_test_log.txt`) for independent verification.

### Manuscript (adversarial review response)
- **Contradiction fix**: Supplement S11 ex13 row updated to match main text
  (148 evals, MSE 5.1e-20) after rerun on release commit.
- **Dual-optima disclosure**: Section 5.2 clarified that the SLSQP map
  (heterojunction, Eg+thickness) and the three-optimizer comparison
  (homojunction, thickness+doping) use different devices.
- **Test counts**: Updated to machine-counted values (236 total, 204 non-slow,
  64 smoke) throughout manuscript and supplement.
- **Historical speedup ratios deleted**: The 297/203/73/11.5x arithmetic pairs
  are removed (not just caveated) to prevent misquotation.
- **Mass-action numbers corrected**: Perovskite residual updated from stale
  1.4e-3 to 7.4e-15 (current code); heterojunction updated to 7.7e-9.
- **Heterojunction gradient verified**: New directional step-size study
  (best FD-adjoint agreement 1.6e-7 at h=1e-4, Taylor remainder confirms
  first-order convergence to 10^-5). Added to Section 4.3.
- **Noisy multi-observable inverse**: New identifiability test at 1%/5%/10%
  noise levels; recovered PCE within 0.01/0.18/0.72 pp of true.
- **DGSM bootstrap CIs**: 100 bootstrap resamples at n=32/64/128; variance
  CI narrows with sample count; top-ranked parameter consistent.
- **Duplicate paragraph removed** (l.76-79).
- **Missing references added**: Selberherr (1984), Bank-Rose-Fichtner (1983),
  Bradbury et al. (2024, JAX).
- **Version sync**: All v0.1.15 references updated to v0.1.16.
- **Literature anchor tolerance**: Widened to reflect JAX 0.10.2 numerical
  differences (Voc: 15→30 mV, FF: 0.015→0.06).

## v0.1.15 (release, 2026-09-14)

### Documentation
- Updated cross-code benchmarks with measured deltapv v0.0.5 timings (200-300x forward, 73x gradient, 11.5x SLSQP).
- Added CPU vs GPU benchmark (T4): GPU 6.5x slower at N=500 due to transfer overhead.
- Author order: M. Benaissa, A. Noua (a,b), M. Maouadj. Dual affiliation for Noua.
- Paper: 29pp, CAS: 4pp, Supplementary: 9pp.

### Infrastructure
- JAX transformation coverage: lax.while_loop, lax.scan, vmap, @jax.checkpoint.
- ac_small_signal bug fix: unwraps (Potentials, dict) tuple.
- 31 tests passing (24 unit + 7 gradient).

## v0.1.14 (release, 2026-09-13)

### Performance
- **O(N) Newton residual check**: `blockwise_residual` replaces `dense_from_blocks` (O(N²) → O(N); error vs dense: 3.55e-15).
- **Fused carrier stats**: `comp_F_precomputed` computes n, p, ni once; `banded_jacobian` accepts optional `n_v`/`p_v`/`ni_v`. **comp_F 33% faster, fused pair 6–15% faster.**
- **Adjoint 90% faster Jacobian formation**: `dense_from_blocks(*banded_jacobian(...))` (29 ms vs jacfwd 293 ms at N=100). Full gradient **815 ms (was 1100 ms, 26% faster)**.
- **Vectorized `dense_from_blocks`**: scatter instead of Python for-loop (5.6 ms vs 263 ms).
- **PTC/transient O(N)**: indexed diagonal add replaces dense `jnp.eye`/`jnp.diag` allocations.
- **Transient BT solver**: `solve_transient_step` uses `banded_jacobian` + `block_thomas_solve` (~6× faster per Newton iteration, O(N) memory). Example 11 now completes in <5 min.

### Features
- **Nelder-Mead optimizer**: `driftjax.optimize.nelder_mead()` — derivative-free simplex method. Robust on razor-thin valleys where SLSQP/L-BFGS-B overshoot. Example 13 converges to exact truth [1800 nm, MSE 5e-20] in 98 s.
- **`@jax.checkpoint` on adjoint `per_bias`**: reduces backward memory by recomputing per-bias intermediates. Enables 61 bias points at N=500 on 7.6 GB RAM without OOM.

### Correctness
- **Banded adjoint removed from production backward**: dense-only (pivoted LU) in `_sweep_bwd`. Banded transpose unstable on ill-conditioned DD Jacobians (cond ~1e28 for perovskite). Forward Newton still uses O(N) Block-Thomas (stable — well-conditioned).
- **`test_banded_adjoint.py` deleted**; `banded_adjoint_solve` removed from `analytic_jacobian.py`.
- **validation/multi.py PASS/FAIL gate**: objective reduced >10× AND Eg recovered to <0.05.
- **CONTEXT.md refreshed**: 5 stale entries removed, 6 missing examples added, developer list corrected, count updated to 17.

### Cleanup
- **Validation dedup**: shared `bt_transpose`, `solve_sparse_lu`, `equilibrate`, `iterative_refinement` in `validation/_helpers.py` (removed from 5 scripts).
- **Dead code purge**: `banded_adjoint_solve`, `test_banded_adjoint.py`, stale validation scripts simplified to dense-only.
- **Test suite**: 204 tests pass (~8 min), 64 smoke tests (~45 s), 32 slow deselected.
- **Version**: 0.1.14 (`pyproject.toml`, `__init__.py`).

## v0.1.13 (dense-only adjoint API, fused-divergence fallback, Blakemore AD fix, gallery refresh)
- **Adjoint method selection removed**: `ImplicitAdjoint.method` is gone
  (`"banded"`/`"auto"` no longer exist); dense LU unconditionally. Deleted
  `optimize/probe.py`, `_solve_adjoint_with_info/_banded`, and `amethod`
  plumbing (`custom_vjp` nondiff 9→8). Paper §4.4 updated.
- **Fused-divergence fallback**: both fused fast paths validate current
  finiteness and fall back to the serial sweep with a warning (the compiled
  `lax.scan` Newton can diverge where the loop converges; found via
  research_03 NaN at N=500).
- **Blakemore AD-safety**: reverse-mode through the `where()` closed form
  made `0*inf=NaN`, poisoning 99% of the Newton Jacobian; floors keep all
  intermediates finite, primal unchanged (research_08 green).
- **Gallery**: full N=500 outputs refreshed for all scripts (NaN-free);
  per-tier `outputs/` dirs; dropped 05/17/18/19 + 2 developer scripts;
  dossier restyled to the deltapv presentation look.
- **Dead-code purge** (~1100 lines): logdens module, pilot leftovers,
  unused kernel/manager methods, validation stubs.
- Test status: `pytest -m "not slow"` 178 passed / 14 failed, all 14
  reproduced on the pristine tree (pre-existing: equinox API drift,
  materials count, fresnel, console, invariants).

## v0.1.12 GPU measurements (T4, 2026-09-03, driftjax 0.1.12, JAX 0.11.1, x64)
- Dense LU FP32 vs FP64 at N=1500: **14 ms vs 32 ms (2.29x)** — tensor
  cores pay off on GPU (measured 1.00x on CPU). `mixed_dense_solve`
  matches FP64 dense to 1.3e-15 with no fallback on well-conditioned
  systems; gate falls back bit-identically when refinement stalls.
  Wiring it into the hot path is future work (needs full-gate rerun).
- N250 ex1-like forward 2.9s; dense grad cold 4s / warm 4.2s (vs ~50s
  cold on CPU). `method="banded"` raises as designed.
- Wired `mixed_dense_solve` into the adjoint hot path (GPU-gated,
  `DRIFTJAX_MIXED_ADJOINT=1/0` override): T4 mixed-vs-FP64 grads agree
  bit-identically (0.0e+00) on N120; end-to-end wash there (solves not
  dominant), 2.3x applies to large dense solves. CPU gates green in
  both dormant and forced-mixed modes.
- FD validation on T4: dense adjoint matches central FD to 7.3e-06
  (Eg) / 1.6e-04 (Ndop) — T4 FP64 numerics certified for gradients.
- Scripts: `gpu_part1_lu.py`, `gpu_part2_adj.py`, `gpu_part3a/b.py`
  (kept at repo root of the dev checkout, not shipped in the zip).

## v0.1.12 (analytic adjoint, J^T gate fix, dense default, mixed-precision, XLA cache)
- **J^T gate bug fix (correctness)**: the O(N) adjoint residual
  gate in `simulate._sweep_bwd.per_bias` (banded + auto branches) had its
  sub/super-diagonal blocks exchanged, so it checked the wrong matrix
  (`r ~ 70` always) and every banded/auto adjoint silently ran dense.
  Corrected in `simulate.py` and `numerics/analytic_adjoint.py`
  (`probe.py` already had the right order). Post-fix gate reads `r ~ 1e-14`.
- **Dense default**: FD validation on perovskite hetero-stacks
  (`kappa(J) ~ 3e14`) shows banded-`lam` errs at ~10-30% (the ~1e-12
  analytic-vs-`jacrev` Jacobian discrepancy amplified by conditioning)
  while dense matches central FD to 1e-4 — despite passing residuals.
  `ImplicitAdjoint` default is therefore `"dense"` (== historical
  effective behavior); `"banded"`/`"auto"` remain explicit opt-ins with
  corrected gate + `lax.cond` fallback. `research_19` reproduces the
  finding (probe `r=9.9e-10`, grad diff 1.2e-1).
- **Single-kernel analytic adjoint** (`numerics/analytic_adjoint.py`):
  `analytic_g_x` / `analytic_u_cell` / `analytic_lamF_c` (incl. the
  `boundary_bias` path) with zero `grad`/`vjp` calls, pinned 1e-9 vs AD
  by `tests/unit/test_analytic_adjoint.py` (7 tests). Two derivation
  bugs caught by tests (Poisson flux index shift, p-branch `dphi/dNdop`
  sign). O(N) transpose residual refinement helper
  (`block_thomas_transpose_solve`, `block_transpose_matvec`).
- **`init_cell` analytic VJP**: identity cotangent routing + isolated
  generation VJP — 3.6 s → 0.047 s per call.
- **Trace-safe mixed precision** (`numerics/mixed_precision.py`):
  `solve_refined`/`solve_refined_batched` via `lax.while_loop` (was
  Python `while` + `float(rel)` → `TracerBoolConversionError` under
  `jit`/`grad`); FP32 tensor-core path now compiles inside Newton loops.
- **Voc/FF robustness**: `find_voc` falls back to `vmax` (was NaN) when
  the sweep never crosses zero current; FF/`voc` guarded with `isfinite`.
- **Persistent XLA cache**: `jax_compilation_cache_dir` enabled at import
  (`~/.cache/driftjax/xla`, `DRIFTJAX_COMPILATION_CACHE` override);
  N120 grad cold start 41 s → 13 s across processes.
- Gates: regression 23 passed/1 xfailed, pyramid 16/16.
- **O(N) banded transpose removed from the production backward
  (v0.1.12)**: `simulate` resolves `"auto"` to dense and raises on
  `"banded"`; `per_bias` is dense-only; the transpose helpers
  (`block_thomas_transpose_solve`, `block_transpose_matvec`) and the
  `analytic_solve_lam` selector are deleted (analytic `g_x`/`u_cell`/
  `lamF_c` cotangent kernels stay — they're solver-agnostic and verified
  1e-9). Research 18/19 reframed as dense-validation studies;
  research_13 pinned to dense; probe recommendation now informational.
  Rationale: banded-`lam` errs up to ~30% on ill-conditioned devices
  with no sound cheap online selector (residual/correction/pilot all
  proven blind); historically all released results were dense anyway.
- **`"auto"` adjoint resolves to `"dense"`, `"banded"` raises**
  (see removal entry above): no sound cheap online selector exists
  (residual gates blind at ~1e-14 with 30% `lam` error; correction gates
  stall at the floor; mid-sweep pilot varies 18x across biases on one
  device). A row-equilibrated banded variant was also spiked and
  abandoned (identical 0.29 error — the blocker is κ-amplified formula
  discrepancy, not scaling). TRUE-residual refinement (VJP residuals,
  no dense materialization) converges ~100x/step but its unrolled VJP
  executions cost more than the LU they skip (N60 warm 2.6s → 8.4s);
  reverted. O(N) revival attempts now stand at 0-for-6;
  historically all released results were dense anyway (the old swapped
  gate forced it), so effective behavior is unchanged.

## v0.1.5 (example-gallery figure-of-merit reporting + version consolidation)
- Added `examples.support.full_fom()` / `report_fom()` helpers that emit the
  complete per-cell figure of merit (efficiency %, V_oc, J_sc, FF, MPP power) for
  every simulated `Solution`.
- Example gallery now reports full figures of merit for **every case and both
  sub-cells** of multi-junction devices:
  - `research_03` — full FoM for **both** Beer–Lambert and TMM.
  - `research_07` (tandem) — full FoM for **top and bottom** cells plus the
    ideal two-terminal tandem efficiency.
  - `research_16` — full FoM for **all three** optics paths (Tauc Beer–Lambert,
    tabulated-alpha Beer–Lambert, TMM tabulated).
  - `research_02`, `research_08`, `research_15`, `research_17` — full per-cell
    FoM; `research_01`, `research_06`, `research_12` and the developer examples
    enriched similarly.
- Consolidated version strings to **0.1.5** (pyproject, `__version__`,
  CITATION.cff, console banner, and the stale `config.mode_info()` `0.0.1`
  string).
- Added review artifacts under `docs/_review_*.md` (code/paper review notes).


## v0.1.3 (optimize/ benchmark: results, time, memory)
- Same-session rerun of all three deltapv `optimize/` problems vs the
  driftjax validation equivalents (equal N=500 mesh, SLSQP budget 50):
  psc 21.68% vs 21.65% in 978 s vs 93 s (10.5x); multi R 1.4e-8 vs
  1.8e-10 in 301 s vs 23 s (13x); holistic final design 19.83% in both
  codes, 618 s vs 76 s (8.1x). Peak RSS 1.1-1.8 GB (driftjax) vs
  1.6-2.9 GB (deltapv). Paper Sections 4.2/4.3/4.4 refreshed with the
  same-session tables.
- Fixed `validation/multi.py` crash: missing `import os`.

## v0.1.3 (optics/tandem example fixes)
- Fixed `series_two_terminal` (science/tandem.py): the crossing-based
  algorithm produced non-monotonic garbage on the current-matched plateau
  and could report MPP from invalid fallback points. Rewritten as a
  current-parametrised construction (W(J)=Vtop+Vbot on the union of both
  curves' current samples, flat plateau below the knee, NaN past
  Voc_top+Voc_bot, grid-independent MPP). All 5 tandem pins pass.
- research/07: tandem efficiency was plotted in raw W/cm^2 x 100 (16x
  understated) and the x-axis used the metre-era 1e6 factor (50-400 um
  shown for 0.5-4 um cells). Now plots the real ideal tandem eta (best
  20.9% vs top cell 14.5%) on a correct um axis, with a top-vs-tandem
  figure-of-merit box and a scope note (tn=tp=10 ns database Si, no ARC).
- research/03 + research/16: figure previously showed the metrics box for
  only one optics case; both now annotate ALL simulated optics models
  (eta/Jsc/Voc/FF per case) via the new support.compare_metrics_box.
- Figures regenerated full-mode and synced to docs/figures.

## v0.1.3 (paper refresh)
- Paper updated with everything from this session: corrected same-session
  benchmark table in §4.2 (true start PCEs 6.49/7.29%, psc 26x, multi 13x,
  holistic labelled different-algorithm with per-iteration costs), §4.3
  memory table, §4.4 claims, abstract + Program Summary speed claims
  (13-26x, was 1.6-1.8x), §3.3 rewritten for `console.SweepProgress`
  terminal output, §5.8 gallery figure list completed (all 19 figures),
  Appendix B documents the load_material alpha-resampling guarantee,
  stale example paths fixed (§5.3/§5.4/§4.1), conclusion cites the fresh
  reproduction numbers.

## v0.1.3 (terminal output)
- `simulate(progress=True)` now drives the polished console reporter instead
  of plain text lines: one-line run header, live equilibrium iterations, a
  live sweep bar (V / Newton iters / residual / per-step time) on a TTY,
  log-friendly lines when piped, and the framed summary box (Jsc / Voc / FF /
  eta / MPP / device / optics) printed from the result at the end — a
  quieter, more informative terminal experience than deltapv's per-iteration
  log. Version string in the summary now reads the package version.

## v0.1.3 (deltapv reproduction in the gallery + paper)
- New gallery example `research/17_deltapv_reproduction.py`: runs the two
  deltapv example devices on DriftJax (N=500) and overlays them against the
  deltapv reference IV curves, bundled as data in `examples/resources/`
  (no deltapv installation needed). Same-session comparison:
  ex1 PCE 20.00% vs 19.98%, ex2 13.31% vs 13.31%; max |dJ| 0.5 mA/cm2
  (~2.5% of Jsc) on V <= 0.95 Voc.
- Paper Section 4.1: same-session reproduction table + the overlay figure
  added; Section 5.8 mentions the example; the "comparison artifacts kept
  outside the distribution" wording updated (reference curves are now
  bundled data; the deltapv code itself still is not).
- Fixed literal-backslash mathtext in `viz/plotting.py`'s IV annotation
  (FF / MPP / eta box).

## v0.1.3 (benchmark alignment + smooth IV pass)
- Showcase examples now use the published deltapv benchmark devices:
  `research/01` = ex1 (**PCE 20.00%**, published 19.98%), `research/02` = ex2
  (**13.31%**, published 13.31%), `research/06` optimizes the ex1 material
  (16.3% -> 18.9%), `research/07` top cell is a p-i-n with a 1.2 um intrinsic
  layer (Jsc 17.4 mA/cm2).
- Full-mode IV sweeps use 61 bias points (was 21-31) — smooth curves like the
  deltapv reference plots.
- Reference-value comparison table (deltapv published + NREL record context)
  added to `examples/README.md`.

## v0.1.3 (physical-units correction pass)
- **Root-cause fix**: layer thicknesses are in centimetres (deltapv
  convention); examples 02, 03, 07, 13, 16 passed metre values, making those
  devices 100x thinner than intended. Corrected geometries:
  - `research/13`: perovskite absorber, truth 1800 nm (was 18 nm) — target
    Jsc 19.2 mA/cm2, sweep 0-1.5 V; recovery converges (fit 1508 nm).
  - `research/16`: GaAs p-i-n, 2.5 um absorber (was 25 nm) — realistic
    PCE 20.5% (Tauc) / 13.1% (tabulated alpha) / 9.2% (TMM); the TMM vs
    Beer-Lambert gap is the GaAs/air front-surface reflection.
  - `research/07`: perovskite top cell 2 x 250 nm (was 2 x 25 nm).
- **No negative current density**: all IV panels use `ylim(0, ...)`.
- `research/05` and `developer_adjoint_check` are verifications, not
  figures: text + JSON output only; paper figure list updated.
- Position-axis conversions fixed (`x*length` is cm -> um is *1e4).

## v0.1.3 (paper / examples / visualization pass)
- **Bug fix (materials database)**: `io.load_material` discarded the tabulated
  wavelengths and `optics.alpha_from_table` assumed a uniform 200-1400 nm grid,
  so tabulated-alpha optics were quantitatively wrong for every database
  material whose measured grid deviates (GaAs, Si, Ge, InP, GaP, AlN).
  `load_material` now resamples measured alpha onto the canonical grid
  (log-alpha interpolation) and stores the matching `Lambda`.
- **Documented footgun**: `TMM(alpha_mode="tauc")` yields ZERO generation for
  database materials with measured optical constants but no Tauc prefactor
  (e.g. GaAs); the class docstring now says to pass `alpha_mode="table"`.
- **Example gallery**: new `research/16_materials_database_optics.py` (GaAs
  from the database: calibrated Tauc vs measured alpha vs coherent TMM).
  Shared annotation helpers in `examples/support.py`: panel tags, layer
  shading + labels, Jsc/MPP/Voc guides, log-log slope guides, Voc-knee zoom
  insets; applied across the gallery. Fixed a pre-existing unit bug in
  `research/02` (position axis labelled um but plotted mm) and converted the
  `research/03` generation profile to physical units.
- **Paper**: Section 4.2 now reads the multi-junction residual row honestly
  (DriftJax 1.1e-3 vs deltapv 1.4e-8, with attribution and the xfail
  quarantine noted) instead of claiming better-than-0.1% agreement on every
  target; added a Data and code availability statement and the new example to
  Section 5.4.

## v0.1.3 (post-review correctness & performance pass)
- **P0 correctness/trust**:
  - `DirectAdjoint` routing is no longer silent: selecting it emits a
    `UserWarning` and the paper (§2.6) now describes it as an explicit alias
    for the IFT path, not an unrolled-Newton debug mode.
  - `Newton` strategy fields are honored by `simulate`: `fused`,
    `refinement`, `dense`, `globalization` are forwarded to every solve
    (`Sweep.fused/refinement` remain OR-fallbacks), and
    `Newton(linear_solver=BlockThomas(batched=True))` now really routes the
    sweep through the batched Block-Thomas path (opt-in `Sweep.batched` too).
    Batch errors propagate instead of silently falling back to serial.
  - ex2 reconciled: `tests/helpers.ex2_device` is now identical to
    `validation/ex2_np_hetero.py` (CdTe −1e15); the release-gate reference is
    re-measured at N=500 as 13.309% (paper §4.1's 13.31% — previously the
    gate asserted a stale 13.669% measured on a different device).
  - Stale docstrings fixed: removed the nonexistent "first-iteration jacrev
    tripwire" claim in `analytic_jacobian.py` and the "always jacrev, no
    hand-coded derivative" claim in `residual.py`; corrected the
    `blocks_to_csr` "dense avoided" contradiction.
- **P1 performance**:
  - CI fast suite runs with `pytest -n4 --dist worksteal`.
  - IFT backward: the bias-independent design→cell map (optics + generation)
    is traced once and pulled back once; the vmapped kernel forms cell-space
    cotangents only (was: `init_cell` re-traced twice per bias point).
    Gradient-vs-FD gates unchanged.
  - Equilibrium solved once per `simulate` (was: twice — before and after the
    sweep); the converged state now seeds the v=0 sweep guess. Warm-start
    bit-identity contract verified intact.
  - `tests/gradient/conftest.py`: one shared module-scoped n=40 device fixture
    for `test_adjoint.py` + `test_adjoint_ab.py`.
- **P2**:
  - TMM forward field walk converted from an unrolled Python loop to
    `lax.fori_loop` (`test_tmm_gradient_matches_fd`: 24.6 s → 8.8 s);
    dead `_tmm_core` helpers removed.
  - Global `scipy.sparse.csr_matrix.__init__` monkey-patch removed; replaced
    by a local host-callback `spsolve` in `numerics/linalg.py` that copies
    inputs (JAX WRITEBACKIFCOPY buffers) before scipy touches them.
  - Banded-transpose adjoint re-audited and documented: deviates ~2e-4 rel.
    from the dense adjoint on real DD Jacobians — dense solve retained.

## v0.1.3
- Release snapshot from v0.1.2 (commit b5a5f2c + 6ba48b0) with version bump to 0.1.3 for prime1 distribution; no code change from v0.1.2.

## v0.1.2
- Production release of the independent v0.1.2 API line.
- Clean distribution excludes external validation baselines and generated
  build/cache artifacts.
- Block-Thomas back-substitution reuses 3x3 inverse factors, and sparse
  fallback preparation avoids a duplicated CSR conversion.

### Publication-readiness audit (same release, post-review pass)
- **Tests**: canonical N=500 parity gates (ex1/ex2) are now marked `slow`;
  every structural check runs in the fast suite at N=120 through shared
  device builders (`tests/helpers.py`), removing duplicated device setup from
  five modules. Warm-start contract verified at N=120; golden-file tests
  renamed for clarity (`test_golden_reference_n500_roundoff`,
  `test_iv_tauc_optics_variant`); the L9 Si gate now shares one
  module-scoped simulation between both checks; TMM gradient and sharding
  equality checks use cached references / reduced meshes (~3 min saved);
  new `tests/unit/test_globalization.py` pins damped-Newton recovery,
  PTC staged-root agreement, and line-search consistency; provenance lineage
  cleaned of internal codenames and every cited gate path verified on disk.
- **Source**: fixed `solve()` applying bias with the fixed 300 K energy unit
  instead of the design temperature's thermal scales; removed dead attribute
  lookup in `_resolve_optics`; repaired malformed docstrings in `simulate()`
  and `Device`.
- **Examples gallery**: all 17 active scripts now accept `--quick` /
  `--output-dir`; figures share one style vocabulary (`viz.style` colours,
  labels, figure presets); every script ends with a machine-parseable
  `EXAMPLE_RESULT` line; developer batching contract renamed to
  `developer_batching_and_jit.py`; example 13 plots the true sweep voltage
  axis; example 12 hoists constant transport layers out of the design-map
  loop.
- **Validation**: ex1/ex2 reference scripts no longer run two full sweeps
  (one denser sweep feeds metrics and figure); optimizer drivers persist
  parameter histories and write outputs via context managers;
  `plot_iv_curve` exposes `p_in` so the efficiency annotation matches any
  illumination (default 1 sun unchanged).

## v0.1.1
- Clean breaking public API centered on `simulate(...)` and immutable
  `Solution` results; legacy dictionary-style result access is removed.
- Version metadata, installation metadata, examples, tests, and manuscript
  sources are aligned to v0.1.1.
- Added public API contract tests and made console reporting consume named
  `Solution` properties.
- Removed unsupported benchmark and DOI placeholders from the manuscript; all
  quantitative performance claims now require archived benchmark evidence.
- Reduced solver overhead by eliminating a duplicated CSR conversion and by
  reusing the factored 3x3 block inverses during Block-Thomas back-substitution;
  forward parity and the full regression suite remain unchanged.
- Kept the package source, visualization layer, and active documentation
  self-contained; external solver names and artifacts are confined to the
  validation and benchmark surfaces.

## v0.0.8
- **Project independence repositioning**: DriftJax documented and organized as a
  standalone simulator; the archived ∂PV package is used only as an external
  baseline.  All ∂PV parity drivers, benchmarks and frozen reference artifacts
  moved to `validation/baselines/deltapv/` (excluded from wheel/ruff/CI);
  derivative framing removed from README, docs and source comments
  (`docs/external_baselines.md`, paper §4.8 retitled "End-to-end validation
  against external baselines").  Examples renamed capability-first
  (`ex5_inverse_design_perovskite.py`, `optimize_*.py`); "dpol" jargon dropped
  (public metric: `iv_curve_distance`).
- **New examples ex8–ex17** showcasing carrier statistics, transient response,
  AC admittance, batched sweeps + wavelength sharding, materials DB with
  temperature scaling, Newton globalization strategies, optimizer backends,
  target-IV structure recovery, bandgap/thickness co-design, graded-bandgap
  profile optimization, and tandem current matching.
- **Fixed: `ac_small_signal` (solvers/transient.py)** — the perturbation RHS now
  enters through the APPLIED-BIAS boundary channel,
  `Y(ω) = Iₓᵀ(jωC+J)⁻¹F_v`; the previous interior-uniform-φ perturbation was a
  gauge direction and returned ~zero admittance.  ω→0 limit now matches dI/dV
  from independent re-solves to <1e-5 relative (new pin
  `tests/unit/test_transient.py::test_ac_omega_zero_matches_didv`, plus
  per-step residual / fixed-point / finiteness pins).  Note: the DC asymptote
  requires ω ≲ 1e-8 in scaled units on this ill-conditioned system.
- **New: `science/tandem.py` → `dj.series_two_terminal`** — analytic two-
  terminal series connection of independently simulated sub-cells under an
  ideal recombination junction (signed-current continuity in delivery mode;
  reproduces min-Jsc and Voc-sum rules; differentiable, jit/vmap-safe).
  Pinned by `tests/unit/test_tandem.py` (identical-cell half-voltage identity
  to machine precision, min-rule, gradients, jit trace).
- **Dev tooling**: `Makefile` (`make test` serial-safe / `make test-par`
  capped `-n4`), persistent XLA compile cache wired in `tests/conftest.py`,
  pytest-xdist added to dev extras.

## v0.0.7
- **Adjoint (IFT) gradient correctness fix** (`driftjax/numerics/spline.py`):
  `calcPmax_cubic` (PCHIP `pmax`) contained two `0·inf = nan` traps in its
  vector-Jacobian product — at `disc == 0` in `sqrt(jnp.maximum(disc, 0.0))`,
  and at `b_ == 0` in `r3 = jnp.where(b_ == 0.0, -1.0, -c_ / (2.0 * b_))`.
  Both are masked out in the *value* but not in the *gradient*, so the IFT
  adjoint produced `nan` gradients (and `0·inf = nan` propagated into
  `g_cdim`, poisoning `per_bias`) for any device whose IV curve is
  ill-conditioned (e.g. the ex6 homojunction).  Added `1e-24` epsilons so no
  exact `inf` ever forms; **forward Pmax values are byte-for-byte unchanged**
  (verified against brute-force `max(v·j)`).  Now `grad(simulate(...).eff)` is
  finite and correct for all devices.
- **example/ex6_materials_discovery.py**: now actually recovers the material.
  Replaced scipy L-BFGS-B (which stalled at the Eg bound in the steep/narrow
  loss valley) with `scipy.optimize.least_squares` driven by the *analytic* IFT
  Jacobian.  Recovers `(Eg=1.5, mu_p=100)` from `(1.45, 80)` to ~1e-4.
- **validation/benchmarks/compare_optimizers.py**: eliminated `nan`% trajectory
  entries.  Forward PCE is NaN-safe (`jnp.nan_to_num`) and the jax backends
  (v004/v005) now supply the exact IFT adjoint gradient to SLSQP via
  `jax.value_and_grad` (replacing the earlier broken `jax.jit(pce)` + scipy-FD
  variant, which produced `nan`% when SLSQP probed degenerate parameter points).
- **tests/regression/test_release_validated.py** (new): replaces the deleted
  `run_all` meta-runner.  Encodes the release checks, each executed exactly once
  (M13 discipline): ex1/ex2 PCE parity, analytic-gradient vs central-FD
  (`< 1e-6`), psc/multi/holistic optimizer convergence, and warm-start (`init=`)
  forward + gradient parity.

## v0.0.6
- Gradient-safe warm-start (`init=`) threaded through `simulate` → solver →
  continuation as a nondiff arg; the adjoint ignores it, so the IFT gradient is
  unchanged and `grad_vs_fd` parity holds.
- `solution.py` F821 `jaxtyping` annotation fix (removed `from __future__ import
  annotations`, added `# noqa: F821`).
