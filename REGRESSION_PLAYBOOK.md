# DriftJax Regression Playbook

How this repo hunts regressions — learned the hard way (Sep 2026 gallery run).
Read this before modifying `src/driftjax`, especially solvers, optics, statistics,
or anything on the `simulate()` hot path. After any change, work the checklist
at the bottom.

## 1. The oracle: committed gallery outputs

`examples/{research,tutorial,developer}/outputs/*.json` are pinned full-resolution (N=500) results. They are
the regression oracle: any full gallery run that disagrees with them is either
a bad script or a real regression. Never "update" an output to match new code
without proving the new number is the correct one (see §6).

## 2. Shrink first: the probe discipline

Never debug by re-running whole examples. Write a minimal probe in `/tmp`
(not the repo) that reproduces one number:

- One sweep, dump `voltages`/`current`, check `isfinite` per bias point.
- Test F (primal) and J (Jacobian) **separately** at a fixed guess — a finite
  residual with a NaN Jacobian and vice versa are different bugs.
- Test statistics/optics primitives directly across their input range
  (e.g. `F_half`/`dF_half` over η ∈ [-30, 20]) before blaming the solver.
- One variable per probe: mesh, steps, `fused`, statistics, optics.

## 3. Bisect with worktrees, never with checkout

To find which commit broke behavior, check out old commits into throwaway
worktrees (`git worktree add /tmp/gal/wt_<name> <sha>`) and run probes with
`PYTHONPATH=/tmp/gal/wt_<name>/src`. The main tree stays on `main` with outputs
intact. Drive `git bisect` manually (one probe per step; full sweeps take
minutes) or with `git bisect run` + a pass/fail probe script. Remove worktrees
when done (`git worktree remove --force`).

Caveats learned:
- Give each probe a **fresh compilation cache** (`DRIFTJAX_COMPILATION_CACHE=<empty dir>`)
  when comparing across versions, to rule cache effects in/out explicitly.
- Run background jobs under `setsid` (tool timeouts kill process groups; `nohup`+`&` is not enough).
- Python buffers stdout when redirected — a silent log means nothing; check `ps` for life.

## 4. Known traps (do not re-learn these)

1. **`Sweep.fused` defaults to `True`** (`problems.py`) and `simulate()` ORs the
   solver and protocol flags. Setting `Newton(fused=False)` does NOT give you
   the serial sweep — you must also pass `Sweep(..., fused=False)`. Any probe
   of "unfused" behavior must set both.
2. **The compiled whole-sweep scan and the serial sweep are different programs.**
   Rounding-level trajectory differences (1e-12 from bias point 1) amplify at
   ill-conditioned high-bias points: the scan can diverge where the loop
   converges. Both fused fast paths (`_simulate_sweep`, `_sweep_fwd`) therefore
   validate `isfinite` on currents and fall back to `_forward` with a warning.
   Keep that guard on any new fast path.
3. **Reverse-mode AD through `jnp.where` forms `0 * inf = NaN`** when the
   unselected branch's tangent overflows (e.g. Blakemore `t**(-2.7)` at `t=0`).
   Any closed form with a branch must keep every intermediate finite **with
   finite tangents** on both sides — check `t**(-3.7)`-type amplifications too,
   not just the primal. Test `dF`-style derivatives, not just values.
4. **Stale outputs lie.** A committed output can predate the code that last
   "modified" it (dirty-tree commits). Bit-identical-to-Boltzmann Blakemore
   numbers were a stale artifact, not proof Blakemore worked. Cross-check
   suspicious numbers against physics (Jsc ∝ ∫G; TMM vs BL within reflection
   losses) and unit tests before trusting them.
5. **NaN signatures discriminate causes.** NaN only at top bias points =
   continuation/Newton trajectory divergence. NaN from point 0 (Jsc NaN) =
   equilibrium/linear-solve failure — look at the Jacobian first.
6. **Huge finite currents past Voc are usually physics, not divergence.**
   With no series resistance, ideal-diode forward current at $V\gg V_{oc}$
   is enormous (research_16: $-6.7\times10^4$ A/cm² at 1.1 V, fully
   converged, resid $\sim10^{-14}$). Check `converged`/`resid` flags before
   assuming solver failure — and scope IV figures to the PV quadrant so
   the tail doesn't compress the science flat.

## 5. Case log (concrete precedents)

- **Fused-routing regression (`ab5194d`, fixed `6694669`).** Symptom: research_03
  `eff=nan`, last bias current non-finite. Bisect → default path rerouted through
  `_forward_fused_scan`. Fix: finite-check + serial fallback, both call sites.
- **Blakemore AD-NaN (fixed `b3c4085`).** Symptom: research_08 Blakemore all-NaN
  from V=0. `F_jacobian` 128880/129600 non-finite → `dF_half` NaN for η≤0 →
  floors at 1e-80 + exp clamp at 700. Primal unchanged.
- **Stale TMM number (no code change).** Committed 10.75% vs fresh 8.93%;
  proved fresh correct via fused==unfused G-integrals + Jsc∝∫G + unit tests.

## 6. Post-change checklist (run every time)

1. `ruff check src/driftjax/<touched>` (I001 in `simulate.py` is pre-existing; leave it).
2. Targeted unit tests for the touched module (e.g. `tests/unit/test_tmm.py`, statistics tests).
3. Minimal probes: failing-case reproduction + a healthy-case control (fast path still taken, no fallback warning where it previously succeeded).
4. Eager-grad probe if the change touches forward code used under `custom_vjp`
   (jitted-grad NaN on ill-conditioned devices is a known separate limitation; do not chase it here).
5. Full gallery at N=500 before release (no reduced modes exist anymore).
6. NaN scan: `grep -hoE "[a-z_]+=(nan|-nan|inf|-inf)"` over all `EXAMPLE_RESULT` lines
   **and** `grep -rl "NaN\|Infinity" examples/*/outputs/*.json` — must be empty.
7. Commit outputs together with the fix; note any intentionally changed numbers
   (and their proof) in the commit message.
