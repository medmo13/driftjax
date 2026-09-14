# DriftJax publication-package audit

Revision date: 14 September 2026.

## Reference article and writing approach

The supplied reference is `prism-uploads/2105.06305v3.pdf`: Mann et al.,
*∂PV: An End-to-End Differentiable Solar-Cell Simulator*, preprint v3,
9 December 2021, subsequently published in *Computer Physics Communications*
272 (2022), 108232. The 27-page uploaded PDF was read locally, including
the model, software architecture, sensitivity/optimisation/inverse examples,
and technical appendices.

The revision adopts its scientific organisation, not its prose or results:

| Reference-paper method | DriftJax adaptation |
| --- | --- |
| Explain the design problem before listing software features | Introduction develops the need for sensitivities and the one-dimensional block-structure opportunity |
| Express the simulator through a compact residual and its implicit derivative | Main article presents the transport equations, block Newton solve, and adjoint with consistent notation |
| Connect mathematical objects to a user-facing simulation workflow | Architecture maps physical inputs, residuals, solvers, custom VJP, and objectives to documented modules |
| Develop a few focused applications | Each main example follows question, setup, objective/method, retained result, and interpretation |
| Put implementation derivations in appendices | Supplement provides scaling, flux derivatives, block elimination, multi-bias adjoints, verification, and auxiliary formulations |
| Compare meaningful outputs and costs | Residuals, gradients, curve differences, objective values, and timing scope are distinguished |

The reference's Appendix E explicitly describes compact band storage and
GMRES with ILU(0). It does not support attributing a dense Newton solver to
the published ∂PV method. The previous contradictory explanation of the
reported baseline speedups is removed from all three manuscripts.

## Reference-code comparison added on 14 September 2026

The shared main/CAS article now includes a dedicated comparison with the
public `romanodev/deltapv` source and a six-component comparison table.
The inspected reference already has analytical Jacobian derivatives,
compact band storage, a dense terminal-current adjoint, and JAX transforms.
The text therefore identifies the documented direct Block-Thomas forward
solve as the principal algorithmic distinction, not analytical derivatives,
implicit differentiation, or dense adjoints in isolation. Additional model
options are distinguished from demonstrated accuracy or performance gains.

Supplement S8 records inspected modules and the moving-branch limitation;
S9 adds matched-physics and identical-Jacobian comparisons, separately from
validation of additional physical models. A repository bibliography entry
records the access date. No commit could be pinned, no reference simulation
was rerun, and the DriftJax implementation remains documentation-based.
The source inspection does not identify the archived v0.0.5 timing revision.

## Established-method citation audit

The main/CAS body and supplement now cite the literature at the relevant
definitions, equations, algorithms, and methodological cautions, rather
than relying on an introduction-only bibliography. The shared database
contains 36 entries (23 added in this pass). The main/CAS body uses 35
distinct references at 54 citation locations; the supplement uses 26 at
35 locations.

| Topic | Supporting references added or reused |
| --- | --- |
| Drift--diffusion equations, densities, contacts, bulk recombination | Gaury et al.; Mann et al.; Shockley and Read; Hall |
| Normalised Fermi--Dirac statistics and approximations | NIST DLMF; Blakemore |
| Optical generation, transfer matrices, photovoltaic metrics | Mann et al.; Byrnes; Shockley and Queisser |
| Conservative discretisation and interface/mesh cautions | Scharfetter and Gummel; Eymard, Gallouët, and Herbin |
| Block elimination, stability, conditioning, refinement | Demmel, Higham, and Schreiber; LAPACK Users' Guide; Carson and Higham |
| Nonlinear continuation and implicit derivatives | Kelley and Keyes; Mann et al.; Blondel et al. |
| JAX differentiation, loops, checkpointing, branch limitations | Official JAX API and methodological documentation |
| Manufactured solutions, convergence, verification versus validation | Salari and Knupp; Oberkampf and Trucano |
| Derivative verification | Official dolfin-adjoint Taylor-test documentation |
| DGSM versus Sobol sampling | Sobol' and Kucherenko; official SciPy Sobol documentation |
| SLSQP, simplex searches, identifiability | Official SciPy SLSQP documentation; Nelder and Mead; Raue et al. |
| Backward differentiation, small-signal response, series tandems | Official SUNDIALS IDA mathematical documentation; Laux; Henry |

Publisher/proceedings records, author/institutional archives, original
research reports, and official software documentation were consulted.
The Kelley--Keyes and Oberkampf--Trucano records are explicitly cited as
the verified technical reports, without mixing report DOIs and journal
publication metadata. Undated documentation has access dates, not invented
publication years. The SLSQP reference identifies the documentation version;
it does not identify the version used for the archived DriftJax runs.

Citations establish the antecedent methods, not the correctness of DriftJax
or its reported timings. Retained numerical results remain attributed to
the supplied example record, and the missing raw-log/source limitations
remain in place. Algebra derived under this manuscript's sign conventions
is distinguished from the cited underlying method. The abstract and
conclusions summarise the study; detailed literature attribution appears
in the corresponding body sections. References to other software's
mathematical documentation do not imply that DriftJax depends on it.

## Canonical source structure

- `DriftJax_paper.tex`: standard single-column article entry point.
- `DriftJax_CAS.tex`: actual Elsevier `cas-sc` entry point, with highlights.
- `DriftJax_content.tex`: the complete scientific body shared by both.
- `DriftJax_abstract.tex`: the abstract shared by both.
- `DriftJax_shared.tex`: notation, author information for article-style
  documents, and class-compatible formatting helpers.
- `DriftJax_supplementary.tex`: standalone technical supplement, S1–S9.
- `cas-refs.bib`: corrected bibliography used by all three documents.
- `DriftJax_paper.md`: source-map and editorial entry point rather than an
  independently maintained, scientifically conflicting manuscript.

The main and CAS scientific text can no longer diverge accidentally because
both wrappers input exactly the same body and abstract. Figures 1–5 are
shared by the main/CAS article; supplementary figures S1–S6 contain the
remaining existing images. All eleven original raster assets are preserved.

## Implementation evidence

No executable source tree, test suite, dependency manifest, raw arrays, or
benchmark scripts were present in this workspace. The new upload is a
reference paper, not DriftJax source code. The documented repository could
not be retrieved in this environment; this is not evidence that the
repository is nonexistent or unavailable to the authors.

Implementation alignment is therefore based on `CODEBASE_DESCRIPTION.md`,
`README.md`, and `CHANGELOG.md`, especially the v0.1.12–v0.1.15 entries:

- Analytical block-tridiagonal assembly and Block-Thomas Newton solve.
- Production dense-only pivoted LU transpose solve; no active banded/auto
  selector in `ImplicitAdjoint()`.
- Solver-independent analytical Boltzmann cotangent kernels.
- Per-bias checkpointing and documented compiled/batched execution paths.
- Optional GPU-gated mixed-precision dense path, refinement, and fallback.
- Documented 25-entry material inventory, without a calibration claim.
- Separate steady-state, transient, AC, tandem, and optimisation capabilities;
  no blanket claim of derivative coverage for all auxiliary solvers.

The manuscripts state this evidence scope. No simulator output, optimisation,
gradient verification experiment, or performance benchmark was rerun.

## Scientific improvements

1. The abstract and introduction now present a specific contribution rather
   than a feature inventory or an editorial audit. Unsupported evidence is
   qualified where relevant and documented systematically in S8–S9.
2. Forward and backward complexities are separated: linear work for a
   fixed-block Newton solve, cubic factorisation and quadratic matrix storage
   for each dense adjoint. Nonlinear iteration counts and bias counts remain
   explicit.
3. Sign conventions, physical units, quasi-Fermi variables, normalised
   Fermi–Dirac statistics, contact normals, and incident-power definitions
   are consistent across the package.
4. The supplement derives Bernoulli derivatives, recombination derivatives,
   the block transpose, complete multi-bias sensitivity contractions, and
   storage terms in potential coordinates. These are mathematical derivations,
   not claims of source-level verification.
5. AC state perturbations are distinguished from terminal admittance,
   including the need for displacement current and compatible frequency units.
6. Convergence, backward error, gradient accuracy, physical validity, and
   cross-code agreement are treated as distinct tests.
7. Design studies now specify the objective and interpretation: coordinate-
   dependent sensitivity ranking; bounded efficiency optimisation; and
   noise-free matched-model curve recovery, not unique experimental inference.
8. Numerical results from the uploaded ∂PV paper were not repurposed as
   DriftJax results. No new measured numbers or synthetic performance claims
   were created.

## Remaining author/source decisions

These are prerequisites for a submission-ready empirical study, not merely
typesetting issues:

1. **Provide the pinned code.** Supply the complete DriftJax release snapshot,
   commit/archive identifier, environment manifest, selected tests, and logs.
2. **Recover device identities and inputs.** The mesh example is labelled
   silicon but elsewhere specifies a 1.5 eV gap. It remains a synthetic
   homojunction pending confirmation.
3. **Reconcile efficiency normalisation.** At N=500 the old metric tuple
   (20.25 mA/cm², 1.050 V, FF 0.846) gives Pmax=17.988075 mW/cm².
   The quoted 20.00% efficiency implies Pin=89.940375 mW/cm². Confirm
   spectrum truncation and the actual power denominator; no guessed metric
   correction was applied to the figures.
4. **Recover cross-code provenance.** Plot labels name earlier DriftJax
   revisions. Match raw curves, physical inputs, norms, and baseline revisions.
   The CdS/CdTe current discrepancy is 3.6 mA/cm², not pointwise sub-percent
   agreement relative to its reported 18.2 mA/cm² current scale.
5. **Resolve timing metadata.** Both N=250 and N=500 are attached to the
   same 0.26 s result; transient mesh assignments also conflict. Supply CPU
   model, repeated trials, synchronisation, exact baseline modifications,
   stopping criteria, and the definition of each timed output.
6. **Reproduce gradients and auxiliary solvers.** Recover FD step sweeps,
   norm definitions, perturbed-solve convergence, current extraction, storage,
   frequency/time scales, and any actual higher-order/transient derivative
   support. The reported multilayer discrepancy near 4.2e-3 does not establish
   a universal relative-error bound below 1e-6.
7. **Reconcile inventory counts.** Release documents report 31 versus 204
   tests and 64 smoke tests without consistent selections. Archive the exact
   collection command; avoid presenting these as one verified suite total.
8. **Confirm author metadata.** The active file had malformed author markup,
   a star next to Benaissa, and text explicitly naming Noua as corresponding
   author. The explicit corresponding-author designation (Noua) is preserved
   consistently in all formats; confirm if Benaissa was intended instead.
   The correct spelling is "Maouadj" (verified and corrected across all documents)”.
   Patterned ORCID placeholders and unconfirmed email addresses were removed
   from the old CAS front matter. Supply confirmed identifiers/contact details.
9. **Regenerate the final figures.** Existing pixels and annotations are
   unchanged. Source-based regeneration should reconcile old version labels,
   material labels, efficiency normalisation, and convergence wording.

## Build procedure

Each entry point is built separately with `latexmk -pdf`, including BibTeX,
using the installed TeX environment. Build outputs and PDF-page previews are
kept in temporary directories outside the project. Only the final three
PDFs are copied back to their existing project filenames after validation.
No dependencies or Python virtual environments are installed.

Typesetting checks cover references, bibliography keys, figures, mathematical
glyphs, and layout. Symbolic checks cover selected newly written algebraic
identities; they are not tests of the DriftJax codebase.

## Final validation results

- `DriftJax_paper.pdf`: 15 pages, five embedded scientific figures.
- `DriftJax_CAS.pdf`: 13 pages including the highlights sheet, with the same
  five figures and the same scientific body and bibliography order.
- `DriftJax_supplementary.pdf`: 17 pages, sections S1–S9, six further figures.
- All three LaTeX and BibTeX runs complete without warnings, undefined
  citations/references, missing mathematical glyphs, or overfull/underfull boxes.
- Main and supplement have 29 and 22 unique compiled labels, respectively;
  35 and 26 bibliography entries are used from the common database.
- The complete code listing remains on one page in both article formats.
- All eleven embedded figure images match the corresponding original asset
  pixels. They appear once each across the article and supplement, with the
  article's five reused in its CAS presentation.
- Page contact sheets and enlarged CAS front matter and supplementary
  storage/AC equations were visually inspected using temporary previews.
- Symbolic checks pass for the Bernoulli derivative, both equilibrium
  Scharfetter–Gummel flux identities, both SRH density derivatives, and the
  potential-coordinate storage Jacobian.
- The final PDFs are saved under their existing project filenames.

CAS-specific compatibility adjustments preserve the class's native math
fonts, use its key-value float options, avoid a spurious zero-width keyword
box warning, and omit an empty ORCID line. No class files were modified.