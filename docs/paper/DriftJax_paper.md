# DriftJax manuscript sources

**Title:** DriftJax: A Structure-Aware Differentiable Drift–Diffusion Solver
for Photovoltaic Device Design

The publication package uses one canonical scientific body instead of
independently maintained Markdown, standard-LaTeX, and CAS manuscripts.

## Edit and compile

- Edit the scientific article in `DriftJax_content.tex`.
- Edit the common abstract in `DriftJax_abstract.tex`.
- Compile `DriftJax_paper.tex` for the standard article.
- Compile `DriftJax_CAS.tex` for the Elsevier CAS version with highlights.
- Edit and compile `DriftJax_supplementary.tex` for Supplementary Sections S1–S9.
- Maintain references in `cas-refs.bib` and shared notation in `DriftJax_shared.tex`.

## Scientific organisation

1. Introduction: design sensitivities and the structured-solver contribution.
2. Model and numerical formulation: transport, block Newton, and implicit adjoint.
3. Software architecture: device-to-objective interface and execution scope.
4. Numerical verification: kernels, residuals, derivatives, and cross-code comparison.
5. Design applications: sensitivity analysis, bounded co-design, and synthetic recovery.
6. Computational performance and scope: cost model and qualified timing reports.
7. Conclusions and publication declarations.

The structure is informed by the uploaded reference paper
`prism-uploads/2105.06305v3.pdf`; its prose and numerical results are not copied.
The supplement contains technical derivations, six additional existing
figures, and explicit reproduction requirements.

## Evidence status

The implementation description follows the supplied v0.1.15 release
documentation. The workspace does not contain executable DriftJax source
or raw benchmark logs. Existing numerical results are retained with their
limitations; they were not rerun or independently certified. See
`MANUSCRIPT_AUDIT.md` for source alignment, author-metadata questions, and
the remaining checks required before submission.
