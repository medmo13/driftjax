# CPC Submission Plan: DriftJax v0.1.15

## Target Venue
**Computer Physics Communications (CPC)** — Program Library (CPiP) track
- Impact Factor: ~3.9
- Scope: Computational physics software
- Precedent: dPV (Mann et al., CPC Vol. 272, 2022) — DriftJax's direct predecessor

## Current State
- Manuscript: 3 PDFs (paper, CAS, supplementary), all compile clean
- Code: 81 source files, 49 test files, 22 example scripts
- Tests: 204 pass (unit/regression), 64 smoke tests
- Git: 2 commits on v0.1.15

## Gap Analysis Summary

### Critical (Must Fix)
| # | Gap | Effort |
|---|-----|--------|
| 1 | Condense abstract to ≤200 words | Small |
| 2 | Add "Running the program" subsection | Small |
| 3 | Add "Requirements" subsection | Small |
| 4 | Update Program Summary with CPC fields (DOI, Homepage) | Small |
| 5 | Archive source to Mendeley Data and obtain DOI | Medium |
| 6 | Remove duplicate text (lines 76-78) | Small |
| 7 | Rename sections to CPC conventions | Small |

### Should Fix
| # | Gap | Effort |
|---|-----|--------|
| 8 | Add quality control / testing subsection | Small |
| 9 | Use CRediT author contribution taxonomy | Small |
| 10 | Add software architecture diagram | Medium |
| 11 | Add benchmark as dedicated subsection | Medium |

### Nice to Have
| # | Gap | Effort |
|---|-----|--------|
| 12 | Include additional figures (optical, temperature) | Medium |
| 13 | Add quick-start code listing | Small |
| 14 | Add known limitations table | Small |

---

## Phase 1: Manuscript Formatting (CPC Style)

### Task 1.1: Condense Abstract
**File:** `docs/DriftJax_abstract.tex`
**Action:** Rewrite to ≤200 words. Remove caveats ("reported timing ratios", "not been isolated"). Focus on:
- What: 1D differentiable DDP solver
- How: SG fluxes, block-tridiagonal Jacobian, Block-Thomas Newton, IFT adjoint
- Verification: Jacobian, conservation, gradient checks
- Applications: sensitivity, optimization, curve fitting

### Task 1.2: Fix Duplicate Text
**File:** `docs/DriftJax_content.tex`
**Action:** Remove duplicate sentence at lines 76-78 ("Detailed derivations...")

### Task 1.3: Rename Sections to CPC Convention
**File:** `docs/DriftJax_content.tex`
**Changes:**
- §3 "Software architecture" → "Implementation"
- §5 "Design applications" → "Example applications"
- §6 "Computational performance and scope" → "Benchmarks" + separate "Limitations" subsection
- §7 "Conclusions" → "Summary"

### Task 1.4: Add "Running the Program" Subsection
**File:** `docs/DriftJax_content.tex`
**Location:** After §3 (Implementation)
**Content:**
```latex
\subsection{Running the program}
DriftJax requires Python $\geq$ 3.11 and JAX $\geq$ 0.10...

Installation:
\begin{lstlisting}
pip install driftjax
\end{lstlisting}

Minimal example:
\begin{lstlisting}
import driftjax as dj
dev = dj.Device(...)
sol = dj.simulate(dev, dj.Sweep())
print(sol.efficiency)
\end{lstlisting}
```

### Task 1.5: Add "Requirements" Subsection
**File:** `docs/DriftJax_content.tex`
**Location:** After "Running the program"
**Content:** Python version, JAX version, dependencies, OS, GPU optional

### Task 1.6: Add "Quality Control" Subsection
**File:** `docs/DriftJax_content.tex`
**Location:** In §4 (Verification), new subsection
**Content:** Test suite (204 unit + 64 smoke), CI/CD, test categories, how to run

### Task 1.7: Update Program Summary Table
**File:** `docs/DriftJax_content.tex`
**Action:** Add CPC-required fields:
- Program Files DOI (after Mendeley upload)
- Program Homepage URL
- External resources (JAX, equinox)
- Running the program reference

### Task 1.8: Use CRediT Author Contributions
**File:** `docs/DriftJax_content.tex`
**Action:** Expand to CRediT taxonomy:
- M. Benaissa: Software, Validation, Writing – original draft
- A. Noua: Conceptualization, Software, Validation, Writing – original draft, Supervision
- M. Maouadj: Supervision, Writing – review & editing

### Task 1.9: Add Acknowledgements/Funding
**File:** `docs/DriftJax_content.tex`
**Action:** Add grant numbers or institutional support

---

## Phase 2: Software Archive

### Task 2.1: Create Release Archive
**Action:** Create `driftjax-v0.1.15.tar.gz` with:
- `src/driftjax/` (all source)
- `tests/` (all tests)
- `examples/` (all examples)
- `docs/` (paper, supplementary)
- `pyproject.toml`, `README.md`, `LICENSE`
- `verification/` (all verification scripts)

### Task 2.2: Upload to Mendeley Data
**Action:**
1. Create account at data.mendeley.com
2. Create draft dataset
3. Upload tar.gz
4. Add metadata (title, description, license=MIT)
5. Save as draft (DO NOT publish)
6. Record DOI for paper

### Task 2.3: Create Zenodo Archive (Backup)
**Action:** Also archive on Zenodo for redundancy

---

## Phase 3: Final Manuscript Assembly

### Task 3.1: Create CPC-Compatible LaTeX
**File:** New `docs/DriftJax_CPC.tex`
**Action:** CPC class file (`cpc.cls`) based wrapper, or adapt existing `DriftJax_paper.tex`

### Task 3.2: Update All Cross-References
**Action:** Ensure all figure/table numbers are consistent across main and supplementary

### Task 3.3: Final PDF Build
**Action:** pdflatex → bibtex → pdflatex → pdflatex for all documents

### Task 3.4: Extract Figures for Submission
**Action:** High-resolution PNG/PDF for each figure

---

## Phase 4: Submission Package

### Task 4.1: Write Cover Letter
**Content:**
- Summary of contribution
- How it differs from dPV
- Key verification results
- Why CPC is the right venue

### Task 4.2: Prepare Submission Checklist
- [ ] Manuscript in LaTeX
- [ ] Figures as separate files
- [ ] Program Summary completed
- [ ] Software archived on Mendeley Data (draft)
- [ ] Open-source license (MIT) in code
- [ ] Availability Statement in manuscript
- [ ] Cover letter
- [ ] Suggested reviewers (optional)

### Task 4.3: Submit via Editorial Manager
**URL:** https://www.editorialmanager.com/COMPHY

---

## Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| CPC class file incompatible with modern LaTeX | Medium | Use article class with CPC formatting guidelines |
| Mendeley Data upload fails | Low | Zenodo as backup |
| Reviewers request additional experiments | Medium | Verification scripts ready to rerun |
| Abstract exceeds 200 words | Low | Edit aggressively |
| Missing CPC-specific sections | Medium | Follow dPV paper structure exactly |

---

## Timeline

| Phase | Duration | Dependencies |
|-------|----------|-------------|
| Phase 1: Manuscript formatting | 1-2 days | None |
| Phase 2: Software archive | 1 day | Phase 1 complete |
| Phase 3: Final assembly | 1 day | Phase 1+2 complete |
| Phase 4: Submission | 1 day | Phase 3 complete |
| **Total** | **4-5 days** | |

---

## Decision Points

1. **CPC vs JOSS?** → CPC (higher impact, proven fit for this type)
2. **Single-column or double-column?** → Single-column for review
3. **Include supplementary in main or separate?** → Separate (CPC allows)
4. **Which figures in main vs supplement?** → 5 in main, 6 in supplement
