# Paper Compilation Instructions

## Prerequisites
Install TeX Live (tested with TeX Live 2024):
```bash
# Ubuntu/Debian:
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended

# Or using tlmgr:
tlmgr install collection-fontsrecommended collection-latexrecommended collection-latexextra

# macOS with MacTeX:
# Download from https://www.tug.org/mactex/
```

## Compilation Steps

```bash
cd /home/med/Desktop/final/open14/driftjax_v0.1.17/docs/paper

# Compile main paper (3 passes for references)
pdflatex DriftJax_paper.tex
bibtex DriftJax_paper  # Process bibliography
pdflatex DriftJax_paper.tex  # Rerun for citation formatting
pdflatex DriftJax_paper.tex  # Rerun for final cross-references

# Compile supplementary material
pdflatex DriftJax_supplementary.tex
bibtex DriftJax_supplementary
pdflatex DriftJax_supplementary.tex
pdflatex DriftJax_supplementary.tex
```

## Output Files
- `DriftJax_paper.pdf` - Main article
- `DriftJax_supplementary.pdf` - Supplementary sections S1-S9

## Dependencies
- `cas-refs.bib` - Bibliography file (includes Selberherr 1984, Gummel 1964, DEVSIM 2023)
- `DriftJax_shared.tex` - Shared notation and commands
- `DriftJax_abstract.tex` - Common abstract
- `DriftJax_content.tex` - Main article body (updated with automatic backend switching)
- Figures: `docs/paper/records/*.png`

## Troubleshooting
If you get "command not found" errors for `pdflatex`, ensure TeX Live is properly installed
and its bin directory is in your PATH.
