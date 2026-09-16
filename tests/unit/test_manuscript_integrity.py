"""Manuscript/release integrity gate (referee Step 30).

Unresolved LaTeX references (`??`) or citations (`[?`) in a submitted
PDF are an automatic submission blocker. This gate scans the paper
sources so the failure surfaces in CI, not in front of a referee.
Also pins the version-string sync (pyproject/__init__/CITATION).
"""

from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent.parent / "docs" / "paper"
TEX_SOURCES = [
    "DriftJax_CAS.tex",
    "DriftJax_paper.tex",
    "DriftJax_supplementary.tex",
    "DriftJax_content.tex",
    "DriftJax_abstract.tex",
    "DriftJax_shared.tex",
    "provenance_table.tex",
]


def test_no_unresolved_placeholders():
    bad = []
    for name in TEX_SOURCES:
        text = (PAPER / name).read_text()
        for i, line in enumerate(text.splitlines(), 1):
            s = line.split("%", 1)[0]  # ignore LaTeX comments
            if "??" in s or "[?" in s:
                bad.append(f"{name}:{i}: {s.strip()[:100]}")
    assert not bad, "unresolved LaTeX placeholders:\n" + "\n".join(bad)


def test_version_strings_in_sync():
    import re

    root = PAPER.parent.parent
    pyproj = (root / "pyproject.toml").read_text()
    init = (root / "src" / "driftjax" / "__init__.py").read_text()
    cff = (root / "CITATION.cff").read_text()
    v_py = re.search(r'version\s*=\s*"([0-9.]+)"', pyproj).group(1)
    v_init = re.search(r'__version__\s*=\s*"([0-9.]+)"', init).group(1)
    v_cff = re.search(r"^version:\s*([0-9.]+)", cff, re.M).group(1)
    assert v_py == v_init == v_cff, (v_py, v_init, v_cff)
