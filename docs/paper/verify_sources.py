#!/usr/bin/env python
"""Verify all paper source files are in place and consistent."""
import os

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
paper_dir = os.path.join(base, "docs", "paper")

required_files = [
    "DriftJax_paper.tex",
    "DriftJax_content.tex",
    "DriftJax_abstract.tex",
    "DriftJax_shared.tex",
    "DriftJax_supplementary.tex",
    "cas-refs.bib",
    "paper.template.tex",
    "BUILD_INSTRUCTIONS.md",
]

print("=== Paper Source Verification ===")
for fname in required_files:
    fpath = os.path.join(paper_dir, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f"  OK: {fname} ({size:,} bytes)")
    else:
        print(f"  MISSING: {fname}")

# Check bib file has our new references
with open(os.path.join(paper_dir, "cas-refs.bib")) as f:
    bib = f.read()
new_refs = ["selberherr1984", "gummel1964", "rodrigue2023", "selberherr1987hot"]
print("\n=== New References Added ===")
for ref in new_refs:
    if ref in bib:
        print(f"  OK: {ref}")
    else:
        print(f"  MISSING: {ref}")

# Check content file mentions native GE fallback
with open(os.path.join(paper_dir, "DriftJax_content.tex")) as f:
    content = f.read()
keywords = ["native_ge", "LAPACK fallback", "automatic backend", "ill-conditioned"]
print("\n=== Content Coverage ===")
for kw in keywords:
    count = content.lower().count(kw.lower())
    status = "OK" if count > 0 else "MISSING"
    print(f"  {status}: '{kw}' ({count} mentions)")
