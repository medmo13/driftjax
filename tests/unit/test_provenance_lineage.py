"""Per-module provenance lineage table (component -> method -> test gate)."""

import pytest

from driftjax.runtime.provenance import LINEAGE, lineage_markdown, register_lineage, report_lineage

pytestmark = pytest.mark.smoke  # no solves: registry bookkeeping, milliseconds


def test_lineage_seeded():
    rows = report_lineage()
    assert len(rows) >= 14
    comps = {r["component"] for r in rows}
    for expected in (
        "checkpointing",
        "fused_kernels",
        "old_parity",
        "analytic_jacobian",
        "files_pad_fixes",
        "lit_scaps",
    ):
        assert expected in comps


def test_lineage_fields():
    for r in report_lineage():
        assert r["component"] and r["source"] and r["basis"] and r["test_gate"]
        # every gate must point at a real test path in this repository
        assert r["test_gate"].startswith("tests/"), r


def test_register_and_markdown():
    register_lineage("demo_module", "demo/codebase", "demo basis", "tests/unit/test_demo.py")
    md = lineage_markdown()
    assert "| component | source | basis | test gate |" in md
    assert "demo_module" in md
    rows = report_lineage()
    assert any(r["component"] == "demo_module" and r["source"] == "demo/codebase" for r in rows)
    del LINEAGE["demo_module"]


def test_seed_gates_exist():
    """Every seeded provenance record must cite an on-disk test module."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for r in report_lineage():
        gate = Path(root) / r["test_gate"].split(" (")[0]
        assert gate.exists(), f"stale gate path: {r['test_gate']}"
