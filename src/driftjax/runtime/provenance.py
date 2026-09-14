"""Provenance: environment + commit + config fingerprint for reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from importlib.metadata import version

import jax

try:
    from datetime import UTC  # Python 3.11+
except ImportError:  # Python 3.10 fallback (AUDIT: was `UTC = UTC` → NameError)
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 — datetime.UTC does not exist on 3.10; ruff's autofix would reintroduce the NameError


def fingerprint() -> str:
    """Config hash — same inputs ⇒ same fingerprint ⇒ bit-comparable runs."""
    h = hashlib.sha256()
    for k, v in sorted(jax.config.values.items()):
        h.update(f"{k}={v}".encode())
    return h.hexdigest()[:16]


def record(extra: dict | None = None) -> dict:
    """Full provenance record (json-serializable)."""
    rec = {
        "package": "driftjax",
        "version": version("driftjax"),
        "jax": jax.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "config_fingerprint": fingerprint(),
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "env": dict(_env_of_interest()),
    }
    if extra:
        rec.update(extra)
    return rec


def _env_of_interest():
    for k in (
        "DRIFTJAX_OPTIMISTIX",
        "DRIFTJAX_BANDED",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "XLA_FLAGS",
    ):
        v = __import__("os").environ.get(k)
        if v is not None:
            yield k, v


def save(path: str, rec: dict | None = None) -> str:
    with open(path, "w") as f:
        json.dump(rec or record(), f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Per-module provenance: component -> method -> executable test gate.
# ---------------------------------------------------------------------------
# Every core component carries an auditable record of *what* it implements,
# *which numerical method* it relies on, and *which test* pins it.
# `report_lineage()` renders the table used in docs and release notes.

LINEAGE: dict[str, dict] = {}


def register_lineage(component: str, source: str, basis: str, test_gate: str) -> None:
    """Register (or overwrite) the provenance record of one component."""
    LINEAGE[component] = {
        "component": component,
        "source": source,
        "basis": basis,
        "test_gate": test_gate,
    }


def report_lineage() -> list[dict]:
    """The component->method->gate table (sorted by component)."""
    _seed_defaults()
    return [LINEAGE[k] for k in sorted(LINEAGE)]


def lineage_markdown() -> str:
    """Markdown table of report_lineage() for docs/reports."""
    rows = report_lineage()
    out = ["| component | source | basis | test gate |", "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['component']} | {r['source']} | {r['basis']} | {r['test_gate']} |")
    return "\n".join(out)


def _seed_defaults() -> None:
    """Seed the core lineage (idempotent)."""
    defaults = {
        "analytic_jacobian": (
            "DriftJax",
            "analytic banded Jacobian, == jacrev ~1e-12",
            "tests/unit/test_analytic_jacobian.py",
        ),
        "banded_solve": (
            "DriftJax",
            "pivoted banded solve (LAPACK dgbsv)",
            "tests/unit/test_banded_solve.py",
        ),
        "checkpointing": (
            "DriftJax",
            "jax.checkpoint solve-chain wrapper",
            "tests/unit/test_checkpointing.py",
        ),
        "fused_kernels": (
            "DriftJax",
            "bernoulli+current+residual fusion",
            "tests/unit/test_fused_kernels.py",
        ),
        "mode_matrix": (
            "DriftJax",
            "explicit statistics x optics x solver dispatch",
            "tests/unit/test_materials.py (smoke)",
        ),
        "old_parity": (
            "reference implementation",
            "published goldens at roundoff parity",
            "tests/regression/test_iv_reference.py",
        ),
        "files_pad_fixes": (
            "DriftJax audit",
            "7 verified bug fixes",
            "tests/regression/test_files_pad_mechanisms.py",
        ),
        "lit_scaps": (
            "SCAPS-1D (Burgelman et al., 2000)",
            "SCAPS CdTe literature anchor",
            "tests/literature/test_scaps_cdte.py",
        ),
        "exact_fd": (
            "DriftJax",
            "128-pt Gauss-Legendre + Sommerfeld",
            "tests/unit/test_statistics.py",
        ),
        "tmm": (
            "DriftJax",
            "coherent 2x2 transfer-matrix optics",
            "tests/property/test_invariants.py",
        ),
        "ptc": (
            "DriftJax",
            "pseudo-transient continuation",
            "tests/unit/test_globalization.py",
        ),
        "transient_ac": (
            "DriftJax",
            "backward-Euler + small-signal AC",
            "tests/property/test_invariants.py",
        ),
        "adjoint": (
            "IFT (implicit function theorem)",
            "differentiable-design gradients via custom_vjp",
            "tests/gradient/test_adjoint_ab.py",
        ),
        "kg_provenance": (
            "DriftJax",
            "per-module lineage table",
            "tests/unit/test_provenance_lineage.py",
        ),
    }
    for comp, (src, basis, gate) in defaults.items():
        if comp not in LINEAGE:
            LINEAGE[comp] = {"component": comp, "source": src, "basis": basis, "test_gate": gate}
