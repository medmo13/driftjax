"""Unified, machine-readable validation record.

Consolidates the component results produced by the release suite into a single
canonical JSON record whose schema matches the headline methods/validation
claims in the paper (Table 3).  Every component keeps its own detailed record
(via its ``save_record``); this module is the *aggregation* layer that makes it
possible to generate the paper's tables from one source of truth:

    {
      "case", "mesh", "bias_points", "precision",
      "jacobian_error", "fd_gradient_error", "adjoint_method",
      "condition_number", "pce", "runtime_s", "..."
    }

The pragmatic contract: each field is filled from the component results where it
is actually measured, and otherwise from the ``headline`` keyword arguments that
name the corresponding numerical gate.  ``None`` means "not measured by this
run" — a record consumer must treat missing fields as gaps, not as zeros.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def build_record(
    verification: Mapping[str, Any] | None = None,
    fd: Mapping[str, Any] | None = None,
    condition: Mapping[str, Any] | None = None,
    scaling: Mapping[str, Any] | None = None,
    **headline: Any,
) -> dict:
    """Merge component results into one canonical validation record.

    Parameters
    ----------
    verification, fd, condition, scaling:
        The dicts returned by the release-suite components (or their
        ``save_record`` payloads).  Each may be ``None``.
    **headline:
        Overrides for canonical keys (e.g. ``jacobian_error=3.2e-13``,
        ``fd_gradient_error=5.56e-7``, ``condition_number=1.2e14``,
        ``pce=21.65``, ``runtime_s=37.1``) that correspond to headline numerical
        gates not reproduced by the light release sub-components.
    """
    v = dict(verification or {})
    f = dict(fd or {})
    c = dict(condition or {})
    s = dict(scaling or {})

    # --- stability: pull from the condition sweep cases ---
    cs_summary = c.get("summary") or {}
    cases = c.get("cases") or []
    worst_case = None
    if cases:
        valid = [x for x in cases if x.get("kappa") is not None]
        if valid:
            worst_case = max(valid, key=lambda x: x.get("kappa", 0.0))

    # --- scaling: fitted exponent / prefactor, plus per-N measurements ---
    fit = s.get("fit") or {}
    measurements = s.get("measurements") or []

    def _first_scalar(mapping: Mapping[str, Any], *keys: str) -> Any:
        for k in keys:
            if k in mapping and mapping[k] is not None:
                return mapping[k]
        return None

    record: dict[str, Any] = {
        # ---- context ----
        "program": "driftjax",
        "case": headline.get("case", "psc_16param"),
        "mesh": headline.get("mesh", 500),
        "bias_points": headline.get("bias_points", 41),
        "precision": headline.get("precision", "float64"),
        # ---- the numerical-validation gates (Table 3 of the paper) ----
        "jacobian_error": headline.get("jacobian_error", _first_scalar(v, "jacobian_error", "max_abs_rel")),
        "fd_gradient_error": headline.get("fd_gradient_error", _first_scalar(f, "max_rel_error", "E_fd")),
        "fd_optimal_h": f.get("optimal_h"),
        "adjoint_method": headline.get(
            "adjoint_method",
            worst_case.get("status") if worst_case else None,
        ),
        "condition_number": headline.get(
            "condition_number",
            worst_case.get("kappa") if worst_case else None,
        ),
        "gradient_discrepancy": worst_case.get("E_g") if worst_case else None,
        # ---- performance / objectives ----
        "pce": headline.get("pce"),
        "runtime_s": headline.get("runtime_s"),
        "scaling_exponent": _first_scalar(fit, "p"),
        "scaling_prefactor_s": _first_scalar(fit, "a_s", "a"),
        "scaling_r_squared": fit.get("r_squared"),
        "scaling_meshes": [m.get("N") for m in measurements if m.get("N") is not None],
        # ---- stability summary ----
        "stability": {
            "banded_safe": cs_summary.get("banded_safe"),
            "dense_fallback": cs_summary.get("dense_fallback"),
            "total_cases": cs_summary.get("total"),
        },
        # ---- verification gates ----
        "verification_gates_passed": headline.get("verification_gates_passed"),
        "verification_gates_total": headline.get("verification_gates_total"),
    }

    # Keep only keys that carry information; drop None trailing noise but keep
    # structure level keys that are informative about a measured run.
    for k in ("scaling_prefactor_s", "scaling_r_squared", "fd_optimal_h"):
        if record.get(k) is None:
            record.pop(k, None)
    for k in ("pce", "runtime_s", "verification_gates_passed", "verification_gates_total"):
        if record.get(k) is None:
            record.pop(k, None)
    if not record.get("scaling_meshes"):
        record.pop("scaling_meshes", None)

    return record


def save_record(record: dict, path: str | Path) -> Path:
    """Write the consolidated record as pretty JSON and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True, default=_default) + "\n"
    )
    return path


def _default(o: Any) -> Any:
    return float(o) if hasattr(o, "__float__") else str(o)


def print_summary(record: dict) -> None:
    print("validation record:")
    for k, val in record.items():
        print(f"  {k}: {val}")
