"""Design constraints (feasibility gates for the psc/optimizer loops)."""

from __future__ import annotations


def material_feasibility(m: dict) -> dict:
    """Physical sanity of a material dict; returns a constraint residual dict."""
    return {
        "nonnegative_bandgap": float(m["Eg"]) >= 0.0,
        "positive_dos": float(m["Nc"]) > 0 and float(m["Nv"]) > 0,
        "positive_mobility": float(m["mn"]) > 0 and float(m["mp"]) > 0,
        "positive_permittivity": float(m["eps"]) > 0.0,
        "positive_lifetimes": float(m["tn"]) > 0 and float(m["tp"]) > 0,
    }


def all_feasible(m: dict) -> bool:
    return all(material_feasibility(m).values())
