"""Mesh-convergence measurement helpers."""

from __future__ import annotations

import numpy as np


def convergence_order(errors, ratios):
    """p from e(h) = C·h^p across two refinements: p = log(e1/e2)/log(r)."""
    p = []
    for e1, e2, r in zip(errors[:-1], errors[1:], ratios, strict=False):
        p.append(float(np.log(e1 / e2) / np.log(r)))
    return np.array(p)
