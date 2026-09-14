"""Shared leaf utilities: tracer detection + canonical solver-stats dict.

Single home for two pieces of logic that were copy-pasted across modules
(``newton`` / ``ptc`` / ``linalg`` / ``mixed_precision`` each carried its own
``_is_tracer``; Newton / line-search / PTC / step-halving each returned a
different stats-dict shape). Leaf module — imports only ``jax`` and stdlib,
so every consumer can import it without cycles.
"""

from __future__ import annotations

from typing import Any, TypedDict

import jax


def is_tracer(x) -> bool:
    """True if ``x`` (or any leaf of it) is a JAX tracer.

    ``design`` is a pytree (``DeviceDesign``) whose *leaves* may be tracers
    under ``jit``/``vmap``/``grad`` while the container itself is not a
    ``Tracer`` subclass, so a naive ``type(x).__name__`` check would miss it.
    """
    if isinstance(x, jax.core.Tracer):
        return True
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(x))


class NewtonStats(TypedDict, total=False):
    """Canonical solver-stats keys (H2: every solver path populates these).

    Conventions (L8): ``resid`` is ‖F‖₂, ``resid_f`` is max|F|, ``error`` is
    the step norm (max|Δx|); ``converged`` is the solver's own flag (never
    recomputed downstream); ``stagnated`` marks best-iterate/rebound exits;
    ``fallback`` names the recovery path (None on the direct path). Solver
    extras (``rejects``/``backtracks``/``dt``/``stall``/…) ride alongside.
    """

    backend: str | None
    iters: int | None
    error: Any
    resid: Any
    resid_f: Any
    converged: bool
    stagnated: bool
    fallback: str | None


def newton_stats(**overrides: Any) -> dict:
    """Build a stats dict with all canonical keys present (None/False defaults)."""
    stats: dict = {
        "backend": None,
        "iters": None,
        "error": None,
        "resid": None,
        "resid_f": None,
        "converged": False,
        "stagnated": False,
        "fallback": None,
    }
    stats.update(overrides)
    return stats
