"""Roofline / cost model for the driftjax solve pipeline (NEW).

Turns the solver structure into FLOP and byte-traffic estimates so kernel
and hardware decisions are made from a model, not vibes.  All quantities
are per Newton iteration unless stated.

Derivation
----------
* residual F:      per node ~ O(1) physics ops × 3 unknowns
* Jacobian J:      jacrev of the residual — flops ≈ c_jac · flops_F (forward
                   AD overhead ~2–3×), banded width W = 13
* linear solve:    Block-Thomas = 2·(N·3³) solves + (3N)·(2·3²) matvecs ⇒
                   ≈ 54N + 54N = O(108N) flops; CSR spsolve similar O(N·W²)
* memory traffic:  J is (3N)² but stored banded ⇒ 13·(3N)·8 bytes; the
                   dense path would move (3N)²·8 — the roofline position of
                   each backend is computed below.
"""

from __future__ import annotations

_W = 13  # banded width of the interleaved DDP Jacobian


def newton_flop_model(n: int, jac_factor: float = 3.0) -> dict:
    """Per-iteration FLOP estimate for the n-node DDP Newton solve."""
    ndof = 3 * n
    f_resid = 60.0 * ndof  # physics residual / node
    f_jac = jac_factor * f_resid  # forward-AD Jacobian
    f_linear = 108.0 * n  # block-Thomas exact O(N)
    return {
        "residual": f_resid,
        "jacobian": f_jac,
        "linear": f_linear,
        "total": f_resid + f_jac + f_linear,
    }


def memory_model(n: int, precision_bytes: int = 8) -> dict:
    """Bytes moved per Newton iteration for each backend."""
    ndof = 3 * n
    dense_J = ndof * ndof * precision_bytes
    banded_J = _W * ndof * precision_bytes
    residual = 3 * ndof * precision_bytes
    return {
        "dense_J": dense_J,
        "banded_J": banded_J,
        "residual": residual,
        "dense_total": dense_J + residual,
        "banded_total": banded_J + residual,
    }


def arithmetic_intensity(n: int, backend: str = "banded", precision_bytes: int = 8) -> float:
    """FLOPs/byte for the linear solve of the given backend."""
    fl = newton_flop_model(n)["linear"]
    mem = memory_model(n, precision_bytes)
    return fl / (mem["banded_J"] if backend == "banded" else mem["dense_J"])


def report(n: int) -> dict:
    return {
        "n": n,
        "flops_per_newton_iter": newton_flop_model(n),
        "bytes_per_newton_iter": memory_model(n),
        "arithmetic_intensity": {
            "banded": arithmetic_intensity(n, "banded"),
            "dense": arithmetic_intensity(n, "dense"),
        },
        "position": "memory-bound on CPU/GPU (AI ~ O(1)); dense J would be the "
        "only compute-bound route and it is O(n²) memory — rejected",
    }
