"""Optimizers: scipy-SLSQP with JAX gradients, preconditioned multi-start,
Nelder-Mead (derivative-free), optax Adam fallback (package-validated pattern)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import scipy.optimize


def slsqp(f, x0, bounds=None, maxiter: int = 50, jac=None, **kw):
    """SLSQP on x ↦ (f(x), jac(x)); f may return (value, grad) tuple.

    Design-failure guard: if f returns a non-finite value (the physics solve
    failed for that probe design), the probe is penalised with a large value
    and a zero gradient so SLSQP's line search rejects it instead of
    propagating NaN (observed in the psc/holistic/ex3 optimizer loops).
    """

    # AUDIT R3: an explicit `jac=True` requests SciPy's tuple protocol, same
    # as None; any other callable is forwarded (previously discarded).
    _use_tuple = jac is None or jac is True
    # L4: SciPy calls fun(x) then jac(x) at the same probe — cache the last
    # objective value so the guarded Jacobian reuses it instead of running
    # a second full physics solve per iteration.
    _cache: dict = {"x": None, "val": None}

    def _as_x(x):
        return np.asarray(x, dtype=np.float64)

    def fun(x):
        xa = _as_x(x)
        out = f(xa)
        if isinstance(out, tuple):
            val, grad = float(out[0]), np.asarray(out[1], dtype=np.float64)
            if not np.isfinite(val) or not np.all(np.isfinite(grad)):
                # H3: under the tuple protocol (default) the penalty must
                # stay a (value, grad) pair — a bare scalar breaks SciPy's
                # jac=True unpacking. Non-finite grads are zeroed likewise.
                _cache["x"], _cache["val"] = xa, 1e10
                return (1e10, np.zeros_like(xa)) if _use_tuple else 1e10
            _cache["x"], _cache["val"] = xa, val
            return (val, grad) if _use_tuple else val
        val = float(out)
        _cache["x"], _cache["val"] = xa, val
        if not np.isfinite(val):
            return (1e10, np.zeros_like(xa)) if _use_tuple else 1e10
        return val

    def _guarded_jac(x):
        # An explicit Jacobian is unguarded by `fun`: at a penalised
        # (non-finite) probe it could still return NaN/inf grad and abort
        # SLSQP, so sanitise it against the objective's finiteness.
        xa = _as_x(x)
        try:
            g = np.asarray(jac(xa), dtype=np.float64)
        except Exception:
            return np.zeros_like(xa)
        if g.shape != xa.shape or not np.all(np.isfinite(g)):
            return np.zeros_like(xa)
        if _cache["x"] is not None and np.array_equal(_cache["x"], xa):
            val = _cache["val"]
        else:
            try:
                out = f(xa)
                val = float(out[0]) if isinstance(out, tuple) else float(out)
            except Exception:
                return np.zeros_like(xa)
        if not np.isfinite(val):
            return np.zeros_like(xa)
        return g

    # AUDIT: an explicitly passed jac callable was discarded
    # (jac=<callable> is not None → False → SciPy finite-differences it).
    # None/True keep the (value, grad)-tuple convention (jac=True).
    return scipy.optimize.minimize(
        fun,
        np.asarray(x0, dtype=np.float64),
        method="SLSQP",
        jac=True if _use_tuple else _guarded_jac,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10},
        **kw,
    )


def _value_grad(f, x):
    """Unpack a (value, grad) objective with validation (L5).

    Returns (val, grad) with non-finite entries zeroed-safe: a failed
    physics probe yields (inf, zeros) so the preconditioner/Adam treat it
    as "no information" instead of propagating NaN (matches slsqp's
    penalise-don't-propagate contract).
    """
    xa = np.asarray(x, dtype=np.float64)
    try:
        out = f(xa)
    except Exception:
        return float("inf"), np.zeros_like(xa)
    if isinstance(out, tuple):
        try:
            val, grad = float(out[0]), np.asarray(out[1], dtype=np.float64)
        except Exception:
            return float("inf"), np.zeros_like(xa)
    else:
        try:
            val = float(out)
        except Exception:
            return float("inf"), np.zeros_like(xa)
        grad = np.zeros_like(xa)
    if not np.isfinite(val) or grad.shape != xa.shape or not np.all(np.isfinite(grad)):
        return float("inf"), np.zeros_like(xa)
    return val, grad


def gradient_scale(f, points):
    """Per-coordinate scale from |∇f| at sample points (preconditioner)."""
    gs = []
    for p in points:
        _, g = _value_grad(f, p)
        gs.append(np.abs(np.asarray(g)))
    G = np.max(np.stack(gs), axis=0)
    return np.where(G > 1e-12, G, 1.0)


def diagonal_scale(gs_max):
    """Scaling map x ↦ x/s from per-coordinate max-|grad|."""
    s = np.where(gs_max > 1e-12, gs_max, 1.0)
    return s


def slsqp_multistart(
    f, starts, bounds=None, maxiter: int = 15, precond: bool = True, gate_tol: float = 1e-4
):
    """Best-of multi-start SLSQP with diagonal preconditioning.

    Preconditioning: measure |∇f| at the starts and optimize y = x/s
    (s = diag scaling) — the DDP objective is badly scaled across
    (mobility, doping, band params)."""
    starts = [np.asarray(s, dtype=np.float64) for s in starts]
    out = None
    if precond:
        s = diagonal_scale(gradient_scale(f, starts))

        def f_y(y):
            val, grad = _value_grad(f, s * np.asarray(y))
            return val, grad * s

        for x0 in starts:
            r = slsqp(
                f_y,
                x0 / s,
                bounds=None
                if bounds is None
                else [(b[0] / si, b[1] / si) for b, si in zip(bounds, s, strict=False)],
                maxiter=maxiter,
            )
            r.x = s * np.asarray(r.x)  # map back
            if out is None or r.fun < out.fun:
                out = r
    else:
        for x0 in starts:
            r = slsqp(f, x0, bounds=bounds, maxiter=maxiter)
            if out is None or r.fun < out.fun:
                out = r
    return out


def adam(f, x0, steps: int = 200, lr: float = 1e-2, gtol: float = 1e-8):
    """Adam minimizer on (value, grad) objectives (optax fallback path)."""
    import optax

    opt = optax.adam(lr)
    x = jnp.asarray(x0, dtype=jnp.float64)
    state = opt.init(x)
    val = float("inf")
    for _ in range(steps):
        # L5: validated unpack — a failed probe holds position (zero grad)
        # instead of poisoning the optimizer state with NaN.
        val, g = _value_grad(f, np.asarray(x))
        updates, state = opt.update(jnp.asarray(g), state, x)
        x = x + updates
        if float(jnp.max(jnp.abs(jnp.asarray(g)))) < gtol:
            break
    return x, float(val)


def nelder_mead(f, x0, bounds=None, maxiter: int = 200, xatol: float = 1e-8,
                fatol: float = 1e-15, adaptive: bool = True):
    """Nelder-Mead simplex minimizer (derivative-free).

    Unlike SLSQP/L-BFGS-B, Nelder-Mead uses no line search and no Hessian
    approximation — it maintains a simplex (triangle in 2D) that naturally
    shrinks to fit narrow valleys.  This makes it robust on objectives with
    razor-thin minima where gradient-based line searches overshoot.

    ``f`` may return a scalar or a (value, grad) tuple; the gradient is
    ignored (only the value is used).  Non-finite values are penalised to
    1e10 to prevent NaN propagation.

    Bounds are enforced by clipping evaluated points; points outside bounds
    receive a penalty value of 1e10.
    """
    lb = np.array([b[0] for b in bounds]) if bounds else None
    ub = np.array([b[1] for b in bounds]) if bounds else None

    def _safe_f(x):
        xa = np.asarray(x, dtype=np.float64)
        if lb is not None and np.any(xa < lb):
            return 1e10
        if ub is not None and np.any(xa > ub):
            return 1e10
        out = f(xa)
        val = float(out[0]) if isinstance(out, tuple) else float(out)
        return val if np.isfinite(val) else 1e10

    return scipy.optimize.minimize(
        _safe_f,
        np.asarray(x0, dtype=np.float64),
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": xatol, "fatol": fatol,
                 "adaptive": adaptive},
    )
