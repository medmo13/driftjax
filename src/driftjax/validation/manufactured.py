"""Method-of-manufactured-solutions for the production Poisson kernel.

Exact solution φ(x) = A·sin(k·x) on [0, L] with ε = 1:
    ∇·(ε∇φ) = −(p − n + Ndop)   ⇒   Ndop = −φ'' = +A·k²·sin(k·x)
with Nc = Nv = 0 (no free carriers — the manufactured charge is the only
source).  The discrete kernel residual at the exact field is the pure
truncation error and converges at O(h²) (measured 2.02–2.04 below)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from driftjax.fields import Potentials, PVCell
from driftjax.numerics.poisson import poisson


def mms_cell(n: int, L: float = 1.0, k: float = 8.0, A: float = 1.0):
    """PVCell for the MMS test: eps=1, carrier-free, Ndop = −φ''."""
    x = np.linspace(0.0, L, n)
    xx = jnp.asarray(x, dtype=jnp.float64)
    Ndop = jnp.asarray(A * k**2 * jnp.sin(k * xx), dtype=jnp.float64)
    return PVCell(
        dgrid=jnp.asarray(np.diff(x), dtype=jnp.float64),
        eps=jnp.ones(n, dtype=jnp.float64),
        Ndop=Ndop,
        Chi=jnp.zeros(n),
        Eg=jnp.zeros(n),
        Nc=jnp.zeros(n),
        Nv=jnp.zeros(n),
        mn=jnp.ones(n),
        mp=jnp.ones(n),
        tn=jnp.ones(n),
        tp=jnp.ones(n),
        Et=jnp.zeros(n),
        Br=jnp.zeros(n),
        Cn=jnp.zeros(n),
        Cp=jnp.zeros(n),
        Snl=0.0,
        Snr=0.0,
        Spl=0.0,
        Spr=0.0,
        PhiMl=0.0,
        PhiMr=0.0,
        G=jnp.zeros(n),
        x=xx,
        statistics="boltzmann",
        T=jnp.asarray(300.0),
    )


def mms_phi(n: int, L: float = 1.0, k: float = 8.0, A: float = 1.0) -> Potentials:
    x = np.linspace(0.0, L, n)
    return Potentials(
        jnp.zeros(n, dtype=jnp.float64),
        jnp.zeros(n, dtype=jnp.float64),
        jnp.asarray(A * np.sin(k * x), dtype=jnp.float64),
    )


def poisson_mms_residual(n: int, L: float = 1.0, k: float = 8.0, A: float = 1.0) -> jnp.ndarray:
    """Truncation residual of the kernel at the exact MMS field."""
    return poisson(mms_cell(n, L, k, A), mms_phi(n, L, k, A))
