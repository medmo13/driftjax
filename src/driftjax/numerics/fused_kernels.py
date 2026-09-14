"""Fused physics kernels — one compiled program per physics op.

comp_F/Jn/Jp are pure-python assemblers; each Newton step would
otherwise dispatch several small JAX programs and shuttle intermediates
through the host.  The fused variants compile the whole assembly chain into
a single XLA program; XLA then fuses the elementwise graphs internally
(3-4x less global-memory traffic on typical device stacks).

Correctness contract (pinned, tests/unit/test_fused_kernels.py): every fused
kernel == its unfused reference to 1e-12 on p-n, hetero, equilibrium and
biased states.  When in doubt, `use_fused=False` (identical results).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from driftjax.fields import BoundaryConditions, Potentials, PVCell
from driftjax.numerics.drift_diffusion import ddn as _ddn
from driftjax.numerics.drift_diffusion import ddp as _ddp
from driftjax.numerics.poisson import poisson as _poisson
from driftjax.numerics.scharfetter_gummel import Jn as _Jn
from driftjax.numerics.scharfetter_gummel import Jp as _Jp
from driftjax.science.contacts import contact_phi, contact_phin, contact_phip


@partial(jax.jit, static_argnames=())
def fused_bernoulli_current(cell: PVCell, pot: Potentials):
    """Single compiled pass for (Jn, Jp) — identical to the SG kernels."""
    return _Jn(cell, pot), _Jp(cell, pot)


@partial(jax.jit, static_argnames=())
def fused_residual(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> jax.Array:
    """Full interleaved 3n residual, computed in one compiled program.

    Layout matches `numerics.residual.comp_F` exactly (interleaved
    [ddn, ddp, pois] interior + Dirichlet contact rows at both ends).
    """
    n = pot.n
    dd_n = _ddn(cell, pot)
    dd_p = _ddp(cell, pot)
    pois = _poisson(cell, pot)
    ct_phin0, ct_phinL = contact_phin(cell, bound, pot)
    ct_phip0, ct_phipL = contact_phip(cell, bound, pot)
    ct_phi0, ct_phiL = contact_phi(cell, bound, pot)
    return jnp.concatenate(
        [
            jnp.expand_dims(ct_phin0, 0),
            jnp.expand_dims(ct_phip0, 0),
            jnp.expand_dims(ct_phi0, 0),
            jnp.stack([dd_n, dd_p, pois], axis=1).reshape(3 * (n - 2)),
            jnp.expand_dims(ct_phinL, 0),
            jnp.expand_dims(ct_phipL, 0),
            jnp.expand_dims(ct_phiL, 0),
        ]
    )


@partial(jax.jit, static_argnames=("alpha_mode", "statistics"))
def _fused_gen_impl(design, ls, alpha_mode, statistics):
    from driftjax.simulator import init_cell

    return init_cell(design, ls, alpha_mode=alpha_mode, statistics=statistics).G


def fused_generation(
    design, ls, alpha_mode: str = "table", statistics: str = "boltzmann"
) -> jax.Array:
    """Generation profile (n,) as one compiled program (== init_cell().G)."""
    return _fused_gen_impl(design, ls, alpha_mode, statistics)


@partial(jax.jit, static_argnames=())
def fused_jacobian_banded(cell: PVCell, bound: BoundaryConditions, pot: Potentials) -> tuple:
    """(A, B, C) block-tridiagonal Jacobian — the pinned analytic backend.

    JIT-compiled so the Jacobian is one XLA program (was alias before).
    Equals banded_jacobian to ~1e-12; see tests/unit/test_fused_kernels.py.
    """
    from driftjax.numerics.analytic_jacobian import banded_jacobian

    if getattr(cell, "statistics", "boltzmann") != "boltzmann":
        raise ValueError(
            f"fused_jacobian_banded is Boltzmann-only; got {getattr(cell, 'statistics', None)!r}. "
            "Use FusedKernelManager(use_fused=False) (dense jacfwd reference)."
        )
    return banded_jacobian(cell, bound, pot)


@partial(jax.jit, static_argnames=())
def fused_residual_and_jacobian(
    cell: PVCell, bound: BoundaryConditions, pot: Potentials
) -> tuple[jax.Array, tuple, tuple, tuple]:
    """Deep-fused residual + banded Jacobian sharing n/p/ni explicitly.

    Computes carrier stats (n, p, ni) once and passes them to both the
    residual and Jacobian paths — no reliance on XLA CSE for sharing.
    """
    from driftjax.numerics.residual import comp_F_precomputed
    from driftjax.numerics.analytic_jacobian import banded_jacobian

    F, n_v, p_v, ni_v = comp_F_precomputed(cell, bound, pot)
    A, B, C = banded_jacobian(cell, bound, pot, n_v=n_v, p_v=p_v, ni_v=ni_v)
    return F, A, B, C


class FusedKernelManager:
    """Interface to fused kernels with exact fallback to the unfused path.

    use_fused=True (default) selects the compiled kernels; False routes to
    the reference implementations (comp_F / F_jacobian / init_cell.G).
    """

    def __init__(self, use_fused: bool = True):
        self.use_fused = use_fused

    def compute_residual(self, cell, bound, pot):
        if self.use_fused:
            return fused_residual(cell, bound, pot)
        from driftjax.numerics.residual import comp_F

        return comp_F(cell, bound, pot)

    def compute_jacobian(self, cell, bound, pot):
        if self.use_fused:
            try:
                return fused_jacobian_banded(cell, bound, pot)
            except ValueError:
                pass  # non-Boltzmann / small-n: exact dense fallback below
        from driftjax.numerics.residual import F_jacobian

        return F_jacobian(cell, bound, pot)
