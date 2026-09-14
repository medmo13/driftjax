"""Adjoint (differentiation) modes (composable ``AbstractAdjoint`` objects).

Mirrors Optimistix: ``ImplicitAdjoint`` (the implicit-function-theorem
adjoint, default) makes ``jax.grad(simulate)`` correct and fast. The
``DirectAdjoint`` name is an explicit alias for the stable IFT path; production
Newton iteration unrolling is not supported.
"""

from __future__ import annotations

import equinox as eqx


class AbstractAdjoint(eqx.Module):
    """Base class for adjoint / differentiation modes."""


class ImplicitAdjoint(AbstractAdjoint):
    """Implicit-function-theorem adjoint (default). Fast and correct.

    The adjoint system ``J^T lambda = g`` is solved with pivoted dense LU.
    (An O(N) banded-transpose path existed but was removed in v0.1.12: it
    produced silent 10-30% gradient errors on ill-conditioned devices that
    no cheap gate could certify. See CHANGELOG.)
    """


class DirectAdjoint(AbstractAdjoint):
    """Alias for the stable IFT adjoint.

    Differentiating *through* the Newton iterations directly is numerically
    unstable for this ill-conditioned DDP Jacobian, and the IFT adjoint is
    mathematically identical for a converged root — so ``DirectAdjoint``
    computes the gradient via the implicit-function theorem.
    """
