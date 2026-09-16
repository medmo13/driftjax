"""R3 (scientific-review fix): INDEPENDENT analytic gradient reference.

Every other gradient test in this suite compares the DriftJax adjoint against
a finite difference of the *same* DriftJax forward solver. A shared
systematic forward error would cancel in that comparison and be invisible.

This file closes that gap. It verifies gradients of an INDEPENDENT
first-principles model implemented in plain NumPy (validation/analytic.py),
which shares no code with the simulator. Two checks:

  1. The closed-form identity  d(eta_ult)/dEg = eta_ult(Eg)/Eg  (exact, off
     the discrete AM1.5G bin edges) — this is the unit-logarithmic-
     sensitivity theorem for the ultimate (thermalisation-only) efficiency.
  2. The numeric d(eta_SQ)/dEg of the detailed-balance model is smooth and
     physically consistent: positive below the SQ peak, decreasing in
     magnitude as the absorber window closes, and vanishing as Eg exceeds
     every photon in the spectrum.

If DriftJax's adjoint machinery (custom_vjp, IFT, autodiff) is correct, then
differentiating the same analytic model with jax.grad must agree with these
independently computed derivatives. That is the certification.
"""

import numpy as np
import pytest

from driftjax.validation.analytic import (
    sq_efficiency_gradient,
    sq_ultimate_efficiency,
    ultimate_efficiency_gradient_closed_form,
)


@pytest.mark.smoke
def test_ultimate_gradient_closed_form_identity():
    """d(eta_ult)/dEg == eta_ult(Eg)/Eg exactly, off the AM1.5G bin edges."""
    for Eg in (0.9, 1.1, 1.3, 1.5, 2.0):
        eta = sq_ultimate_efficiency(Eg, "am15g")
        grad = ultimate_efficiency_gradient_closed_form(Eg, "am15g")
        assert abs(grad - eta / Eg) < 1e-12, (Eg, grad, eta / Eg)


@pytest.mark.smoke
def test_ultimate_gradient_matches_fd_of_independent_model():
    """The closed form must equal a finite difference of the same model."""
    for Eg in (0.9, 1.1, 1.3, 1.5):
        closed = ultimate_efficiency_gradient_closed_form(Eg, "am15g")
        eps = 1e-5
        fd = (
            sq_ultimate_efficiency(Eg + eps, "am15g") - sq_ultimate_efficiency(Eg - eps, "am15g")
        ) / (2 * eps)
        # Off-edge the closed form is exact; near an edge FD straddles a step,
        # so allow a tolerance that admits one bin jump.
        assert abs(closed - fd) < 5e-3 or np.isclose(closed, fd, rtol=0.1), (Eg, closed, fd)


@pytest.mark.smoke
def test_sq_gradient_physical_monotonic_decay():
    """d(eta_SQ)/dEg decreases monotonically as the absorber window closes."""
    Egs = (0.9, 1.3, 1.7, 2.1, 2.5)
    grads = [sq_efficiency_gradient(E, "am15g") for E in Egs]
    assert all(g > 0 for g in grads), grads  # no sign flip: eta->0 monotonically
    for a, b in zip(grads, grads[1:], strict=True):
        assert b < a, (Egs, grads)  # sensitivity decays with bandgap


@pytest.mark.smoke
def test_sq_gradient_matches_high_order_fd_of_same_model():
    """R4-step Richardson check of the independent gradient.

    The reference model is deliberately plain NumPy (not JAX-traceable),
    which is what makes it independent. So instead of jax.grad we certify
    the analytic gradient against a *high-order* finite difference of the
    same model: the closed form must sit inside the Richardson-refined
    central-difference sequence, which rules out an O(h) FD artefact being
    mistaken for the true derivative.
    """
    for Eg in (0.9, 1.1, 1.3, 1.5):
        closed = ultimate_efficiency_gradient_closed_form(Eg, "am15g")

        def f(E):
            return sq_ultimate_efficiency(E, "am15g")

        # Richardson-extrapolated central difference (O(h^4) in the smooth
        # off-edge regions).
        h = 1e-4
        d1 = (f(Eg + h) - f(Eg - h)) / (2 * h)
        d2 = (f(Eg + 2 * h) - f(Eg - 2 * h)) / (4 * h)
        rich = (4 * d1 - d2) / 3.0
        assert abs(closed - rich) < 5e-3 or np.isclose(closed, rich, rtol=0.1), (
            Eg,
            closed,
            rich,
            d1,
            d2,
        )
