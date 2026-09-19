"""Simulation result container.

``Solution`` is an immutable PyTree carrying the IV curve, efficiency and the
per-bias potentials, so it flows cleanly through ``jit``/``grad``/``vmap`` and
exposes convenient methods (``iv_curve()``, ``at_bias()``, ``plot()``).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from driftjax.fields import Potentials, PVCell


class Solution(eqx.Module):
    """Immutable result of an equilibrium solve or bias sweep."""

    voltages: Float[Array, "n"]  # bias grid (V)  # noqa: F821
    current: Float[Array, "n"]  # current density (A/cm^2)  # noqa: F821
    potentials: list  # per-bias ``Potentials`` (PyTree list)
    cell: PVCell
    eff: float  # power conversion efficiency (fraction)
    voc: float  # open-circuit voltage (V)
    ff: float  # fill factor
    jsc: float  # short-circuit current density (A/cm^2)
    pmax: float  # max power (dimensionless solar-cell units)
    eq_pot: Potentials  # equilibrium potential
    P_in: Float[Array, "nlam"]  # incident power per wavelength (W/m^2)  # noqa: F821
    protocol: str = eqx.field(static=True, default="sweep")
    # R1 (scientific-review fix): the efficiency denominator, made explicit.
    # eta = Pmax / sum(P_in). The raw embedded AM1.5G table sums to
    # 899.9168 W/m^2 (DriftJax default, preserves all golden test outputs identical);
    # spectrum(normalize=True) gives the standard 1-sun 1000 W/m^2 value.
    # Read this before quoting an absolute efficiency: the two conventions
    # differ by ~11% relative.
    p_in_total_wm2: object = 899.9167906040965  # float on concrete paths, 0-d array under trace
    # H1 failure semantics: True only when every bias residual was
    # verified (concrete path). P0-4 contract: under jit/grad/vmap the
    # audit cannot concretize, so converged stays at its default True and
    # MUST NOT be read as "verified" there — gradient_status is the
    # authoritative tri-state (CERTIFIED / UNRELIABLE / UNVERIFIED), and
    # converged=True + gradient_status=UNVERIFIED means "not audited".
    # Never trust eff/voc/ff when converged is False; never treat a
    # gradient as certified unless gradient_status is CERTIFIED.
    converged: bool = True
    # max|F| over biases from the post-hoc audit (0.0 when unverified).
    max_residual: float = 0.0
    # Per-bias max|F| from the post-hoc audit (empty when unverified).
    per_bias_residuals: list = eqx.field(default_factory=list)
    # R2 provenance: per-bias truncated-SVD (lstsq) fallback flags from the
    # serial sweep (True = dgbsv failed and lstsq stepped on that bias;
    # False = clean banded solves; None entries / empty = unverified, e.g.
    # under jit/grad/vmap, fused-scan or batched paths).
    fallback_used: list = eqx.field(default_factory=list)
    # Implicit-Voc status: True when the sweep bracketed a current sign
    # change (J(Voc) = 0 refined by bisection on converged evaluations);
    # False means Voc is NaN (above sweep range) or unverified (traced).
    voc_bracketed: bool = False
    # Tight secant slope dJ/dV at the refined root (dimensionless current
    # per dimensionless volt; NaN when unrefined/unverified). Feeds the
    # VJP implicit term dVoc/dtheta = -J_theta/J_V; NaN selects the coarse
    # bracket-secant fallback there.
    voc_JV: float = float("nan")

    @property
    def efficiency(self):
        """Power-conversion efficiency as a fraction.

        Always returns the computed value.  Check ``converged`` and
        ``max_residual`` to assess reliability; for ill-conditioned
        systems (cond(J) >> 1/eps) the solver may do its best yet
        still have a large residual — this does NOT mean the IV curve
        or efficiency is wrong, only that the residual cannot be
        driven below the conditioning floor.

        The denominator is ``p_in_total_wm2`` (raw AM1.5G = 899.9168 W/m^2
        by default).  Use :attr:`efficiency_standard` for the 1000 W/m^2
        1-sun convention.
        """
        return self.eff

    @property
    def efficiency_standard(self):
        """Efficiency with the denominator relabelled to 1000 W/m^2 (fraction).

        R1 (scientific-review fix): eta_standard = eta * P_in_total / 1000.
        This is the *convention rescaling* of the same converged solve, i.e.
        "this Pmax expressed against the 1-sun denominator", which is what
        makes the number comparable across codes that normalise differently.

        It is NOT identical to re-solving with ``spectrum(normalize=True)``:
        Pmax is only approximately linear in the photon flux, so scaling the
        flux by k scales Pmax by ~k with a small superlinear deviation (about
        0.06% relative at 1 sun for a typical cell).  For the exact
        1000 W/m^2 efficiency, pass ``ls=spectrum(normalize=True)`` to
        ``simulate``.  The difference is documented, not hidden.
        """
        denom = self.p_in_total_wm2
        if denom is None:
            denom = 899.9167906040965
        denom = float(jnp.asarray(denom))
        if not (denom > 0.0):
            return float("nan")
        return float(jnp.asarray(self.eff)) * denom / 1000.0

    @property
    def currents(self):
        """Current-density samples in A cm⁻²."""
        return self.current

    @property
    def gradient_status(self):
        """Gradient certification status: CERTIFIED / UNRELIABLE / UNVERIFIED.

        Step 36 (minimum viable failure semantics): CERTIFIED when the sweep
        converged with no fallback (clean pivoted-banded solves throughout);
        UNRELIABLE when unconverged (gradients must not be trusted) or when
        any bias used the truncated-SVD fallback (minimum-norm
        least-squares steps, not exact Newton steps — see fallback_used);
        UNVERIFIED when the audit could not run (jit/grad/vmap, fused-scan
        or batched paths with empty diagnostics).  Full certification with
        adjoint residuals and condition estimates remains Tier-C work.
        """
        if not self.per_bias_residuals and not self.fallback_used:
            return "UNVERIFIED"
        if not self.converged:
            return "UNRELIABLE"
        if any(x is True for x in self.fallback_used):
            return "UNRELIABLE"
        return "CERTIFIED"

    def iv_curve(self):
        """Return ``(voltages[V], current[A/cm^2])``."""
        return self.voltages, self.current

    def at_bias(self, v):
        """Return the ``Potentials`` at the bias point closest to ``v`` (V).

        Concrete-path only: ``int(argmin(...))`` concretizes the bias grid,
        so this raises under ``jit``/``grad``/``vmap``. M7: inside traced
        code, stack the potentials first (``potentials`` is a Python list,
        so plain indexing concretizes too), e.g.
        ``jax.tree.map(lambda *ps: jnp.stack([pot2vec(p) for p in ps]),
        *sol.potentials)`` evaluated concretely, then gather rows by index.
        """
        if self.voltages.size == 0:
            raise ValueError("an equilibrium solution has no bias points")
        if any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves((self.voltages, v))
        ):
            raise ValueError(
                "at_bias is concrete-path only; index sol.potentials directly under jit/grad/vmap"
            )
        idx = int(jnp.argmin(jnp.abs(self.voltages - float(v))))
        return self.potentials[idx]

    def plot(self, path: str = "iv.png", kind: str = "iv"):
        """Render a publication-style figure and return its path."""
        from driftjax.viz import plotting

        if kind == "iv":
            try:
                import numpy as _np

                p_in = float(_np.sum(_np.asarray(self.P_in, float)))
            except Exception:
                p_in = None
            return plotting.plot_iv_curve(
                self.voltages,
                self.current,
                path=path,
                title=f"PCE = {float(self.eff) * 100:.2f}%",
                p_in=p_in,
            )
        if kind == "band":
            return plotting.plot_band_diagram(self.cell, self.eq_pot, path=path)
        raise ValueError(f"unknown plot kind {kind!r}")
