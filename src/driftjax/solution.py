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

    @property
    def efficiency(self):
        """Power-conversion efficiency as a fraction."""
        return self.eff

    @property
    def currents(self):
        """Current-density samples in A cm⁻²."""
        return self.current

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
