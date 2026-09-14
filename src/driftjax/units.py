"""Physical constants and dimensionless scaling for driftjax.

The whole simulator works in dimensionless quantities.  The scaling uses the
thermal voltage Vt = k_B T / q as the energy scale (Debye-unit convention;
cf. Mann et al., CPC 2021): the density scale is n₀ = 1e19 cm⁻³;
lengths are scaled by L₀ = √(ε₀ k_B T / (q² n₀)) so that the Poisson operator
has O(1) coefficients.  Any redesign that changed this scaling would have to
re-derive every transport coefficient; the package proved it out over
~10 iterations, so it is kept verbatim and unit-checked in
``tests/unit/test_units.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy import constants as _const

F64 = Float[Array, "..."]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

T = 300.0  # temperature (K)
nm = 1e-9  # metre
cm = 1e-2  # metre

kB = _const.k  # J/K   Boltzmann constant
hc = _const.c * _const.h  # J·m   Planck × speed of light
eV = _const.eV  # J     electronvolt
q = _const.e  # C     elementary charge
eps0 = _const.epsilon_0 * 1e-2  # C/(V·cm)  (cgs-consistent permittivity)

# ---------------------------------------------------------------------------
# Scaling base units
# ---------------------------------------------------------------------------


def thermal_scales(temperature: float | jax.Array) -> dict:
    """Dimensionless scaling units at a given temperature (I5).

    Everything the solver touches is dimensionless; the mapping back to
    physical units depends on temperature through Vt = kT/q, the Debye
    length and the current/generation scales.  ``thermal_scales(T)``
    returns {length, energy, current, gratedens, time, velocity, Vt} for
    that T, and the module globals below are pinned to T = 300 K (the
    package convention) — bit-identical to the pre-I5 constants.

    Known first-order-only model: Eg, Nc/Nv and mobilities are kept
    temperature-independent; the dominant Voc(T) ≈ −2 mV/K behaviour comes
    from the Vt scaling alone.
    """
    density_ = jnp.float64(1e19)  # 1/cm³
    mobility_ = jnp.float64(1.0)  # cm²/(V·s)
    length_ = jnp.sqrt(eps0 * kB * temperature / (q**2 * density_))  # cm
    energy_ = jnp.float64(kB * temperature / q)  # eV  ≡ Vt
    time_ = eps0 / q / density_ / mobility_  # s  (T-independent)
    velocity_ = length_ / time_  # cm/s
    current_ = kB * temperature * density_ * mobility_ / length_  # A/cm²
    gratedens_ = density_ * mobility_ * kB * temperature / (q * length_**2)
    return {
        "length": length_,
        "energy": energy_,
        "current": current_,
        "gratedens": gratedens_,
        "time": time_,
        "velocity": velocity_,
        "Vt": energy_,
    }


_SCALES_300 = thermal_scales(300.0)
length = _SCALES_300["length"]  # cm
density = jnp.float64(1e19)  # 1/cm³
energy = _SCALES_300["energy"]  # eV  ≡ Vt
mobility = jnp.float64(1.0)  # cm²/(V·s)
time = _SCALES_300["time"]  # s
velocity = _SCALES_300["velocity"]  # cm/s
current = _SCALES_300["current"]  # A/cm²
gratedens = _SCALES_300["gratedens"]  # 1/(cm³·s)
Vt = energy  # thermal voltage (eV)

# ---------------------------------------------------------------------------
# Per-quantity scaling table:  dimensionless = physical / scale
# ---------------------------------------------------------------------------

scales: dict[str, F64] = {
    "grid": length,
    "dgrid": length,
    "eps": jnp.float64(1.0),  # relative permittivity
    "Chi": energy,
    "Eg": energy,
    "Et": energy,
    "Nc": density,
    "Nv": density,
    "Ndop": density,
    "mn": mobility,
    "mp": mobility,
    "tn": time,
    "tp": time,
    "Br": 1 / (time * density),  # cm³/s
    "Cn": 1 / (time * density**2),  # cm⁶/s
    "Cp": 1 / (time * density**2),
    "A": jnp.float64(1.0),  # Tauc prefactor (optical, unscaled)
    "alpha": jnp.float64(1.0),  # m⁻¹ (optical, unscaled)
    "G": gratedens,
    "Snl": velocity,
    "Snr": velocity,
    "Spl": velocity,
    "Spr": velocity,
    "PhiMl": energy,
    "PhiMr": energy,
    "P_in": jnp.float64(1.0),  # W/m² (optical, unscaled)
    "Lambda": jnp.float64(1.0),  # m (optical, unscaled)
}


def scale(param_name: str, value: float) -> float:
    """Dimensionless value of a physical quantity."""
    return value / float(scales[param_name])


# Dimensional consistency checks (kept as readable asserts for documentation):
#   current:  q·n₀·μ·Vt/L0  =  C·cm⁻³·cm²/(V·s)·(J/C)/cm = J/(cm³·s)·cm²·cm⁻¹·... = A/cm²  ✓
#   grate:    n₀·μ·Vt/(q·L0²) = 1/(cm³·s) ✓
#   time:     ε₀/(q·n₀·μ)  →  s ✓
