"""Optics API (composable ``AbstractOptics`` objects)."""

from __future__ import annotations

import equinox as eqx

from driftjax.fields import DeviceDesign, LightSource
from driftjax.science.optics import (
    _normalize_alpha_mode,  # M8: single canonical normalizer (was duplicated here)
    fresnel_generation,
    generation_profile,
    tmm_generation,
)


class AbstractOptics(eqx.Module):
    """Base class for optical generation models."""

    def generation(self, design: DeviceDesign, ls: LightSource):
        raise NotImplementedError


class BeerLambert(AbstractOptics):
    """Beer–Lambert generation (Tauc or tabulated absorption coefficient)."""

    alpha_mode: str = "tauc"  # "tauc" or "table" or "beer-lambert"

    def __init__(self, alpha_mode: str = "tauc"):
        super().__init__()
        am = _normalize_alpha_mode(alpha_mode)
        if am not in ("tauc", "table", "beer-lambert"):
            raise ValueError(f"BeerLambert alpha_mode must be tauc/table/beer-lambert, got {alpha_mode!r}")
        self.alpha_mode = am

    def generation(self, design: DeviceDesign, ls: LightSource):
        return generation_profile(design, ls, alpha_mode=self.alpha_mode)


class TMM(AbstractOptics):
    """Coherent transfer-matrix optics.

    ``alpha_mode="tauc"`` (default) evaluates alpha from the material's Tauc
    prefactor ``A``; a database material with measured optical constants but
    ``A = null`` (e.g. GaAs) then yields ZERO generation. Pass
    ``TMM(alpha_mode="table")`` for such materials.
    """

    alpha_mode: str = "tauc"

    def __init__(self, alpha_mode: str = "tauc"):
        super().__init__()
        am = _normalize_alpha_mode(alpha_mode)
        if am not in ("tauc", "table"):
            raise ValueError(f"TMM alpha_mode must be tauc/table, got {alpha_mode!r}")
        self.alpha_mode = am

    def generation(self, design: DeviceDesign, ls: LightSource):
        return tmm_generation(design, ls, alpha_mode=self.alpha_mode)


class Fresnel(AbstractOptics):
    """Incoherent double-pass Beer-Lambert with Fresnel front loss + rear reflector."""

    alpha_mode: str = "tauc"
    rear_reflectance: float = 0.9

    def __init__(self, alpha_mode: str = "tauc", rear_reflectance: float = 0.9):
        super().__init__()
        am = _normalize_alpha_mode(alpha_mode)
        if am not in ("tauc", "table"):
            raise ValueError(f"Fresnel alpha_mode must be tauc/table, got {alpha_mode!r}")
        self.alpha_mode = am
        self.rear_reflectance = float(rear_reflectance)

    def generation(self, design: DeviceDesign, ls: LightSource):
        return fresnel_generation(
            design, ls, alpha_mode=self.alpha_mode, rear_reflectance=self.rear_reflectance
        )


def apply_optics(optics: AbstractOptics, design: DeviceDesign, ls: LightSource):
    """Dispatch to the optics model's generation profile."""
    return optics.generation(design, ls)
