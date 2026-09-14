"""Shared device/material builders for the DriftJax test suite.

Single source of truth for the canonical validation devices so that every
test module exercises identical physics (and a mesh-size change is a
one-line edit).  All builders return *baked* ``DeviceDesign`` objects or
:class:`~driftjax.simulator.Device` pytrees ready for ``dj.simulate``.

Conventions (mirroring the paper's validation pyramid):
    * ``ex1``  — Si-like homojunction (deltapv ex1 geometry).
    * ``ex2``  — CdS/CdTe heterojunction (deltapv ex2 / SCAPS anchor).
    * ``pn``   — symmetric p-n diode on one material (fast unit tests).
"""

from __future__ import annotations

import driftjax as dj

# Canonical ex1 absorber/contact parameters (validated reference values).
EX1_MATERIAL = dict(
    Chi=3.9,
    Eg=1.5,
    eps=9.4,
    Nc=8e17,
    Nv=1.8e19,
    mn=100,
    mp=100,
    Et=0,
    tn=1e-8,
    tp=1e-8,
    A=20000.0,
)

# Canonical CdS window / CdTe absorber pair (SCAPS-1D comparison).
CDS_MATERIAL = dict(
    Nc=2.2e18,
    Nv=1.8e19,
    Eg=2.4,
    eps=10,
    Et=0,
    mn=100,
    mp=25,
    tn=1e-8,
    tp=1e-13,
    Chi=4.0,
    A=10000.0,
)
CDTE_MATERIAL = dict(
    Nc=8e17,
    Nv=1.8e19,
    Eg=1.5,
    eps=9.4,
    Et=0,
    mn=320,
    mp=40,
    tn=5e-9,
    tp=5e-9,
    Chi=3.9,
    A=10000.0,
)


def ex1_device(n_points: int = 500) -> dj.Device:
    """Symmetric n-p homojunction on the ex1 material at ``n_points`` nodes."""
    mat = dj.material(**EX1_MATERIAL)
    return dj.Device(
        layers=list(zip([1e-4, 1e-4], [mat, mat], [1e17, -1e17], strict=False)),
        n_points=n_points,
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
    )


def ex2_device(n_points: int = 500) -> dj.Device:
    """CdS(25 nm)/CdTe(4 um) heterojunction at ``n_points`` nodes.

    Geometry is IDENTICAL to validation/ex2_np_hetero.py (the canonical
    deltapv-ex2 / SCAPS anchor): CdTe absorber doped -1e15 cm^-3.  The release
    gates in tests/regression/test_release_validated.py assert against this
    same problem, so the paper's Section 4.1 numbers and the executable gates
    describe one device.
    """
    cds, cdte = dj.material(**CDS_MATERIAL), dj.material(**CDTE_MATERIAL)
    return dj.Device(
        layers=list(zip([2.5e-6, 4e-4], [cds, cdte], [1e17, -1e15], strict=False)),
        n_points=n_points,
        Snl=1.16e7,
        Snr=1.16e7,
        Spl=1.16e7,
        Spr=1.16e7,
    )


def pn_device(
    n_points: int = 60,
    thickness: float = 2e-4,
    doping: float = 1e17,
    material_kw: dict | None = None,
) -> dj.Device:
    """Symmetric p-n step dopant profile on one material (default ex1)."""
    mat = dj.material(**(material_kw or EX1_MATERIAL))
    return dj.Device(
        layers=[
            (thickness / 2, mat, doping),
            (thickness / 2, mat, -doping),
        ],
        n_points=n_points,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )


def si_homojunction_device(n_points: int = 200) -> dj.Device:
    """Planar crystalline-Si homojunction used by the L9 experimental gate."""
    si = dj.material(
        Chi=3.9,
        Eg=1.12,
        eps=11.7,
        Nc=2.8e19,
        Nv=1.04e19,
        mn=1.08,
        mp=0.56,
        Et=0,
        tn=1e-6,
        tp=1e-6,
        A=20000.0,
    )
    return dj.Device(
        layers=list(zip([1e-4, 1e-4], [si, si], [1e16, -1e16], strict=False)),
        n_points=n_points,
        Snl=1e7,
        Snr=0,
        Spl=0,
        Spr=1e7,
        PhiMl=-1.0,
        PhiMr=-1.0,
    )
