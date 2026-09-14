"""Solar spectra: AM1.5G (rescaled to exactly 1000 W/m²), white, mono, custom.

Known package bug (F-08, fixed here): the embedded AM1.5G table summed to
~899.9 W/m²; the flux is renormalised so that Σ P(λ_i) == 1000 W/m² by
construction.  The absolute photocurrent depends on this:
The raw AM1.5G table sums to ~899.9 W/m²; normalize=True scales to 1000 W/m².

Units note (AUDIT R3): the table entries are *bin-integrated* powers per
wavelength point [W/m²], NOT spectral densities [W/m²/nm] (peak entries are
~40 W/m² over ~30 nm bins; a Δλ-weighted trapezoid gives ~32000, which is
unphysical). The unweighted sum is therefore the correct normalisation —
do not "fix" this with a trapezoid rule.
"""

from __future__ import annotations

import jax.numpy as jnp

from driftjax.fields import LightSource

# Tabulated AM1.5G reference spectrum (λ in nm, P in W/m²)
_LAMBDA_AM15G = jnp.array(
    [
        296.58786520846525,
        301.86932586906727,
        307.3413862642167,
        314.0101663463744,
        322.2157186534005,
        331.6588606552103,
        343.16563211861506,
        356.4718787589982,
        371.43844861016,
        388.4562139889681,
        406.85926227327474,
        426.0481717998874,
        447.62901892787136,
        469.92116394102163,
        494.07400235193575,
        519.8819737412621,
        547.21371249863,
        576.0744631926423,
        606.46754270433,
        638.2129109456624,
        671.2959617860226,
        705.7863072495503,
        742.0507337674146,
        780.5554341808609,
        818.2455840936051,
        858.0879046742818,
        897.7411488242675,
        940.8662643813793,
        987.4902295689035,
        1030.0610336012219,
        1074.091374437584,
        1117.2269901074096,
        1173.443596282667,
        1218.9715095166084,
        1264.2332426026787,
        1307.3216392819909,
        1342.224915652247,
        1447.3678952624173,
        1497.1669408915898,
        1542.2086303659403,
        1591.7043105789685,
        1642.3988363909662,
        1692.8888886873513,
        1741.451683376098,
        1783.2390162887173,
        1848.737471241858,
        1963.9220689859833,
        2001.8975232780376,
        2052.9961725177836,
        2105.726153145771,
        2157.036887098164,
        2209.6681561997557,
        2260.783110191019,
        2311.1416799890626,
        2359.5013086654512,
        2405.924012276593,
        2447.2596671376177,
        2479.1748474211004,
        2508.9873785326154,
        2803.94419627386,
        2862.6037651092292,
        2898.589860171952,
        2935.189959763852,
        2977.422243585477,
        3016.3189651212374,
        3065.497960229955,
        3118.9606513702056,
        3161.8065262433493,
        3196.3617805534977,
        3254.417788924162,
        3295.710821301024,
        3346.8175838814655,
        3390.141776338096,
        3430.816643341351,
        3470.4895400476826,
        3508.7437587983923,
        3546.544332677495,
        3583.80383634895,
        3619.463180983516,
        3652.9771254767124,
        3687.57586456887,
        3717.861131147382,
        3747.7584839248098,
        3776.2656006382517,
        3803.1349830975896,
        3828.321455229104,
        3851.8040704128543,
        3873.6506783527016,
        3894.7021530916877,
        3912.8548933604898,
        3929.98782245093,
        3944.9901304159,
        3958.1676145712195,
        3969.5282751392497,
        3979.1727953185414,
        3986.99564951234,
        3992.9791614959527,
        3997.1413552005,
        3999.457603085752,
    ]
)

_P_IN_AM15G = jnp.array(
    [
        0.00012889204787158213,
        0.008777279582043227,
        0.10107774097655463,
        0.46722334523923387,
        1.2150988129071114,
        2.4725889291277845,
        3.5265164985065622,
        4.936507253041806,
        7.269751112678899,
        8.927357155341136,
        16.020906918535356,
        19.34033100690697,
        26.057509884831223,
        30.829282730150954,
        33.39515661592565,
        35.44747476918117,
        38.125496404177476,
        39.540923343124724,
        40.8404938686826,
        41.596166567916164,
        41.940750589024525,
        39.789515395817496,
        38.39403584669429,
        37.691423788156115,
        35.18840644601061,
        35.75215291789887,
        30.259804321835226,
        18.202504474049892,
        27.752491142563077,
        28.337421416452848,
        25.1872011944878,
        12.673966009585598,
        17.23007013474619,
        19.802853014125215,
        18.19642379589036,
        13.500050082734614,
        3.001238136332884,
        2.6573289581422364,
        8.217048544364276,
        12.530664298026352,
        11.933102731756618,
        11.141729752106082,
        9.840969492071611,
        7.532064618549481,
        3.399772792777609,
        0.045292634528007025,
        1.2942960011011828,
        2.6679591131597404,
        3.7979450072049112,
        4.513945457451602,
        4.317987777546656,
        3.90667388108776,
        3.43007058789695,
        2.8661884962618718,
        2.1310126280587927,
        1.529495517033794,
        0.9838948727965099,
        0.3344187742565935,
        0.042086112090700394,
        4.5965977504444245e-6,
        0.002519090572158028,
        0.03909265151751956,
        0.13365361767129383,
        0.17652719137748085,
        0.21141755386263275,
        0.14834619599764395,
        0.19075355865114366,
        0.35892368435944627,
        0.13533761251828436,
        0.16353225285698836,
        0.1822229755593091,
        0.22581063882710814,
        0.3320492714611748,
        0.3977721572943606,
        0.4374505055593256,
        0.4476869896126664,
        0.377551925311501,
        0.35114326073602675,
        0.35871733567086955,
        0.3452873527071678,
        0.2949742205561386,
        0.30177237871160434,
        0.2706572565634048,
        0.2483643691929065,
        0.23175334821947552,
        0.21865531905695917,
        0.19395293482122686,
        0.1529684011833061,
        0.14372812778454383,
        0.1280118754702922,
        0.1138792546671928,
        0.10644202367603008,
        0.09530732687067581,
        0.08099285747487681,
        0.06504601291568587,
        0.05127412754488824,
        0.03704777100973708,
        0.023219685557830824,
        0.009917010541730737,
    ]
)


def spectrum(normalize: bool = True) -> LightSource:
    """AM1.5G spectrum.

    ``normalize=True`` (default): total power exactly 1000 W/m² (1-sun,
    driftjax convention.  ``normalize=False`` returns the
    raw embedded table (Σ ≈ 899.9 W/m²) — the Beer–Lambert convention,
    so the regression against driftjax goldens reproduces
    bit-comparable IV curves.
    """
    P = _P_IN_AM15G * (1000.0 / jnp.sum(_P_IN_AM15G)) if normalize else _P_IN_AM15G
    return LightSource(Lambda=_LAMBDA_AM15G, P_in=P, kind="sun")


def white(n_points: int = 100, lam_lo: float = 400.0, lam_hi: float = 800.0) -> LightSource:
    """Flat spectrum (W/m² per wavelength point), lam in nm.

    Normalised to 1 sun (Σ P = 1000 W/m²).  (AUDIT: the old flat
    200 W/m²-per-point table integrated to 20 000 W/m² ≈ 20 suns.)
    """
    # L2: degenerate grids (n_points < 1, inverted range, non-finite edges)
    # silently produce empty/NaN spectra and a 0/0 efficiency downstream.
    n = int(n_points)
    lo, hi = float(lam_lo), float(lam_hi)
    if n < 1:
        raise ValueError(f"white needs n_points >= 1, got {n_points!r}")
    if not (lo == lo and hi == hi) or not (hi > lo and lo > 0):
        raise ValueError(f"white needs 0 < lam_lo < lam_hi (nm), got {lam_lo!r}, {lam_hi!r}")
    lam = jnp.linspace(lo, hi, n)
    return LightSource(
        Lambda=lam,
        P_in=jnp.full(n, 1000.0 / n, dtype=jnp.float64),
        kind="white",
    )


def monochromatic(wavelength_m: float) -> LightSource:
    """Monochromatic source at a wavelength in metres.

    The package convention for ``LightSource.Lambda`` is **nanometres**
    (every optics consumer multiplies by 1e-9; see ``science/optics.py``),
    so the metres-based API is converted here.  Before v0.1.14 the raw
    metres value was stored unconverted — ``monochromatic(5e-07)`` produced
    a λ = 5e-16 m "photon" (≈5e15 eV), for which every material is
    transparent: the source silently rendered the cell dark.
    """
    # L2: non-positive wavelengths give hc/0 = inf in photonflux (silent
    # zero flux); fail loudly instead.
    wl = float(wavelength_m)
    if not (wl > 0.0):
        raise ValueError(f"monochromatic needs wavelength_m > 0, got {wavelength_m!r}")
    return LightSource(
        Lambda=jnp.array([wl * 1e9]),
        P_in=jnp.array([1000.0], dtype=jnp.float64),
        kind="mono",
    )
