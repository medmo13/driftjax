"""Optical generation: Beer–Lambert (tauc / tabulated α) and optional TMM.

Beer–Lambert
------------
G(x) = Σ_λ φ₀(λ)·α(λ,x)·exp(−∫₀ˣ α(λ,x′)dx′)   (per-interval, node-averaged).

The Tauc model uses α = A·√(hν − Eg) with the band edge smoothed over
~2.6 meV (softplus) so AD never differentiates exactly at the kink
(√′ = ∞) — a real gradient bug in the original package (F-10).

Transfer-Matrix Method
----------------------
Coherent 2×2 chain over the same interval layers; k = αλ/4π so TMM reduces
to Beer–Lambert in the thick-cell limit and adds interference for thin
films.  (Validated against a direct Maxwell-continuity solve and the 1-slab
Airy formula in the package test suite; ported here with identical math.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax, vmap

from driftjax.fields import DeviceDesign, LightSource
from driftjax.units import cm, gratedens, hc, kB, length, q
from driftjax.units import eV as _eV_J

TAUC_SMOOTH_W = 0.1  # Vt-units (~2.6 meV) Urbach-like smoothing


def photonflux(ls: LightSource) -> jax.Array:
    """Photon flux [1/(m²·s)] from power density [W/m²].

    L3: Lambda ≤ 0 gives hc/0 = inf → silent zero flux. This runs under
    jit (no Python raise allowed), so construction sites validate
    (white/monochromatic); direct LightSource builds with Lambda ≤ 0 are
    caller error and yield zero flux for those bins by construction.
    """
    E_photon = hc / (ls.Lambda * 1e-9)
    return ls.P_in / E_photon


def alpha_tauc(design: DeviceDesign, ls: LightSource) -> jax.Array:
    """Tauc absorption (n_λ, n_node) in m⁻¹: A·√(hν − Eg), edge-smoothed.

    Convention (driftjax standard): design.A is in
    [cm⁻¹·eV⁻¹ᐟ²]; A_si = A/cm/√(eV) turns it into [m⁻¹·J⁻¹ᐟ²].
    """
    vt = kB * design.T / q  # trace-safe Vt(T) (design.T is a field)
    dE_J = (hc / (ls.Lambda * 1e-9))[:, None] - (design.Eg * vt * q)[None, :]
    x = dE_J / (TAUC_SMOOTH_W * vt * q)
    # softplus edge (logaddexp) keeps alpha >= 0 and smooth, BUT for photons
    # far below the band edge `exp(x)` underflows to 0, so logaddexp -> 0 and
    # dE_soft -> 0 exactly.  Reverse-mode then sees sqrt'(0)=inf multiplied by
    # the (also ~0) sigmoid chain -> inf*0 = NaN.  Floor the sqrt argument so
    # the derivative is always finite (the floor is ~1e-21 of a nominal edge
    # value, so it does not change physics; alpha simply asymptotes to 0).
    dE_soft = TAUC_SMOOTH_W * vt * q * jnp.logaddexp(0.0, x)
    A_si = design.A[None, :] / cm / jnp.sqrt(_eV_J)
    return A_si * jnp.sqrt(dE_soft + 1e-40)


def alpha_from_table(design: DeviceDesign, ls: LightSource) -> jax.Array:
    """Tabulated α (n_tab, n_node) interpolated onto the light-source λ grid.

    Tab λs are the package's canonical uniform 200–1400 nm grid —
    ``io.load_material`` resamples measured tables onto it at load time
    (log-α interpolation), so the assumption is exact for database materials.
    Queries are clamped to the tab interval (extrapolation diverges).
    Vectorized as one XLA program (no per-node Python loop) — fused with
    generation when called via ``fused_generation``.
    """
    if design.alpha.size == 0 or design.alpha.ndim != 2:
        return alpha_tauc(design, ls)
    tab_lam = jnp.linspace(200.0, 1400.0, design.alpha.shape[0]) * 1e-9
    lam_q = ls.Lambda * 1e-9

    def interp_col(a1):
        return jnp.interp(lam_q, tab_lam, a1)

    return vmap(interp_col, in_axes=1, out_axes=1)(design.alpha)


def _gen_per_lambda(x_m: jax.Array, phi0: jax.Array, alpha: jax.Array) -> jax.Array:
    """Per-wavelength Beer–Lambert generation (n_λ, n_node) [1/(m³·s)]."""
    dx = jnp.diff(x_m)
    if alpha.ndim == 1 or alpha.shape[1] == 1:
        ai = jnp.tile(alpha[:, 0:1], (1, dx.shape[0]))
    else:
        ai = alpha[:, :-1]
    tau = jnp.cumsum(ai * dx[None, :], axis=1)
    tau_prep = jnp.concatenate([jnp.zeros((phi0.shape[0], 1)), tau], axis=1)
    phi_right = phi0[:, None] * jnp.exp(-tau_prep[:, :-1])
    G_int = ai * phi_right
    G_l = jnp.concatenate([G_int, G_int[:, -1:]], axis=1)
    G_r = jnp.concatenate([G_int[:, :1], G_int], axis=1)
    return 0.5 * (G_l + G_r)


def beer_lambert_G(design: DeviceDesign, ls: LightSource, alpha_mode: str = "tauc") -> jax.Array:
    """Dimensionless generation profile (n_node,) via Beer–Lambert."""
    from driftjax.units import thermal_scales as _ts

    x_m = design.x * _ts(design.T)["length"] * cm
    phi0 = photonflux(ls)
    alpha = alpha_tauc(design, ls) if alpha_mode == "tauc" else alpha_from_table(design, ls)
    G_raw = _gen_per_lambda(x_m, phi0, alpha)
    G_m3s = jnp.sum(G_raw, axis=0)
    return G_m3s * 1e-6 / gratedens


def _fresnel_per_lambda(x_m, phi0, alpha, n_tilde, rear_reflectance: float = 0.9):
    """Double-pass generation with front-surface Fresnel loss and a rear
    reflector (n_λ, n_node) [1/(m³·s)].

    Front: Φ_f = Φ₀·(1 − R_f), R_f = |(1−ñ)/(1+ñ)|² at the front surface.
    Rear:  the unabsorbed Φ_f·e^{−τ_L} is reflected with power reflectance
    R_b and traverses the absorber a second time.  With τ(x) the optical
    depth from the front:

        G(x) = α(x)·Φ₀·(1−R_f)·[e^{−τ(x)} + R_b·e^{−2τ_L}·e^{+τ(x)}]

    Exact double-pass absorptance per wavelength:

        A = (1−R_f)·(1−e^{−τ_L})·(1 + R_b·e^{−τ_L})

    R_b = 0 gives front-reflection-only single pass; R_b = 1 a perfect
    rear mirror.  (AUDIT: the ``rear_reflectance`` argument was previously
    accepted by ``fresnel_generation`` but never forwarded here, so every
    call silently used a perfect mirror.)
    """
    dx = jnp.diff(x_m)
    if alpha.ndim == 1 or alpha.shape[1] == 1:
        ai = jnp.tile(alpha[:, 0:1], (1, dx.shape[0]))
    else:
        ai = alpha[:, :-1]
    tau = jnp.cumsum(ai * dx[None, :], axis=1)
    tau_prep = jnp.concatenate([jnp.zeros((phi0.shape[0], 1)), tau], axis=1)
    tau_L = tau[:, -1:]
    R_f = jnp.abs((1.0 - n_tilde) / (1.0 + n_tilde)) ** 2
    phi_f = phi0[:, None] * (1.0 - R_f)[:, None]
    # forward + backward (rear-reflected) photon fluxes at each interval edge
    Rb = jnp.asarray(rear_reflectance, dtype=jnp.float64)
    phi_fwd = phi_f * jnp.exp(-tau_prep[:, :-1])
    phi_bwd = Rb * phi_f * jnp.exp(-2.0 * tau_L) * jnp.exp(+tau_prep[:, :-1])
    G_int = ai * (phi_fwd + phi_bwd)
    G_l = jnp.concatenate([G_int, G_int[:, -1:]], axis=1)
    G_r = jnp.concatenate([G_int[:, :1], G_int], axis=1)
    return 0.5 * (G_l + G_r)


def fresnel_generation(
    design: DeviceDesign, ls: LightSource, alpha_mode: str = "tauc", rear_reflectance: float = 0.9
) -> jax.Array:
    """H2: Beer–Lambert + Fresnel — front-surface reflection loss and a rear
    reflector (double pass).  Dimensionless generation (n_node,).

    rear_reflectance = 0  →  front-reflection-only Beer–Lambert
    rear_reflectance = 1  →  perfect rear mirror (maximal double pass)
    The default 0.9 models a typical back reflector.
    """
    from driftjax.units import thermal_scales as _ts

    x_m = design.x * _ts(design.T)["length"] * cm
    phi0 = photonflux(ls)
    alpha = alpha_tauc(design, ls) if alpha_mode == "tauc" else alpha_from_table(design, ls)
    n_real, k = _refractive_index(design, ls, alpha_mode)
    n_tilde = n_real[:, 0] + 1j * k[:, 0]
    G_raw = _fresnel_per_lambda(x_m, phi0, alpha, n_tilde, rear_reflectance)
    G_m3s = jnp.sum(G_raw, axis=0)
    return G_m3s * 1e-6 / gratedens


# ---------------------------------------------------------------------------
# TMM (optional, coherent optics)
# ---------------------------------------------------------------------------

_GL4_N = jnp.asarray(
    [0.06943184420297371, 0.33000947820757187, 0.6699905217924281, 0.9305681557970263]
)
_GL4_W = jnp.asarray(
    [0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269]
)


def _refractive_index(design: DeviceDesign, ls: LightSource, alpha_mode: str):
    """(n, k) arrays of shape (n_λ, n_node): n = √eps, k = αλ/4π."""
    alpha = alpha_tauc(design, ls) if alpha_mode == "tauc" else alpha_from_table(design, ls)
    if alpha.ndim == 1:
        alpha = alpha[:, None]
    n_real = jnp.broadcast_to(jnp.sqrt(jnp.maximum(design.eps, 0.0))[None, :], alpha.shape)
    k = alpha * (ls.Lambda[:, None] * 1e-9) / (4.0 * jnp.pi)
    return n_real, k


def _tmm_core(n_layer, k_layer, d_m, lam_m, flux):
    """Per-wavelength (G_node, R, T_power): generation nodes plus the
    coherent power reflectance/transmittance from the same chain matrix.
    Air is assumed on both sides (free-standing stack), matching the
    package TMM convention; validated analytically (1-slab Airy) in
    tests/unit/test_tmm.py."""
    L = d_m.shape[0]
    n_layer = n_layer[:L]
    k_layer = k_layer[:L]
    n_tilde = n_layer + 1j * k_layer
    kappa = 2.0 * jnp.pi * n_tilde / lam_m

    def interface(a, b):
        r = (a - b) / (a + b)
        tau = (a + b) / (2.0 * b)
        return tau * jnp.array([[1.0, -r], [-r, 1.0]])

    n_air = 1.0 + 0j

    # Stable R/T via the Airy recursion (exact for any stack; immune to the
    # e^{+κL} growth that breaks the transfer-matrix chain on thick lossy
    # cells — verified identical to the chain to 1e-16 on thin stacks and
    # stable to 1 mm). H1 audit: the previous chain used an inverted
    # propagation phase (diag(e^{−iκd}, e^{iκd})) plus r = T[1,0]/T[0,0],
    # which is only self-consistent for a SINGLE slab — for multilayers it
    # over-shot ∫G·dx vs 1−R−T by up to 3.5×.
    r_eff = (n_tilde[L - 1] - n_air) / (n_tilde[L - 1] + n_air)
    t_eff = 2.0 * n_tilde[L - 1] / (n_tilde[L - 1] + n_air)

    def _airy_body(i, carry):
        r_eff_, t_eff_ = carry
        j = L - 1 - i
        # dynamic indexing for tracer j
        n_j = lax.dynamic_index_in_dim(n_tilde, j, keepdims=False)
        n_jm1 = lax.dynamic_index_in_dim(n_tilde, j - 1, keepdims=False)
        k_j = lax.dynamic_index_in_dim(kappa, j, keepdims=False)
        d_j = lax.dynamic_index_in_dim(d_m, j, keepdims=False)
        r_j = (n_jm1 - n_j) / (n_jm1 + n_j)
        t_j = 2.0 * n_jm1 / (n_jm1 + n_j)
        e2 = jnp.exp(2j * k_j * d_j)
        denom = 1.0 + r_j * r_eff_ * e2
        r_eff_n = (r_j + r_eff_ * e2) / denom
        t_eff_n = t_eff_ * t_j * jnp.exp(1j * k_j * d_j) / denom
        return (r_eff_n, t_eff_n)

    if L > 1:
        r_eff, t_eff = lax.fori_loop(0, L - 1, _airy_body, (r_eff, t_eff))
    r0 = (n_air - n_tilde[0]) / (n_air + n_tilde[0])
    t0 = 2.0 / (n_air + n_tilde[0])
    e2 = jnp.exp(2j * kappa[0] * d_m[0])
    denom = 1.0 + r0 * r_eff * e2
    r_total = (r0 + r_eff * e2) / denom
    t_total = t_eff * t0 * jnp.exp(1j * kappa[0] * d_m[0]) / denom
    R_pow = jnp.abs(r_total) ** 2
    T_pow = jnp.abs(t_total) ** 2
    inc = jnp.array([1.0 + 0j, r_total])

    # Forward field walk across interfaces, one lax.fori_loop instead of an
    # unrolled Python loop: keeps the jaxpr O(1) in the layer count (the loop
    # is re-entered once per wavelength under vmap, so the unrolled form made
    # the TMM trace O(n_lambda * L)).
    E_plus = jnp.zeros(L, dtype=jnp.complex128)
    E_minus = jnp.zeros(L, dtype=jnp.complex128)
    field = interface(n_air, n_tilde[0]) @ inc
    E_plus = E_plus.at[0].set(field[0])
    E_minus = E_minus.at[0].set(field[1])

    def _walk_body(j, carry):
        field, Ep, Em = carry
        n_j = lax.dynamic_index_in_dim(n_tilde, j, keepdims=False)
        n_jp1 = lax.dynamic_index_in_dim(n_tilde, j + 1, keepdims=False)
        kap_j = lax.dynamic_index_in_dim(kappa, j, keepdims=False)
        d_j = lax.dynamic_index_in_dim(d_m, j, keepdims=False)
        d_prop = jnp.exp(1j * kap_j * d_j)
        prop = jnp.array([[d_prop, 0.0], [0.0, 1.0 / d_prop]])
        field = interface(n_j, n_jp1) @ (prop @ field)
        Ep = Ep.at[j + 1].set(field[0])
        Em = Em.at[j + 1].set(field[1])
        return (field, Ep, Em)

    if L > 1:
        _, E_plus, E_minus = lax.fori_loop(0, L - 1, _walk_body, (field, E_plus, E_minus))

    x_gl = _GL4_N[None, :] * d_m[:, None]
    e_pos = jnp.exp((1j * kappa.real[:, None] - kappa.imag[:, None]) * x_gl)
    e_neg = jnp.exp((-1j * kappa.real[:, None] + kappa.imag[:, None]) * x_gl)
    field = E_plus[:, None] * e_pos + E_minus[:, None] * e_neg
    mean_E2 = jnp.sum(_GL4_W[None, :] * jnp.abs(field) ** 2, axis=1)

    alpha = 4.0 * jnp.pi * k_layer / lam_m
    G_int = flux * alpha * n_layer * mean_E2
    G_l = jnp.concatenate([G_int, G_int[-1:]])
    G_r = jnp.concatenate([G_int[:1], G_int])
    return 0.5 * (G_l + G_r), R_pow, T_pow


def _tmm_per_wavelength(n_layer, k_layer, d_m, lam_m, flux) -> jax.Array:
    """TMM generation for one wavelength.  Returns (n_node,) [1/(m³·s)]."""
    return _tmm_core(n_layer, k_layer, d_m, lam_m, flux)[0]


def tmm_generation(design: DeviceDesign, ls: LightSource, alpha_mode: str = "table") -> jax.Array:
    """Dimensionless generation via TMM (n_node,)."""
    n_real, k = _refractive_index(design, ls, alpha_mode)
    d_m = design.dgrid * length * cm
    phi0 = photonflux(ls)

    def per_lambda(lam_nm, n_lam, k_lam, flux_lam):
        return _tmm_per_wavelength(n_lam, k_lam, d_m, lam_nm * 1e-9, flux_lam)

    G_m3s = jnp.sum(vmap(per_lambda)(ls.Lambda, n_real, k, phi0), axis=0)
    return G_m3s * 1e-6 / gratedens


def tmm_rt(design: DeviceDesign, ls: LightSource, alpha_mode: str = "table"):
    """(R, T) power reflectance/transmittance per wavelength (air both sides).

    R and T are (n_λ,) — the coherent Airy result for the whole stack —
    so `1 - R - T` is the spectral absorptance, and

        Σ_λ A(λ)·Φ₀(λ)·(hν)  vs  Σ_j G_j·Δx·(hν)

    is the energy-conservation identity tested in tests/unit/test_tmm.py.
    """
    n_real, k = _refractive_index(design, ls, alpha_mode)
    d_m = design.dgrid * length * cm

    def per_lambda(lam_nm, n_lam, k_lam):
        _, R, T = _tmm_core(n_lam, k_lam, d_m, lam_nm * 1e-9, 1.0)
        return R, T

    R, T = vmap(per_lambda)(ls.Lambda, n_real, k)
    return R, T


def _generation_beer_lambert(design: DeviceDesign, ls: LightSource) -> jax.Array:
    """Generation profile (Beer–Lambert, hard-kink Tauc
    absorption A·√(hν−Eg) (0 below the edge) and per-node Beer–Lambert
    φ·α with flux integrated from the left face,
    ``optical.compute_G`` conventions (spectrum must be the raw 899.9 W/m²
    table for Beer–Lambert convention)."""
    from driftjax.units import thermal_scales as _ts

    x_m = design.x * _ts(design.T)["length"] * cm
    dx = jnp.diff(x_m)
    vt = _ts(design.T)["energy"]
    Eg_si = design.Eg * vt * q
    A_si = design.A / cm / jnp.sqrt(_eV_J)

    def per_lambda(lam_nm, phi0):
        dE = hc / (lam_nm * 1e-9) - Eg_si
        alpha = jnp.where(dE > 0, A_si * jnp.sqrt(jnp.abs(dE)), 0.0)
        phi = phi0 * jnp.exp(-jnp.cumsum(jnp.concatenate([jnp.zeros(1), alpha[:-1] * dx])))
        return phi * alpha

    G_m3s = jnp.sum(vmap(per_lambda)(ls.Lambda, photonflux(ls)), axis=0)
    return G_m3s * 1e-6 / gratedens


def _normalize_alpha_mode(mode: str) -> str:
    if not isinstance(mode, str):
        return mode
    m = mode.strip().lower().replace("_", "-")
    if m in ("beerlambert", "bl"):
        m = "beer-lambert"
    return m


def generation_profile(
    design: DeviceDesign, ls: LightSource, alpha_mode: str = "table"
) -> jax.Array:
    """Dispatch: "tmm" → coherent TMM; "beer-lambert" → Beer–Lambert;
    "table"/"tauc" → Beer–Lambert (smoothed-edge tauc / tabulated)."""
    alpha_mode = _normalize_alpha_mode(alpha_mode)
    valid = {"tmm", "beer-lambert", "fresnel", "table", "tauc"}
    if alpha_mode not in valid:
        raise ValueError(f"Unknown alpha_mode {alpha_mode!r}, expected one of {sorted(valid)}")
    if alpha_mode == "tmm":
        src = "tauc" if design.alpha.size == 0 else "table"
        return tmm_generation(design, ls, alpha_mode=src)
    if alpha_mode == "beer-lambert":
        return _generation_beer_lambert(design, ls)
    if alpha_mode == "fresnel":
        src = "tauc" if design.alpha.size == 0 else "table"
        return fresnel_generation(design, ls, alpha_mode=src)
    return beer_lambert_G(design, ls, alpha_mode=alpha_mode)
