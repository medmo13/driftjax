"""High-level simulator API: Device → relax / sweep / solve."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from driftjax.fields import DeviceDesign, LightSource, Material, Potentials, PVCell
from driftjax.io import material_to_dict
from driftjax.science.carrier_statistics import resolve_statistics
from driftjax.science.contacts import boundary_bias, boundary_eq
from driftjax.science.optics import _normalize_alpha_mode, generation_profile
from driftjax.science.spectrum import spectrum
from driftjax.solvers.continuation import equilibrium_guess, find_voc, total_current
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.units import current, density, mobility


def _to_dict(mat):
    """Convert a Material object to a dict (for internal numerics code)."""
    if isinstance(mat, dict):
        return mat
    return material_to_dict(mat)


def _make_design(
    n_points: int,
    Ls: list,
    mats: list,
    Ns: list,
    Snl: float | jnp.ndarray = 0.0,
    Snr: float | jnp.ndarray = 0.0,
    Spl: float | jnp.ndarray = 0.0,
    Spr: float | jnp.ndarray = 0.0,
    PhiMl: float | jnp.ndarray = 0.0,
    PhiMr: float | jnp.ndarray = 0.0,
    x_custom=None,
    T: float = 300.0,
) -> DeviceDesign:
    """DeviceDesign on a uniform grid.  Ls in cm; mats are Material objects or
    dicts; Ns are doping densities (cm⁻³, positive = donor, negative = acceptor);
    SRVs in cm/s; work functions in eV (0 → ohmic); T in K (I5: the
    dimensionless design is built with the thermal scales of T)."""
    from driftjax.units import thermal_scales

    sc = thermal_scales(T)
    total = sum(Ls)
    x_cm = jnp.linspace(0.0, total, n_points) if x_custom is None else x_custom
    n = n_points
    x_dimless = x_cm / jnp.asarray(sc["length"], dtype=jnp.float64)

    per_node_scaled = {
        "Chi": sc["energy"],
        "Eg": sc["energy"],
        "Et": sc["energy"],
        "Nc": density,
        "Nv": density,
        "Ndop": density,
        "mn": mobility,
        "mp": mobility,
        "tn": sc["time"],
        "tp": sc["time"],
        "Br": 1.0 / (sc["time"] * density),
        "Cn": 1.0 / (sc["time"] * density**2),
        "Cp": 1.0 / (sc["time"] * density**2),
    }
    per_node_raw = ["eps", "A"]

    d = {
        "x": x_dimless,
        "dgrid": jnp.diff(x_dimless),
        "alpha": jnp.zeros((0, 0), dtype=jnp.float64),
    }
    for key in per_node_scaled:
        d[key] = jnp.zeros((n,), dtype=jnp.float64)
    for key in per_node_raw:
        d[key] = jnp.zeros((n,), dtype=jnp.float64)

    start = 0.0
    n_layers = len(Ls)
    for idx, (t, mat, dop) in enumerate(zip(Ls, mats, Ns, strict=False)):
        mat_d = _to_dict(mat)
        is_last = idx == n_layers - 1
        # Use half-open interval to avoid double-counting interface node
        if is_last:
            mask = (x_cm >= start) & (x_cm <= start + t)
        else:
            mask = (x_cm >= start) & (x_cm < start + t)
        start += t
        for key, scl in per_node_scaled.items():
            src = jnp.asarray(mat_d.get(key, 0.0), dtype=jnp.float64)
            d[key] = jnp.where(mask, jnp.expand_dims(src / scl, 0), d[key])
        for key in per_node_raw:
            src = jnp.asarray(mat_d.get(key, 0.0), dtype=jnp.float64)
            d[key] = jnp.where(mask, jnp.expand_dims(src, 0), d[key])
        d["Ndop"] = jnp.where(
            mask, jnp.expand_dims(jnp.asarray(dop, dtype=jnp.float64) / density, 0), d["Ndop"]
        )
        if "alpha" in mat_d and jnp.asarray(mat_d["alpha"]).size:
            arr = jnp.asarray(mat_d["alpha"], dtype=jnp.float64)
            n_tab = arr.shape[0]
            if d["alpha"].size == 0:
                d["alpha"] = jnp.zeros((n_tab, n), dtype=jnp.float64)
            if d["alpha"].shape[0] != n_tab:
                # Resample new table to existing grid size (canonical 200-1400)
                x_src = jnp.linspace(200.0, 1400.0, n_tab)
                x_dst = jnp.linspace(200.0, 1400.0, d["alpha"].shape[0])
                arr = jnp.interp(x_dst, x_src, arr)
                n_tab = d["alpha"].shape[0]
            d["alpha"] = jnp.where(
                mask[None, :],
                jnp.broadcast_to(arr.reshape(n_tab, 1), (n_tab, n)),
                d["alpha"],
            )

    d["Snl"] = jnp.asarray(Snl, dtype=jnp.float64) / sc["velocity"]
    d["Snr"] = jnp.asarray(Snr, dtype=jnp.float64) / sc["velocity"]
    d["Spl"] = jnp.asarray(Spl, dtype=jnp.float64) / sc["velocity"]
    d["Spr"] = jnp.asarray(Spr, dtype=jnp.float64) / sc["velocity"]
    d["PhiMl"] = jnp.asarray(PhiMl, dtype=jnp.float64) / sc["energy"]
    d["PhiMr"] = jnp.asarray(PhiMr, dtype=jnp.float64) / sc["energy"]
    d["T"] = jnp.asarray(T, dtype=jnp.float64)
    return DeviceDesign(**d)


# ---------------------------------------------------------------------------
# Device — compositional class wrapping _make_design
# ---------------------------------------------------------------------------


class Device(eqx.Module):
    """Compositional solar-cell device definition (a JAX PyTree).

    Stores the baked :class:`DeviceDesign` (``_design``, a valid differentiable
    PyTree) plus the source materials/thicknesses/doping needed to rebuild it.
    Interface parameters and the two materials are *dynamic* leaves, so the
    device is ``jax.grad``/``jax.vmap``/``jax.jit``-safe; ``n_points`` and
    ``alpha_mode`` are static.

    Example
    -------
    >>> si = dj.load_material("Si")
    >>> dev = dj.Device(n_points=200,
    ...                 layers=[(1e-4, si, 1e17), (1e-4, si, -1e17)],
    ...                 Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)
    """

    _design: DeviceDesign
    n_points: int = eqx.field(static=True)
    alpha_mode: str = eqx.field(static=True)
    mat_l: Material
    mat_r: Material
    mats: tuple = eqx.field(static=False)
    Ls: jnp.ndarray
    Ns: jnp.ndarray
    Snl: jnp.ndarray
    Snr: jnp.ndarray
    Spl: jnp.ndarray
    Spr: jnp.ndarray
    PhiMl: jnp.ndarray
    PhiMr: jnp.ndarray
    T: jnp.ndarray

    def __init__(
        self,
        n_points: int,
        layers: list[tuple[float, Any, float]],
        Snl: float = 0.0,
        Snr: float = 0.0,
        Spl: float = 0.0,
        Spr: float = 0.0,
        PhiMl: float = 0.0,
        PhiMr: float = 0.0,
        x_custom=None,
        T: float = 300.0,
        alpha_mode: str = "beer-lambert",
    ):
        import jax.numpy as _jnp

        # NB: Ls/Ns are kept as plain Python lists (NOT float()-concretized) so
        # the Device stays trace-safe: tracer thickness/doping flows straight
        # through to the differentiable design.  This is what enables
        # jax.grad(lambda x: simulate(x2des(x), ...).eff) for optimizers.
        Ls_list = [th for th, _mat, _dop in layers]
        mats = [mat for _thickness, mat, _dop in layers]
        Ns_list = [dop for _thickness, _mat, dop in layers]
        # M8: single canonical normalizer (was a third inline copy here).
        _am = _normalize_alpha_mode(alpha_mode)
        if _am not in ("beer-lambert", "table", "tauc", "tmm", "fresnel"):
            raise ValueError(
                f"Device alpha_mode must be beer-lambert/table/tauc/tmm/fresnel, got {alpha_mode!r}"
            )
        if int(n_points) < 3:
            raise ValueError(f"Device requires n_points >= 3, got {n_points}")
        # L2: T <= 0 gives vt = 0 → inf/NaN thermal scales. Concrete check
        # only (T may be a tracer on the differentiable-device path).
        try:
            _Tf = float(T)
        except Exception:
            _Tf = None
        if _Tf is not None and not (_Tf > 0.0):
            raise ValueError(f"Device requires T > 0 K, got {T!r}")
        self.n_points = int(n_points)
        self.mat_l = mats[0]
        self.mat_r = mats[1] if len(mats) > 1 else mats[0]
        self.mats = tuple(mats)
        self.alpha_mode = _am
        self.Ls = _jnp.asarray(Ls_list, dtype=_jnp.float64)
        self.Ns = _jnp.asarray(Ns_list, dtype=_jnp.float64)
        self.Snl = _jnp.asarray(Snl, dtype=_jnp.float64)
        self.Snr = _jnp.asarray(Snr, dtype=_jnp.float64)
        self.Spl = _jnp.asarray(Spl, dtype=_jnp.float64)
        self.Spr = _jnp.asarray(Spr, dtype=_jnp.float64)
        self.PhiMl = _jnp.asarray(PhiMl, dtype=_jnp.float64)
        self.PhiMr = _jnp.asarray(PhiMr, dtype=_jnp.float64)
        self.T = _jnp.asarray(T, dtype=_jnp.float64)
        self._design = _make_design(
            n_points=n_points,
            Ls=Ls_list,
            mats=mats,
            Ns=Ns_list,
            Snl=Snl,
            Snr=Snr,
            Spl=Spl,
            Spr=Spr,
            PhiMl=PhiMl,
            PhiMr=PhiMr,
            x_custom=x_custom,
            T=T,
        )

    def design(self) -> DeviceDesign:
        """Materialise the Device into a DeviceDesign (cached)."""
        return self._design

    def with_temperature(self, T: float) -> Device:
        """Return a copy at a different temperature (K), rebuilding the design."""
        import equinox as _eqx

        # with_temperature is an eager-only helper (not on the traced path):
        # keep Ls/Ns as lists, but don't force float() on tracer elements.
        Ls_list = list(self.Ls) if getattr(self.Ls, "ndim", 0) == 1 else [self.Ls]
        Ns_list = list(self.Ns) if getattr(self.Ns, "ndim", 0) == 1 else [self.Ns]
        mats_list = (
            list(self.mats)
            if hasattr(self, "mats") and self.mats is not None
            else [self.mat_l, self.mat_r]
        )
        new_design = _make_design(
            n_points=self.n_points,
            Ls=Ls_list,
            mats=mats_list,
            Ns=Ns_list,
            Snl=self.Snl,
            Snr=self.Snr,
            Spl=self.Spl,
            Spr=self.Spr,
            PhiMl=self.PhiMl,
            PhiMr=self.PhiMr,
            T=T,
        )
        out = _eqx.tree_at(
            lambda d: (d._design, d.T, d.mat_l, d.mat_r, d.mats),
            self,
            (
                new_design,
                jnp.asarray(T, dtype=jnp.float64),
                mats_list[0],
                mats_list[1] if len(mats_list) > 1 else mats_list[0],
                tuple(mats_list),
            ),
        )
        return out

    def __getattr__(self, name):
        if name.startswith("_") or name in (
            "_design",
            "n_points",
            "alpha_mode",
            "mat_l",
            "mat_r",
            "Ls",
            "Ns",
            "Snl",
            "Snr",
            "Spl",
            "Spr",
            "PhiMl",
            "PhiMr",
            "T",
        ):
            raise AttributeError(name)
        return getattr(self._design, name)


# ---------------------------------------------------------------------------
# Core simulation functions
# ---------------------------------------------------------------------------


def _init_cell_impl(
    design: DeviceDesign,
    ls: LightSource,
    alpha_mode: str = "table",
    statistics: str = "boltzmann",
    optics=None,
    fused: bool = False,
) -> PVCell:
    """Bake a design: generation profile + statistics mode (impl)."""
    if ls is None:
        ls = spectrum(normalize=False)
    mode = _resolve_statistics_gated(statistics)
    if fused:
        from driftjax.numerics.fused_kernels import fused_generation as _fg

        if optics is not None:
            G = optics.generation(design, ls)
        else:
            G = _fg(design, ls, alpha_mode=alpha_mode, statistics=statistics)
    elif optics is not None:
        G = optics.generation(design, ls)
    else:
        G = generation_profile(design, ls, alpha_mode=alpha_mode)
    d = {
        "dgrid": design.dgrid,
        "x": design.x,
        "G": G,
        **{
            k: getattr(design, k)
            for k in [
                "eps",
                "Chi",
                "Eg",
                "Nc",
                "Nv",
                "mn",
                "mp",
                "tn",
                "tp",
                "Et",
                "Br",
                "Cn",
                "Cp",
                "Ndop",
                "Snl",
                "Snr",
                "Spl",
                "Spr",
                "PhiMl",
                "PhiMr",
            ]
        },
    }
    return PVCell(**d, statistics=mode, T=design.T)


_FD_EXPERIMENTAL_WARNED = False


def _resolve_statistics_gated(statistics: str) -> str:
    """resolve_statistics + one-time experimental gate for non-Boltzmann modes.

    P0-2: Fermi-Dirac transport/recombination thermodynamic consistency is
    unvalidated (classical SRH/radiative/Auger closures are used regardless
    of statistics mode), and Blakemore is documented-inaccurate (~60% at
    eta=1). Both remain available but emit a single UserWarning per session
    so they cannot be mistaken for validated production physics. The
    validated scientific model is Boltzmann-only (paper §4, preliminary
    section); resolve_statistics itself stays a pure mapper.
    """
    global _FD_EXPERIMENTAL_WARNED
    mode = resolve_statistics(statistics)
    if mode != "boltzmann" and not _FD_EXPERIMENTAL_WARNED:
        import warnings

        warnings.warn(
            f"statistics={mode!r} is EXPERIMENTAL: Fermi-Dirac/Blakemore "
            "transport-recombination thermodynamic consistency is not "
            "validated (classical closures used regardless of mode). "
            "Validated scope is Boltzmann-only.",
            UserWarning,
            stacklevel=3,
        )
        _FD_EXPERIMENTAL_WARNED = True
    return mode


import equinox as _eqx_vjp


@_eqx_vjp.filter_custom_vjp
def init_cell(design, ls, alpha_mode="table", statistics="boltzmann", optics=None, fused=False):
    """Bake a design: generation profile + statistics mode (analytic VJP for dG/ddesign)."""
    return _init_cell_impl(design, ls, alpha_mode, statistics, optics, fused)


def _generation_only(design, ls, alpha_mode, statistics, optics, fused):
    """Just the generation profile (no PVCell assembly) — cheap VJP target."""
    if fused:
        from driftjax.numerics.fused_kernels import fused_generation as _fg

        if optics is not None:
            return optics.generation(design, ls)
        return _fg(design, ls, alpha_mode=alpha_mode, statistics=statistics)
    if optics is not None:
        return optics.generation(design, ls)
    return generation_profile(design, ls, alpha_mode=alpha_mode)


_DIRECT_FIELDS = (
    "eps",
    "Chi",
    "Eg",
    "Nc",
    "Nv",
    "mn",
    "mp",
    "tn",
    "tp",
    "Et",
    "Br",
    "Cn",
    "Cp",
    "Ndop",
    "Snl",
    "Snr",
    "Spl",
    "Spr",
    "PhiMl",
    "PhiMr",
)


@init_cell.def_fwd
def _init_cell_fwd(perturbed, design, ls, *args, **kwargs):
    cell = _init_cell_impl(design, ls, *args, **kwargs)
    return cell, (design, ls, args, kwargs)


@init_cell.def_bwd
def _init_cell_bwd(res, g_cell, perturbed, design, ls, *args, **kwargs):
    saved_design, saved_ls, saved_args, saved_kwargs = res
    if g_cell is None:
        return jax.tree.map(lambda _: None, design)
    alpha_mode = kwargs.get("alpha_mode", "table")
    statistics = kwargs.get("statistics", "boltzmann")
    optics = kwargs.get("optics")
    fused = kwargs.get("fused", False)
    g_G = getattr(g_cell, "G", None)
    # 1) Generation-only VJP: only G depends on design via the generation chain.
    #    For Beer-Lambert, G depends on Eg, A, x, dgrid, T (and via alpha mode
    #    also Ndop). The VJP of the generation fn returns nonzero gradients
    #    only for those fields, and zero for the others.
    if g_G is not None:
        _, vjp_gen = _eqx_vjp.filter_vjp(
            lambda d: _generation_only(d, saved_ls, alpha_mode, statistics, optics, fused),
            saved_design,
        )
        (g_design_gen,) = vjp_gen(g_G)
    else:
        # zeros for array leaves, None elsewhere (same tree structure, so the
        # downstream zip over saved_design paths still aligns leaf-for-leaf)
        g_design_gen = jax.tree.map(
            lambda x: jnp.zeros_like(x) if isinstance(x, jax.Array) else None,
            saved_design,
        )
    # 2) Identity cotangents for directly-copied fields. PVCell mirrors these
    #    from design, so their VJP is identity (d field_i / d field_i = 1).
    _PVCELL_DIRECT = (
        "dgrid",
        "x",
        "eps",
        "Chi",
        "Eg",
        "Nc",
        "Nv",
        "mn",
        "mp",
        "tn",
        "tp",
        "Et",
        "Br",
        "Cn",
        "Cp",
        "Ndop",
        "Snl",
        "Snr",
        "Spl",
        "Spr",
        "PhiMl",
        "PhiMr",
        "T",
    )
    g_design_leaves = jax.tree_util.tree_leaves(g_design_gen)
    paths_and_leaves = jax.tree_util.tree_flatten_with_path(saved_design)[0]
    result_leaves = []
    for path, design_leaf in paths_and_leaves:
        field_name = getattr(path[0], "name", None) if len(path) > 0 else None
        g_gen = g_design_leaves[len(result_leaves)]
        if field_name is not None and field_name in _PVCELL_DIRECT and hasattr(g_cell, field_name):
            g_identity = getattr(g_cell, field_name)
        else:
            g_identity = None
        if g_identity is not None and g_gen is not None:
            result_leaves.append(g_gen + g_identity)
        elif g_gen is not None:
            result_leaves.append(g_gen)
        elif g_identity is not None:
            result_leaves.append(g_identity)
        else:
            result_leaves.append(jnp.zeros_like(design_leaf))
    g_design = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(saved_design), result_leaves
    )
    return g_design


def equilibrium(
    design: DeviceDesign, ls: LightSource, alpha_mode: str = "table", statistics: str = "boltzmann"
):
    """Two-stage equilibrium: Poisson-only φ, then full 3N system at V=0."""
    cell = init_cell(design, ls, alpha_mode=alpha_mode, statistics=statistics)
    bound = boundary_eq(cell)
    pot_eq0 = solve_eq(cell, bound, equilibrium_guess(cell).phi)
    pot, _info = solve_newton(cell, bound, pot_eq0)
    return cell, pot


# ---------------------------------------------------------------------------
# relax / sweep / solve — the three core operations
# ---------------------------------------------------------------------------


def _resolve_optics(design_or_device, ls, alpha_mode):
    """Resolve illumination + optics model for a Device when not given explicitly.

    A Device carries its own optics (``alpha_mode``); when a bare DeviceDesign
    is passed, fall back to AM1.5G with ``alpha_mode="table"``.
    Normalizes alias strings (beer_lambert vs beer-lambert).
    """
    if alpha_mode is not None and isinstance(alpha_mode, str):
        alpha_mode = alpha_mode.strip().lower().replace("_", "-")
        if alpha_mode in ("beerlambert", "bl"):
            alpha_mode = "beer-lambert"
    if isinstance(design_or_device, Device):
        alpha_mode = alpha_mode or getattr(design_or_device, "alpha_mode", "table")
    else:
        alpha_mode = alpha_mode or "table"
    return ls or spectrum(), alpha_mode


def relax(
    design_or_device,
    ls: LightSource | None = None,
    alpha_mode: str | None = None,
    statistics: str = "boltzmann",
    T: float | None = None,
):
    """Equilibrium relaxation: compute the thermal-equilibrium state.

    Parameters
    ----------
    design_or_device : DeviceDesign or Device
    ls : LightSource, optional — defaults to AM1.5G
    alpha_mode : str
    statistics : str
    T : float, optional — override design temperature (K)

    Returns
    -------
    (cell, pot) : tuple of PVCell and Potentials
    """
    design = design_or_device.design() if isinstance(design_or_device, Device) else design_or_device
    ls, alpha_mode = _resolve_optics(design_or_device, ls, alpha_mode)
    if T is not None:
        design = design.with_temperature(T)
    return equilibrium(design, ls, alpha_mode=alpha_mode, statistics=statistics)


def sweep(
    design_or_device,
    ls: LightSource | None = None,
    vmax: float = 1.1,
    n_steps: int = 41,
    alpha_mode: str | None = None,
    statistics: str = "boltzmann",
    refinement: bool = False,
    fused: bool = False,
    T: float | None = None,
    progress=None,
    init=None,
):
    """Full IV sweep.

    Returns: iv = (v [V], j [A/cm²]); plus iv_dim =
    (v, j dimensionless), voc [V], jsc_density [A/cm²], jsc (dimensionless),
    ff, eff, pmax [W/m²], pots, cell.

    Physical conventions: j[V] = j_dim·current (A/cm², current = 1.096e6);
    eff = pmax/(Σ P_in); jsc from the v=0 point when the sweep starts there.

    T (I5): override the design temperature (K); None → design.T.  The
    design is re-scaled via with_temperature so the whole dimensionless
    pipeline (mesh, potentials, lifetimes, generation) uses the thermal
    scales of T.

    progress: terminal reporting (never affects numerics).
      * None (default)     — silent (library hygiene; tests/logs unaffected)
      * True / "live"      — driftjax.console.SweepProgress: live per-bias
                             bar on a TTY, per-step lines under --verbose,
                             nothing when piped
      * your own object    — with update(i, **info) and close()
      * callable fn        — fn(i, info_dict) per bias step

    Info keys per step: v_v (V), iters, resid, backend, fallback, converged,
    dt (s), compiling (first JIT pass).
    """
    from driftjax.units import thermal_scales

    design = design_or_device.design() if isinstance(design_or_device, Device) else design_or_device
    ls, alpha_mode = _resolve_optics(design_or_device, ls, alpha_mode)
    if T is not None:
        design = design.with_temperature(T)
    cell = init_cell(design, ls, alpha_mode=alpha_mode, statistics=statistics)
    sc = thermal_scales(float(jnp.asarray(cell.T)))
    vmax_dim = vmax / sc["energy"]

    prog: object | None = None
    if progress is True or progress == "live":
        from driftjax import console

        prog = console.SweepProgress(n_steps, vmax_v=vmax, label=f"IV sweep ({alpha_mode})")
    elif progress is not None:
        if hasattr(progress, "update"):
            prog = progress  # user object with update/iter/close
        else:
            from driftjax.console import _CallableProgress

            prog = _CallableProgress(progress)  # plain callable fn(i, info)

    voltages, currents, pots, _fb = _sweep(
        cell,
        vmax_dim,
        n_steps,
        tol=1e-10,
        refinement=refinement,
        fused=fused,
        progress=prog,
        init=init,
        v_scale=sc["energy"],
    )
    del _fb  # R2 per-bias fallback flags live on Solution (simulate path)
    if prog is not None and hasattr(prog, "close"):
        try:
            prog.close()
        except Exception:
            pass

    voc_dim = find_voc(voltages, currents)
    jsc = jnp.abs(currents[0])
    pmax_dim, vmax_dim_out = _mpp(voltages, currents)
    # H5: NaN-safe ff (matches simulate.py) — beyond-range voc_dim is NaN
    # and must propagate as NaN, not as NaN/eps noise.
    ff = jnp.where(jnp.isfinite(voc_dim), pmax_dim / (voc_dim * jsc + 1e-30), jnp.nan)
    # Physical output units: volts and A/cm²
    v_volts = voltages * sc["energy"]
    j_phys = currents * sc["current"]
    pmax_phys_wm2 = pmax_dim * sc["energy"] * sc["current"] * 1e4  # W/m²
    return {
        "iv": (v_volts, j_phys),
        "iv_dim": (voltages, currents),
        "voc": voc_dim * sc["energy"],
        "voc_v": voc_dim * sc["energy"],
        "jsc": jsc,
        "jsc_density": jsc * sc["current"],
        "ff": ff,
        "eff": pmax_phys_wm2 / jnp.sum(ls.P_in),
        "pmax": pmax_dim,
        "pots": pots,
        "cell": cell,
        "eq": (
            solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi) if len(pots) else None
        ),
    }


# (legacy ``simulate`` alias removed — use :func:`sweep`)


def solve(
    design_or_device,
    bias: float,
    ls: LightSource | None = None,
    pot_ini: Potentials | None = None,
    alpha_mode: str | None = None,
    statistics: str = "boltzmann",
    T: float | None = None,
):
    """Single-bias-point solve.

    Parameters
    ----------
    design_or_device : DeviceDesign or Device
    bias : float — applied bias (V)
    ls : LightSource, optional — defaults to AM1.5G
    pot_ini : Potentials, optional — initial guess; if None, uses equilibrium
    alpha_mode : str
    statistics : str
    T : float, optional — override design temperature (K)

    Returns
    -------
    (j, pot) : tuple of current density (A/cm²) and Potentials
    """
    design = design_or_device.design() if isinstance(design_or_device, Device) else design_or_device
    ls, alpha_mode = _resolve_optics(design_or_device, ls, alpha_mode)
    if T is not None:
        design = design.with_temperature(T)
    cell = init_cell(design, ls, alpha_mode=alpha_mode, statistics=statistics)
    # Bias is applied in volts; convert with the thermal scales of the actual
    # design temperature (not the module-level T = 300 K constants), so
    # ``solve(dev, 0.6, T=350)`` applies the same physical voltage as at 300 K.
    from driftjax.units import thermal_scales

    sc_scale = thermal_scales(float(jnp.asarray(cell.T)))["energy"]
    if pot_ini is None:
        bound = boundary_eq(cell)
        pot_ini = solve_eq(cell, bound, equilibrium_guess(cell).phi)
    bound = boundary_bias(cell, bias / sc_scale)
    pot, _ = solve_newton(cell, bound, pot_ini)
    j = total_current(cell, pot) * current
    return j, pot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sweep(*args, **kwargs):
    """Thin wrapper around solvers.continuation.sweep to avoid name collision."""
    from driftjax.solvers.continuation import sweep as _continuation_sweep

    return _continuation_sweep(*args, **kwargs)


def _mpp(voltages, currents, tau=None):
    """Maximum-power point of the P = V·J curve.

    When ``tau`` is ``None`` (the default) the MPP is found by a
    per-segment cubic-PCHIP maximisation followed by a hard
    ``argmax`` over the candidate set.  That selection is *discrete*:
    if the global maximiser switches between segments as a design
    parameter changes the returned ``pmax`` is only Lipschitz, and
    its gradient is a subgradient that is discontinuous at the
    switch.  This is usually fine for a *solved* IV curve, but it
    makes the efficiency objective non-smooth whenever it is used
    inside an adjoint/JVP/VJP pass.  The warning below is emitted
    once per concrete call so the user cannot miss it.

    To obtain a globally smooth objective, pass a positive ``tau``
    (see :func:`driftjax.numerics.spline.calcPmax_smooth`): the
    soft-maximum is smooth everywhere and recovers the hard MPP as
    ``tau -> 0``.
    """
    import warnings

    from driftjax.numerics.spline import calcPmax_cubic, calcPmax_smooth

    if tau is None:
        # Warn only on the concrete path: inside jit/grad/vmap the voltages
        # are tracers and the warning would fire once per mapped element.
        from jax.core import Tracer

        if not isinstance(voltages, Tracer):
            warnings.warn(
                "Default MPP objective is the HARD argmax over PCHIP "
                "candidates (no smoothing tau).  The resulting pmax is "
                "only Lipschitz in the design: its gradient is a "
                "subgradient and is discontinuous at segment switches. "
                "Use Sweep(mpp_tau=...) for a smooth objective inside "
                "adjoint/JVP/VJP differentiation.  See the R5 note in "
                "SCIENTIFIC_REVIEW.md.",
                UserWarning,
                stacklevel=2,
            )
        return calcPmax_cubic(jnp.asarray(voltages), jnp.asarray(currents))
    return calcPmax_smooth(jnp.asarray(voltages), jnp.asarray(currents), tau)


# ---------------------------------------------------------------------------
# I5 — temperature + spectrum API
# ---------------------------------------------------------------------------


def temperature_sweep(
    design: DeviceDesign, ls: LightSource | None = None, Ts=(290.0, 300.0, 310.0, 320.0), **sim_kw
):
    """I5: simulate a design at several temperatures (K).

    Returns a dict {T: :class:`driftjax.solution.Solution`}.
    Voc(T) is the headline observable: for the constant-Eg / constant-τ
    model it drops ≈ 3.0–4.0 mV/K (the −2 mV/K textbook value additionally
    assumes Eg(T) and Nc(T) ∝ T^{3/2}).
    """
    from driftjax.problems import Sweep
    from driftjax.simulate import simulate

    sweep_kw = dict(sim_kw)
    protocol = Sweep(
        vmax=sweep_kw.pop("vmax", 1.1),
        n_steps=sweep_kw.pop("n_steps", 41),
    )

    return {float(T): simulate(design, protocol, ls=ls, T=float(T), **sweep_kw) for T in Ts}


def spectral_sensitivity(
    design: DeviceDesign,
    ls: LightSource | None = None,
    n_bins: int = 12,
    rel_step: float = 2e-2,
    **sim_kw,
):
    """I5: ∂η/∂Φ(λ) by central finite differences on binned photon flux.

    The spectrum is split into ``n_bins`` wavelength bins; each bin's
    P_in is perturbed by ±rel_step and the efficiency change is measured
    end-to-end.  Returns (n_bins, 2) array: [λ_center (nm), dη/dΦ_bin],
    where Φ_bin is the bin's integrated incident power (W/m²), so the
    units are 1/(W/m²).  Sub-bandgap bins come out ≈ 0 — the classic
    spectral-sensitivity signature.
    """
    ls = spectrum() if ls is None else ls
    lam_nm = np.asarray(ls.Lambda)  # already in nm (package convention)
    edges = np.linspace(float(lam_nm.min()), float(lam_nm.max()), n_bins + 1)
    idx = np.clip(np.digitize(lam_nm, edges) - 1, 0, n_bins - 1)
    P = np.asarray(ls.P_in, dtype=np.float64).copy()

    def eta(P_arr):
        ls_p = LightSource(Lambda=ls.Lambda, P_in=jnp.asarray(P_arr))
        return float(sweep(design, ls_p, **sim_kw)["eff"])

    out = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        Pp, Pm = P.copy(), P.copy()
        Pp[m] *= 1.0 + rel_step
        Pm[m] *= 1.0 - rel_step
        d_phi = float(np.sum(P[m]) * rel_step)  # W/m² in this bin
        d_eta = eta(Pp) - eta(Pm)
        out.append((float(np.mean(lam_nm[m])), d_eta / (2.0 * d_phi)))
    return np.asarray(out)
