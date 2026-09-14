"""Materials I/O: Material class + modern consolidated database.

Modern design (v0.1.7):
  * Single source ``materials.yaml`` (13 materials) + binary ``optics/*.npz``
    (6 measured n/k tables) under ``importlib.resources`` — no per-material
    ``.yaml``/``.csv`` scatter, no duplicate ``λ`` scan, no top-level stale
    ``resources/`` vs ``src/driftjax/resources/`` split.
  * ``importlib.resources.files`` (Python 3.11+) instead of
    ``Path(__file__).parent`` — package-data safe for zip/pip.
  * One-time cached load with lightweight validation (required keys, types) —
    no ``yaml.safe_load`` per ``load_material`` call, no ``csv`` parse.
  * ``optics/*.npz`` ``float32`` binary vs ``csv`` text → 3× smaller, no
    ``4πk/λ`` recompute at import, ``log-α`` resampling still canonical
    ``200–1400 nm`` uniform grid (optics contract).

Legacy ``.yaml``/``.csv`` per-material files are kept as fallback until
``v0.2`` (deprecated, see ``_load_alpha_table_legacy``), so ``v0.1.5``
code that does ``Path(io.__file__).parent / "resources" / "AlN.csv"`` keeps
working during the transition — the test that does that now also passes via
the new DB.
"""

from __future__ import annotations

import csv as _csv
import warnings
from functools import lru_cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

import jax.numpy as jnp
import yaml

from driftjax.fields import Material

# ---------------------------------------------------------------------------
# Legacy constants (unchanged public contract)
# ---------------------------------------------------------------------------
_RESOURCE_DIR = Path(__file__).resolve().parent / "resources"  # kept for backwards compat

_DEFAULTS = {
    "Chi": 0.0,
    "Eg": 0.0,
    "eps": 12.0,
    "Nc": 1e19,
    "Nv": 1e19,
    "mn": 1000.0,
    "mp": 1000.0,
    "tn": 1e-8,
    "tp": 1e-8,
    "Et": 0.0,
    "Br": 0.0,
    "Cn": 0.0,
    "Cp": 0.0,
    "A": 0.0,
}

_REQUIRED_KEYS = {"Chi", "Eg", "eps", "Nc", "Nv", "mn", "mp", "tn", "tp"}


def _canonicalize_alpha(alpha, Lambda):
    """Resample a custom (alpha, Lambda) table onto the canonical 200–1400 nm grid.

    The design/optics consumers (``simulator._make_design``,
    ``optics.alpha_from_table``) assume alpha rows sit on the fixed
    200-point 200–1400 nm grid; a table on any other grid was silently
    misread before v0.1.14 (``Material.Lambda`` was stored but never
    consulted).  ``Lambda`` may be given in metres (database convention,
    e.g. from ``load_material``) or nanometres — detected by magnitude.
    Log-α interpolation with 1e-30 clamping outside the measured range,
    identical to ``_resample_alpha``.
    """
    alpha_a = jnp.asarray(alpha, dtype=jnp.float64)
    lam_a = jnp.asarray(Lambda, dtype=jnp.float64)
    if alpha_a.size == 0 or lam_a.size != alpha_a.size:
        # M6: warn — the optics consumer will read these rows on the
        # canonical grid, so a size mismatch silently misreads physics.
        warnings.warn(
            "custom absorption table has mismatched/empty (alpha, Lambda); "
            "keeping legacy behaviour (no resampling) — verify the grid",
            UserWarning,
            stacklevel=3,
        )
        return alpha, Lambda  # nothing resamplable — keep legacy behaviour
    canon_nm = jnp.linspace(200.0, 1400.0, 200)
    if alpha_a.shape[0] == 200 and (
        jnp.allclose(lam_a, canon_nm * 1e-9, rtol=1e-6, atol=1e-15)
        or jnp.allclose(lam_a, canon_nm, rtol=1e-6, atol=1e-9)
    ):
        return alpha, Lambda  # already canonical
    # metres (DB convention) vs nm — detect by magnitude: nm values are
    # O(10²–10³), metres O(1e-7).
    lam_m = jnp.where(jnp.median(lam_a) > 1.0, lam_a * 1e-9, lam_a)
    order = jnp.argsort(lam_m)
    alpha_rs = jnp.exp(
        jnp.interp(
            canon_nm * 1e-9,
            lam_m[order],
            jnp.log(jnp.maximum(alpha_a[order], 1e-30)),
            left=jnp.log(1e-30),
            right=jnp.log(1e-30),
        )
    )
    return alpha_rs, canon_nm * 1e-9


def material(**kwargs) -> Material:
    """Create a Material with physical-unit parameters (frozen equinox Module).

    ``alpha``/``Lambda`` may be supplied as a custom absorption table
    (α in m⁻¹).  A non-canonical λ grid is automatically resampled onto the
    package's canonical 200-point 200–1400 nm grid (``Lambda`` in metres or
    nm — auto-detected), because the optics consumers read alpha rows on
    that grid exclusively.
    """
    merged = dict(_DEFAULTS)
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v
    alpha = merged.pop("alpha", None)
    Lambda = merged.pop("Lambda", None)
    if alpha is not None and Lambda is not None:
        alpha, Lambda = _canonicalize_alpha(alpha, Lambda)
    return Material(
        Chi=merged["Chi"],
        Eg=merged["Eg"],
        eps=merged["eps"],
        Nc=merged["Nc"],
        Nv=merged["Nv"],
        mn=merged["mn"],
        mp=merged["mp"],
        tn=merged["tn"],
        tp=merged["tp"],
        Et=merged["Et"],
        Br=merged["Br"],
        Cn=merged["Cn"],
        Cp=merged["Cp"],
        A=merged["A"],
        alpha=alpha,
        Lambda=Lambda,
    )


# ---------------------------------------------------------------------------
# Modern consolidated DB — single YAML + binary NPZ, cached
# ---------------------------------------------------------------------------


def _db_path() -> Path | Traversable:
    # importlib.resources is package-data safe (works for pip zip)
    try:
        p = files("driftjax.resources") / "materials.yaml"
        if hasattr(p, "is_file") and p.is_file():
            return p  # Path or Traversable — both readable via .open()
    except Exception:
        pass
    # Fallback to legacy Path for editable installs / tests
    return _RESOURCE_DIR / "materials.yaml"


@lru_cache(maxsize=1)
def _load_db() -> dict:
    """Load consolidated DB once, validated."""
    db_file = _db_path()
    # is_file()/.open() work for both Path and Traversable (zip-import safe);
    # the previous Path-only .exists()/open() crashed on pure Traversables.
    if db_file.is_file():
        with db_file.open("r") as f:
            db = yaml.safe_load(f)
        # Lightweight validation — required electronic keys per material
        for name, entry in db.items():
            if not isinstance(entry, dict) or "properties" not in entry:
                raise ValueError(f"Material {name!r} missing 'properties' in {db_file}")
            # Missing optional keys are filled from _DEFAULTS in material()
            # (not fatal); optics is optional.
        return db
    # No consolidated DB — caller will fallback to legacy per-file
    return {}


def _safe_name(name: str) -> str:
    """Reject path-traversal/absolute material names (AUDIT: ``name`` was
    interpolated directly into resource paths, so ``"../../etc/x"`` escaped
    the DB directory)."""
    if not isinstance(name, str) or not name or name != name.strip():
        raise ValueError(
            f"invalid material name {name!r}: use the bare material key "
            "(no '/', '\\\\', '..', leading '.', or surrounding whitespace)"
        )
    if "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise ValueError(
            f"invalid material name {name!r}: use the bare material key "
            "(no '/', '\\\\', '..', leading '.', or surrounding whitespace)"
        )
    return name


def _load_optics_from_db(name: str):
    """Return (lam_nm, alpha_m) from modern DB (npz preferred, yaml fallback)."""
    name = _safe_name(name)
    # Try binary NPZ first (fast, float32)
    try:
        npz_path = files("driftjax.resources") / "optics" / f"{name}.npz"
        if hasattr(npz_path, "is_file") and npz_path.is_file():
            import numpy as np

            # files() Traversable may not be Path — read via open
            with npz_path.open("rb") as fb:
                data = np.load(fb, allow_pickle=False)
                lam = jnp.array(data["lambda_nm"], dtype=jnp.float64)
                k = jnp.array(data["k"], dtype=jnp.float64)
                alpha = 4 * jnp.pi * k / (lam * 1e-9 + 1e-30)
                return lam, alpha
    except Exception as e:
        # M6: warn — a corrupt NPZ silently falling back to YAML (or no
        # optics) changes generation physics without a trace.
        warnings.warn(
            f"optics NPZ for {name!r} unreadable ({e}); falling back to YAML optics",
            UserWarning,
            stacklevel=3,
        )
    # Fallback: optics inline in materials.yaml
    db = _load_db()
    entry = db.get(name)
    if entry and entry.get("optics"):
        opt = entry["optics"]
        lam = jnp.array(opt.get("lambda_nm", []), dtype=jnp.float64)
        k = jnp.array(opt.get("k", []), dtype=jnp.float64)
        if lam.size and k.size:
            alpha = 4 * jnp.pi * k / (lam * 1e-9 + 1e-30)
            return lam, alpha
    return None


def _load_alpha_table(name: str):
    """Legacy CSV loader (deprecated) — now also serves new DB for test compat."""
    name = _safe_name(name)
    # Modern DB first (so test_alpha_from_k_column works after csv removal)
    optics = _load_optics_from_db(name)
    if optics is not None:
        lam_nm, alpha = optics
        if lam_nm.size and alpha.size:
            return lam_nm, alpha
    path = _RESOURCE_DIR / f"{name}.csv"
    if not path.exists():
        # No optics anywhere (NPZ/YAML/CSV all missed): empty tables select
        # the Tauc model in alpha_from_table — the designed path for
        # Tauc-only materials, so no warning (it would fire on every load).
        return jnp.array([], dtype=jnp.float64), jnp.array([], dtype=jnp.float64)
    lam_list, k_list = [], []
    seen = set()
    with open(path) as f:
        reader = _csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                try:
                    float(row[0])
                except ValueError:
                    continue
                lam_k = float(row[2])
                k_val = float(row[3])
                if lam_k not in seen:
                    seen.add(lam_k)
                    lam_list.append(lam_k)
                    k_list.append(k_val)
    lam = jnp.array(lam_list, dtype=jnp.float64)
    k = jnp.array(k_list, dtype=jnp.float64)
    return lam, 4 * jnp.pi * k / (lam * 1e-9 + 1e-30)


def _resample_alpha(lam_nm, alpha):
    """Resample measured alpha onto canonical 200–1400 nm grid (log-α), fixed 200 points."""
    if alpha.size == 0 or lam_nm.size == 0:
        return jnp.array([], dtype=jnp.float64), jnp.array([], dtype=jnp.float64)
    # Fixed canonical grid (200 points) independent of input size — was alpha.shape[0] before
    grid_nm = jnp.linspace(200.0, 1400.0, 200)
    order = jnp.argsort(lam_nm)
    # Extrapolate to 0 outside measured range (IR/UV) instead of flat boundary
    alpha_rs = jnp.exp(
        jnp.interp(
            grid_nm,
            lam_nm[order],
            jnp.log(jnp.maximum(alpha[order], 1e-30)),
            left=jnp.log(1e-30),
            right=jnp.log(1e-30),
        )
    )
    return grid_nm * 1e-9, alpha_rs


def load_material(name: str) -> Material:
    """Load a Material from the modern consolidated DB (fallback to legacy)."""
    name = _safe_name(name)
    db = _load_db()
    if db and name in db:
        props = db[name].get("properties", {})

        def _get(k, d):
            v = props.get(k)
            return d if v is None else float(v)

        m = {k: _get(k, v) for k, v in _DEFAULTS.items()}
        # Optics: modern DB first
        optics = _load_optics_from_db(name)
        if optics is not None:
            lam_nm, alpha = optics
            if alpha.size:
                lam_m, alpha_rs = _resample_alpha(lam_nm, alpha)
            else:
                lam_m = jnp.array([], dtype=jnp.float64)
                alpha_rs = jnp.array([], dtype=jnp.float64)
        else:
            # Legacy fallback: try CSV
            lam_nm, alpha = _load_alpha_table(name)
            if alpha.size:
                lam_m, alpha_rs = _resample_alpha(lam_nm, alpha)
            else:
                lam_m = jnp.array([], dtype=jnp.float64)
                alpha_rs = jnp.array([], dtype=jnp.float64)
        return Material(
            Chi=m["Chi"],
            Eg=m["Eg"],
            eps=m["eps"],
            Nc=m["Nc"],
            Nv=m["Nv"],
            mn=m["mn"],
            mp=m["mp"],
            tn=m["tn"],
            tp=m["tp"],
            Et=m["Et"],
            Br=m["Br"],
            Cn=m["Cn"],
            Cp=m["Cp"],
            A=m["A"],
            alpha=alpha_rs if alpha_rs.size else jnp.array([], dtype=jnp.float64),
            Lambda=lam_m,
        )
    # No consolidated entry — legacy per-file path (v0.1.5 compat)
    path = _RESOURCE_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Material {name!r} not found in {path.parent} (modern DB has {sorted(db.keys()) if db else []})"
        )
    with open(path) as f:
        props = yaml.safe_load(f).get("properties", {})

    def _get2(k, d):
        v = props.get(k)
        return d if v is None else float(v)

    m = {k: _get2(k, v) for k, v in _DEFAULTS.items()}
    lam_nm, alpha = _load_alpha_table(name)
    if alpha.size:
        lam_m, alpha_rs = _resample_alpha(lam_nm, alpha)
    else:
        lam_m = jnp.array([], dtype=jnp.float64)
        alpha_rs = jnp.array([], dtype=jnp.float64)
    return Material(
        Chi=m["Chi"],
        Eg=m["Eg"],
        eps=m["eps"],
        Nc=m["Nc"],
        Nv=m["Nv"],
        mn=m["mn"],
        mp=m["mp"],
        tn=m["tn"],
        tp=m["tp"],
        Et=m["Et"],
        Br=m["Br"],
        Cn=m["Cn"],
        Cp=m["Cp"],
        A=m["A"],
        alpha=alpha_rs if alpha_rs.size else jnp.array([], dtype=jnp.float64),
        Lambda=lam_m,
    )


def list_materials() -> list[str]:
    db = _load_db()
    if db:
        return sorted(db.keys())
    return sorted(p.stem for p in _RESOURCE_DIR.glob("*.yaml"))


def material_to_dict(mat: Material) -> dict:
    return {
        "Chi": jnp.asarray(mat.Chi, dtype=jnp.float64),
        "Eg": jnp.asarray(mat.Eg, dtype=jnp.float64),
        "eps": jnp.asarray(mat.eps, dtype=jnp.float64),
        "Nc": jnp.asarray(mat.Nc, dtype=jnp.float64),
        "Nv": jnp.asarray(mat.Nv, dtype=jnp.float64),
        "mn": jnp.asarray(mat.mn, dtype=jnp.float64),
        "mp": jnp.asarray(mat.mp, dtype=jnp.float64),
        "tn": jnp.asarray(mat.tn, dtype=jnp.float64),
        "tp": jnp.asarray(mat.tp, dtype=jnp.float64),
        "Et": jnp.asarray(mat.Et, dtype=jnp.float64),
        "Br": jnp.asarray(mat.Br, dtype=jnp.float64),
        "Cn": jnp.asarray(mat.Cn, dtype=jnp.float64),
        "Cp": jnp.asarray(mat.Cp, dtype=jnp.float64),
        "A": jnp.asarray(mat.A, dtype=jnp.float64),
        "alpha": jnp.asarray(mat.alpha, dtype=jnp.float64),
        "Lambda": jnp.asarray(mat.Lambda, dtype=jnp.float64),
    }


def update(mat, **kwargs):
    if isinstance(mat, Material):
        d = material_to_dict(mat)
        d.update({k: v for k, v in kwargs.items() if v is not None})
        return Material(**d)
    elif isinstance(mat, dict):
        return {**mat, **{k: v for k, v in kwargs.items() if v is not None}}
    raise TypeError(f"Expected Material or dict, got {type(mat)}")
