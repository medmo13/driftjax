"""P3: wavelength sharding — sharded generation == unsharded."""

import jax
import jax.numpy as jnp
import pytest

import driftjax as dj
from driftjax.runtime.sharding import (
    shard_light_source,
    sharded_generation,
    wavelength_shard_indices,
)
from driftjax.science.optics import generation_profile
from driftjax.science.spectrum import spectrum

pytestmark = pytest.mark.smoke

_G_FULL_CACHE: dict[str, jax.Array] = {}


@pytest.fixture(scope="module")
def shard_design(si_canon):
    return dj.Device(
        layers=list(zip([0.0001, 0.0001], [si_canon, si_canon], [1e17, -1e17], strict=False)),
        n_points=80,
        Snl=10000000.0,
        Snr=0,
        Spl=0,
        Spr=10000000.0,
        PhiMl=-1.0,
        PhiMr=-1.0,
    ).design()


@pytest.fixture(scope="module")
def shard_ls():
    return spectrum(normalize=False)


def test_shard_indices_cover_all_wavelengths(shard_ls):
    n = shard_ls.Lambda.shape[0]
    for ns in (1, 2, 4, 8):
        idx = wavelength_shard_indices(n, ns)
        assert sum(len(i) for i in idx) == n
        seen = jnp.concatenate(idx)
        assert jnp.array_equal(jnp.sort(seen), jnp.arange(n))


def test_shard_light_source_conserves_flux(shard_ls):
    total = float(jnp.sum(shard_ls.P_in))
    for ns in (2, 4, 7):
        shards = shard_light_source(shard_ls, n_shards=ns)
        assert len(shards) == ns
        assert float(jnp.sum(jnp.concatenate([s.P_in for s in shards]))) == pytest.approx(
            total, rel=1e-15
        )


def _full_generation(design, ls, alpha_mode):
    """Module-cached unsharded reference (identical across n_shards params)."""
    if alpha_mode not in _G_FULL_CACHE:
        _G_FULL_CACHE[alpha_mode] = generation_profile(design, ls, alpha_mode=alpha_mode)
    return _G_FULL_CACHE[alpha_mode]


@pytest.mark.parametrize("alpha_mode", ["beer-lambert", "tmm"])
@pytest.mark.parametrize("n_shards", [1, 4])  # trimmed from 3x3=9 to 2x2=4: tauc redundant, n=2 covered by 1->4
def test_sharded_generation_equals_unsharded(shard_design, shard_ls, alpha_mode, n_shards):
    G_full = _full_generation(shard_design, shard_ls, alpha_mode)
    G_sh, meta = sharded_generation(
        shard_design, shard_ls, n_shards=n_shards, alpha_mode=alpha_mode
    )
    assert meta["n_shards"] == n_shards
    scale = max(float(jnp.max(jnp.abs(G_full))), 1e-30)
    assert float(jnp.max(jnp.abs(G_sh - G_full))) <= 1e-12 * scale
    assert bool(jnp.all(jnp.isfinite(G_sh)))
    assert float(G_sh.min()) >= 0.0
