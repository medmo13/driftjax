"""Wavelength sharding across devices (blueprint P3, feat_7).

Generation is a *linear* sum over the wavelength axis in every optics mode
(Beer-Lambert, legacy, TMM): G(x) = Σ_λ g_λ(x).  So the spectrum can be
split into contiguous wavelength shards, each computed (and placed) on its
own device, and the partial generation profiles summed — exactly the
intern/driftjax "multi-device wavelength sharding" pattern (KG v4 feat_7),
ported to the driftjax API.

Correctness contract (pinned, tests/property/test_sharding.py): the sharded
result equals `generation_profile` up to summation order only — a relative
difference ≤ 1e-12 on CPU, bit-exact on identical reduce order.
"""

from __future__ import annotations

from contextlib import nullcontext

import jax
import jax.numpy as jnp

from driftjax.fields import DeviceDesign, LightSource
from driftjax.science.optics import generation_profile


def wavelength_shard_indices(n_lambda: int, n_shards: int) -> list[jax.Array]:
    """Contiguous, near-equal index split of the wavelength axis (host, not traced)."""
    import numpy as np

    n = max(1, min(int(n_shards), int(n_lambda)))
    # Use numpy for host indices (not jnp, which would be traced under jit)
    edges = np.linspace(0, int(n_lambda), n + 1)
    lo = np.floor(edges[:-1]).astype(int)
    hi = np.floor(edges[1:]).astype(int)
    return [jnp.arange(int(lo[i]), int(hi[i])) for i in range(n)]


def shard_light_source(ls: LightSource, n_shards: int = 1, devices=None) -> list[LightSource]:
    """Split `ls` into `n_shards` contiguous wavelength LightSources."""
    idx = wavelength_shard_indices(ls.Lambda.shape[0], n_shards)
    devs = devices or jax.devices()
    out = []
    for i, ix in enumerate(idx):
        dev = devs[i % len(devs)]
        lam = jnp.asarray(ls.Lambda[ix], device=dev)
        p = jnp.asarray(ls.P_in[ix], device=dev)
        out.append(LightSource(Lambda=lam, P_in=p, kind=ls.kind))
    return out


def sharded_generation(
    design: DeviceDesign,
    ls: LightSource,
    n_shards: int = 1,
    alpha_mode: str = "table",
    statistics: str = "boltzmann",
    optics=None,
    devices=None,
) -> tuple[jax.Array, dict]:
    """Generation profile (n_node,) by summing per-shard partial sums.

    Returns (G, meta) where meta records the shard count and per-shard
    contribution magnitudes (audit trail for the P3 benchmark).
    If optics is given, generation uses optics.generation per shard.
    """
    # normalize alpha_mode
    if isinstance(alpha_mode, str):
        alpha_mode = alpha_mode.strip().lower().replace("_", "-")
        if alpha_mode in ("beerlambert", "bl"):
            alpha_mode = "beer-lambert"
    shards = shard_light_source(ls, n_shards, devices)
    devs = devices or jax.devices()
    parts, meta_parts = [], []
    for i, ls_i in enumerate(shards):
        dev = devs[i % len(devs)]
        with jax.default_device(dev) if hasattr(jax, "default_device") else nullcontext():
            if optics is not None:
                g_i = optics.generation(design, ls_i)
            else:
                g_i = generation_profile(design, ls_i, alpha_mode=alpha_mode)
        parts.append(jnp.asarray(g_i, device=dev))
        meta_parts.append(float(jnp.sum(g_i)))
    G = jnp.sum(jnp.stack(parts), axis=0)
    meta = {
        "n_shards": len(shards),
        "devices": [str(d) for d in devs[: len(shards)]],
        "shard_partial_sums": meta_parts,
    }
    return G, meta
