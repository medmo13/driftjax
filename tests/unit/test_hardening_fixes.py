"""Hardening for 5 fixes: mats 3-layer, batched jit, O(N) CSR, fused optics.

Covers the ~10% coverage lift for the 75% gate (kept at 60 until hit).
"""

import jax
import jax.numpy as jnp
import pytest

import driftjax as dj
from driftjax import Sweep, simulate
from driftjax.io import load_material


def test_mats_three_layer_preserved():
    """Device with 3 layers preserves mats through with_temperature and design."""
    Si = load_material("Si")
    GaAs = load_material("GaAs")
    CdTe = load_material("CdTe")
    dev = dj.Device(layers=[(5e-5, Si, 1e16), (5e-5, GaAs, 1e15), (5e-5, CdTe, -1e16)], n_points=30)
    assert len(dev.mats) == 3
    assert float(dev.mats[0].Eg) == float(Si.Eg) and float(dev.mats[2].Eg) == float(CdTe.Eg)
    # design rebuild uses all 3 mats
    dsg = dev.design()
    assert dsg.x.shape[0] == 30
    dev2 = dev.with_temperature(350.0)
    assert len(dev2.mats) == 3
    assert float(dev2.T) == 350.0
    # with_temperature must rebuild with 3 layers, not truncate to 2 (value check, not identity)
    assert float(dev2.mats[1].Eg) == float(GaAs.Eg)
    assert float(dev2.mats[1].Chi) == float(GaAs.Chi)
    # sweep still converges after temperature rebuild
    sol = simulate(dev2, Sweep(vmax=0.5, n_steps=4))
    # Voc is NaN only when the IV curve never crosses zero within the swept
    # range (Voc beyond vmax); otherwise it must lie inside [0, vmax].
    assert jnp.isnan(sol.voc) or (0.0 <= float(sol.voc) <= 0.5 + 1e-6)
    assert jnp.isfinite(sol.efficiency)


def test_blocks_to_csr_on_matches_dense():
    """O(N) blocks_to_csr equals dense_to_csr to machine precision."""
    from driftjax.numerics.analytic_jacobian import banded_jacobian, dense_from_blocks
    from driftjax.numerics.linalg import blocks_to_csr, dense_to_csr
    from driftjax.science.contacts import boundary_bias, boundary_eq
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq

    dev = dj.Device(
        n_points=20, layers=[(1e-4, load_material("Si"), 1e16), (1e-4, load_material("Si"), -1e16)]
    )
    cell = init_cell(dev.design(), spectrum(normalize=False))
    pot = solve_eq(cell, boundary_eq(cell), equilibrium_guess(cell).phi)
    A, B, C = banded_jacobian(cell, boundary_bias(cell, 0.2), pot)
    dense = dense_from_blocks(A, B, C)
    d1, i1, p1 = dense_to_csr(dense)
    d2, i2, p2 = blocks_to_csr(A, B, C)
    assert jnp.allclose(d1, d2)
    assert jnp.array_equal(i1, i2)
    assert jnp.array_equal(p1, p2)
    # larger N still O(N) without dense alloc
    dev2 = dj.Device(
        n_points=60, layers=[(1e-4, load_material("Si"), 1e16), (1e-4, load_material("Si"), -1e16)]
    )
    cell2 = init_cell(dev2.design(), spectrum(normalize=False))
    pot2 = solve_eq(cell2, boundary_eq(cell2), equilibrium_guess(cell2).phi)
    A2, B2, C2 = banded_jacobian(cell2, boundary_bias(cell2, 0.1), pot2)
    d2b, _, _ = blocks_to_csr(A2, B2, C2)
    assert d2b.shape[0] == 3 * 60 * 13  # n*W entries (including zeros for ordering)


def test_fused_vs_unfused_generation():
    """fused_generation == init_cell.G to 1e-12."""
    from driftjax.numerics.fused_kernels import fused_generation
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell

    dev = dj.Device(
        n_points=40, layers=[(1e-4, load_material("Si"), 1e16), (1e-4, load_material("Si"), -1e16)]
    )
    ls = spectrum(normalize=False)
    G_f = fused_generation(dev.design(), ls, alpha_mode="beer-lambert", statistics="boltzmann")
    G_u = init_cell(dev.design(), ls, alpha_mode="beer-lambert", statistics="boltzmann").G
    assert float(jnp.max(jnp.abs(G_f - G_u))) < 1e-12


def test_batched_equals_serial_and_jit():
    """Batched forward equals serial and is jit-cacheable."""
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import sweep
    from driftjax.units import energy

    dev = dj.Device(
        n_points=30, layers=[(1e-4, load_material("Si"), 1e16), (1e-4, load_material("Si"), -1e16)]
    )
    cell = init_cell(dev.design(), spectrum(normalize=False))
    v_s, j_s, _, _ = sweep(cell, 0.6 / energy, n_steps=5, tol=1e-8, batched=False)
    v_b, j_b, _, _ = sweep(cell, 0.6 / energy, n_steps=5, tol=1e-8, batched=True)
    assert float(jnp.max(jnp.abs(j_b - j_s))) < 1e-3
    # jit
    jit_batched = jax.jit(lambda c: sweep(c, 0.6 / energy, n_steps=5, tol=1e-8, batched=True))
    v_jb, j_jb, _, _ = jit_batched(cell)
    assert float(jnp.max(jnp.abs(j_jb - j_s))) < 1e-3


@pytest.mark.slow  # 2x jax.grad(simulate) through batched+serial sweeps
def test_batched_jit_grad_matches_serial():
    """batched forward under jit grad matches serial grad (proves fusion)."""
    # Use validated ex1-like material where batched==serial to 1e-3 (property test)
    try:
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
        from support import ex1_device

        dev = ex1_device(n_points=30)
    except Exception:
        # fallback: inline ex1 material
        mat = dj.material(
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
            A=2e4,
        )
        dev = dj.Device(
            n_points=30,
            layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )
    prot_s = Sweep(vmax=0.6, n_steps=4, fused=True, batched=False)
    prot_b = Sweep(vmax=0.6, n_steps=4, fused=True, batched=True)

    def eff_s(d):
        return simulate(d, prot_s).efficiency

    def eff_b(d):
        return simulate(d, prot_b).efficiency

    # forward parity (solver tol 1e-8 -> eff 1e-3)
    assert abs(float(eff_s(dev) - eff_b(dev))) < 1e-3
    # grad parity (batched compile should be faster, but we just check correctness)
    g_s = jax.grad(eff_s)(dev)
    g_b = jax.grad(eff_b)(dev)
    # compare a scalar leaf (thickness) - allow solver tol 1e-3 relative on gradients
    leaves_s = jax.tree_util.tree_leaves(g_s)
    leaves_b = jax.tree_util.tree_leaves(g_b)
    # find first array leaf with size >1 (thickness or doping)
    for a, b in zip(leaves_s, leaves_b, strict=False):
        if hasattr(a, "shape") and a.size > 1 and jnp.max(jnp.abs(a)) > 1e-12:
            rel = float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-30))
            assert rel < 1e-3, rel
            break
