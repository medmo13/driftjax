"""Method of manufactured solutions: FV Poisson truncation is O(h²)."""

import jax.numpy as jnp

from driftjax.validation.manufactured import poisson_mms_residual


def test_poisson_mms_second_order():
    errs = [float(jnp.max(jnp.abs(poisson_mms_residual(n, k=8.0)))) for n in (40, 80, 160, 320)]
    p = [float(jnp.log(errs[i] / errs[i + 1]) / jnp.log(2.0)) for i in range(3)]
    assert all(pp > 1.8 for pp in p), (errs, p)


def test_poisson_mms_higher_k_larger_error():
    e_lo = float(jnp.max(jnp.abs(poisson_mms_residual(80, k=4.0))))
    e_hi = float(jnp.max(jnp.abs(poisson_mms_residual(80, k=16.0))))
    assert e_hi > e_lo


def test_poisson_mms_vanish_low_k():
    r = poisson_mms_residual(200, k=0.1)
    assert float(jnp.max(jnp.abs(r))) < 1e-08
