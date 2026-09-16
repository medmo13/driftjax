"""Pivoted banded solver (LAPACK dgbsv) == dense solve on synthetic systems."""

import jax.numpy as jnp
import numpy as np

from driftjax.numerics.banded_solve import (
    _blocks_to_lapack_banded,
    banded_solve,
    banded_solve_with_info,
)


def _synthetic_blocks(key_seed=0, n=8):
    rng = np.random.default_rng(key_seed)
    A = jnp.asarray(rng.normal(size=(n, 3, 3)))
    B = jnp.asarray(rng.normal(size=(n - 1, 3, 3)) * 0.1)
    C = jnp.asarray(rng.normal(size=(n - 1, 3, 3)) * 0.1)
    # Diagonally dominant => nonsingular.
    A = A + 6.0 * jnp.eye(3)[None, :, :]
    b = jnp.asarray(rng.normal(size=(n, 3)))
    return A, B, C, b


def _dense_from(A, B, C):
    from driftjax.numerics.analytic_jacobian import dense_from_blocks

    return dense_from_blocks(A, B, C)


def test_banded_matches_dense():
    A, B, C, b = _synthetic_blocks()
    x = banded_solve(A, B, C, b).reshape(-1)
    J = _dense_from(A, B, C)
    xd = jnp.linalg.solve(J, b.reshape(-1))
    assert float(jnp.max(jnp.abs(x - xd))) < 1e-10


def test_banded_residual_small():
    A, B, C, b = _synthetic_blocks(key_seed=1, n=30)
    x = banded_solve(A, B, C, b).reshape(-1)
    J = _dense_from(A, B, C)
    rel = float(jnp.linalg.norm(J @ x - b.reshape(-1)) / (jnp.linalg.norm(b) + 1e-30))
    assert rel < 1e-12


def test_jvp_through_banded_solve_raises_loudly():
    """Forward Newton steps run through a host callback and are opaque to AD.

    Gradients must flow via the custom_vjp implicit-adjoint path, never by
    differentiating through banded_solve.  This test pins the loud failure
    mode: jvp must raise, never silently return wrong tangents.
    """
    import jax

    A, B, C, b = _synthetic_blocks(n=4)
    with __import__("pytest").raises(ValueError, match="[Pp]ure callback"):
        jax.jvp(lambda A_: banded_solve(A_, B, C, b), (A,), (jnp.ones_like(A),))


def test_banded_transpose_matches_dense():
    """banded_transpose solves J^T·x=g to dense accuracy on exact structure."""
    from driftjax.numerics.banded_solve import banded_transpose

    A, B, C, b = _synthetic_blocks(key_seed=3, n=30)
    x = banded_transpose(A, B, C, b.reshape(-1))
    J = _dense_from(A, B, C)
    rel = float(jnp.linalg.norm(J.T @ x - b.reshape(-1)) / (jnp.linalg.norm(b) + 1e-30))
    assert rel < 1e-12


def test_lapack_banded_layout_matches_dense():
    """_blocks_to_lapack_banded must reproduce the dense matrix entries."""
    A, B, C, _ = _synthetic_blocks(key_seed=2, n=5)
    ab, kl, ku = _blocks_to_lapack_banded(A, B, C)
    J = np.asarray(_dense_from(A, B, C))
    ab_np = np.asarray(ab)
    N = J.shape[0]
    for j in range(N):
        for i in range(max(0, j - ku), min(N, j + kl + 1)):
            assert ab_np[ku + i - j, j] == J[i, j]


def test_with_info_clean_path_reports_no_fallback():
    """R2: well-conditioned systems must report used_lstsq=False."""
    A, B, C, b = _synthetic_blocks(key_seed=5, n=8)
    x, info = banded_solve_with_info(A, B, C, b)
    assert bool(info["used_lstsq"]) is False
    assert bool(info["used_zeros"]) is False
    J = _dense_from(A, B, C)
    rel = float(jnp.linalg.norm(J @ x.reshape(-1) - b.reshape(-1)) / (jnp.linalg.norm(b) + 1e-30))
    assert rel < 1e-12


def test_nan_storage_returns_zeros_and_flags():
    """N2: non-finite banded storage must return zeros AND flag used_zeros.

    The zero step alone would satisfy a step-norm gate with huge ||F||, so
    the flag (not the values) is what lets callers detect the failure: the
    Newton loops OR it into last_stats["lstsq"]/rebound handling and the
    post-hoc audit in simulate() is the certified detector. This test pins
    both halves of that contract.
    """
    A, B, C, b = _synthetic_blocks(key_seed=6, n=6)
    A = A.at[2].set(jnp.full((3, 3), jnp.nan))
    x, info = banded_solve_with_info(A, B, C, b)
    assert bool(info["used_zeros"]) is True
    assert float(jnp.max(jnp.abs(x))) == 0.0
    # Legacy entry point preserves the zeros behavior (callers detect via
    # the residual audit, never via the values themselves).
    x_legacy = banded_solve(A, B, C, b)
    assert float(jnp.max(jnp.abs(x_legacy))) == 0.0


def test_failed_linear_solve_never_certifies():
    """P0-solver: a zeros/nonfinite banded step must NEVER satisfy the
    step-norm gate directly. Mechanism (pinned here): a zero step yields
    relative linear residual EXACTLY 1.0 (||F||/||F|| for F != 0), which
    always trips the ``use_dense`` gate (1.0 > 1e-4), so the dense fallback
    engages; if dense also fails the step is NaN and the NaN/best-iterate
    machinery handles it. The only zeros-kept case is F == 0 (already at
    the root). Forces the zeros path by stubbing banded_solve_with_info,
    then checks (a) the kept step reroutes to dense rescue (backend 2),
    never a direct banded certification, and (b) the full solve either
    converges legitimately (small residual) or reports converged=False.
    """
    import driftjax.numerics.banded_solve as _bs

    A0, B0, C0, b0 = _synthetic_blocks(key_seed=7, n=6)
    real = _bs.banded_solve_with_info

    def _stub(A, B, C, b):
        x, info = real(A, B, C, b)
        return jnp.zeros_like(x), {"used_lstsq": False, "used_zeros": jnp.array(True)}

    _bs.banded_solve_with_info = _stub
    try:
        import driftjax as dj
        from driftjax.science.contacts import boundary_bias
        from driftjax.science.spectrum import spectrum as _spec
        from driftjax.simulator import init_cell

        mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
        dev = dj.Device(
            n_points=8,
            layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
            Snl=1e7,
            Snr=0.0,
            Spl=0.0,
            Spr=1e7,
        )
        cell = init_cell(dev.design(), _spec(normalize=False))
        bound = boundary_bias(cell, 0.1)
        from driftjax.solvers.continuation import equilibrium_guess
        from driftjax.solvers.newton import solve_eq, solve_newton

        pot0 = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
        # (a) direct impl call (no jit cache): stubbed zeros on a
        # NON-converged state -> dense rescue (backend code 2), finite
        # error, never a banded certification of a zero step.
        from driftjax.fields import pot2vec
        from driftjax.solvers.newton import _step_newton_impl

        out = _step_newton_impl(cell, bound, pot2vec(pot0), False, False, True, False)
        _xn, _xe, _xr, _xrf, _xl, _xbc, _xls = out
        assert int(_xbc) == 2  # dense rescue engaged
        assert float(_xe) < float("inf")
        # (b) full solve with stub: converges legitimately via dense rescue
        # or reports honestly; either way no false certification with huge
        # ||F||.
        pot, last = solve_newton(cell, bound, pot0, max_steps=50)
        from driftjax.numerics.residual import comp_F

        r_final = float(jnp.max(jnp.abs(comp_F(cell, bound, pot))))
        if bool(last.get("converged", False)):
            assert r_final < 1e-6, (r_final, last)
    finally:
        _bs.banded_solve_with_info = real
