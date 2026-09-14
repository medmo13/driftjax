"""Tier IV — V12: Adjoint consistency verification.

Tests that the dense adjoint solve is correct (residual < tolerance).
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp


def test_adjoint_consistency() -> dict:
    """Verify adjoint equation residual with dense solve."""
    import driftjax as dj
    from driftjax.fields import pot2vec, vec2pot
    from driftjax.numerics.residual import F_jacobian
    from driftjax.numerics.scharfetter_gummel import Jn, Jp
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.units import energy

    ls = spectrum()
    mat = dj.material(Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19,
                       mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4)
    dev = dj.Device(n_points=50, layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)],
                     Snl=1e7, Snr=0, Spl=0, Spr=1e7)
    cell = init_cell(dev.design(), ls)
    bound = boundary_bias(cell, 0.4 / energy)
    pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)
    pot, _ = solve_newton(cell, bound, pot_eq, tol=1e-10)

    def objective(pot_vec):
        p = vec2pot(pot_vec)
        return jnp.sum(Jn(cell, p) + Jp(cell, p))

    g_vec = pot2vec(pot)
    g_obj = jax.grad(objective)(g_vec)

    J_dense = F_jacobian(cell, bound, pot)
    lam_dense = jnp.linalg.solve(J_dense.T, g_obj)
    residual_dense = float(jnp.linalg.norm(J_dense.T @ lam_dense - g_obj) / (jnp.linalg.norm(g_obj) + 1e-30))

    return {
        "test": "V12_adjoint_consistency",
        "adjoint_residual_dense": residual_dense,
        "passed": residual_dense < 1e-6,
    }


def save_record(result: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)


def print_summary(result: dict) -> None:
    print(f"\n{'='*60}")
    print("TIER IV — V12: Adjoint Consistency")
    print(f"{'='*60}")
    print(f"  Adjoint residual (dense):   {result['adjoint_residual_dense']:.2e}")
    print(f"  Status: {'PASS' if result['passed'] else 'FAIL'}")
    print()


if __name__ == "__main__":
    r = test_adjoint_consistency()
    print_summary(r)
