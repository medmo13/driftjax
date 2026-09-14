"""Test pivoted banded/sparse LU vs Block-Thomas vs Dense for perovskite adjoint"""

import jax.numpy as jnp
import numpy as np
from _helpers import bt_transpose, equilibrate, solve_sparse_lu

import driftjax as dj
from driftjax import BeerLambert, Sweep
from driftjax.numerics.block_thomas import extract_blocks
from driftjax.numerics.residual import F_jacobian
from driftjax.science.contacts import boundary_bias


def test():
    m = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18, mn=130.0, mp=160.0, A=2e4)
    mt = dj.material(Eg=1.6, Chi=3.9, eps=20.0, Nc=1e18, Nv=1e18, mn=100.0, mp=100.0, A=2e4)
    dev_h = dj.Device(
        n_points=15, layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)], Snl=1e7, Snr=0.0, Spl=0.0, Spr=1e7
    )
    dev_p = dj.Device(
        n_points=15,
        layers=[(2e-5, mt, 1e18), (6e-5, m, -1e18), (2e-5, m, 1e18)],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )
    for label, dev in [("Homojunction", dev_h), ("Perovskite 3-layer", dev_p)]:
        print(f"\n=== {label} ===")
        s = dj.simulate(dev, Sweep(n_steps=5), optics=BeerLambert("tauc"))
        cell = s.cell
        for pot, vb in zip(s.potentials, s.voltages, strict=False):
            J = F_jacobian(cell, boundary_bias(cell, vb), pot)
            g = jnp.ones(J.shape[0])
            # Dense
            lam_d = jnp.linalg.solve(J.T, g)
            r_d = (
                float(jnp.linalg.norm(J.T @ lam_d - g) / (jnp.linalg.norm(g) + 1e-30))
                if np.all(np.isfinite(np.array(lam_d)))
                else float("inf")
            )
            # BT
            A, B, C = extract_blocks(J)
            lam_bt = bt_transpose(A, B, C, g)
            r_bt = (
                float(jnp.linalg.norm(J.T @ lam_bt - g) / (jnp.linalg.norm(g) + 1e-30))
                if np.all(np.isfinite(np.array(lam_bt)))
                else float("inf")
            )
            # Sparse pivoted
            lam_sp, r_sp = solve_sparse_lu(J.T, g)
            # Equilibrated sparse
            J_tilde, Dr, Dc = equilibrate(J.T)
            g_tilde = Dr @ np.array(g)
            lam_eq_tilde, r_eq_tilde = solve_sparse_lu(J_tilde, g_tilde)
            if lam_eq_tilde is not None:
                lam_eq = Dc @ np.array(lam_eq_tilde)
                r_eq = (
                    float(
                        np.linalg.norm(np.array(J.T) @ lam_eq - np.array(g))
                        / (np.linalg.norm(g) + 1e-30)
                    )
                    if np.all(np.isfinite(lam_eq))
                    else float("inf")
                )
            else:
                r_eq = float("inf")
            try:
                c = float(np.linalg.cond(np.array(J)))
            except Exception:
                c = float("inf")
            print(
                f"V={vb:.2f} dense r={r_d:.2e} BT r={r_bt:.2e} sparse r={r_sp:.2e} equil-sparse r={r_eq:.2e} cond {c:.2e}"
            )


if __name__ == "__main__":
    test()
