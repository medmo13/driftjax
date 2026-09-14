"""N=120 perovskite harness - 6 solvers on real optimization device"""

# Import perov device maker from validation/psc
import importlib.util
import pathlib
import time

import jax.numpy as jnp
import numpy as np
from _helpers import bt_transpose, equilibrate, iterative_refinement, solve_sparse_lu

import driftjax as dj
from driftjax import BeerLambert, Sweep
from driftjax.numerics.block_thomas import extract_blocks
from driftjax.numerics.residual import F_jacobian
from driftjax.science.contacts import boundary_bias

spec = importlib.util.spec_from_file_location("psc", str(pathlib.Path(__file__).parent / "psc.py"))
psc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psc)
N, NS = 120, 20
X0 = psc.X0
dev = psc.x2des(jnp.array(X0))
print(f"Perovskite device: N={N}, X0 PCE start will be computed")

# Get cell and potentials via simulate at V sweep
s = dj.simulate(dev, Sweep(n_steps=5, vmax=1.0), optics=BeerLambert())
cell = s.cell
print(f"Simulated {len(s.potentials)} bias points, voltages {np.array(s.voltages)}")

# Test at each bias
for idx, (pot, vb) in enumerate(zip(s.potentials, s.voltages, strict=False)):
    J = F_jacobian(cell, boundary_bias(cell, vb), pot)
    g = jnp.ones(J.shape[0])
    print(f"\n--- V={vb:.3f} (idx {idx}) n={J.shape[0] // 3} cond~", end=" ")
    try:
        c = float(np.linalg.cond(np.array(J)))
        print(f"{c:.2e} ---")
    except Exception:
        print("inf ---")
    # Dense
    t0 = time.time()
    lam_d = jnp.linalg.solve(J.T, g)
    t_d = time.time() - t0
    r_d = (
        float(jnp.linalg.norm(J.T @ lam_d - g) / (jnp.linalg.norm(g) + 1e-30))
        if np.all(np.isfinite(np.array(lam_d)))
        else float("inf")
    )
    print(f"  Dense        r={r_d:.2e} t={t_d * 1000:.1f}ms")
    # BT
    A, B, C = extract_blocks(J)
    t0 = time.time()
    lam_bt = bt_transpose(A, B, C, g)
    t_bt = time.time() - t0
    r_bt = (
        float(jnp.linalg.norm(J.T @ lam_bt - g) / (jnp.linalg.norm(g) + 1e-30))
        if np.all(np.isfinite(np.array(lam_bt)))
        else float("inf")
    )
    print(f"  BT           r={r_bt:.2e} t={t_bt * 1000:.1f}ms")
    # Scaled BT
    J_tilde, Dr, Dc = equilibrate(J.T)
    g_tilde = Dr @ np.array(g)
    # Need to solve scaled BT: J_tilde^T? Actually for equilibration, solve Dr*J*Dc * y = Dr*g, then lam = Dc*y, where y solves J_tilde^T? Wait J is original, JT is J.T, we equilibrate JT
    # For JT, equilibration: JT_tilde = Dr @ JT @ Dc, solve JT_tilde @ y = Dr @ g, then lam = Dc @ y, but using BT on JT_tilde's blocks?
    # For simplicity, test scaled dense
    # Scaled BT via scaled blocks (approx)
    # We skip detailed scaled BT for now, just test scaled sparse
    # Pivoted sparse (SuperLU)
    t0 = time.time()
    lam_sp, r_sp = solve_sparse_lu(J.T, g)
    t_sp = time.time() - t0
    print(f"  Sparse LU    r={r_sp:.2e} t={t_sp * 1000:.1f}ms")
    # Scaled sparse
    J_T_tilde, Dr2, Dc2 = equilibrate(J.T)
    t0 = time.time()
    lam_sp_s, r_sp_s = solve_sparse_lu(J_T_tilde, Dr2 @ np.array(g))
    if lam_sp_s is not None:
        lam_sp_s_unscaled = Dc2 @ np.array(lam_sp_s)
        r_sp_s_unscaled = float(
            np.linalg.norm(np.array(J.T) @ lam_sp_s_unscaled - np.array(g))
            / (np.linalg.norm(g) + 1e-30)
        )
    else:
        r_sp_s_unscaled = float("inf")
    t_sp_s = time.time() - t0
    print(f"  Scaled Sparse r={r_sp_s_unscaled:.2e} t={t_sp_s * 1000:.1f}ms")
    # Banded + IR
    if lam_bt is not None and np.all(np.isfinite(np.array(lam_bt))):
        lam_ir, r_ir = iterative_refinement(J.T, g, lam_bt, iters=2)
        print(f"  BT+IR(2)     r={r_ir:.2e}")
    else:
        print("  BT+IR(2)     r=inf (BT failed)")

print("\nDone")
