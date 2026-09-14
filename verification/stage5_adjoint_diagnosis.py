#!/usr/bin/env python3
"""Stage 5: Structured-adjoint diagnosis — compare 4 linear solvers on saved Jacobians.

Compares Block-Thomas vs Dense LU on the forward system J x = b,
then Dense LU on J^T λ = b, reports condition numbers, and inspects
the most ill-conditioned case in detail.
"""

import sys
import os
import numpy as np

# Ensure float64 and correct path
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from driftjax.fields import Potentials, PVCell, BoundaryConditions, pot2vec, vec2pot
from driftjax.numerics.residual import comp_F, F_jacobian
from driftjax.numerics.analytic_jacobian import (
    banded_jacobian, dense_from_blocks, blockwise_matvec, blockwise_matvec_transpose,
)
from driftjax.numerics.block_thomas import block_thomas_solve, extract_blocks
from driftjax.simulator import Device, init_cell, equilibrium, _make_design
from driftjax.science.contacts import boundary_eq, boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.units import thermal_scales, density

# ── helpers ──────────────────────────────────────────────────────────────────

def make_device(name, n_points, layers, Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7, T=300.0):
    """Create a driftjax Device."""
    return Device(
        n_points=n_points,
        layers=layers,
        Snl=Snl, Snr=Snr, Spl=Spl, Spr=Spr,
        T=T,
    )

def solve_eq_and_get_pot(cell, bound):
    """Solve equilibrium and return converged potentials."""
    from driftjax.solvers.newton import solve_eq, solve_newton
    from driftjax.solvers.continuation import equilibrium_guess
    phi_ini = equilibrium_guess(cell).phi
    pot_eq = solve_eq(cell, bound, phi_ini)
    pot, _ = solve_newton(cell, bound, pot_eq, tol=1e-12)
    return pot

def solve_at_bias(cell, pot_eq, v_bias_dim):
    """Solve at a bias point starting from equilibrium."""
    from driftjax.solvers.newton import solve_newton
    bound = boundary_bias(cell, v_bias_dim)
    pot, _ = solve_newton(cell, bound, pot_eq, tol=1e-12)
    return pot, bound

def cond_number(J):
    """Condition number via SVD (on CPU, float64)."""
    J_np = np.array(J, dtype=np.float64)
    s = np.linalg.svd(J_np, compute_uv=False)
    return float(s[0] / max(s[-1], 1e-300))

def svd_values(J):
    """Singular values of a matrix."""
    J_np = np.array(J, dtype=np.float64)
    return np.linalg.svd(J_np, compute_uv=False)

def rel_residual(J, x, b):
    """Relative residual ||Jx - b|| / ||b||."""
    return float(jnp.linalg.norm(J @ x - b) / (jnp.linalg.norm(b) + 1e-300))

def rel_error(x_true, x_test):
    """Relative error ||x_true - x_test|| / ||x_true||."""
    nrm = float(jnp.linalg.norm(x_true))
    if nrm < 1e-300:
        return float(jnp.linalg.norm(x_test))
    return float(jnp.linalg.norm(x_true - x_test) / nrm)

# ── device definitions ───────────────────────────────────────────────────────

Si = None  # loaded lazily

def get_Si():
    global Si
    if Si is None:
        from driftjax.io import load_material
        Si = load_material("Si")
    return Si

def get_CdS():
    from driftjax.io import load_material
    return load_material("CdS")

def get_CdTe():
    from driftjax.io import load_material
    return load_material("CdTe")

def build_devices():
    """Build the 5 test devices."""
    si = get_Si()
    cds = get_CdS()
    cdte = get_CdTe()

    devices = {}

    # 1. Simple homojunction N=200
    devices["homo_N200"] = make_device(
        "homo_N200", n_points=200,
        layers=[(1e-4, si, 1e16), (1e-4, si, -1e16)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )

    # 2. Fine homojunction N=500
    devices["homo_N500"] = make_device(
        "homo_N500", n_points=500,
        layers=[(1e-4, si, 1e16), (1e-4, si, -1e16)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )

    # 3. Heterojunction N=500 (CdS/CdTe)
    devices["hetero_N500"] = make_device(
        "hetero_N500", n_points=500,
        layers=[(5e-5, cds, 1e17), (2e-4, cdte, -1e16)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )

    # 4. Highly doped N=500, doping=1e18
    devices["highdop_N500"] = make_device(
        "highdop_N500", n_points=500,
        layers=[(1e-4, si, 1e18), (1e-4, si, -1e18)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )

    # 5. Ill-conditioned: extreme SRV contrast
    devices["illcond_N500"] = make_device(
        "illcond_N500", n_points=500,
        layers=[(1e-4, si, 1e16), (1e-4, si, -1e16)],
        Snl=1e3, Snr=1e9, Spl=1e9, Spr=1e3,
    )

    return devices

# ── main analysis ────────────────────────────────────────────────────────────

def analyze_device(name, dev, v_bias_volt=0.8):
    """Full analysis of one device at equilibrium and bias."""
    sc = thermal_scales(300.0)
    v_bias_dim = v_bias_volt / sc["energy"]

    ls = spectrum(normalize=False)
    cell = init_cell(dev._design, ls, alpha_mode="table", statistics="boltzmann")
    bound_eq = boundary_eq(cell)

    print(f"\n{'='*80}")
    print(f"  DEVICE: {name}  (N={dev.n_points})")
    print(f"{'='*80}")

    # --- Equilibrium ---
    print(f"\n  --- Equilibrium ---")
    pot_eq = solve_eq_and_get_pot(cell, bound_eq)
    x_eq = pot2vec(pot_eq)
    F_eq = comp_F(cell, bound_eq, pot_eq)
    print(f"  ||F_eq|| = {float(jnp.linalg.norm(F_eq)):.6e}")

    J_eq = F_jacobian(cell, bound_eq, pot_eq)
    cond_eq = cond_number(J_eq)
    print(f"  cond(J_eq) = {cond_eq:.4e}")

    b_eq = -F_eq

    # Block-Thomas forward
    A_eq, B_eq, C_eq = banded_jacobian(cell, bound_eq, pot_eq)
    x_bt_eq = block_thomas_solve(A_eq, B_eq, C_eq, b_eq.reshape(-1, 3)).reshape(-1)
    resid_bt_eq = rel_residual(J_eq, x_bt_eq, b_eq)

    # Dense LU forward
    x_dense_eq = jnp.linalg.solve(J_eq, b_eq)
    resid_dense_eq = rel_residual(J_eq, x_dense_eq, b_eq)

    print(f"  Forward solve Jx=b:")
    print(f"    Block-Thomas:  rel_resid = {resid_bt_eq:.6e}")
    print(f"    Dense LU:      rel_resid = {resid_dense_eq:.6e}")
    print(f"    ||x_bt - x_dense||/||x_dense|| = {rel_error(x_dense_eq, x_bt_eq):.6e}")

    # Adjoint: J^T λ = g_x (g_x = random for diagnosis)
    rng_key = jax.random.PRNGKey(42)
    g_x = jax.random.normal(rng_key, shape=b_eq.shape, dtype=jnp.float64)

    # Dense LU on J^T
    x_adj_dense = jnp.linalg.solve(J_eq.T, g_x)
    resid_adj_dense = rel_residual(J_eq.T, x_adj_dense, g_x)
    print(f"  Adjoint solve J^T λ = g:")
    print(f"    Dense LU:      rel_resid = {resid_adj_dense:.6e}")

    # J^T via block structure (transpose of blocks)
    A_T = A_eq.transpose(0, 2, 1)
    B_T = C_eq.transpose(0, 2, 1)  # sub of J^T = super of J transposed
    C_T = B_eq.transpose(0, 2, 1)  # super of J^T = sub of J transposed
    x_adj_bt = block_thomas_solve(A_T, B_T, C_T, g_x.reshape(-1, 3)).reshape(-1)
    resid_adj_bt = rel_residual(J_eq.T, x_adj_bt, g_x)
    print(f"    Block-Thomas:  rel_resid = {resid_adj_bt:.6e}")
    print(f"    ||x_bt_adj - x_dense_adj||/||x_dense_adj|| = {rel_error(x_adj_dense, x_adj_bt):.6e}")

    # Adjoint gradient via jax.grad
    try:
        def eff_fn(design):
            from driftjax.simulator import _make_design as _mkd
            from driftjax.simulate import simulate
            from driftjax.problems import Sweep
            sol = simulate(design, Sweep(vmax=1.0, n_steps=11))
            return sol.efficiency
        grad_eff = jax.grad(eff_fn)(dev._design)
        grad_norm = float(jnp.sqrt(sum(jnp.sum(v**2) for v in jax.tree.leaves(grad_eff) if isinstance(v, jax.Array))))
        print(f"  jax.grad(efficiency).norm = {grad_norm:.6e}")
    except Exception as e:
        print(f"  jax.grad(efficiency) FAILED: {e}")
        grad_norm = float('nan')

    # --- Biased ---
    print(f"\n  --- Bias V={v_bias_volt}V ---")
    pot_b, bound_b = solve_at_bias(cell, pot_eq, v_bias_dim)
    x_b = pot2vec(pot_b)
    F_b = comp_F(cell, bound_b, pot_b)
    print(f"  ||F_bias|| = {float(jnp.linalg.norm(F_b)):.6e}")

    J_b = F_jacobian(cell, bound_b, pot_b)
    cond_b = cond_number(J_b)
    print(f"  cond(J_bias) = {cond_b:.4e}")

    b_b = -F_b

    # Block-Thomas forward
    A_b, B_b, C_b = banded_jacobian(cell, bound_b, pot_b)
    x_bt_b = block_thomas_solve(A_b, B_b, C_b, b_b.reshape(-1, 3)).reshape(-1)
    resid_bt_b = rel_residual(J_b, x_bt_b, b_b)

    # Dense LU forward
    x_dense_b = jnp.linalg.solve(J_b, b_b)
    resid_dense_b = rel_residual(J_b, x_dense_b, b_b)

    print(f"  Forward solve Jx=b:")
    print(f"    Block-Thomas:  rel_resid = {resid_bt_b:.6e}")
    print(f"    Dense LU:      rel_resid = {resid_dense_b:.6e}")
    print(f"    ||x_bt - x_dense||/||x_dense|| = {rel_error(x_dense_b, x_bt_b):.6e}")

    # Adjoint at bias
    rng_key2 = jax.random.PRNGKey(123)
    g_x_b = jax.random.normal(rng_key2, shape=b_b.shape, dtype=jnp.float64)

    x_adj_dense_b = jnp.linalg.solve(J_b.T, g_x_b)
    resid_adj_dense_b = rel_residual(J_b.T, x_adj_dense_b, g_x_b)

    A_T_b = A_b.transpose(0, 2, 1)
    B_T_b = C_b.transpose(0, 2, 1)
    C_T_b = B_b.transpose(0, 2, 1)
    x_adj_bt_b = block_thomas_solve(A_T_b, B_T_b, C_T_b, g_x_b.reshape(-1, 3)).reshape(-1)
    resid_adj_bt_b = rel_residual(J_b.T, x_adj_bt_b, g_x_b)

    print(f"  Adjoint solve J^T λ = g:")
    print(f"    Dense LU:      rel_resid = {resid_adj_dense_b:.6e}")
    print(f"    Block-Thomas:  rel_resid = {resid_adj_bt_b:.6e}")
    print(f"    ||x_bt_adj - x_dense_adj||/||x_dense_adj|| = {rel_error(x_adj_dense_b, x_adj_bt_b):.6e}")

    # Analytic vs AD Jacobian match
    J_analytic_dense = dense_from_blocks(A_b, B_b, C_b)
    jac_diff = float(jnp.max(jnp.abs(J_b - J_analytic_dense)))
    print(f"  Analytic vs AD Jacobian max|diff| = {jac_diff:.6e}")

    return {
        "name": name,
        "N": dev.n_points,
        "cond_eq": cond_eq,
        "cond_bias": cond_b,
        "resid_bt_eq": resid_bt_eq,
        "resid_dense_eq": resid_dense_eq,
        "resid_bt_bias": resid_bt_b,
        "resid_dense_bias": resid_dense_b,
        "resid_adj_bt_bias": resid_adj_bt_b,
        "resid_adj_dense_bias": resid_adj_dense_b,
        "jac_diff": jac_diff,
        "grad_norm": grad_norm,
        "cell": cell,
        "pot_eq": pot_eq,
        "pot_bias": pot_b,
        "bound_bias": bound_b,
        "J_bias": J_b,
        "J_eq": J_eq,
        "A_bias": A_b,
        "B_bias": B_b,
        "C_bias": C_b,
        "b_bias": b_b,
        "g_x_bias": g_x_b,
    }


def deep_dive_illcond(info):
    """Deep analysis of the most ill-conditioned case."""
    print(f"\n{'='*80}")
    print(f"  DEEP DIVE: {info['name']}")
    print(f"{'='*80}")

    J = info["J_bias"]
    J_eq = info["J_eq"]
    A, B, C = info["A_bias"], info["B_bias"], info["C_bias"]
    b = info["b_bias"]
    g = info["g_x_bias"]

    # a. Save J, J^T, b
    save_dir = os.path.join(os.path.dirname(__file__), "stage5_matrices")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "J_bias.npy"), np.array(J))
    np.save(os.path.join(save_dir, "Jt_bias.npy"), np.array(J.T))
    np.save(os.path.join(save_dir, "b_bias.npy"), np.array(b))
    np.save(os.path.join(save_dir, "g_x.npy"), np.array(g))
    print(f"  Saved J, J^T, b, g_x to {save_dir}/")

    # b. Singular values of J and J^T
    sJ = svd_values(J)
    sJt = svd_values(J.T)
    print(f"\n  Singular values of J (top 5):     {sJ[:5]}")
    print(f"  Singular values of J^T (top 5):   {sJt[:5]}")
    print(f"  ||sJ - sJt|| = {np.linalg.norm(sJ - sJt):.6e}  (should be 0)")
    print(f"  cond(J) = {sJ[0]/max(sJ[-1], 1e-300):.4e}")

    # c. Block-Thomas on J^T with transposed blocks
    A_T = A.transpose(0, 2, 1)
    B_T = C.transpose(0, 2, 1)
    C_T = B.transpose(0, 2, 1)

    x_dense_adj = jnp.linalg.solve(J.T, g)
    x_bt_adj = block_thomas_solve(A_T, B_T, C_T, g.reshape(-1, 3)).reshape(-1)

    resid_dense_adj = rel_residual(J.T, x_dense_adj, g)
    resid_bt_adj = rel_residual(J.T, x_bt_adj, g)

    print(f"\n  Block-Thomas on J^T (transposed blocks):")
    print(f"    Dense LU residual:   {resid_dense_adj:.6e}")
    print(f"    BT residual:         {resid_bt_adj:.6e}")
    print(f"    ||x_bt - x_dense||/||x_dense|| = {rel_error(x_dense_adj, x_bt_adj):.6e}")

    # Also verify: J^T @ x_bt_adj ≈ g ?
    print(f"    ||J^T @ x_bt - g|| = {float(jnp.linalg.norm(J.T @ x_bt_adj - g)):.6e}")

    # d. Analytic vs AD Jacobian
    J_analytic_dense = dense_from_blocks(A, B, C)
    jac_diff_max = float(jnp.max(jnp.abs(J - J_analytic_dense)))
    jac_diff_norm = float(jnp.linalg.norm(J - J_analytic_dense))
    jac_norm = float(jnp.linalg.norm(J))
    print(f"\n  Analytic vs AD Jacobian:")
    print(f"    max|diff| = {jac_diff_max:.6e}")
    print(f"    ||diff||  = {jac_diff_norm:.6e}")
    print(f"    ||J||     = {jac_norm:.6e}")
    print(f"    relative  = {jac_diff_norm / max(jac_norm, 1e-300):.6e}")

    # e. Scaling: diagonal preconditioning with J + ε·I
    print(f"\n  Regularization analysis (J + ε·I):")
    kappa = sJ[0] / max(sJ[-1], 1e-300)
    for eps_scale in [0, 1e-2, 1e-1, 1e0, 1e1]:
        eps = eps_scale * sJ[-1]  # scale by smallest SV
        J_reg = J + eps * jnp.eye(J.shape[0])
        cond_reg = cond_number(J_reg)
        x_d = jnp.linalg.solve(J_reg, b)
        A_r, B_r, C_r = extract_blocks(J_reg)
        x_t = block_thomas_solve(A_r, B_r, C_r, b.reshape(-1, 3)).reshape(-1)
        rd = rel_residual(J_reg, x_d, b)
        rt = rel_residual(J_reg, x_t, b)
        re = rel_error(x_d, x_t)
        print(f"    ε={eps_scale:.2e}·σ_min: cond={cond_reg:.4e}  Dense_resid={rd:.4e}  BT_resid={rt:.4e}  rel_err={re:.4e}")

    # f. Row-equilibration check
    print(f"\n  Row-equilibration analysis:")
    row_max = jnp.max(jnp.abs(J), axis=1)
    print(f"    min row max|J_i*| = {float(jnp.min(row_max)):.4e}")
    print(f"    max row max|J_i*| = {float(jnp.max(row_max)):.4e}")
    print(f"    ratio max/min     = {float(jnp.max(row_max)/jnp.min(row_max)):.4e}")

    D = jnp.diag(1.0 / jnp.maximum(row_max, 1e-300))
    J_eq = D @ J
    b_eq = D @ b
    cond_eq_j = cond_number(J_eq)
    x_d_eq = jnp.linalg.solve(J_eq, b_eq)
    A_eq, B_eq, C_eq = extract_blocks(J_eq)
    x_t_eq = block_thomas_solve(A_eq, B_eq, C_eq, b_eq.reshape(-1, 3)).reshape(-1)
    print(f"    After row-eq: cond = {cond_eq_j:.4e}")
    print(f"    Dense resid: {rel_residual(J_eq, x_d_eq, b_eq):.4e}")
    print(f"    BT resid:    {rel_residual(J_eq, x_t_eq, b_eq):.4e}")
    print(f"    BT rel_err:  {rel_error(x_d_eq, x_t_eq):.4e}")

    # Summary of determinant checks on blocks
    a_det, b_det, c_det = A[..., 0, 0], A[..., 0, 1], A[..., 0, 2]
    d_det, e_det, f_det = A[..., 1, 0], A[..., 1, 1], A[..., 1, 2]
    g_det, h_det, i_det = A[..., 2, 0], A[..., 2, 1], A[..., 2, 2]
    det = a_det * (e_det * i_det - f_det * h_det) - b_det * (d_det * i_det - f_det * g_det) + c_det * (d_det * h_det - e_det * g_det)
    print(f"\n  Block determinants:")
    print(f"    min|det(A_i)| = {float(jnp.min(jnp.abs(det))):.6e}")
    print(f"    max|det(A_i)| = {float(jnp.max(jnp.abs(det))):.6e}")
    print(f"    mean|det(A_i)| = {float(jnp.mean(jnp.abs(det))):.6e}")
    n_singular = int(jnp.sum(jnp.abs(det) < 1e-30))
    print(f"    blocks with |det| < 1e-30: {n_singular}/{len(det)}")


def main():
    print("="*80)
    print("  STAGE 5: STRUCTURED-ADJOINT DIAGNOSIS")
    print("  DriftJax v0.1.15 — Jacobian solver comparison")
    print("="*80)

    devices = build_devices()
    all_results = {}

    for name, dev in devices.items():
        try:
            info = analyze_device(name, dev)
            all_results[name] = info
        except Exception as e:
            print(f"\n  *** FAILED on {name}: {e}")
            import traceback
            traceback.print_exc()

    # Deep dive: both most ill-conditioned AND worst adjoint
    if all_results:
        worst_cond = max(all_results.items(), key=lambda kv: kv[1]["cond_bias"])
        worst_adj = max(all_results.items(), key=lambda kv: max(kv[1]["resid_adj_bt_bias"], kv[1]["resid_adj_dense_bias"]))
        print(f"\n{'#'*80}")
        print(f"  Most ill-conditioned (by cond): {worst_cond[0]} (cond = {worst_cond[1]['cond_bias']:.4e})")
        print(f"{'#'*80}")
        deep_dive_illcond(worst_cond[1])

        if worst_adj[0] != worst_cond[0]:
            print(f"\n{'#'*80}")
            print(f"  Worst adjoint accuracy: {worst_adj[0]} (adj_resid = {worst_adj[1]['resid_adj_dense_bias']:.4e})")
            print(f"{'#'*80}")
            deep_dive_illcond(worst_adj[1])

    # Summary table
    print(f"\n{'='*80}")
    print("  SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Device':<20} {'N':>4} {'cond_eq':>12} {'cond_bias':>12} {'res_BT_eq':>12} {'res_D_eq':>12} {'res_BT_b':>12} {'res_D_b':>12} {'adj_BT':>12} {'adj_D':>12} {'jac_diff':>12}")
    print("-"*148)
    for name, r in all_results.items():
        print(f"{name:<20} {r['N']:>4} {r['cond_eq']:>12.4e} {r['cond_bias']:>12.4e} "
              f"{r['resid_bt_eq']:>12.4e} {r['resid_dense_eq']:>12.4e} "
              f"{r['resid_bt_bias']:>12.4e} {r['resid_dense_bias']:>12.4e} "
              f"{r['resid_adj_bt_bias']:>12.4e} {r['resid_adj_dense_bias']:>12.4e} "
              f"{r['jac_diff']:>12.4e}")

    print("\n  Done.")

if __name__ == "__main__":
    main()
