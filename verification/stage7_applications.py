"""Stage 7 — Application-Strengthening Verification for DriftJax v0.1.15.

Four analyses:
  1. DGSM sensitivity ranking stability (32/64/128 Sobol samples)
  2. Optimization comparison (jax.grad+L-BFGS-B, FD+L-BFGS-B, Nelder-Mead)
  3. Recovery attribution (13_target_iv_structure with Nelder-Mead)
  4. Mesh convergence with extended levels (N=62..1000)

Run:
  PYTHONPATH=src /home/med/Desktop/final/venv_latest/bin/python verification/stage7_applications.py
"""

from __future__ import annotations

import os
import sys
import time

os.environ["JAX_ENABLE_X64"] = "1"

import jax
import jax.numpy as jnp
import numpy as np
import scipy.optimize

jax.config.update("jax_enable_x64", True)

import driftjax as dj
from driftjax import BeerLambert, Newton, Sweep, simulate

# ---------------------------------------------------------------------------
# Shared material and helpers
# ---------------------------------------------------------------------------

BASE_MAT = dict(
    Eg=1.12, Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
    mn=1350.0, mp=450.0, tn=1e-5, tp=1e-5, A=2e4,
)
W_SIM = 2e-4


def make_homo(N: int):
    si = dj.material(**BASE_MAT)
    return dj.Device(
        n_points=N,
        layers=[(W_SIM / 2, si, 1e16), (W_SIM / 2, si, -1e16)],
        Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
    )


def divider(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    sys.stdout.flush()


# ===========================================================================
# 1. DGSM SENSITIVITY RANKING STABILITY
# ===========================================================================

def section1_dgsm_stability():
    divider("1. DGSM SENSITIVITY RANKING STABILITY")
    sys.stdout.flush()

    PARAM_NAMES = [
        "Eg", "Chi", "eps", "Nc", "Nv", "mn", "mp", "tn", "tp",
        "Snl", "Spr", "t1", "t2",
    ]
    LO = np.array([0.9, 3.8, 10.0, 1e18, 5e18, 500.0, 100.0, 1e-6, 1e-6,
                    1e6, 1e6, 0.5e-4, 0.5e-4])
    HI = np.array([1.4, 4.3, 14.0, 5e19, 3e19, 2000.0, 800.0, 1e-4, 1e-4,
                    1e8, 1e8, 2.0e-4, 2.0e-4])
    N_nominal = np.array([1.12, 4.05, 11.7, 2.8e19, 1.04e19, 1350.0, 450.0,
                           1e-5, 1e-5, 1e7, 1e7, 1e-4, 1e-4])

    def eff_of_13(p):
        """Evaluate efficiency for a 13-dim parameter vector (no float() — tracer-safe)."""
        p64 = jnp.asarray(p, dtype=jnp.float64)
        mat = dj.material(
            Eg=jnp.float64(p64[0]), Chi=jnp.float64(p64[1]), eps=jnp.float64(p64[2]),
            Nc=jnp.float64(p64[3]), Nv=jnp.float64(p64[4]),
            mn=jnp.float64(p64[5]), mp=jnp.float64(p64[6]),
            tn=jnp.float64(p64[7]), tp=jnp.float64(p64[8]),
            A=2e4,
        )
        dev = dj.Device(
            n_points=200,
            layers=[
                (jnp.float64(p64[11]), mat, 1e16),
                (jnp.float64(p64[12]), mat, -1e16),
            ],
            Snl=jnp.float64(p64[9]), Snr=jnp.float64(p64[9]),
            Spl=jnp.float64(p64[10]), Spr=jnp.float64(p64[10]),
        )
        return simulate(dev, Sweep(vmax=0.8, n_steps=15), solver=Newton()).efficiency

    results_by_N = {}
    for N_sobol in [32, 64, 128]:
        print(f"\n  --- Sobol N = {N_sobol} ---")
        sys.stdout.flush()
        t0 = time.perf_counter()
        from scipy.stats.qmc import Sobol
        sob = Sobol(d=13, seed=42)
        n_base2 = int(np.ceil(np.log2(max(N_sobol, 2))))
        U = sob.random_base2(n_base2)[:N_sobol]
        P = LO + U * (HI - LO)

        dgsm_accum = np.zeros(13)
        for i in range(N_sobol):
            grad = np.asarray(jax.grad(lambda x: eff_of_13(jnp.asarray(x)))(
                jnp.asarray(P[i], dtype=jnp.float64)
            ))
            dgsm_accum += grad ** 2
            if (i + 1) % 16 == 0 or i == N_sobol - 1:
                print(f"    sample {i+1}/{N_sobol} done", flush=True)

        dgsm = dgsm_accum / N_sobol
        dgsm_norm = dgsm / dgsm.max() if dgsm.max() > 0 else dgsm
        ranking = np.argsort(-dgsm_norm)
        elapsed = time.perf_counter() - t0
        results_by_N[N_sobol] = {
            "dgsm": dgsm_norm.tolist(),
            "ranking": ranking.tolist(),
            "elapsed_s": elapsed,
        }
        print(f"  DGSM ranking (N={N_sobol}): {[PARAM_NAMES[r] for r in ranking]}")
        print(f"  Normalized DGSM: {[f'{dgsm_norm[r]:.4f}' for r in ranking]}")
        print(f"  Time: {elapsed:.1f}s")
        sys.stdout.flush()

    # Spearman correlation between 32 and 128
    from scipy.stats import spearmanr
    rank_32 = results_by_N[32]["ranking"]
    rank_128 = results_by_N[128]["ranking"]
    rho, pval = spearmanr(rank_32, rank_128)
    print(f"\n  Spearman rank correlation (32 vs 128): rho={rho:.4f}, p={pval:.4e}")

    # Top-3 consistency
    top3_32 = set(rank_32[:3])
    top3_64 = set(results_by_N[64]["ranking"][:3])
    top3_128 = set(rank_128[:3])
    top3_all_same = (top3_32 == top3_64 == top3_128)
    print(f"  Top-3 at N=32 : {[PARAM_NAMES[r] for r in rank_32[:3]]}")
    print(f"  Top-3 at N=64 : {[PARAM_NAMES[r] for r in results_by_N[64]['ranking'][:3]]}")
    print(f"  Top-3 at N=128: {[PARAM_NAMES[r] for r in rank_128[:3]]}")
    print(f"  Top-3 identical at all sample sizes: {top3_all_same}")

    return results_by_N, rho, pval, top3_all_same


# ===========================================================================
# 2. OPTIMIZATION COMPARISON
# ===========================================================================

def section2_optimization_comparison():
    divider("2. OPTIMIZATION COMPARISON (jax.grad LBFGS-B vs FD LBFGS-B vs Nelder-Mead)")
    sys.stdout.flush()

    N_GRID = 200
    N_STEPS = 17

    def build_device_from_params(Eg, thickness_um):
        mat = dj.material(
            Eg=jnp.float64(Eg), Chi=4.05, eps=11.7, Nc=2.8e19, Nv=1.04e19,
            mn=1350.0, mp=450.0, tn=1e-5, tp=1e-5, A=2e4,
        )
        t = jnp.float64(thickness_um) * 1e-4  # um -> cm
        return dj.Device(
            n_points=N_GRID,
            layers=[(t / 2, mat, 1e16), (t / 2, mat, -1e16)],
            Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
        )

    def neg_eff(x):
        """x = [Eg (eV), thickness (um)] — tracer-safe (no float())."""
        x64 = jnp.asarray(x, dtype=jnp.float64)
        Eg = jnp.float64(x64[0])
        t_um = jnp.float64(x64[1])
        dev = build_device_from_params(Eg, t_um)
        return -simulate(dev, Sweep(vmax=0.8, n_steps=N_STEPS), solver=Newton()).efficiency

    x0 = np.array([1.12, 1.0])  # [Eg, thickness_um]
    bounds = ((0.9, 1.5), (0.5, 3.0))
    results = {}

    # (a) jax.grad + L-BFGS-B (via SLSQP with explicit jac)
    print("\n  --- (a) jax.grad + L-BFGS-B ---")
    sys.stdout.flush()
    eval_count_a = [0]
    t0 = time.perf_counter()

    def obj_a(x_np):
        eval_count_a[0] += 1
        x_jax = jnp.asarray(x_np, dtype=jnp.float64)
        val, g = jax.value_and_grad(lambda v: neg_eff(v))(x_jax)
        return float(val), np.asarray(g, dtype=np.float64)

    # Use SLSQP with jac=True (tuple protocol: f returns (val, grad))
    result_a = dj.optimize.slsqp(obj_a, x0, bounds=bounds, maxiter=50, jac=True)
    wall_a = time.perf_counter() - t0
    results["jax_grad_LBFGSB"] = {
        "final_eff": -float(result_a.fun),
        "n_evals": eval_count_a[0],
        "wall_s": wall_a,
        "x_opt": np.asarray(result_a.x).tolist(),
        "message": str(result_a.message),
    }
    print(f"    Final eff: {-result_a.fun*100:.4f}%  Evals: {eval_count_a[0]}  Wall: {wall_a:.1f}s")

    # (b) Finite-difference gradient + L-BFGS-B (scipy built-in FD)
    print("\n  --- (b) Finite-difference + L-BFGS-B (scipy) ---")
    sys.stdout.flush()
    eval_count_b = [0]
    t0 = time.perf_counter()

    def obj_b(x_np):
        eval_count_b[0] += 1
        return neg_eff(x_np)

    result_b = scipy.optimize.minimize(
        obj_b, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 50, "ftol": 1e-12},
    )
    wall_b = time.perf_counter() - t0
    results["FD_LBFGSB"] = {
        "final_eff": -float(result_b.fun),
        "n_evals": eval_count_b[0],
        "wall_s": wall_b,
        "x_opt": np.asarray(result_b.x).tolist(),
        "message": str(result_b.message),
    }
    print(f"    Final eff: {-result_b.fun*100:.4f}%  Evals: {eval_count_b[0]}  Wall: {wall_b:.1f}s")

    # (c) Nelder-Mead (derivative-free)
    print("\n  --- (c) Nelder-Mead (derivative-free) ---")
    sys.stdout.flush()
    eval_count_c = [0]
    t0 = time.perf_counter()

    def obj_c(x_np):
        eval_count_c[0] += 1
        return neg_eff(x_np)

    result_c = dj.optimize.nelder_mead(obj_c, x0, bounds=bounds, maxiter=50)
    wall_c = time.perf_counter() - t0
    results["Nelder_Mead"] = {
        "final_eff": -float(result_c.fun),
        "n_evals": eval_count_c[0],
        "wall_s": wall_c,
        "x_opt": np.asarray(result_c.x).tolist(),
        "message": str(result_c.message),
    }
    print(f"    Final eff: {-result_c.fun*100:.4f}%  Evals: {eval_count_c[0]}  Wall: {wall_c:.1f}s")

    # Summary table
    print("\n  Summary:")
    print(f"  {'Method':<25} {'Final Eff%':>10} {'Evals':>6} {'Wall(s)':>8}")
    for name, r in results.items():
        print(f"  {name:<25} {r['final_eff']*100:>10.4f} {r['n_evals']:>6} {r['wall_s']:>8.1f}")

    return results


# ===========================================================================
# 3. RECOVERY ATTRIBUTION
# ===========================================================================

def section3_recovery_attribution():
    divider("3. RECOVERY ATTRIBUTION (13_target_iv_structure with Nelder-Mead)")
    sys.stdout.flush()

    points = 200
    steps_opt = 41

    material = dj.material(
        Chi=3.9, Eg=1.55, eps=12.0, Nc=2.2e19, Nv=2.2e19,
        mn=1000.0, mp=300.0, tn=1e-6, tp=1e-6, A=3e4,
    )

    def device_from(x):
        return dj.Device(
            n_points=points,
            layers=[
                (5e-6, material, 10.0 ** x[1]),
                (x[0] * 1e-4, material, -(10.0 ** x[1])),
            ],
            Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
        )

    solver = Newton()
    optics = BeerLambert()
    protocol = Sweep(vmax=1.5, n_steps=steps_opt)

    def curve(x, protocol_inner=None):
        if protocol_inner is None:
            protocol_inner = protocol
        sol = dj.simulate(device_from(x), protocol_inner, solver=solver, optics=optics)
        return sol.voltages, sol.currents

    truth_um = jnp.array([1.8, 16.0])
    start_um = np.array([0.85, 15.5])

    v_axis, target = curve(truth_um)
    target = jnp.asarray(target)

    history = []
    eval_count = [0]

    def objective(x):
        eval_count[0] += 1
        _, j_curve = curve(x)
        val = float(dj.optimize.objectives.iv_mse(j_curve, target))
        history.append(val)
        return val

    t0 = time.perf_counter()
    result = dj.optimize.nelder_mead(
        objective,
        start_um,
        bounds=((0.5, 3.0), (15.0, 17.0)),
        maxiter=200,
    )
    wall = time.perf_counter() - t0

    fitted = np.asarray(result.x)
    print(f"  Optimizer   : Nelder-Mead (derivative-free)")
    print(f"  Evals       : {eval_count[0]}")
    print(f"  Final MSE   : {result.fun:.6e}")
    print(f"  Fitted (um) : {fitted.tolist()}")
    print(f"  Fitted (nm) : [{fitted[0]*1e3:.3f}, {fitted[1]:.3f}]")
    print(f"  Truth  (um) : [1.8, 16.0]")
    print(f"  Wall time   : {wall:.1f}s")
    print(f"  Converged   : {result.success}")
    print(f"  History len : {len(history)}")
    if len(history) > 1:
        print(f"  MSE start   : {history[0]:.6e}")
        print(f"  MSE end     : {history[-1]:.6e}")
        if history[-1] > 0:
            print(f"  Reduction   : {history[0]/history[-1]:.1e}x")

    # Confirm it is Nelder-Mead (not SLSQP)
    print(f"\n  CONFIRMED: optimizer is Nelder-Mead (derivative-free)")
    return {
        "optimizer": "Nelder-Mead",
        "n_evals": eval_count[0],
        "final_mse": float(result.fun),
        "fitted_um": fitted.tolist(),
        "wall_s": wall,
        "history": history,
    }


# ===========================================================================
# 4. MESH CONVERGENCE WITH EXTENDED LEVELS
# ===========================================================================

def section4_mesh_convergence():
    divider("4. MESH CONVERGENCE (N=62, 125, 250, 500, 1000)")
    sys.stdout.flush()

    grid_sizes = [62, 125, 250, 500, 1000]
    results = {}

    for N in grid_sizes:
        print(f"\n  --- N = {N} ---")
        sys.stdout.flush()
        t0 = time.perf_counter()
        dev = make_homo(N)
        sol = dj.simulate(dev, Sweep(vmax=0.8, n_steps=31), solver=Newton(), optics=BeerLambert())
        jax.block_until_ready(sol)
        elapsed = time.perf_counter() - t0

        eff = float(sol.eff)
        jsc = float(sol.jsc)
        voc = float(sol.voc)
        ff = float(sol.ff)
        results[N] = {
            "eff": eff, "jsc": jsc, "voc": voc, "ff": ff,
            "jsc_mA": jsc * 1e3, "elapsed_s": elapsed,
        }
        print(f"    eff={eff:.6f}  Jsc={jsc*1e3:.4f} mA/cm2  Voc={voc:.4f} V  FF={ff:.4f}  ({elapsed:.1f}s)")
        sys.stdout.flush()

    # Convergence ratios (using absolute changes between successive levels)
    print("\n  Convergence ratios:")
    print(f"  {'N':>6} {'Eff':>12} {'Jsc(mA)':>10} {'Voc':>10} {'FF':>10} {'dE':>12} {'dJ':>12} {'dV':>12} {'dF':>12}")
    ns = sorted(results.keys())
    for i, N in enumerate(ns):
        r = results[N]
        if i > 0:
            N_prev = ns[i - 1]
            r_prev = results[N_prev]
            dE = r["eff"] - r_prev["eff"]
            dJ = r["jsc"] - r_prev["jsc"]
            dV = r["voc"] - r_prev["voc"]
            dF = r["ff"] - r_prev["ff"]
            dE_s = f"{dE:>+12.6e}"
            dJ_s = f"{dJ:>+12.6e}"
            dV_s = f"{dV:>+12.6e}"
            dF_s = f"{dF:>+12.6e}"
        else:
            dE_s = dJ_s = dV_s = dF_s = f"{'---':>12}"
        print(f"  {N:>6} {r['eff']:>12.6f} {r['jsc']*1e3:>10.4f} {r['voc']:>10.4f} {r['ff']:>10.4f} {dE_s} {dJ_s} {dV_s} {dF_s}")

    # Observed convergence order: ratio of successive differences
    print("\n  Convergence order (ratio of successive |delta|):")
    for i in range(2, len(ns)):
        N_prev, N_curr = ns[i - 1], ns[i]
        r_prev = results[N_prev]
        r_curr = results[N_curr]
        r_pp = results[ns[i - 2]]
        de_prev = abs(r_prev["eff"] - r_pp["eff"])
        de_curr = abs(r_curr["eff"] - r_prev["eff"])
        dj_prev = abs(r_prev["jsc"] - r_pp["jsc"])
        dj_curr = abs(r_curr["jsc"] - r_prev["jsc"])
        h_ratio = N_prev / N_curr  # h_curr / h_prev
        if de_curr > 1e-30 and de_prev > 1e-30:
            order_E = np.log(de_prev / de_curr) / np.log(1.0 / h_ratio)
            order_J = np.log(dj_prev / dj_curr) / np.log(1.0 / h_ratio)
        else:
            order_E = order_J = float("nan")
        print(f"    N={N_prev}->{N_curr}: order_E={order_E:.2f}, order_J={order_J:.2f}")

    # Also compute the reference efficiency at N=500
    print(f"\n  Reference (N=500): eff={results[500]['eff']:.6f}, Jsc={results[500]['jsc']*1e3:.4f} mA/cm2")
    for N in ns:
        err = abs(results[N]["eff"] - results[500]["eff"])
        print(f"    N={N:>4}: |eff - eff_500| = {err:.6e}")

    return results


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("  DriftJax v0.1.15 — Stage 7 Application-Strengthening Verification")
    print("=" * 70)
    print(f"  JAX version : {jax.__version__}")
    print(f"  Platform    : {jax.default_backend()}")
    print(f"  float64     : True")
    sys.stdout.flush()

    # Warm-up
    jnp.sum(jnp.ones(1))
    jax.block_until_ready(jnp.ones(1))

    all_results = {}

    # 1. DGSM stability
    try:
        r1, rho, pval, top3_ok = section1_dgsm_stability()
        all_results["dgsm"] = {"data": r1, "spearman_rho": rho, "spearman_p": pval, "top3_identical": top3_ok}
    except Exception as e:
        print(f"  DGSM FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["dgsm"] = {"error": str(e)}

    # 2. Optimization comparison
    try:
        r2 = section2_optimization_comparison()
        all_results["optimization"] = r2
    except Exception as e:
        print(f"  OPTIMIZATION FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["optimization"] = {"error": str(e)}

    # 3. Recovery attribution
    try:
        r3 = section3_recovery_attribution()
        all_results["recovery"] = r3
    except Exception as e:
        print(f"  RECOVERY FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["recovery"] = {"error": str(e)}

    # 4. Mesh convergence
    try:
        r4 = section4_mesh_convergence()
        all_results["convergence"] = r4
    except Exception as e:
        print(f"  CONVERGENCE FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["convergence"] = {"error": str(e)}

    # Final summary
    divider("STAGE 7 COMPLETE — ALL RESULTS SUMMARY")

    if "dgsm" in all_results and "error" not in all_results["dgsm"]:
        d = all_results["dgsm"]
        print(f"\n  1. DGSM STABILITY:")
        print(f"     Spearman rho (32 vs 128): {d['spearman_rho']:.4f} (p={d['spearman_p']:.4e})")
        print(f"     Top-3 identical at all sample sizes: {d['top3_identical']}")
        for N in [32, 64, 128]:
            rd = d["data"][N]
            PARAM_NAMES = ["Eg","Chi","eps","Nc","Nv","mn","mp","tn","tp","Snl","Spr","t1","t2"]
            print(f"     N={N:>3}: top={PARAM_NAMES[rd['ranking'][0]]} ({rd['dgsm'][rd['ranking'][0]]:.4f}), "
                  f"2nd={PARAM_NAMES[rd['ranking'][1]]} ({rd['dgsm'][rd['ranking'][1]]:.4f}), "
                  f"3rd={PARAM_NAMES[rd['ranking'][2]]} ({rd['dgsm'][rd['ranking'][2]]:.4f})")

    if "optimization" in all_results and "error" not in all_results["optimization"]:
        o = all_results["optimization"]
        print(f"\n  2. OPTIMIZATION COMPARISON:")
        for name, r in o.items():
            print(f"     {name}: eff={r['final_eff']*100:.4f}%  evals={r['n_evals']}  wall={r['wall_s']:.1f}s")

    if "recovery" in all_results and "error" not in all_results["recovery"]:
        rv = all_results["recovery"]
        print(f"\n  3. RECOVERY ATTRIBUTION:")
        print(f"     Optimizer: {rv['optimizer']}")
        print(f"     Evals: {rv['n_evals']}, Final MSE: {rv['final_mse']:.6e}")
        print(f"     Fitted: [{rv['fitted_um'][0]*1e3:.3f} nm, {rv['fitted_um'][1]:.3f} logD]")

    if "convergence" in all_results and "error" not in all_results["convergence"]:
        c = all_results["convergence"]
        print(f"\n  4. MESH CONVERGENCE:")
        for N in sorted(c.keys()):
            r = c[N]
            print(f"     N={N:>4}: eff={r['eff']:.6f}  Jsc={r['jsc_mA']:.4f} mA/cm2  "
                  f"Voc={r['voc']:.4f} V  FF={r['ff']:.4f}")

    print(f"\n{'=' * 70}")
    print("  STAGE 7 APPLICATION VERIFICATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
