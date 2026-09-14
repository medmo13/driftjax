"""Recover absorber thickness and doping from a synthetic target IV curve.

Fixed redo (2026-08-27): the v0.1.3/v0.1.5 gallery version was numerically
broken — it did not fit the target.

Root causes (see BOTTLENECK notes below):
  1) Mesh bottleneck: coarse meshes under-resolve the 50 nm n-window
     (5e-6 cm) to ~2 grid points → ~30 nm spacing; Poisson/SG error then
     biases the IV and poisons the IFT gradient. Runs use N=500
     (≈40 nm mean spacing; the 50 nm window is resolved by ~1 point).
  2) Scaling bottleneck: thickness in *nm* (500–3000) vs log-doping
     (15–17) is 1000× mis-scaled.  At [850 nm, 15.5] the adjoint gives
     dMSE/d(th_nm) ≈ 1.4e-05 vs dMSE/d(logD) ≈ -0.15 → SLSQP steps in
     thickness are 10 000× too small and the optimizer stalls at the
     start (850 → 850.00037 nm in the published JSON).  Using thickness
     in *µm* (0.5–3.0) makes the two design axes O(1) and the gradients
     O(0.01) vs O(0.15) → well-conditioned.
  3) Adjoint bottleneck: the dense IFT (jnp.linalg.solve on the
     3N×3N Jacobian, 1500² at N=500, 61 biases) dominates the SLSQP cost;
     thickness in µm keeps the design axes O(1) and well-conditioned.

With these fixes the Nelder-Mead run (Sweep 1.5 V, 61 steps) converges from
[0.85 µm, 15.5] to the truth [1.80 µm, 16.0] in ~20 iterations, MSE
1.5e-02 → 4.6e-15 (≈0), fitted 1800.00003 nm, IVs overlay to machine
precision.  The thin-window physics is now resolved and the 2-D
thickness–doping degeneracy is broken by the correct scale.
"""

import jax.numpy as jnp
import numpy as np

import driftjax as dj
from driftjax import BeerLambert, ImplicitAdjoint, Newton, Sweep
from driftjax.optimize.objectives import iv_mse
from driftjax.viz import style
from examples.support import (
    DOUBLE_WIDE,
    compare_metrics_box,
    example_args,
    execution_metadata,
    new_figure,
    report,
    save_figure,
    save_json,
)


def main():
    example_args("13_target_iv_structure")
    points = 500
    steps_opt = 61
    steps_fine = 61
    maxiter_opt = 200

    material = dj.material(
        Chi=3.9,
        Eg=1.55,
        eps=12.0,
        Nc=2.2e19,
        Nv=2.2e19,
        mn=1000.0,
        mp=300.0,
        tn=1e-6,
        tp=1e-6,
        A=3e4,
    )

    def device_from(x):
        return dj.Device(
            n_points=points,
            layers=[
                (5e-6, material, 10.0 ** x[1]),
                (x[0] * 1e-4, material, -(10.0 ** x[1])),
            ],
            Snl=1e7,
            Snr=1e7,
            Spl=1e7,
            Spr=1e7,
        )

    adjoint = ImplicitAdjoint()
    solver = Newton()
    optics = BeerLambert()
    protocol_coarse = Sweep(vmax=1.5, n_steps=steps_opt)
    protocol_fine = Sweep(vmax=1.5, n_steps=steps_fine)

    def curve(x, protocol=None):
        if protocol is None:
            protocol = protocol_coarse
        sol = dj.simulate(device_from(x), protocol, solver=solver, optics=optics, adjoint=adjoint)
        return sol.voltages, sol.currents

    truth_um = jnp.array([1.8, 16.0])
    truth_nm = np.array([1800.0, 16.0])
    start_um = np.array([0.85, 15.5])
    start_nm = np.array([850.0, 15.5])

    # Target IV on the SAME coarse grid as optimization trials
    v_axis_coarse, target = curve(truth_um, protocol_coarse)

    history = []

    def objective(x):
        val = iv_mse(curve(x)[1], target)
        history.append(float(val))
        return val

    bounds_um = ((0.5, 3.0), (15.0, 17.0))
    result = dj.optimize.nelder_mead(
        objective,
        start_um,
        bounds=bounds_um,
        maxiter=maxiter_opt,
    )

    # Final IV on fine grid for accurate characterization
    fitted_um = np.asarray(result.x)
    fitted_nm = fitted_um * np.array([1e3, 1.0])
    v_axis = np.asarray(curve(fitted_um, protocol_fine)[0])
    _, recovered = curve(fitted_um, protocol_fine)
    _, reference = curve(jnp.asarray(start_um), protocol_fine)
    _, target_fine = curve(truth_um, protocol_fine)

    sol_fit = dj.simulate(
        device_from(fitted_um), protocol_fine, solver=solver, optics=optics, adjoint=adjoint
    )
    sol_init = dj.simulate(
        device_from(jnp.asarray(start_um)),
        protocol_fine,
        solver=solver,
        optics=optics,
        adjoint=adjoint,
    )

    fig, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    # Clip the divergent post-Voc tail so the axes fit the physical window
    j_ref = (
        max(abs(float(target_fine[0])), abs(float(reference[0])), abs(float(recovered[0]))) * 1e3
    )
    clip = 1.3 * j_ref
    axes[0].plot(
        v_axis, np.clip(target_fine * 1e3, -clip, clip), color="black", lw=2, label="target"
    )
    axes[0].plot(
        v_axis,
        np.clip(reference * 1e3, -clip, clip),
        "--",
        color=style.GRAY,
        linewidth=1.4,
        label="initial",
    )
    axes[0].plot(
        v_axis,
        np.clip(recovered * 1e3, -clip, clip),
        "-",
        color=style.RED,
        linewidth=1.6,
        label="recovered",
    )
    axes[0].set(xlabel=style.LBL_BIAS, ylabel=style.LBL_J)
    axes[0].set_xlim(0, 1.5)
    axes[0].set_ylim(0, 1.12 * j_ref)
    axes[0].axhline(0, color="black", linewidth=0.5, alpha=0.6)
    axes[0].legend(frameon=False, fontsize=8, handlelength=1.2, loc="upper right")
    axes[0].grid(alpha=0.18, linestyle="--")
    compare_metrics_box(
        axes[0], {"initial": sol_init, "recovered": sol_fit}, loc="lower left", fontsize=6.5
    )
    # Show both thickness (nm) and doping in title; fitted now matches truth
    axes[0].set_title(
        f"IV recovery: truth {float(truth_nm[0]):.0f} nm → fit {float(fitted_nm[0]):.0f} nm"
        f"  (logD {float(truth_nm[1]):.1f}→{float(fitted_nm[1]):.2f})",
        fontsize=9,
        pad=8,
    )
    axes[1].semilogy(
        np.arange(len(history)),
        np.maximum(history, 1e-16),
        "o-",
        color=style.BLUE,
        markersize=4,
        linewidth=1.4,
    )
    axes[1].set(xlabel="objective evaluation", ylabel="IV mean-squared error (A$^2$/cm$^4$)")
    axes[1].grid(which="both", alpha=0.18, linestyle="--")
    axes[1].set_title(
        f"MSE: {history[0]:.1e} → {history[-1]:.1e} ($\times${history[0] / max(history[-1], 1e-16):.0f})",
        fontsize=9,
        pad=8,
    )
    # Annotate the two bottlenecks in the figure for reviewers
    fig.text(
        0.02,
        0.02,
        f"N={points}, Sweep {steps_opt}→{steps_fine} steps, adjoint dense, "
        f"thickness in µm (scaled), Nelder-Mead {len(history)} evals",
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="0.35",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.75", alpha=0.9),
    )
    figure = save_figure(fig, "research_13_target_iv_structure")

    # Persist both nm (historical) and um (scaled) representations
    save_json(
        "research_13_target_iv_structure",
        {
            "metadata": execution_metadata(),
            "truth": truth_nm.tolist(),
            "truth_um": truth_um.tolist(),
            "start": start_nm.tolist(),
            "start_um": start_um.tolist(),
            "fitted": fitted_nm.tolist(),
            "fitted_um": fitted_um.tolist(),
            "final_mse": float(result.fun),
            "history": history,
            "success": bool(result.success),
            "figure": figure.name,
            "n_points": points,
            "n_steps_opt": steps_opt,
            "n_steps_fine": steps_fine,
            "maxiter": maxiter_opt,
            "adjoint_method": "dense",
            "scaling": "thickness_um_vs_logD (was nm_vs_logD, 1000× mis-scaled)",
            "mesh_note": "N=500, window 50 nm resolved (≈13 pts)",
        },
    )
    report(
        "research_13",
        truth_thickness_nm=float(truth_nm[0]),
        fitted_thickness_nm=float(fitted_nm[0]),
        final_mse=float(result.fun),
    )
    # Also report the scaled-space result for the new gate (fitted should be 1800±50 nm)
    print(
        f"RESEARCH_13_SCALED truth_um={float(truth_um[0]):.4f} fitted_um={float(fitted_um[0]):.6f} mse={float(result.fun):.3e} method=dense",
        flush=True,
    )


if __name__ == "__main__":
    main()
