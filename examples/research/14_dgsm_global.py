"""14 — DGSM global sensitivity via forward-mode AD (research).

Question: Which material parameters drive PCE globally vs locally?
Mirrors Mann2021 Sec 4.1 — DGSM = E[(dPCE/dp)^2] via one adjoint per Sobol sample.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import driftjax as dj
from driftjax import Newton, Sweep, simulate
from examples.support import (
    DOUBLE_WIDE,
    example_args,
    execution_metadata,
    new_figure,
    report,
    save_figure,
    save_json,
)

NAMES = [r"$\mu_n$", r"$\mu_p$", r"$\log_{10}\tau_n$", r"$\log_{10}\tau_p$"]
LO = np.array([50.0, 50.0, -9.0, -9.0])
HI = np.array([200.0, 200.0, -7.0, -7.0])
BASE = dict(
    Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, Et=0, A=2e4, mn=100.0, mp=100.0, tn=1e-8, tp=1e-8
)
mat0 = dj.material(**BASE)  # constant nominal material (never mutated)


def main():
    example_args("14_dgsm_global")
    # Mesh/steps live here (no module-global mutation).  N=32 Sobol samples
    # for the DGSM rank ordering.
    npoints, nsteps = (500, 15)
    N = 32

    def eff_of(p):
        mv = eqx.tree_at(
            lambda mm: (mm.mn, mm.mp, mm.tn, mm.tp), mat0, (p[0], p[1], 10 ** p[2], 10 ** p[3])
        )
        dev = dj.Device(
            n_points=npoints,
            layers=[(1e-4, mv, 1e17), (1e-4, mv, -1e17)],
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )
        return simulate(dev, Sweep(vmax=1.1, n_steps=nsteps), solver=Newton()).efficiency

    # Every sample design shares array shapes: serial jitted singles compile
    # once then run seconds each. (A 32-way vmap of N=500 sweep Jacobians
    # compiles one monstrous XLA program — OOM on small machines.)
    g_loc = np.asarray(jax.jacobian(eff_of)(jnp.array([100.0, 100.0, -8.0, -8.0])))
    grad_1 = jax.jacobian(eff_of)
    from scipy.stats.qmc import Sobol

    sob = Sobol(d=4, seed=12345)
    # power of two for base2
    U = sob.random_base2(m=int(np.log2(N))) if N in (4, 8, 16, 32, 64) else sob.random(N)
    if U.shape[0] != N:
        U = sob.random(N)
    P = LO + U * (HI - LO)
    G = np.asarray([grad_1(p) for p in jnp.asarray(P)])
    dgsm = np.mean(G**2, axis=0)
    dgsm_n = dgsm / dgsm.max() if dgsm.max() > 0 else dgsm
    order = np.argsort(-dgsm_n)

    fig, axes = new_figure(nrows=1, ncols=2, **DOUBLE_WIDE)
    ax0, ax1 = axes
    ax0.bar(
        range(4),
        dgsm_n,
        color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"],
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax0.set_xticks(range(4), NAMES)
    ax0.set_ylabel(r"$S_i / \max S$ (DGSM)")
    ax0.set_ylim(0, 1.15)
    ax0.grid(axis="y", alpha=0.18, linestyle="--")
    ax0.set_title(
        rf"DGSM global (N={N} Sobol, $E[(\partial\eta/\partial p)^2]$)", fontsize=8, pad=8
    )
    ax1.bar(
        range(4),
        np.abs(g_loc) / np.max(np.abs(g_loc)),
        color=["#0072B2", "#E69F00", "#009E73", "#CC79A7"],
        edgecolor="black",
        linewidth=0.5,
        width=0.6,
    )
    ax1.set_xticks(range(4), NAMES)
    ax1.set_ylabel(r"$|\partial\eta/\partial p| / \max$ (local)")
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis="y", alpha=0.18, linestyle="--")
    ax1.set_title("Local at nominal", fontsize=9, pad=8)
    fig.suptitle("Global vs local PCE sensitivity (IFT adjoint)")

    figure_path = save_figure(fig, "research_14_dgsm_global")
    payload = {
        "N": int(N),
        "NAMES": list(map(str, NAMES)),
        "dgsm_normalized": dgsm_n.tolist(),
        "g_loc": g_loc.tolist(),
        "order": order.tolist(),
        "most_influential_global": str(NAMES[int(order[0])]),
        "caption": "DGSM (Sobol) vs local: global ranking from IFT gradients; bars normalized to max.",
        "figure": str(figure_path.name),
        "metadata": execution_metadata(),
    }
    save_json("research_14_dgsm_global", payload)
    report("research_14_dgsm_global", N=N, max_dgsm=float(dgsm_n[order[0]]))


if __name__ == "__main__":
    main()
