"""Tutorial 03: your first gradient.

Differentiate the efficiency with respect to the design in one line,
rank the most sensitive parameters, and check one component against a
finite difference. This is the primitive every inverse-design example
is built on.
"""

import jax
import jax.numpy as jnp

import driftjax as dj
from driftjax.viz import plot_dossier
from examples import support
from examples.support import (
    ex1_device,
    example_args,
    execution_metadata,
    report,
    save_json,
    solution_metrics,
)


def main():
    example_args("tutorial_03_first_gradient")
    points = 500

    dev = ex1_device(points)
    design = dev.design()

    def eff(d):
        return dj.simulate(d, dj.Sweep(vmax=1.1, n_steps=21)).efficiency

    g = jax.grad(eff)(design)
    leaves = [
        (p[0].name if hasattr(p[0], "name") else str(p[0]), a)
        for p, a in jax.tree_util.tree_flatten_with_path(g)[0]
        if a.size
    ]
    ranked = sorted(
        ((n, float(jnp.max(jnp.abs(a)))) for n, a in leaves), key=lambda kv: kv[1], reverse=True
    )[:5]
    print("top sensitivities:", [(n, f"{v:.3e}") for n, v in ranked], flush=True)

    # Finite-difference spot check on the band gap at mid-device.
    name, idx = "Eg", points // 2
    arr = getattr(design, name)
    h = float(abs(arr[idx])) * 1e-6
    import equinox as eqx

    dp = eqx.tree_at(lambda m: getattr(m, name), design, arr.at[idx].add(h))
    dm = eqx.tree_at(lambda m: getattr(m, name), design, arr.at[idx].add(-h))
    fd = (float(eff(dp)) - float(eff(dm))) / (2 * h)
    an = float(getattr(g, name)[idx])
    print(f"FD check {name}[{idx}]: adjoint={an:.5e} FD={fd:.5e}", flush=True)

    sol = dj.simulate(design, dj.Sweep(vmax=1.1, n_steps=61))
    fig_path = plot_dossier(
        sol,
        path=str(support.OUTPUT_ROOT / "tutorial_03_dossier.png"),
        title=f"Si p-n gradient device, N={points}",
    )
    save_json(
        "tutorial_03_first_gradient",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            "top_sensitivities": dict(ranked),
            "fd_check": {"name": name, "index": idx, "adjoint": an, "fd": fd},
            "figure": str(fig_path),
            **solution_metrics(sol),
        },
    )
    report("tutorial_03", fd_rel_diff=abs(an - fd) / max(1e-12, abs(fd)), **solution_metrics(sol))


if __name__ == "__main__":
    main()
