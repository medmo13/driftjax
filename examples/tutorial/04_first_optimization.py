"""Tutorial 04: your first optimization.

Maximize efficiency over two parameters (electron/hole mobility) with
SLSQP using the exact adjoint gradient, starting from a
collection-limited device. Before/after dossiers show what the
optimizer changed. The pattern scales directly to the 16-parameter
perovskite problems in research/06 and research/12. (Geometry such as
thickness must stay concrete — the mesh cannot be a differentiation
tracer — so material and doping parameters are the differentiable set.)
"""

import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import driftjax as dj
from driftjax.viz import plot_dossier
from examples import support
from examples.support import (
    ex1_material,
    example_args,
    execution_metadata,
    report,
    report_fom,
    save_json,
    solution_metrics,
)


def main():
    example_args("tutorial_04_first_optimization")
    points = 500
    maxiter = 25

    mat = ex1_material()

    def device_from(x):
        logmu_n, logmu_p = x[0], x[1]
        import equinox as eqx

        m = eqx.tree_at(lambda mm: (mm.mn, mm.mp), mat, (10.0**logmu_n, 10.0**logmu_p))
        return dj.Device(
            layers=[(1e-4, m, 1e17), (1e-4, m, -1e17)],
            n_points=points,
            Snl=1e7,
            Snr=0,
            Spl=0,
            Spr=1e7,
        )

    import jax

    vg = jax.jit(
        jax.value_and_grad(
            lambda x: -dj.simulate(device_from(x), dj.Sweep(vmax=1.2, n_steps=21)).efficiency
        )
    )

    x0 = np.array([0.7, 0.7])
    bounds = [(0.7, 3.0), (0.7, 3.0)]
    hist = []
    state = {}

    # scipy calls fun and jac separately at the SAME x; memoize so each
    # point costs ONE simulate+adjoint instead of two.
    def value_grad(x):
        xa = np.asarray(x, dtype=float)
        if state.get("x") is None or not np.array_equal(xa, state["x"]):
            v, g = vg(jnp.asarray(xa))
            state.update(x=xa, v=float(v), g=np.asarray(g))
            hist.append(float(v))
        return state["v"], state["g"]

    res = minimize(
        lambda x: value_grad(x)[0],
        x0,
        jac=lambda x: value_grad(x)[1],
        method="SLSQP",
        bounds=bounds,
        options={"maxiter": maxiter},
    )
    x_opt = res.x

    sol_init = dj.simulate(device_from(x0), dj.Sweep(vmax=1.2, n_steps=61))
    sol_opt = dj.simulate(device_from(x_opt), dj.Sweep(vmax=1.2, n_steps=61))
    p_init = plot_dossier(
        sol_init,
        path=str(support.OUTPUT_ROOT / "tutorial_04_init.png"),
        title="Before optimization",
    )
    p_opt = plot_dossier(
        sol_opt, path=str(support.OUTPUT_ROOT / "tutorial_04_opt.png"), title="After optimization"
    )
    save_json(
        "tutorial_04_first_optimization",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            "x0": [float(v) for v in x0],
            "x_opt": [float(v) for v in x_opt],
            "figures": [str(p_init), str(p_opt)],
            "init": solution_metrics(sol_init),
            "opt": solution_metrics(sol_opt),
        },
    )
    report(
        "tutorial_04",
        init_eff=solution_metrics(sol_init)["efficiency_fraction"],
        opt_eff=solution_metrics(sol_opt)["efficiency_fraction"],
    )
    report_fom("tutorial_04", init=sol_init, opt=sol_opt)


if __name__ == "__main__":
    main()
