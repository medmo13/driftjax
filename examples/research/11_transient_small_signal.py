"""Research example 11: transient step response and small-signal admittance.

Question: does the AC zero-frequency limit agree with the DC terminal
sensitivity? A voltage-step transient (ramp-hold) shows the charging
timescale, and the small-signal admittance spectrum Y(ω) is checked
against dI/dV from the IV sweep — the time-domain/frequency-domain
consistency every transient solver must satisfy.
"""

import jax.numpy as jnp
import numpy as np

import driftjax as dj
from driftjax.science.contacts import boundary_bias
from driftjax.solvers.continuation import equilibrium_guess
from driftjax.solvers.newton import solve_eq, solve_newton
from driftjax.solvers.transient import ac_small_signal, solve_transient
from driftjax.viz import style
from examples.support import (
    example_args,
    execution_metadata,
    new_figure,
    pn_device,
    report,
    save_figure,
    save_json,
    si_material,
    solution_metrics,
    tag_panels,
)


def main():
    example_args("11_transient_small_signal")
    points = 500
    n_steps = 40
    n_omega = 16

    dev = pn_device(si_material(), points=points)
    sol = dj.simulate(dev, dj.Sweep(vmax=0.6, n_steps=21))
    cell, pot_eq = sol.cell, sol.eq_pot

    # Time-domain: 0 -> 0.3 V ramp-hold step response.
    times, currents, _ = solve_transient(cell, 0.3 / 0.02585, pot_eq,
                                         t_final=1e-3, n_steps=n_steps)
    times = np.asarray(times)
    currents = np.asarray(currents)

    # Frequency-domain: admittance at the 0.3 V operating point, all in
    # dimensionless units. The DD Jacobian is ill-conditioned, so "low
    # frequency" means REALLY low (omega << 1e-6) for the DC asymptote.
    from driftjax.numerics.scharfetter_gummel import Jn, Jp

    v_dc = 0.3 / 0.02585
    pot_dc = solve_eq(cell, boundary_bias(cell, v_dc),
                      equilibrium_guess(cell).phi)
    pot_dc, _ = solve_newton(cell, boundary_bias(cell, v_dc), pot_dc)
    omegas = np.logspace(-12, -6, n_omega)
    y = np.asarray(ac_small_signal(cell, v_dc, pot_dc, omegas))

    # DC reference: central difference of warm-started re-solves
    # (dimensionless currents and bias, like tests/unit/test_transient.py).
    h = 2e-3
    xa, _ = solve_newton(cell, boundary_bias(cell, v_dc + h), pot_dc)
    xb, _ = solve_newton(cell, boundary_bias(cell, v_dc - h), pot_dc)
    i_a = float(jnp.mean(Jn(cell, xa) + Jp(cell, xa)))
    i_b = float(jnp.mean(Jn(cell, xb) + Jp(cell, xb)))
    didv_dc = (i_a - i_b) / (2.0 * h)
    y0 = float(np.real(y[0]))
    rel = abs(y0 - didv_dc) / max(abs(float(didv_dc)), 1e-30)

    fig, axes = new_figure(nrows=1, ncols=2, width=9.0, height=3.8)
    ax0, ax1 = axes.ravel()
    ax0.plot(times * 1e6, currents, lw=1.6, color=style.SERIES[0])
    ax0.set(xlabel="time / µs", ylabel="current (dimensionless)")
    ax0.grid(alpha=0.2, linestyle="--")
    ax1.loglog(omegas, np.abs(y), "o-", ms=4, lw=1.4, color=style.SERIES[1])
    ax1.axhline(abs(float(didv_dc)), color="black", ls="--", lw=1.0,
                label=f"DC dI/dV (rel diff {rel:.1e})")
    ax1.set(xlabel="angular frequency (dimensionless)", ylabel="|Y| (dimensionless)")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(alpha=0.2, which="both", linestyle="--")
    tag_panels(axes.ravel())
    fp = save_figure(fig, "research_11_transient_small_signal")
    save_json("research_11_transient_small_signal", {
        "metadata": execution_metadata(),
        "n_points": points, "y0_vs_didv_rel": float(rel),
        "t_settle_us": float(times[int(np.argmax(currents > 0.9 * currents[-1]))] * 1e6)
        if currents[-1] > 0 else 0.0,
        "figure": fp.name, **solution_metrics(sol),
    })
    report("research_11", y0_vs_didv_rel=float(rel),
           **solution_metrics(sol))


if __name__ == "__main__":
    main()
