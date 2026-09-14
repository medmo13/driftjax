"""Tutorial 02: bands and charge, equilibrium vs bias.

Render two dossiers for the same device — equilibrium bands/Fermi level
versus a biased operating point with quasi-Fermi level splitting — and
see where the photovoltage comes from. Introduces ``at_bias`` and the
``show_qfl`` dossier option.
"""

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
    example_args("tutorial_02_bands_and_charge")
    points = 500

    dev = ex1_device(points)
    sol = dj.simulate(dev, dj.Sweep(vmax=1.1, n_steps=61))

    p_eq = plot_dossier(
        sol,
        path=str(support.OUTPUT_ROOT / "tutorial_02_equilibrium.png"),
        title=f"Si p-n at equilibrium, N={points}",
    )
    pot_bias = sol.at_bias(0.5)
    p_qfl = plot_dossier(
        sol,
        pot_bias,
        show_qfl=True,
        path=str(support.OUTPUT_ROOT / "tutorial_02_qfl.png"),
        title="Si p-n at V = 0.5 V — quasi-Fermi level splitting",
    )
    save_json(
        "tutorial_02_bands_and_charge",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            "bias_V": 0.5,
            "figures": [str(p_eq), str(p_qfl)],
            **solution_metrics(sol),
        },
    )
    report("tutorial_02", **solution_metrics(sol))


if __name__ == "__main__":
    main()
