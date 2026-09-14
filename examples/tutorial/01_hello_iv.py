"""Tutorial 01: hello, IV curve.

Build a silicon p-n device, run one illuminated sweep, and render the
canonical 4-panel dossier (J-V, layers, bands, charge). The place to
start: every later tutorial reuses this exact five-line pattern.
"""

import driftjax as dj
from driftjax.viz import plot_dossier
from examples import support
from examples.support import (
    ex1_device,
    example_args,
    execution_metadata,
    report,
    report_fom,
    run_timed,
    save_json,
    solution_metrics,
)


def main():
    example_args("tutorial_01_hello_iv")
    points = 500

    dev = ex1_device(points)
    sol, wall = run_timed(dj.simulate, dev, dj.Sweep(vmax=1.1, n_steps=61))

    fig_path = plot_dossier(
        sol,
        path=str(support.OUTPUT_ROOT / "tutorial_01_dossier.png"),
        title=f"Si p-n, N={points}",
    )
    save_json(
        "tutorial_01_hello_iv",
        {
            "metadata": execution_metadata(),
            "n_points": points,
            "wall_s": wall,
            "figure": str(fig_path),
            **solution_metrics(sol),
        },
    )
    report("tutorial_01", wall_s=wall, **solution_metrics(sol))
    report_fom("tutorial_01", sol=sol)


if __name__ == "__main__":
    main()
