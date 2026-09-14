"""ex1_np_junction.py — canonical Si-like n-p homojunction validation.

Reference forward problem (deltapv ex1 geometry) on the DriftJax API with
Beer-Lambert absorption. Writes the IV figure, a results JSON, and one
``RESULT_JSON`` stdout line for benchmark harnesses.
"""

import json
from pathlib import Path

import numpy as np

import driftjax as dj
from driftjax import BeerLambert, Newton, Sweep, simulate
from driftjax.viz import plot_iv_curve

RESULTS_DIR = Path(__file__).resolve().parent
N_POINTS = 500

material = dj.material(
    Chi=3.9,
    Eg=1.5,
    eps=9.4,
    Nc=8e17,
    Nv=1.8e19,
    mn=100,
    mp=100,
    Et=0,
    tn=1e-8,
    tp=1e-8,
    A=20000.0,
)
des = dj.Device(
    layers=list(zip([0.0001, 0.0001], [material, material], [1e17, -1e17], strict=False)),
    n_points=N_POINTS,
    Snl=1e7,
    Snr=0,
    Spl=0,
    Spr=1e7,
)


def run(vmax=1.1, n_steps=25, alpha_mode="beer-lambert"):
    sol = simulate(
        des,
        Sweep(vmax=vmax, n_steps=n_steps),
        solver=Newton(),
        optics=BeerLambert(alpha_mode=alpha_mode),
    )
    return dict(
        eff_pct=float(sol.eff) * 100,
        jsc_mA_cm2=float(sol.jsc) * 1000.0,
        voc_V=float(sol.voc),
        ff=float(sol.ff),
        vmax=vmax,
        n_steps=n_steps,
        alpha_mode=alpha_mode,
        n_points=N_POINTS,
    )


if __name__ == "__main__":
    # A single denser sweep feeds both the reported metrics and the figure,
    # so the title FoM and the plotted MPP corner always agree (a coarse
    # 25-step metrics sweep undershoots the MPP by ~0.5%).
    sol = simulate(
        des,
        Sweep(vmax=1.1, n_steps=101),
        solver=Newton(),
        optics=BeerLambert(alpha_mode="beer-lambert"),
    )
    res = dict(
        eff_pct=float(sol.eff) * 100,
        jsc_mA_cm2=float(sol.jsc) * 1000.0,
        voc_V=float(sol.voc),
        ff=float(sol.ff),
        vmax=1.1,
        n_steps=101,
        alpha_mode="beer-lambert",
        n_points=N_POINTS,
    )
    res["script"] = "ex1_np_junction"
    v, j = np.asarray(sol.voltages), np.asarray(sol.current)
    plot_iv_curve(
        v,
        j,
        str(RESULTS_DIR / "ex1_iv.png"),
        title=(
            f"eff={res['eff_pct']:.2f}%  Jsc={res['jsc_mA_cm2']:.1f} "
            f"Voc={res['voc_V']:.3f} V  FF={res['ff']:.3f}"
        ),
        p_in=float(np.sum(np.asarray(sol.P_in, float))),
    )
    from driftjax.viz.io import save_results

    save_results({**res, "voltages": v.tolist(), "current_mA_cm2": (j * 1000.0).tolist()})
    print("RESULT_JSON " + json.dumps(res))
