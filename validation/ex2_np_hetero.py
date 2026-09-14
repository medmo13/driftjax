"""ex2_np_hetero.py — canonical CdS/CdTe heterojunction validation.

Reference forward problem (deltapv ex2 / SCAPS geometry) on the DriftJax API
with Beer-Lambert absorption. Writes the IV figure, a results JSON, and one
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

CdS = dj.material(
    Nc=2.2e18,
    Nv=1.8e19,
    Eg=2.4,
    eps=10,
    Et=0,
    mn=100,
    mp=25,
    tn=1e-8,
    tp=1e-13,
    Chi=4.0,
    A=10000.0,
)
CdTe = dj.material(
    Nc=8e17,
    Nv=1.8e19,
    Eg=1.5,
    eps=9.4,
    Et=0,
    mn=320,
    mp=40,
    tn=5e-9,
    tp=5e-9,
    Chi=3.9,
    A=10000.0,
)
des = dj.Device(
    layers=list(zip([2.5e-6, 4e-4], [CdS, CdTe], [1e17, -1e15], strict=False)),
    n_points=N_POINTS,
    Snl=1.16e7,
    Snr=1.16e7,
    Spl=1.16e7,
    Spr=1.16e7,
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
    # One denser sweep feeds both the figure and the reported numbers;
    # see ex1 for the consistency rationale.
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
    res["script"] = "ex2_np_hetero"
    v, j = np.asarray(sol.voltages), np.asarray(sol.current)
    plot_iv_curve(
        v,
        j,
        str(RESULTS_DIR / "ex2_iv.png"),
        title=(
            f"eff={res['eff_pct']:.2f}%  Jsc={res['jsc_mA_cm2']:.1f} "
            f"Voc={res['voc_V']:.3f} V  FF={res['ff']:.3f}"
        ),
        p_in=float(np.sum(np.asarray(sol.P_in, float))),
    )
    from driftjax.viz.io import save_results

    save_results({**res, "voltages": v.tolist(), "current_mA_cm2": (j * 1000.0).tolist()})
    print("RESULT_JSON " + json.dumps(res))
