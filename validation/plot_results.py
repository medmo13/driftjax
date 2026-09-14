"""plot_results.py - aggregate optimizer convergence histories.

Plots the convergence histories saved by psc.py /
optimize_material_recovery.py / optimize_holistic.py
(validation/results/<name>.json).

Run:  PYTHONPATH=src python validation/plot_results.py
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from driftjax.viz import style
style.apply("presentation")


def load(name):
    try:
        return json.loads(open(__file__.rsplit("/", 1)[0] + f"/results/{name}.json").read())
    except Exception as e:
        print(f"skip {name}: {e}")
        return None


for name, ylabel in [("perovskite", "PCE / %"), ("holistic", "PCE / %"),
                    ("material_recovery", "R (lower=better)")]:
    d = load(name)
    if d is None:
        continue
    ys = np.array(d["history"])
    plt.figure(figsize=(8, 5))
    plt.plot(ys, "k-o", ms=3)
    plt.xlabel("SLSQP iteration")
    plt.ylabel(ylabel)
    plt.title(f"{name}: {ys[0]:.3f} -> {ys[-1]:.3f}")
    plt.tight_layout()
    out = __file__.rsplit("/", 1)[0] + f"/results/{name}_convergence.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"plotted {out}")
