"""Validate driftjax analytic IV-curve gradient vs central finite differences.

The design is fully parameterised by a flat vector x (Eg, Chi, eps, log10 Nc,
log10 Nv, mp for each of two layers, then t1, t2, log10 Nd, log10 Na). We build
the device from x inside the differentiated function (the exact pattern used by
the optimize/ examples) and compare jax.grad(pce)(x) against a central finite
difference for all 16 parameters.

Run:  PYTHONPATH=src python validation/grad_vs_fd.py
"""

import time

import jax
import jax.numpy as jnp

import driftjax as dj
from driftjax.optics.api import BeerLambert
from driftjax.problems import Sweep


def build_device(x, mn=130.0):
    m1 = dj.material(
        Eg=x[0], Chi=x[1], eps=x[2], Nc=10 ** x[3], Nv=10 ** x[4], mn=mn, mp=x[5], A=2e4
    )
    m2 = dj.material(
        Eg=x[6], Chi=x[7], eps=x[8], Nc=10 ** x[9], Nv=10 ** x[10], mn=mn, mp=x[11], A=2e4
    )
    return dj.Device(
        n_points=20,
        layers=[(x[12], m1, 10 ** x[13]), (x[14], m2, -(10 ** x[15]))],
        Snl=1e7,
        Snr=0.0,
        Spl=0.0,
        Spr=1e7,
    )


def pce(x):
    s = dj.simulate(build_device(x), Sweep(n_steps=5), optics=BeerLambert("tauc"), progress=False)
    return s.eff


NAMES = [
    "Eg1",
    "Chi1",
    "eps1",
    "lNc1",
    "lNv1",
    "mp1",
    "Eg2",
    "Chi2",
    "eps2",
    "lNc2",
    "lNv2",
    "mp2",
    "t1",
    "t2",
    "lNd",
    "lNa",
]
X0 = jnp.array(
    [1.4, 3.0, 10.0, 18.0, 18.0, 160.0, 1.4, 3.0, 10.0, 18.0, 18.0, 160.0, 1e-4, 1e-4, 17.0, 17.0]
)


def main():
    print("pce(X0) =", float(pce(X0)))
    t0 = time.time()
    g = jax.grad(pce)(X0)
    print("jax.grad(pce) compile+run: %.1fs" % (time.time() - t0))

    eps = 1e-6
    SIG = 1e-8  # below this magnitude the parameter is at the FD/AD noise floor
    rows = []
    for i, nm in enumerate(NAMES):
        xi = X0.at[i].set(X0[i] + eps)
        xm = X0.at[i].set(X0[i] - eps)
        fp, fm = float(pce(xi)), float(pce(xm))
        fd = (fp - fm) / (2 * eps)
        an = float(g[i])
        mag = max(abs(an), abs(fd))
        rel = abs(an - fd) / mag if mag > 0 else 0.0
        rows.append((nm, an, fd, rel, mag))

    rows.sort(key=lambda r: -r[3])
    print()
    print(f"{'param':<6} {'analytic':>16} {'FD':>16} {'rel.err':>12}  note")
    worst = 0.0
    for nm, an, fd, rel, mag in rows:
        note = "" if mag > SIG else "(noise floor)"
        print(f"{nm:<6} {an:>16.6e} {fd:>16.6e} {rel:>12.2e}  {note}")
        if mag > SIG:
            worst = max(worst, rel)
    print()
    print(f"max rel.err over SIGNIFICANT params (|g| > {SIG:.0e}) = {worst:.2e}")
    assert worst < 1e-4, "analytic gradient disagrees with finite differences on significant params"
    print(
        "PASS: analytic gradient matches central FD to < 1e-4 relative error on all significant params"
    )


if __name__ == "__main__":
    main()
