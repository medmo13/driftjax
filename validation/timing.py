"""Timing of the analytic IV-curve gradient (the performance fix).

Before the fix, jax.grad(simulate) re-traced init_cell/optics/newton once per
bias point (a Python for-loop), so compiling + running the gradient took many
minutes. After the fix the bias loop is vectorised with jax.vmap, so the whole
sweep-adjoint is a single traced graph. This script reports the one-time
compile cost and the cached (repeated) cost.

Run:  PYTHONPATH=src python validation/timing.py
"""
import time

import jax
import jax.numpy as jnp

import driftjax as dj
from driftjax.optics.api import BeerLambert
from driftjax.problems import Sweep


def build_device(x, mn=130.0):
    m1 = dj.material(Eg=x[0], Chi=x[1], eps=x[2], Nc=10 ** x[3],
                                  Nv=10 ** x[4], mn=mn, mp=x[5], A=2e4)
    m2 = dj.material(Eg=x[6], Chi=x[7], eps=x[8], Nc=10 ** x[9],
                                  Nv=10 ** x[10], mn=mn, mp=x[11], A=2e4)
    return dj.Device(n_points=20,
                     layers=[(x[12], m1, 10 ** x[13]), (x[14], m2, -10 ** x[15])],
                     Snl=1e7, Snr=0.0, Spl=0.0, Spr=1e7)


def pce(x):
    s = dj.simulate(build_device(x), Sweep(n_steps=5),
                    optics=BeerLambert("tauc"), progress=False)
    return s.eff


X0 = jnp.array([1.4, 3.0, 10.0, 18.0, 18.0, 160.0,
               1.4, 3.0, 10.0, 18.0, 18.0, 160.0,
               1e-4, 1e-4, 17.0, 17.0])


def main():
    t = time.time(); g1 = jax.grad(pce)(X0); c1 = time.time() - t
    t = time.time(); g2 = jax.grad(pce)(X0); r2 = time.time() - t
    jitg = jax.jit(jax.grad(pce))
    t = time.time(); gj = jitg(X0); cj = time.time() - t
    t = time.time(); gj2 = jitg(X0); rj = time.time() - t
    print("grad call 1 (compile+run): %.2fs" % c1)
    print("grad call 2 (cached)     : %.3fs" % r2)
    print("jit  call 1 (compile+run): %.2fs" % cj)
    print("jit  call 2 (cached)     : %.3fs" % rj)
    print("grad[0]=%.4e (Eg1)  grad[12]=%.4e (t1)" % (float(g1[0]), float(g1[12])))


if __name__ == "__main__":
    main()
