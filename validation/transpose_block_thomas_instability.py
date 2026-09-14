"""Why the adjoint uses the dense solve, not the O(N) transposed Block-Thomas.

The drift-diffusion Jacobian J (3n x 3n, block-tridiagonal: diag A, super B,
sub C) has a STABLE, exact forward solve via block_thomas_solve. The adjoint
needs J^T lam = g. The O(N) idea is to feed J^T's blocks into the forward
block-thomas elimination. J^T is also block-tridiagonal, with blocks
    diag = A^T,  super = C^T,  sub = B^T          (each 3x3 block transposed).

Findings (see validation/transpose_block_thomas_instability.py output):
  * On a clean synthetic block-tridiagonal the O(N) transpose solve is EXACT
    (residual 8.9e-16) -- the algorithm is mathematically sound.
  * On a real homojunction Jacobian it is stable too (residual ~1e-14).
  * BUT on the ill-conditioned 3-layer perovskite Jacobian (kappa ~ 1e14, the
    actual optimization target) it blows up: residual 1e6 at V=0 and NaN near
    open-circuit. This is exactly the 5.77x gradient error from commit 56ee78d.

The dense jnp.linalg.solve(J.T, g) (with partial pivoting) is uniformly stable
across all architectures -- the same direct solver deltapv uses. So the adjoint
keeps the dense solve; the bias loop is vectorised with jax.vmap for compile
speed only. (The forward newton step still uses the fast O(N) block_thomas.)

Run:  PYTHONPATH=src python validation/transpose_block_thomas_instability.py
"""
import jax.numpy as jnp
import jax.random as jr
import numpy as onp

import driftjax as dj
from driftjax.numerics.block_thomas import block_thomas_solve, extract_blocks
from driftjax.numerics.residual import F_jacobian
from driftjax.optics.api import BeerLambert
from driftjax.problems import Sweep
from driftjax.science.contacts import boundary_bias

from _helpers import bt_transpose


def synthetic():
    print("=== Synthetic block-tridiagonal (clean, well-conditioned) ===")
    key = jr.PRNGKey(0)

    def rand(*s):
        nonlocal key
        key, k2 = jr.split(key)
        return jr.normal(k2, s)

    N = 20
    A = rand(N, 3, 3) + 5.0 * jnp.eye(3)[None]
    B = rand(N - 1, 3, 3)
    C = rand(N - 1, 3, 3)
    Jn = onp.zeros((3 * N, 3 * N))
    for i in range(N):
        Jn[3 * i:3 * i + 3, 3 * i:3 * i + 3] = onp.asarray(A[i])
    for i in range(N - 1):
        Jn[3 * i:3 * i + 3, 3 * i + 3:3 * i + 6] = onp.asarray(B[i])
        Jn[3 * i + 3:3 * i + 6, 3 * i:3 * i + 3] = onp.asarray(C[i])
    J = jnp.asarray(Jn)
    g = jnp.ones(3 * N)
    # forward
    x_bt = block_thomas_solve(A, B, C, g.reshape(N, 3)).reshape(-1)
    x_de = jnp.linalg.solve(J, g)
    fwd = float(jnp.max(jnp.abs(x_bt - x_de)))
    # transpose (correct super/sub ordering)
    lam_bt = bt_transpose(A, B, C, g)
    lam_de = jnp.linalg.solve(J.T, g)
    res = float(jnp.max(jnp.abs(J.T @ lam_bt - g)))
    err = float(jnp.max(jnp.abs(lam_bt - lam_de)))
    print("  forward  block-thomas vs dense : max|err| = %.2e (expect ~1e-16)" % fwd)
    print("  transpose block-thomas vs dense: residual = %.2e, lam err = %.2e (expect ~1e-16)"
          % (res, err))


def real_dev(dev, label):
    print("=== %s ===" % label)
    s = dj.simulate(dev, Sweep(n_steps=5), optics=BeerLambert("tauc"), progress=False)
    cell = s.cell
    worst = 0.0
    for pot, vb in zip(s.potentials, s.voltages):
        J = F_jacobian(cell, boundary_bias(cell, vb), pot)
        A, B, C = extract_blocks(J)
        n = J.shape[0]
        g = jnp.ones(n)
        lam_bt = bt_transpose(A, B, C, g)
        lam_de = jnp.linalg.solve(J.T, g)
        res = float(jnp.max(jnp.abs(J.T @ lam_bt - g)))
        err = float(jnp.max(jnp.abs(lam_bt - lam_de)))
        flag = "" if res < 1e-6 else "  <-- UNSTABLE"
        print("  V=%6.3f  transpose_residual=%.3e  lam_vs_dense=%.3e%s"
              % (vb, res, err, flag))
        if not jnp.isnan(res):
            worst = max(worst, res)
    print("  -> max transpose residual = %s" % ("%.2e" % worst if worst > 0 else "NaN (diverged)"))


def main():
    synthetic()
    mat = dj.material(Eg=1.4, Chi=3.0, eps=10.0, Nc=1e18, Nv=1e18,
                                   mn=130.0, mp=160.0, A=2e4)
    dev_h = dj.Device(n_points=15,
                      layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e17)],
                      Snl=1e7, Snr=0.0, Spl=0.0, Spr=1e7)
    mt = dj.material(Eg=1.6, Chi=3.9, eps=20.0, Nc=1e18, Nv=1e18,
                                  mn=100.0, mp=100.0, A=2e4)
    dev_3 = dj.Device(n_points=15,
                      layers=[(2e-5, mt, 1e18), (6e-5, mat, -1e18), (2e-5, mat, 1e18)],
                      Snl=1e7, Snr=0.0, Spl=0.0, Spr=1e7)
    real_dev(dev_h, "Homojunction (n_points=15) -- well-conditioned")
    real_dev(dev_3, "3-layer perovskite (n_points=15) -- ill-conditioned target")
    print()
    print("Conclusion: the O(N) transposed block-Thomas is exact on clean / homojunction")
    print("Jacobians but UNSTABLE on the ill-conditioned 3-layer perovskite (the real")
    print("optimization target). The adjoint therefore uses the exact, uniformly stable")
    print("dense jnp.linalg.solve(J.T, g) -- matching deltapv. The forward newton step")
    print("still uses the fast O(N) block_thomas solve.")


if __name__ == "__main__":
    main()
