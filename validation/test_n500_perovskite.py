"""N=500 perovskite - production mesh, full comparison"""
import jax.numpy as jnp, jax, numpy as np, time
import driftjax as dj
from driftjax import Sweep, BeerLambert
from driftjax.numerics.residual import F_jacobian
from driftjax.numerics.block_thomas import block_thomas_solve, extract_blocks
from driftjax.science.contacts import boundary_bias

# Perovskite N=500 device (production)
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("psc", str(pathlib.Path(__file__).parent / "psc.py"))
psc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psc)
# Monkey-patch N globally for x2des
orig_N = psc.N
psc.N = 500
dev = psc.x2des(jax.numpy.array(psc.X0))
print(f"Perovskite N=500 device created, layers {len(dev.layers) if hasattr(dev,'layers') else 'unknown'}")
s = dj.simulate(dev, Sweep(n_steps=5, vmax=1.0), optics=BeerLambert())
cell = s.cell
print(f"Simulated {len(s.potentials)} biases, n={cell.x.shape[0]}")
for idx, (pot, vb) in enumerate(zip(s.potentials, s.voltages)):
    if idx not in [2]: continue  # just V=0.5 for brevity
    J = F_jacobian(cell, boundary_bias(cell, vb), pot)
    n = J.shape[0]
    print(f"\nV={vb:.2f} J {n}x{n} ({n//3} points) cond est...")
    try:
        c = float(np.linalg.cond(np.array(J)))
        print(f"cond {c:.2e}")
    except Exception as e:
        print(f"cond failed {e}")
        c = float('inf')
    g = jnp.ones(n)
    # Warmup
    A,B,C = extract_blocks(J)
    def bt(g_):
        At=jnp.transpose(A,(0,2,1)); Ct=jnp.transpose(C,(0,2,1)); Bt=jnp.transpose(B,(0,2,1))
        return block_thomas_solve(At,Ct,Bt,g_.reshape(A.shape[0],3)).reshape(-1)
    import jax
    jbt = jax.jit(bt)
    jdense = jax.jit(lambda JT, gg: jnp.linalg.solve(JT, gg))
    # Warmup
    _ = jbt(g).block_until_ready()
    _ = jdense(J.T, g).block_until_ready()
    # Time warm
    t0=time.time()
    for _ in range(3):
        jbt(g).block_until_ready()
    t_bt = (time.time()-t0)/3*1000
    t0=time.time()
    for _ in range(3):
        jdense(J.T, g).block_until_ready()
    t_d = (time.time()-t0)/3*1000
    # Sparse (not jitted, outside)
    import scipy.sparse, scipy.sparse.linalg
    t0=time.time()
    JT_sp = scipy.sparse.csc_matrix(np.array(J.T))
    lu = scipy.sparse.linalg.splu(JT_sp)
    lam_sp = lu.solve(np.array(g))
    t_sp = (time.time()-t0)*1000
    # Residuals
    lam_bt = jbt(g)
    lam_d = jdense(J.T, g)
    r_bt = float(jnp.linalg.norm(J.T @ lam_bt - g)/(jnp.linalg.norm(g)+1e-30))
    r_d = float(jnp.linalg.norm(J.T @ lam_d - g)/(jnp.linalg.norm(g)+1e-30))
    r_sp = float(np.linalg.norm(JT_sp @ lam_sp - np.array(g))/(np.linalg.norm(g)+1e-30))
    print(f"  BT warm r={r_bt:.2e} t={t_bt:.1f}ms")
    print(f"  Dense warm r={r_d:.2e} t={t_d:.1f}ms")
    print(f"  Sparse r={r_sp:.2e} t={t_sp:.1f}ms")
    # Memory
    mem_dense = n*n*8/1e6
    mem_banded = (2*5+5+1)*n*8/1e6 if n>0 else 0
    print(f"  Memory dense {mem_dense:.1f} MB vs banded {n*3*3*8/1e6:.1f} MB (A,B,C) / {mem_banded:.1f} MB (band storage)")
    print(f"  Flops dense ~{int(2*(n**3)/3):.2e} vs BT ~{120*n:.2e} (108N)")

# Restore
psc.N = orig_N
