import os
os.environ.pop('DRIFTJAX_ROW_EQUIL', None)
os.environ['OMP_NUM_THREADS']='1'
os.environ['JAX_ENABLE_X64']='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intraop_parallelism_threads=1'
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from driftjax.numerics.banded_solve import banded_solve
np.random.seed(2); rs=np.random.RandomState(2); n=14
A=jnp.asarray(rs.randn(n,3,3)*0.4); A=A.at[:,0,0].set(A[:,0,0]+7); A=A.at[:,1,1].set(A[:,1,1]+7); A=A.at[:,2,2].set(A[:,2,2]+7)
B=jnp.asarray(rs.randn(n-1,3,3)*0.4)
C=jnp.asarray(rs.randn(n-1,3,3)*0.4)
b=jnp.asarray(rs.randn(n,3))
xb=banded_solve(A,B,C,b)
os.environ['DRIFTJAX_ROW_EQUIL']='1'
xo=banded_solve(A,B,C,b)
os.environ.pop('DRIFTJAX_ROW_EQUIL', None)
r1=float(jnp.linalg.norm(xb.reshape(-1)-xo.reshape(-1))/(jnp.linalg.norm(xb)+1e-30))
print('forward golden-safe rel_diff:', r1, 'bit-identical:', r1<1e-12)
