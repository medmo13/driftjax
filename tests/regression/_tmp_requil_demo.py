import os, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from driftjax.numerics.banded_solve import adjoint_banded_solve_blocks
np.random.seed(0); n=16
rs=np.random.RandomState(0)
A=jnp.asarray(rs.randn(n,3,3)*0.3); A=A.at[:,0,0].set(A[:,0,0]+6); A=A.at[:,1,1].set(A[:,1,1]+6); A=A.at[:,2,2].set(A[:,2,2]+6)
B=jnp.asarray(rs.randn(n-1,3,3)*0.3)
C=jnp.asarray(rs.randn(n-1,3,3)*0.3)
g=jnp.asarray(rs.randn(n,3))
os.environ.pop("DRIFTJAX_ROW_EQUIL",None)
lam_off,fb_off=adjoint_banded_solve_blocks(A,B,C,g)
os.environ["DRIFTJAX_ROW_EQUIL"]="1"
lam_on,fb_on=adjoint_banded_solve_blocks(A,B,C,g)
os.environ.pop("DRIFTJAX_ROW_EQUIL",None)
diff=float(jnp.linalg.norm(lam_off.reshape(-1)-lam_on.reshape(-1))/(jnp.linalg.norm(lam_off)+1e-30))
print("well-conditioned: fb off", bool(fb_off), "on", bool(fb_on), "rel_diff", diff)
print("golden-safe bit-identical:", diff<1e-12)
