import time
t0 = time.time()
import driftjax as dj
from driftjax import BeerLambert, ImplicitAdjoint, Newton, Sweep
from driftjax.optimize.objectives import iv_mse
import numpy as np
import jax.numpy as jnp

points = 250
steps_opt = 31
maxiter_opt = 80

material = dj.material(Chi=3.9, Eg=1.55, eps=12.0, Nc=2.2e19, Nv=2.2e19, mn=1000.0, mp=300.0, tn=1e-6, tp=1e-6, A=3e4)

def device_from(x):
    return dj.Device(n_points=points, layers=[(5e-6, material, 10.0**x[1]), (x[0]*1e-4, material, -(10.0**x[1]))], Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7)

adjoint = ImplicitAdjoint()
solver = Newton(backend="native_ge_eq")
optics = BeerLambert()
protocol_coarse = Sweep(vmax=1.5, n_steps=steps_opt)

def curve(x, protocol=None):
    if protocol is None: protocol = protocol_coarse
    sol = dj.simulate(device_from(x), protocol, solver=solver, optics=optics, adjoint=adjoint)
    return sol.voltages, sol.currents

truth_um = jnp.array([1.8, 16.0])
v_axis_coarse, target = curve(truth_um, protocol_coarse)
print(f"curve eval: {time.time()-t0:.1f}s")

t1 = time.time()
def objective(x):
    return iv_mse(curve(x)[1], target)

start_um = np.array([0.85, 15.5])
bounds_um = ((0.5, 3.0), (15.0, 17.0))
result = dj.optimize.nelder_mead(objective, start_um, bounds=bounds_um, maxiter=maxiter_opt)
print(f"optimization: {time.time()-t1:.1f}s")
print(f"Total: {time.time()-t0:.1f}s")
print(f"optimized: {result}")
