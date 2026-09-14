# API Reference

## Core Simulation

### `simulate(device, protocol, *, solver, optics, adjoint, progress, statistics, T, ls, init)`

Run a full device simulation.

**Parameters:**
- `device` — `Device` object defining the layer structure
- `protocol` — `Sweep(vmax, n_steps)` or `Equilibrium()`
- `solver` — `Newton()` (default), `Newton(globalization="ls")`, etc.
- `optics` — `BeerLambert(alpha_mode)` or `TMM(table)`
- `adjoint` — `ImplicitAdjoint(method="dense"|"banded")`
- `progress` — `True` for live progress reporting
- `ls` — custom `LightSource` (default: AM1.5G)

**Returns:** `Solution` with `.efficiency`, `.voc`, `.jsc`, `.ff`, `.pmax`, `.voltages`, `.current`

```python
sol = dj.simulate(dev, dj.Sweep(vmax=1.0))
print(f"PCE = {sol.efficiency:.2f}%, Voc = {sol.voc:.3f} V")
```

### `Device(n_points, layers, Snl, Snr, Spl, Spr, T, work_functions)`

Define a semiconductor device.

**Parameters:**
- `n_points` — number of mesh nodes
- `layers` — list of `(thickness, Material, doping)` tuples
- `Snl/Snr/Spl/Spr` — surface recombination velocities (cm/s)
- `T` — temperature (K, default 300)
- `work_functions` — contact work functions (eV)

```python
dev = dj.Device(
    n_points=200,
    layers=[(1e-4, Si, 1e17), (1e-4, Si, -1e17)],
    Snl=1e7, Snr=1e7, Spl=1e7, Spr=1e7,
)
```

### `Material(Chi, Eg, eps, Nc, Nv, mn, mp, tn, tp, A, Br, Cn, Cp, Et, name)`

Define a semiconductor material.

**Defaults:** SRH lifetime tn=tp=1e-8 s, trap level Et=0 (midgap), Auger coefficients Cn=Cp=1e-30 cm⁶/s.

```python
Si = dj.load_material("Si")
GaAs = dj.load_material("GaAs")
custom = dj.material(Eg=1.5, Chi=3.9, eps=9.4, Nc=8e17, Nv=1.8e19)
```

## Protocols

### `Sweep(vmax=1.0, n_steps=50)`

Voltage sweep from 0 to V_max.

### `Equilibrium()`

Equilibrium solve (zero bias, no illumination).

## Solvers

### `Newton()`

Default Newton solver with log-damping and auto globalization.

**Options:**
- `globalization` — `"auto"` (default), `"logdamp"`, `"ls"`, `"ptc"`
- `tol` — step norm tolerance (default 1e-8)
- `f_tol` — residual gate tolerance (default None)
- `max_steps` — maximum iterations (default 100)

### `ImplicitAdjoint(method="banded")`

IFT adjoint for gradient computation.

**Options:**
- `method` — `"banded"` (fast, O(N)) or `"dense"` (robust, O(N²))

## Optics

### `BeerLambert(alpha_mode="beer-lambert")`

Beer–Lambert absorption model.

**Options:**
- `alpha_mode` — `"beer-lambert"` (Tauc α), `"table"` (measured n/k), `"tauc"` (explicit Tauc)

### `TMM(table=None)`

Coherent transfer-matrix method for thin-film stacks.

## Spectrum

### `spectrum(normalize=True)`

AM1.5G solar spectrum.

### `white(suns=1.0)`

Flat spectrum at specified suns intensity.

### `monochromatic(wavelength_nm)`

Single-wavelength source.

## Solution

### `Solution` attributes

- `.efficiency` — power conversion efficiency (%)
- `.voc` — open-circuit voltage (V)
- `.jsc` — short-circuit current density (A/cm²)
- `.ff` — fill factor
- `.pmax` — maximum power density (W/cm²)
- `.voltages` — voltage array (V)
- `.current` — current density array (A/cm²)

### `Solution.at_bias(voltage)`

Interpolate the solution at a specific voltage.

### `Solution.plot()`

Plot the J–V curve.

## Utilities

### `material(**kwargs)`

Create a custom material from keyword arguments.

### `load_material(name)`

Load a material from the database (25 materials: Si, GaAs, CdTe, MAPbI₃, CIGS, etc.).

### `units`

Physical constants and unit conversion factors.

### `thermal_scales(T)`

Dimensionless scaling factors for temperature T.

## Inverse Design

### `maximize_efficiency(device, bounds, method="slsqp")`

Maximize efficiency via gradient-based optimization.

**Options:**
- `method` — `"slsqp"` (default), `"lbfgs"`, `"adam"`
- `bounds` — list of `(min, max)` tuples for each design parameter

```python
result = dj.optimize.maximize_efficiency(dev, bounds=[(0.5, 2.0)])
```

## Gradient

```python
import jax

# Efficiency gradient w.r.t. device design
g = jax.grad(lambda d: dj.simulate(d).efficiency)(dev)

# Access gradient components
g_eg = g.layers[0][1].Eg  # dPCE/dEg for first layer
g_chi = g.layers[0][1].Chi  # dPCE/dChi for first layer
```

## Validation

```python
# Gradient vs finite difference
from driftjax.validation import check_grad_fd
check_grad_fd(dev, eps=1e-7)

# Analytic Jacobian vs AD reference
from driftjax.validation import check_analytic_jacobian
check_analytic_jacobian(cell, bound, pot)
```
