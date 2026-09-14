# Optics — Generation Profiles

DriftJax supports multiple optical models for computing the photogeneration profile G(x) within the device.

## Beer–Lambert

The simplest model: monochromatic or broadband absorption with exponential decay.

### Tauc-Law Absorption

For amorphous and direct-gap materials, the absorption coefficient follows:

```
α(E) = A · (E − E_g)^{1/2}     (Tauc law)
```

smoothed with a softplus function near the band edge for differentiability:

```
α(E) = A · softplus((E − E_g) / kT)^{1/2}
```

### Tabulated Absorption

For materials with measured n/k data (Aspnes tables), the absorption is interpolated from the table.

### Generation Profile

The generation rate at depth x for a broadband source is:

```
G(x) = ∫ Φ(E) · α(E) · e^{−α(E)·x} dE
```

where Φ(E) is the incident photon flux.

**Code:** `science/optics.py:beer_lambert_G(), alpha_tauc(), alpha_from_table()`

### Alpha Mode Normalization

The `alpha_mode` parameter controls how absorption is normalized:

- `"beer-lambert"`: standard Beer–Lambert with the raw α
- `"table"`: interpolated from measured n/k data
- `"tauc"`: Tauc-law α with softplus smoothing

The normalization ensures that the total absorbed power matches the incident spectrum.

## Transfer-Matrix Method (TMM)

For coherent thin-film stacks, the TMM computes the exact electromagnetic field distribution.

### How It Works

1. Each layer is characterized by its complex refractive index ñ = n + ik and thickness d
2. The transfer matrix for each layer is:

```
M_j = | cos(δ_j)    i·sin(δ_j)/ñ_j |
      | i·ñ_j·sin(δ_j)    cos(δ_j)  |
```

where δ_j = 2π·ñ_j·d_j/λ is the phase thickness.

3. The total transfer matrix is the product M = M_1 · M_2 · … · M_N
4. The reflection and transmission coefficients are extracted from M

### Generation from TMM

The generation rate at depth x within layer j is:

```
G_j(x) = (2π/λ) · Im(ñ_j) · |E_j(x)|²
```

where |E_j(x)|² is the squared electric field magnitude computed from the TMM fields.

**Code:** `science/optics.py:tmm_generation()`, `optics/api.py:TMM`

## Fresnel Generation

A simplified model for single-interface reflection losses:

```
G(x) = (1 − R) · α · e^{−α·x}
```

where R is the Fresnel reflectance at the front surface.

**Code:** `science/optics.py:fresnel_generation()`

## Light Source

### AM1.5G Spectrum

The standard solar spectrum is provided as a tabulated photon flux:

```python
ls = dj.spectrum()  # AM1.5G, normalized to 1000 W/m²
```

### White Light (1 Sun)

A flat spectrum with total power equal to 1 sun:

```python
ls = dj.white(1.0)  # 1 sun, flat spectrum
```

### Monochromatic

A single wavelength:

```python
ls = dj.monochromatic(550)  # 550 nm
```

**Code:** `science/spectrum.py:spectrum(), white(), monochromatic()`

## Generation in the Simulation

The generation profile is computed once during `init_cell` and stored in `cell.G`:

```python
cell = dj.simulator.init_cell(design, ls, alpha_mode="beer-lambert")
# cell.G is the (N,) generation profile
```

The generation enters the drift-diffusion equations as a source term:

```
−R + G + ∂_x J_n = 0     (electron continuity)
 R − G + ∂_x J_p = 0     (hole continuity)
```

## Tandem Cells

For two-terminal tandem cells, the current is limited by the sub-cell with lower photocurrent:

```python
sol = dj.simulate(tandem_dev, dj.Sweep())
```

The tandem model stacks two devices in series, with the same current through both and voltages adding.

**Code:** `science/tandem.py:series_two_terminal()`
