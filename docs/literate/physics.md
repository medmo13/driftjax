# Physics — The Van Roosbroeck System

DriftJax solves the **Van Roosbroeck system** — the coupled drift–diffusion–Poisson equations that govern charge transport in semiconductor devices. This document explains the mathematical formulation and how it maps to the code.

## The Governing Equations

In dimensionless form (energies in units of thermal voltage V_t, densities in units of a reference density, lengths in Debye length L_0), the system is:

### Poisson's Equation

```
∇·(ε ∇φ) = −(p − n + N_dop)
```

where φ is the electrostatic potential, n and p are electron and hole concentrations, and N_dop is the net doping density.

**Code:** `numerics/poisson.py:poisson()`

The permittivity on each face is the **harmonic mean** `2ab/(a+b)` — the only average preserving D = εE continuity across a jump in ε. This is computed by `harmonic_ave_eps()`.

### Electron Continuity

```
−R + G + ∂_x J_n = 0
```

where J_n is the electron current density and R is the recombination rate.

**Code:** `numerics/drift_diffusion.py:ddn()`

### Hole Continuity

```
R − G + ∂_x J_p = 0
```

**Code:** `numerics/drift_diffusion.py:ddp()`

## Scharfetter–Gummel Discretization

The current fluxes J_n and J_p use the **Scharfetter–Gummel (SG) scheme**, which is the natural finite-volume discretization for drift–diffusion. The SG flux between nodes i and i+1 is:

```
J_n = (mn / d) · [B(ψ_n,i+1 − ψ_n,i) · n_i − B(ψ_n,i − ψ_n,i+1) · n_i+1]
```

where B(x) = x / (e^x − 1) is the **Bernoulli function** and ψ_n = χ + ln(N_c) + φ is the quasi-Fermi level for electrons.

**Code:** `numerics/scharfetter_gummel.py:Jn(), Jp()`

The Bernoulli function has a removable singularity at x = 0, handled via Taylor expansion for |x| < 1e-5.

## Carrier Statistics

The carrier concentrations relate to the quasi-Fermi levels through the **statistics function**:

### Boltzmann (default)

```
n = N_c · e^(φ_n − χ)     p = N_v · e^(χ + E_g − φ_p)
```

### Fermi–Dirac (Nilsson approximation)

```
n = N_c · F_{1/2}(η_n) / Γ(3/2)     (via Padé approximant)
```

### Blakemore

```
n = N_c · (exp(η_n) / (1 + α·exp(η_n)))     (γ = 2 for GaAs)
```

**Code:** `science/carrier_statistics.py:n(), p(), ni()`

The intrinsic carrier density n_i is computed from the bandgap:

```
n_i² = N_c · N_v · e^(−E_g / V_t)
```

## Recombination

Three recombination mechanisms are implemented, all proportional to (np − n_i²) by detailed balance:

### SRH (Shockley–Read–Hall)

```
R_SRH = (np − n_i²) / [τ_p(n + n_1) + τ_n(p + p_1)]
```

where n_1 = n_i·e^{E_t}, p_1 = n_i·e^{−E_t}, and E_t is the trap level relative to the intrinsic level.

### Radiative

```
R_rad = B_r · (np − n_i²)
```

### Auger

```
R_Auger = (C_n·n + C_p·p) · (np − n_i²)
```

**Code:** `science/recombination.py:srh(), radiative(), auger(), total()`

The total recombination R = R_SRH + R_rad + R_Auger is used in both the residual (drift-diffusion equations) and the Jacobian (its partial derivatives with respect to φ_n, φ_p, φ).

## Boundary Conditions

The device contacts use **surface-recombination-velocity (SRV)** boundary conditions:

```
J_n(L) = S_nr · (n(L) − n_0)     (right contact)
J_n(0) = S_nl · (n(0) − n_0)     (left contact)
```

and similarly for holes. The electrostatic potential boundary is Dirichlet (fixed by the applied bias and work function).

**Code:** `science/contacts.py:contact_phin(), contact_phip(), contact_phi()`

The equilibrium boundary (`boundary_eq`) sets the built-in potential from the work function difference. The biased boundary (`boundary_bias`) adds the applied voltage.

## Units and Scaling

All internal quantities are **dimensionless**:

| Quantity | Scaling |
|----------|---------|
| Energy | V_t = kT/q (thermal voltage) |
| Length | L_0 = sqrt(ε·V_t / (q·N_ref)) (Debye length) |
| Density | N_ref (reference doping) |
| Current | q·N_ref·D_n / L_0 |

Physical units are applied only at the library boundary (`units.py`). The scaling is temperature-dependent via `thermal_scales(T)`.

## DOF Layout

The unknown field is a single **interleaved vector**:

```
[φn₀, φp₀, φ₀, φn₁, φp₁, φ₁, …, φn_{N-1}, φp_{N-1}, φ_{N-1}]
```

This layout is defined **only** in `pot2vec`/`vec2pot` (`fields.py`). Changing it requires re-deriving the analytic Jacobian and residual.
