# 3-Layer Perovskite Stress Test (regression)

**WARNING: This is NOT a solar-cell simulation. It is a singular-matrix stress test.**

The n-p-n stack (n+/p/n+ with electron-selective contacts at both ends) has
**no hole-collection path**: photogenerated holes in the intrinsic region
cannot reach either contact, so Jsc ~ 0 and Voc = nan. The device does not
produce a physical IV curve.

**Why this test exists:** the 3-layer n-p-n structure produces a Jacobian with
kappa ~ 4e45 (structurally rank-deficient). It is used to exercise:
1. The pivoted-banded LAPACK (dgbsv) -> truncated-SVD (lstsq) fallback path.
2. The solver_fallback.json archived record (converged=True, max_residual=2.19e-8
   with the *default* raw spectrum).

**Known fragility:** with spectrum(normalize=True) (1000 W/m2), the +11%
incident-drive pushes the singular device past the lstsq fallback recovery
radius, causing divergence.

**Positive control:** see test_perovskite_pin.py -- a structurally-sound
p-i-n perovskite (non-selective ohmic contacts) that converges to machine
precision with a textbook diode J-V.

**Do NOT cite the 3-layer device PCE as a device-performance number.**
