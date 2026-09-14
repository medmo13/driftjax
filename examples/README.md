# DriftJax v0.1.3 example gallery

The active gallery is organized around scientific questions and API contracts.
Each script writes one publication-style PNG (300 dpi, shared
`driftjax.viz.style` look) and a JSON sidecar to `outputs/` inside the script's
own tier (`research/outputs/`, `tutorial/outputs/`, `developer/outputs/`) — except
the two pure verification scripts (`research/05`, `developer_adjoint_check`),
which report a single number as text + JSON and deliberately produce no
figure.  Layer thicknesses follow the deltapv convention of the library:
**centimetres** (`Device(layers=[(thickness_cm, material, doping_cm3), ...])`). The JSON
records the result payload plus software/hardware metadata; the final stdout
line of every script is a machine-parseable `EXAMPLE_RESULT <name> k=v ...`
summary for CI assertions.

Run from the repository root:

```bash
export JAX_ENABLE_X64=1
export MPLBACKEND=Agg
export PYTHONPATH=src:.
python examples/research/01_device_iv.py
```

All scripts run at full production resolution and accept `--output-dir`
(isolated reproducibility runs), so gallery figures and release numbers
describe the same discretisation.

Every IV figure reports the full figure-of-merit set (eta, V_oc, J_sc, FF and
the MPP point); device examples show deltapv-style equilibrium band diagrams
(with the electrostatic bending), carrier/charge densities, and per-layer
material bars; the inverse-design examples (06, 12, 13, 15) plot the
optimizer iteration history alongside the design trajectory.

Figures share one annotation vocabulary (`examples/support.py`): panel tags
`(a)`, `(b)`, …; layer shading with layer names on 1-D profiles; $J_{sc}$ /
MPP / $V_{oc}$ guides on IV curves; slope-guide segments on log–log
convergence plots; and $V_{oc}$-knee zoom insets.

## Reference values (how the gallery numbers compare)

The showcase devices are the **published deltapv benchmarks**, so every
headline number is citable against the reference code and the literature
(NREL best-research-cell chart; values are records for *optimized* lab
cells with ARC, texture and light trapping that these 1-D models omit):

| device (example) | driftjax | deltapv published | literature context |
|---|---|---|---|
| ex1 Eg=1.5 p-n (01, 06) | **20.0%** (Jsc 20.3, Voc 1.05, FF 0.85) | 19.98% | Si record 27.3%; this Eg=1.5 model is not c-Si |
| ex2 CdS/CdTe (02) | **13.3%** | 13.31% | CdTe record 22.4% (needs graded absorption, back field) |
| perovskite p-i-n (12) | **~20%** | — | perovskite record 26.7% |
| tandem top cell (07) | top Jsc 17.4 mA/cm²; best ideal 2-terminal tandem η 20.9% (t_bot = 120 µm) | — | tandem record 34.6% |
| inverse recovery (13) | fit 1732 nm vs truth 1800 nm | — | — |

The remaining examples (04, 08, 09, 11, 14, developer/*) use the plain
2 µm Si `pn_device` with symmetric Sn = 1e7 cm/s contacts; its ~6.6% PCE is
the correct physics for that structure (thin Si, no light trapping, strong
contact recombination) — those examples demonstrate solver features, not
record efficiencies.

## Research workflows

| Script | Scientific question | Primary output |
|---|---|---|
| `research/01_device_iv.py` | Is the mesh converged for reported IV metrics? | IV curves and convergence error |
| `research/02_heterojunction_physics.py` | How do band offsets relate to a heterojunction IV? | band alignment and IV |
| `research/03_optical_model_comparison.py` | Does coherent optics change the predicted device response? | Beer–Lambert/TMM comparison |
| `research/04_temperature_recombination.py` | How does temperature affect performance under fixed material assumptions? | temperature trends |
| `research/06_constrained_inverse_design.py` | Which bounded thickness/doping combination improves efficiency? | derivative-free design trajectory |
| `research/07_tandem_current_matching.py` | How does bottom-cell thickness affect ideal series matching? | mismatch and ideal tandem $\eta$ |
| `research/08_statistics_regime_map.py` | When do carrier-statistics assumptions change device predictions? | statistics regime map |
| `research/09_srv_contacts.py` | How much do slow contacts cost in Voc and FF? | SRV sweep trends + IV family |
| `research/10_generation_profile.py` | How deep does the light reach? | generation profiles vs thickness/gap |
| `research/11_transient_small_signal.py` | Does the AC zero-frequency limit agree with DC sensitivity? | transient and admittance |
| `research/12_material_thickness_design_map.py` | How do bandgap and thickness trade off under bounds? | adjoint-driven co-design map |
| `research/13_target_iv_structure.py` | Can structure be recovered from a target IV curve? | inverse recovery |
| `research/14_dgsm_global.py` | Which parameters matter globally (derivative-based sensitivity)? | DGSM ranking |
| `research/15_deltapv_crosscode.py` | Do DriftJax and ∂PV agree, and what does each cost? | IV overlays + timing table |
| `research/16_graded_doping.py` | Does grading the junction help collection? | abrupt vs graded IV + field |

## Tutorial workflows

| Script | What you learn | Output |
|---|---|---|
| `tutorial/01_hello_iv.py` | Build a device, run a sweep, read FoM | 4-panel dossier |
| `tutorial/02_bands_and_charge.py` | Equilibrium vs biased bands, QFL splitting | two dossiers |
| `tutorial/03_first_gradient.py` | One-line adjoint gradient + FD check | sensitivities + dossier |
| `tutorial/04_first_optimization.py` | 2-parameter SLSQP with exact gradients | before/after dossiers |

## Developer workflows

| Script | Contract being exercised |
|---|---|
| `developer/developer_adjoint_check.py` | differentiability of a material parameter through the solve | text + JSON verification (no figure) |
| `developer/developer_batching_and_jit.py` | JIT warm-up, batched bias solves, and wavelength sharding |

The old numbered scripts and their generated artifacts are retained only as
`archive/v0.1.1-original/`; they are not part of the active gallery or release
quality gate.
