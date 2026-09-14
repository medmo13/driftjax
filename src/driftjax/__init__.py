"""driftjax — 1-D drift-diffusion–Poisson solar-cell simulator in JAX.

End-to-end differentiable 1-D drift-diffusion–Poisson solar-cell simulator in
JAX.  The public surface is a single composable entry point, ``simulate``::

    from driftjax import Device, simulate, Sweep, Newton, BeerLambert
    sol = simulate(dev, Sweep())            # -> Solution (jit/grad/vmap-safe)
    eff = simulate(dev).efficiency          # power conversion efficiency
    grad = jax.grad(lambda d: simulate(d).efficiency)(dev)   # transparent IFT gradient

See README.md and docs/ for the design record.
"""

import logging
import os

import jax

jax.config.update("jax_enable_x64", True)

# Persistent XLA compilation cache. Cold `jax.grad(simulate)` compiles
# hundreds of kernels (~44s); the disk cache makes repeat processes reuse
# them. Correctness-unaffecting. Override with DRIFTJAX_COMPILATION_CACHE
# (<path> to use, "" to disable).
_cache_dir = os.environ.get("DRIFTJAX_COMPILATION_CACHE", None)
if _cache_dir is None:
    _base = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))
    _cache_dir = os.path.join(_base, "driftjax", "xla")
if _cache_dir:
    try:
        os.makedirs(_cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
        # Cache even sub-second kernel compiles (default floor is 1.0s).
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception:
        pass

logger = logging.getLogger("driftjax")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

from driftjax import (
    config as config,
)
from driftjax import (
    io as io,
)
from driftjax import (
    optimize as optimize,  # (slsqp, slsqp_multistart, adam)
)
from driftjax import (
    simulator as simulator,
)
from driftjax.adjoint.api import DirectAdjoint, ImplicitAdjoint
from driftjax.config import DEFAULT_MODE, Mode, mode_info
from driftjax.fields import Material
from driftjax.io import load_material, material
from driftjax.numerics.fused_kernels import (
    FusedKernelManager,
    fused_bernoulli_current,
    fused_generation,
    fused_jacobian_banded,
    fused_residual,
)
from driftjax.optics.api import TMM, BeerLambert, Fresnel
from driftjax.problems import Equilibrium, Sweep
from driftjax.runtime import provenance, sharding
from driftjax.runtime.provenance import (
    register_lineage,
    report_lineage,
)
from driftjax.runtime.sharding import sharded_generation
from driftjax.science.spectrum import spectrum as _am15g
from driftjax.science.tandem import series_two_terminal
from driftjax.simulator import Device


def AM15G():
    """The ASTM G-173 AM1.5G global-tilt spectrum (raw, 1-sun, not rescaled)."""
    return _am15g(normalize=False)


from driftjax.simulate import simulate  # (new, composable entry point)
from driftjax.solution import Solution
from driftjax.solvers.api import BlockThomas, Newton
from driftjax.units import Vt, current, energy
from driftjax.viz.plotting import (
    plot_band_diagram,
    plot_bars,
    plot_charge,
    plot_iv_curve,
)

__version__ = "0.1.16"


__all__ = [
    "config",
    "io",
    "optimize",
    "simulator",
    "DEFAULT_MODE",
    "Mode",
    "mode_info",
    "Material",
    "Device",
    "material",
    "load_material",
    "FusedKernelManager",
    "fused_bernoulli_current",
    "fused_generation",
    "fused_jacobian_banded",
    "fused_residual",
    "provenance",
    "sharding",
    "register_lineage",
    "report_lineage",
    "sharded_generation",
    "series_two_terminal",
    "AM15G",
    "simulate",
    "Solution",
    "Equilibrium",
    "Sweep",
    "Newton",
    "BlockThomas",
    "BeerLambert",
    "TMM",
    "Fresnel",
    "ImplicitAdjoint",
    "DirectAdjoint",
    "Vt",
    "current",
    "energy",
    "plot_band_diagram",
    "plot_bars",
    "plot_charge",
    "plot_iv_curve",
]
