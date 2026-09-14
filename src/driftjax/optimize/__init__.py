"""Design optimization API: optimizers (SLSQP package, Adam) + feasibility."""

from driftjax.optimize.constraints import all_feasible, material_feasibility
from driftjax.optimize.objectives import efficiency, iv_curve_distance, iv_mse
from driftjax.optimize.optimizers import (
    adam,
    diagonal_scale,
    gradient_scale,
    nelder_mead,
    slsqp,
    slsqp_multistart,
)

__all__ = [
    "slsqp",
    "slsqp_multistart",
    "nelder_mead",
    "adam",
    "material_feasibility",
    "all_feasible",
    "efficiency",
    "iv_mse",
    "iv_curve_distance",
    "gradient_scale",
    "diagonal_scale",
]
