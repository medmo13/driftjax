"""Developer example: a minimal implicit-differentiation check.

This is a VERIFICATION, not a research figure: it produces one number (the
relative error between the implicit gradient and a central finite
difference), so it reports text + a JSON sidecar instead of a plot.  The
one-sentence statement this example backs is:

    "d(efficiency)/dE_g computed by the IFT adjoint matches a central finite
    difference to < 1e-4 relative error on the canonical p-n device."
"""

import dataclasses

import jax
import jax.numpy as jnp

import driftjax as dj
from examples.support import (
    example_args,
    execution_metadata,
    pn_device,
    report,
    save_json,
    si_material,
)


def main():
    example_args("developer_adjoint_check")
    points = 500

    def make_device(eg):
        material = dataclasses.replace(si_material(), Eg=eg)
        return pn_device(material, points=points)

    def objective(eg):
        return dj.simulate(make_device(eg), dj.Sweep(vmax=0.75, n_steps=9)).efficiency

    eg = jnp.asarray(1.12)
    gradient = float(jax.grad(objective)(eg))
    step = 1e-4
    finite_difference = float((objective(eg + step) - objective(eg - step)) / (2 * step))
    relative_error = abs(gradient - finite_difference) / max(abs(finite_difference), 1e-14)

    print()
    print("Implicit-differentiation check (d efficiency / d Eg):")
    print(f"  IFT adjoint        = {gradient: .6e}")
    print(f"  central FD         = {finite_difference: .6e}")
    print(f"  relative error     = {relative_error:.2e}  (gate < 1e-4)")
    print()
    save_json(
        "developer_adjoint_check",
        {
            "metadata": execution_metadata(),
            "bandgap_eV": float(eg),
            "implicit_gradient": gradient,
            "finite_difference": finite_difference,
            "relative_error": relative_error,
            "figure": None,  # verification result is reported as text, not plotted
        },
    )
    report("developer_adjoint_check", grad_fd_relative_error=float(relative_error))


if __name__ == "__main__":
    main()
