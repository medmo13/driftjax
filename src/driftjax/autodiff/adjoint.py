"""Adjoint differentiation through the Newton solve (implicit function theorem).

Design-gradient adjoint (IFT) with full design→physics chain.
chain: F̃(des, x) = comp_F(init_cell(des, ls), boundary_bias(...), x) so the
gradient captures the optics channel (generation), the boundary channel
(contact potentials / neq / peq) and the transport channel.

   1. forward:  F̃(x*, p) = 0
   2. adjoint:  λ = solve(Jᵀ, g_x),  J = ∂F̃/∂x,  g_x = ∂L/∂x
   3. IFT:      dL/dp = g_p − λᵀ·F_p

Known limitation (package, PLAN §11): at V = 0 the equilibrium Jacobian is
degenerate (cond ~ 1e18).  Verify gradients at a small forward bias
(V ≳ 0.3 V).
"""

from __future__ import annotations

import jax

from driftjax.fields import DeviceDesign, LightSource, Potentials, pot2vec, vec2pot
from driftjax.numerics.residual import F_jacobian, comp_F
from driftjax.science.contacts import boundary_bias
from driftjax.science.spectrum import spectrum
from driftjax.simulator import Device, init_cell
from driftjax.solvers.newton import solve_newton


def _cell_of(des: DeviceDesign, ls: LightSource, alpha_mode: str = "table", optics=None):
    if optics is not None:
        return init_cell(des, ls, alpha_mode=alpha_mode, optics=optics)
    return init_cell(des, ls, alpha_mode=alpha_mode)


def forward_solve(
    des: DeviceDesign,
    ls: LightSource,
    v_applied: float,
    pot_ini: Potentials,
    tol: float = 1e-12,
    alpha_mode: str = "table",
    optics=None,
) -> Potentials:
    """Solve F̃(des, x) = 0 with the production Newton path."""
    cell = _cell_of(des, ls, alpha_mode, optics=optics)
    bound = boundary_bias(cell, v_applied)
    return solve_newton(cell, bound, pot_ini, tol=tol)[0]


def _solve_adjoint(J, g_x, tol=1e-8):
    """λ = solve(Jᵀ, g_x) via dense pivoted LU (legacy helper).

    Use ``driftjax.simulate(..., adjoint=...)`` for production gradients.
    """
    from driftjax.solvers.newton import _linear_solve
    return _linear_solve(J.T, g_x)[0]


def manual_adjoint_grad(
    loss_fn,
    design_or_device,
    v_applied: float,
    pot_ini: Potentials,
    ls: LightSource | None = None,
    alpha_mode: str | None = None,
    tol: float = 1e-12,
    optics=None,
):
    """(L, grad_pytree): dL/ddes by hand-rolled IFT through the full chain.

    Accepts either a DeviceDesign or a Device; when a Device is given, the
    illumination and optics model default to the device\'s own (beer-lambert).
    """
    alpha_mode_resolved: str | None = alpha_mode
    if isinstance(design_or_device, Device):
        ls = ls or spectrum()
        des = design_or_device.design()
        if alpha_mode_resolved is None:
            alpha_mode_resolved = getattr(des, "alpha_mode", "table")
        if optics is None and hasattr(design_or_device, "alpha_mode"):
            # allow Device optics to flow through when not explicitly given
            optics = getattr(design_or_device, "_optics", None)
    else:
        ls = ls or spectrum()
        des = design_or_device
    if alpha_mode_resolved is None:
        alpha_mode_resolved = "table"
    cell = _cell_of(des, ls, alpha_mode_resolved, optics=optics)
    bound = boundary_bias(cell, v_applied)

    def F_wrapped(d, xv):
        c = _cell_of(d, ls, alpha_mode_resolved, optics=optics)
        return comp_F(c, boundary_bias(c, v_applied), vec2pot(xv))

    def loss_flat(d, xv):
        return loss_fn(_cell_of(d, ls, alpha_mode_resolved, optics=optics), vec2pot(xv))

    pot = forward_solve(des, ls, v_applied, pot_ini, tol=tol, alpha_mode=alpha_mode_resolved, optics=optics)
    x = pot2vec(pot)

    L = loss_fn(cell, pot)
    g_x = jax.grad(lambda xv: loss_flat(des, xv))(x)
    J = F_jacobian(cell, bound, pot)
    lam = _solve_adjoint(J, g_x)

    g_c = jax.grad(lambda d: loss_flat(d, x))(des)
    # Only lam^T @ dF/ddes is needed: contract the adjoint through a VJP
    # (ONE reverse pass) instead of materialising dF/ddes with jacrev
    # (500 reverse passes = one per residual row).
    _, F_vjp = jax.vjp(lambda d: F_wrapped(d, x), des)
    lam_Fc = F_vjp(lam)[0]
    grad = jax.tree.map(lambda a, b: a - b, g_c, lam_Fc)
    return L, grad
