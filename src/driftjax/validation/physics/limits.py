"""Tier III — Physical model limit tests."""

from __future__ import annotations

import jax.numpy as jnp


def test_equilibrium_limit() -> dict:
    """V8: At equilibrium, verify np → n_i² and recombination → 0."""
    import driftjax as dj
    from driftjax.science.carrier_statistics import n as _n
    from driftjax.science.carrier_statistics import ni
    from driftjax.science.carrier_statistics import p as _p
    from driftjax.science.contacts import boundary_bias
    from driftjax.science.recombination import total as recomb_total
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell
    from driftjax.solvers.continuation import equilibrium_guess
    from driftjax.solvers.newton import solve_eq

    ls = spectrum()
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )
    dev = dj.Device(
        n_points=100, layers=[(4e-5, mat, 1e17), (1e-4, mat, -1e15)], Snl=1e7, Snr=0, Spl=0, Spr=1e7
    )
    cell = init_cell(dev.design(), ls)
    pot_eq = solve_eq(cell, boundary_bias(cell, 0.0), equilibrium_guess(cell).phi)

    n_arr = _n(cell, pot_eq)
    p_arr = _p(cell, pot_eq)
    ni_val = ni(cell)
    ni2 = ni_val**2
    np_error = float(jnp.max(jnp.abs(n_arr * p_arr - ni2) / (ni2 + 1e-30)))

    R_total = recomb_total(cell, pot_eq)
    R_max = float(jnp.max(jnp.abs(R_total)))
    R_rel = R_max / (float(jnp.mean(jnp.abs(n_arr * p_arr))) + 1e-30)

    return {
        "test": "V8_equilibrium_limit",
        "np_over_ni2_error": np_error,
        "R_max": R_max,
        "passed": np_error < 1e-3 and R_rel < 1e-3,
    }


def test_optical_limiting() -> dict:
    """V9: Optical limiting — front generation should exceed back."""
    import driftjax as dj
    from driftjax.science.spectrum import spectrum
    from driftjax.simulator import init_cell

    ls = spectrum()
    mat = dj.material(
        Chi=3.9, Eg=1.5, eps=9.4, Nc=8e17, Nv=1.8e19, mn=100, mp=100, tn=1e-8, tp=1e-8, A=1e4
    )
    dev = dj.Device(
        n_points=100, layers=[(1e-4, mat, 1e17), (1e-4, mat, -1e15)], Snl=1e7, Snr=0, Spl=0, Spr=1e7
    )
    cell = init_cell(dev.design(), ls)

    G = cell.G
    front = float(jnp.mean(jnp.abs(G[: len(G) // 4])))
    back = float(jnp.mean(jnp.abs(G[3 * len(G) // 4 :])))
    ratio = front / (back + 1e-30)

    return {"test": "V9_optical_limiting", "front_over_back_ratio": ratio, "passed": ratio > 1.0}


def print_summary(results: list[dict]) -> None:
    print(f"\n{'=' * 60}")
    print("TIER III — Physical Model Limits")
    print(f"{'=' * 60}")
    for r in results:
        status = "PASS" if r.get("passed", False) else "FAIL"
        print(f"  [{status}] {r['test']}")
        for k, v in r.items():
            if k not in ("test", "passed"):
                print(f"    {k}: {v}")
    print()


if __name__ == "__main__":
    print_summary([test_equilibrium_limit(), test_optical_limiting()])
