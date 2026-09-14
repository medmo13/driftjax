"""Contract tests for the v0.1.3 public simulation API."""

import pytest

import driftjax as dj


def _device():
    material = dj.load_material("Si")
    return dj.Device(
        layers=[(1e-4, material, 1e17), (1e-4, material, -1e17)],
        n_points=24,
        Snl=1e7,
        Snr=1e7,
        Spl=1e7,
        Spr=1e7,
    )


def test_solution_exposes_named_properties():
    solution = dj.simulate(_device(), dj.Sweep(vmax=0.7, n_steps=7))
    voltages, currents = solution.iv_curve()

    assert voltages.shape == currents.shape == (7,)
    assert solution.efficiency == solution.eff
    assert solution.currents is solution.current
    assert solution.at_bias(0.3) is solution.potentials[3]


def test_equilibrium_result_rejects_bias_lookup():
    solution = dj.simulate(_device(), dj.Equilibrium())

    with pytest.raises(ValueError, match="no bias points"):
        solution.at_bias(0.0)


def test_unknown_protocol_is_rejected():
    with pytest.raises(TypeError, match="protocol"):
        dj.simulate(_device(), object())
