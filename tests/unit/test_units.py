"""Dimensionless scaling vs the driftjax's published values."""

import pytest

from driftjax.units import current, density, energy, length, mobility, time, velocity

pytestmark = pytest.mark.smoke  # no solves: unit-scale arithmetic, milliseconds


def test_scale_values():
    assert abs(float(length) - 3.78e-08) < 1e-09
    assert abs(float(energy) - 0.02585) < 0.001
    assert abs(float(density) - 1e19) < 1e-08
    assert abs(float(mobility) - 1.0) < 1e-12
    assert abs(float(current) - 1095000.0) < 0.1 * 1095000.0
    assert float(velocity) > 0 and float(time) > 0


def test_scale_consistency():
    assert abs(float(velocity) - float(length) / float(time)) < 1e-12 * float(velocity)
