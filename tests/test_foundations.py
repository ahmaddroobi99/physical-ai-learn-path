import numpy as np

from physical_ai_lab.foundations import (
    meters_per_second_to_kmh,
    meters_to_centimeters,
    normalize,
)


def test_normalize_maps_endpoints_to_zero_and_one():
    values = np.array([1.2, 1.4, 1.1, 1.8, 2.0], dtype=np.float32)
    scaled = normalize(values)
    assert scaled.dtype == np.float32
    assert scaled[np.argmin(values)] == 0.0
    assert scaled[np.argmax(values)] == 1.0


def test_normalize_constant_input_is_zero():
    assert np.all(normalize(np.array([3.0, 3.0, 3.0])) == 0.0)


def test_unit_conversions():
    assert np.allclose(meters_to_centimeters([0.7, 1.4]), [70.0, 140.0])
    assert np.isclose(float(meters_per_second_to_kmh([1.0])[0]), 3.6)
