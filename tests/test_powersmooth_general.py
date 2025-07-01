import numpy as np
from numpy.testing import assert_allclose

from powersmooth.powersmooth import (
    powersmooth_general,
    upsample_with_mask,
    upsample_to_uniform,
)


# Helper polynomial functions

def constant_func(x):
    return np.full_like(x, 2.5)


def linear_func(x):
    return 1.3 * x + 0.5


def quadratic_func(x):
    return 0.5 * x**2 + 1.0 * x + 0.1


def cubic_func(x):
    return -0.2 * x**3 + 0.5 * x**2 + 1.0 * x + 2.0


def _check_unchanged(x, y, weights):
    """Helper asserting that smoothing keeps the values at sample points."""
    smooth = powersmooth_general(x, y, weights)

    xu, yu, mask = upsample_with_mask(x, y, dx=0.05)
    smooth_up = powersmooth_general(xu, yu, weights, mask)

    xu2, yu2, mask2 = upsample_to_uniform(x, y, dx=0.05)
    smooth_up2 = powersmooth_general(xu2, yu2, weights, mask2)

    assert_allclose(smooth, y, rtol=1e-6)
    assert_allclose(smooth_up[mask == 1], yu[mask == 1], rtol=1e-6)
    assert_allclose(smooth_up2[mask2 == 1], yu2[mask2 == 1], rtol=1e-6)


# Constant function should remain constant for any weights

def test_constant_uniform_and_nonuniform():
    x_uniform = np.linspace(0.0, 1.0, 6)
    x_nonuniform = np.array([0.0, 0.2, 0.35, 0.7, 1.0])
    weights = {1: 0.5, 2: 0.2, 3: 0.1}

    for x in (x_uniform, x_nonuniform):
        y = constant_func(x)
        _check_unchanged(x, y, weights)


def test_linear_no_first_derivative_penalty():
    x_uniform = np.linspace(0.0, 1.0, 6)
    x_nonuniform = np.array([0.0, 0.15, 0.4, 0.65, 1.0])
    weights = {2: 0.1, 3: 1e-5}  # first derivative weight zero

    for x in (x_uniform, x_nonuniform):
        y = linear_func(x)
        _check_unchanged(x, y, weights)


def test_linear_with_first_derivative_penalty_changes():
    x = np.linspace(0.0, 1.0, 6)
    y = linear_func(x)
    result = powersmooth_general(x, y, {1: 1.0})
    assert np.max(np.abs(result - y)) > 1e-6


def test_quadratic_no_second_derivative_penalty():
    x_uniform = np.linspace(0.0, 1.0, 6)
    x_nonuniform = np.array([0.0, 0.2, 0.55, 0.85, 1.0])
    weights = {2:1e-10,3: 1e-2} #small second derivative penalty for stability

    for x in (x_uniform, x_nonuniform):
        y = quadratic_func(x)
        _check_unchanged(x, y, weights)


def test_quadratic_with_second_derivative_penalty_changes():
    x = np.linspace(0.0, 1.0, 6)
    y = quadratic_func(x)
    result = powersmooth_general(x, y, {2: 1.0})
    assert np.max(np.abs(result - y)) > 1e-6


def test_cubic_no_penalty():
    x_uniform = np.linspace(0.0, 1.0, 6)
    x_nonuniform = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    weights = {2: 1e-12,3: 1e-12} #small penalty for stability

    for x in (x_uniform, x_nonuniform):
        y = cubic_func(x)
        _check_unchanged(x, y, weights)


def test_cubic_with_third_derivative_penalty_changes():
    x = np.linspace(0.0, 1.0, 6)
    y = cubic_func(x)
    result = powersmooth_general(x, y, {3: 1.0})
    assert np.max(np.abs(result - y)) > 1e-6

if __name__ == "__main__":
    tests = [
        test_constant_uniform_and_nonuniform,
        test_linear_no_first_derivative_penalty,
        test_linear_with_first_derivative_penalty_changes,
        test_quadratic_no_second_derivative_penalty,
        test_quadratic_with_second_derivative_penalty_changes,
        test_cubic_no_penalty,
        test_cubic_with_third_derivative_penalty_changes,
    ]

    for test in tests:
        print(f"Running {test.__name__}...")
        test()
        print(f"✓ {test.__name__} passed")