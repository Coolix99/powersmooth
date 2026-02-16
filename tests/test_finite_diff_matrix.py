import numpy as np
from numpy.testing import assert_allclose

from powersmooth.powersmooth import finite_diff_matrix, upsample_with_mask


def test_finite_diff_uniform():
    x = np.arange(6, dtype=float)

    D1 = finite_diff_matrix(x, 1).toarray()
    expected1 = np.array([
        [-1.5,  2.0, -0.5,  0.0,  0.0,  0.0],  # forward
        [-0.5,  0.0,  0.5,  0.0,  0.0,  0.0],
        [ 0.0, -0.5,  0.0,  0.5,  0.0,  0.0],
        [ 0.0,  0.0, -0.5,  0.0,  0.5,  0.0],
        [ 0.0,  0.0,  0.0, -0.5,  0.0,  0.5],
        [ 0.0,  0.0,  0.0,  0.5, -2.0,  1.5],  # backward
    ])
    assert_allclose(D1, expected1)

    D2 = finite_diff_matrix(x, 2).toarray()
    expected2 = np.array([
        [ 1.0, -2.0,  1.0,  0.0,  0.0,  0.0],  # forward
        [ 1.0, -2.0,  1.0,  0.0,  0.0,  0.0],
        [ 0.0,  1.0, -2.0,  1.0,  0.0,  0.0],
        [ 0.0,  0.0,  1.0, -2.0,  1.0,  0.0],
        [ 0.0,  0.0,  0.0,  1.0, -2.0,  1.0],
        [ 0.0,  0.0,  0.0,  1.0, -2.0,  1.0],  # backward
    ])
    assert_allclose(D2, expected2)

    D3 = finite_diff_matrix(x, 3).toarray()
    expected3 = np.array([
        [-2.5,  9.0, -12.0, 7.0, -1.5, 0.0],
        [-1.5,  5.0, -6.0,  3.0, -0.5, 0.0],
        [-0.5,  1.0,  0.0, -1.0,  0.5, 0.0],
        [ 0.0, -0.5,  1.0,  0.0, -1.0, 0.5],
        [ 0.0,  0.5, -3.0,  6.0, -5.0, 1.5],
        [ 0.0,  1.5, -7.0, 12.0, -9.0, 2.5],
    ])
    assert_allclose(D3, expected3)


def test_upsample_with_mask():
    x = np.array([0.0, 2.0])
    y = np.array([1.0, 2.0])
    dx = 0.6

    x_new, y_new, mask_new = upsample_with_mask(x, y, dx)

    expected_x = np.array([0.0, 0.6, 1.0, 1.4, 2.0])
    expected_y = np.array([1.0, 0.0, 0.0, 0.0, 2.0])
    expected_mask = np.array([1, 0, 0, 0, 1])

    assert_allclose(x_new, expected_x)
    assert_allclose(y_new, expected_y)
    assert_allclose(mask_new, expected_mask)

if __name__ == "__main__":
    print("Running test_finite_diff_uniform...")
    test_finite_diff_uniform()
    print("✓ test_finite_diff_uniform passed")

    print("Running test_upsample_with_mask...")
    test_upsample_with_mask()
    print("✓ test_upsample_with_mask passed")
