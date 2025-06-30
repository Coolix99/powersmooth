import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def finite_diff_matrix(x: np.ndarray, order: int) -> sp.csr_matrix:
    """
    Construct a sparse finite difference matrix for non-uniformly spaced data.

    Parameters:
    - x: 1D array of positions (non-uniform spacing allowed)
    - order: Derivative order (1, 2, or 3)

    Returns:
    - Sparse CSR matrix D such that D @ y approximates the derivative y^(order)
    """
    n = len(x)
    rows, cols, data = [], [], []

    if order == 1:
        for i in range(1, n-1):
            dx = x[i+1] - x[i-1]
            rows += [i, i]
            cols += [i-1, i+1]
            data += [-1/dx, 1/dx]
    elif order == 2:
        for i in range(1, n-1):
            dx1 = x[i] - x[i-1]
            dx2 = x[i+1] - x[i]
            rows += [i, i, i]
            cols += [i-1, i, i+1]
            c1 = 2.0 / (dx1 * (dx1 + dx2))
            c2 = -2.0 / (dx1 * dx2)
            c3 = 2.0 / (dx2 * (dx1 + dx2))
            data += [c1, c2, c3]
    elif order == 3:
        for i in range(2, n-2):
            dx1 = x[i] - x[i-1]
            dx2 = x[i+1] - x[i]
            dx3 = x[i+2] - x[i+1]
            denom1 = dx1 * (dx1 + dx2) * (dx1 + dx2 + dx3)
            denom2 = dx2 * (dx1 + dx2) * (dx2 + dx3)
            denom3 = dx3 * (dx2 + dx3) * (dx1 + dx2 + dx3)
            rows += [i]*4
            cols += [i-1, i, i+1, i+2]
            data += [-1/denom1, 1/denom1 + 1/denom2, -1/denom2 - 1/denom3, 1/denom3]
    else:
        raise ValueError("Only 1st, 2nd, 3rd derivatives supported")

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

def powersmooth_general(x: np.ndarray,
                         y: np.ndarray,
                         weights: dict,
                         mask: np.ndarray = None) -> np.ndarray:
    """
    Perform smoothing on non-uniformly spaced data with derivative-based regularization.

    Parameters:
    - x: 1D array of positions (non-uniformly spaced)
    - y: 1D array of observations at positions x
    - weights: dict mapping derivative order to penalty weight (e.g., {1: 0.1, 2: 0.01})
    - mask: Optional array (same shape as y) indicating where data fidelity should apply (1=True, 0=False)

    Returns:
    - Smoothed version of y as 1D array
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    assert len(x) == len(y), "x and y must have same length"
    n = len(x)

    if mask is None:
        mask = np.ones(n)
    else:
        mask = np.asarray(mask).astype(float).flatten()

    A0 = sp.diags(mask, 0, shape=(n, n), format='csr')
    b = mask * y

    A_penalty = sp.csr_matrix((n, n))

    for order, w in weights.items():
        Dk = finite_diff_matrix(x, order)
        A_penalty += w * (Dk.T @ Dk)

    A = A0 + A_penalty

    y_smooth = spla.spsolve(A, b)
    return y_smooth

def upsample_with_mask(x: np.ndarray, y: np.ndarray, dx: float) -> tuple:
    """
    Densify a non-uniform dataset by inserting intermediate points between known values.

    Parameters:
    - x: 1D array of original positions
    - y: 1D array of values at positions x
    - dx: Desired approximate spacing between new points

    Returns:
    - x_new: 1D array with original and inserted positions
    - y_new: 1D array with original y values and zeros at new positions
    - mask_new: 1D array with 1 at original points and 0 at interpolated positions
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    assert len(x) == len(y), "x and y must have the same length"

    x_new = []
    y_new = []
    mask_new = []

    for i in range(len(x) - 1):
        x_start = x[i]
        x_end = x[i + 1]
        segment = [x_start]

        num_points = int(np.floor((x_end - x_start) / dx))
        if num_points > 0:
            segment += list(np.linspace(x_start + dx, x_end - dx, num_points))
        
        x_new.extend(segment)
        y_new.extend([y[i]] + [0] * num_points)
        mask_new.extend([1] + [0] * num_points)

    x_new.append(x[-1])
    y_new.append(y[-1])
    mask_new.append(1)

    return np.array(x_new), np.array(y_new), np.array(mask_new)
