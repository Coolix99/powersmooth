import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def powersmooth(vec: np.ndarray, order: int, weight: float) -> np.ndarray:
    """
    Smooth noisy time-series data using quadratic programming with a Bayesian prior
    on the (order+1)-th derivative.

    Parameters:
    - vec: 1D input array (will be reshaped to column vector)
    - order: Derivative order for smoothing (0, 1, or 2)
    - weight: Weight of the smoothing term

    Returns:
    - Smoothed version of vec
    """
    vec = np.asarray(vec).flatten()
    n = len(vec)

    A0 = sp.eye(n, format='csc')

    if order == 0:
        main_diag = np.r_[1, 2*np.ones(n-2), 1]
        upper_diag = -2 * np.ones(n-1)
        A1 = sp.diags([main_diag, upper_diag], [0, 1], shape=(n, n), format='csc')

    elif order == 1:
        d0 = np.r_[1, 5, 6*np.ones(n-4), 5, 1]
        d1 = -4 * np.r_[0, 1, 2*np.ones(n-3), 1]
        d2 = 2 * np.r_[0, 0, np.ones(n-2)]
        A1 = sp.diags([d0, d1, d2], [0, 1, 2], shape=(n, n), format='csc')

    elif order == 2:
        d0 = np.r_[1, 10, 19, 20*np.ones(n-6), 19, 10, 1]
        d1 = -6 * np.r_[0, 1, 4, 5*np.ones(n-5), 4, 1]
        d2 = 6 * np.r_[0, 0, 1, 2*np.ones(n-4), 1]
        d3 = -2 * np.ones(n)
        A1 = sp.diags([d0, d1, d2, d3], [0, 1, 2, 3], shape=(n, n), format='csc')

    else:
        raise ValueError(f"Order={order} not supported.")

    A = A0 + weight * A1
    A = 0.5 * (A + A.T)  # Ensure symmetry
    b = -2 * vec

    vecs = spla.spsolve(A, -b / 2)
    return vecs

def powersmooth_upsampled(vec: np.ndarray, order: int, weight: float, factor: int) -> np.ndarray:
    vec = np.asarray(vec).flatten()
    n = len(vec)
    new_n = (n - 1) * (factor + 1) + 1

    A0_diag = np.zeros(new_n)
    A0_diag[::factor + 1] = 1  # Fidelity only at original indices
    A0 = sp.diags(A0_diag, 0, shape=(new_n, new_n), format='csc')

    if order == 0:
        main_diag = np.r_[1, 2*np.ones(new_n-2), 1]
        upper_diag = -2 * np.ones(new_n-1)
        A1 = sp.diags([main_diag, upper_diag], [0, 1], shape=(new_n, new_n), format='csc')
    elif order == 1:
        d0 = np.r_[1, 5, 6*np.ones(new_n-4), 5, 1]
        d1 = -4 * np.r_[0, 1, 2*np.ones(new_n-3), 1]
        d2 = 2 * np.r_[0, 0, np.ones(new_n-2)]
        A1 = sp.diags([d0, d1, d2], [0, 1, 2], shape=(new_n, new_n), format='csc')
    elif order == 2:
        d0 = np.r_[1, 10, 19, 20*np.ones(new_n-6), 19, 10, 1]
        d1 = -6 * np.r_[0, 1, 4, 5*np.ones(new_n-5), 4, 1]
        d2 = 6 * np.r_[0, 0, 1, 2*np.ones(new_n-4), 1]
        d3 = -2 * np.ones(new_n)
        A1 = sp.diags([d0, d1, d2, d3], [0, 1, 2, 3], shape=(new_n, new_n), format='csc')
    else:
        raise ValueError(f"Order={order} not supported.")

    A = A0 + weight * A1
    A = 0.5 * (A + A.T)

    b = np.zeros(new_n)
    b[::factor + 1] = -2 * vec

    vecs = spla.spsolve(A, -b / 2)
    return vecs



