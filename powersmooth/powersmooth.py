import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings

def _fd_weights_vandermonde(xi: np.ndarray, x0: float, m: int) -> np.ndarray:
    """
    Compute FD weights w_j so that
        f^(m)(x0) ≈ sum_j w_j * f(xi[j]),
    by solving the Vandermonde system:
        sum_j w_j*(xi[j]-x0)^k = k! * δ_{k,m},  k=0..n-1.
    xi : array of n stencil points
    x0 : center point
    m  : derivative order
    returns: w of length n
    """
    n = len(xi)
    t = xi - x0
    # build moment matrix A[k,j] = t[j]^k
    A = np.vstack([t**k for k in range(n)])      # shape (n,n)
    # RHS: [0,0,..., m!, ...,0] (1 at k=m times m!)
    rhs = np.array([np.math.factorial(k) if k == m else 0 for k in range(n)])
    return np.linalg.solve(A, rhs)               # shape (n,)

def finite_diff_matrix(x: np.ndarray, order: int) -> sp.csr_matrix:
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
            dx1, dx2 = x[i]-x[i-1], x[i+1]-x[i]
            rows += [i]*3
            cols += [i-1, i, i+1]
            data += [
                2.0 / (dx1*(dx1+dx2)),
                -2.0/(dx1*dx2),
                2.0 / (dx2*(dx1+dx2))
            ]

    elif order == 3:
        for i in range(2, n-2):
            idx = np.arange(i-2, i+3)       # [i-2, i-1, i, i+1, i+2]
            xi  = x[idx]
            w   = _fd_weights_vandermonde(xi, x[i], 3)
            rows.extend([i]*5)
            cols.extend(idx.tolist())
            data.extend(w.tolist())


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
    cond = np.linalg.cond(A.toarray())
    if cond > 1e12:
        warnings.warn(
            f"Matrix A is ill-conditioned (cond={cond:.2e}); solution may be unstable.",
            category=RuntimeWarning,
            stacklevel=2
        )
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

def upsample_with_exact_data_inclusion(x: np.ndarray, y: np.ndarray, dx: float, atol=1e-8):
    """
    Create a uniform grid that includes all original x values,
    inserting them if needed, and prepare data/mask for smoothing.

    Parameters
    ----------
    x : array_like
        Original sample positions (non-uniform).
    y : array_like
        Corresponding values.
    dx : float
        Approximate spacing of the uniform grid.
    atol : float
        Absolute tolerance for matching grid points.

    Returns
    -------
    tuple
        x_dense : np.ndarray
            Uniform grid + inserted original points (sorted)
        y_dense : np.ndarray
            y values at original positions, zeros elsewhere
        mask : np.ndarray
            1 where original y-values are placed, 0 elsewhere
        inserted_mask : np.ndarray
            True where original points were inserted (to optionally remove after smoothing)
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    assert len(x) == len(y)

    # Create base uniform grid
    x_uniform = np.arange(x[0], x[-1] + dx * 0.5, dx)

    # Identify x[i] not close to any x_uniform
    close_mask = np.isclose(x[:, None], x_uniform[None, :], atol=atol).any(axis=1)
    x_missing = x[~close_mask]

    # Merge and sort
    x_dense = np.concatenate([x_uniform, x_missing])
    x_dense = np.unique(np.sort(x_dense))  # ensure strict order and no duplicates

    # Place y values and build masks
    y_dense = np.zeros_like(x_dense)
    mask = np.zeros_like(x_dense, dtype=int)
    inserted_mask = np.zeros_like(x_dense, dtype=bool)

    for xi, yi in zip(x, y):
        idx = np.argmin(np.abs(x_dense - xi))
        if np.abs(x_dense[idx] - xi) <= atol:
            y_dense[idx] = yi
            mask[idx] = 1
        else:
            raise RuntimeError("Failed to match original point — should not happen.")

    # mark points that were not in the original grid
    inserted_mask = ~np.isclose(x_dense[:, None], x_uniform[None, :], atol=atol).any(axis=1)

    return x_dense, y_dense, mask, inserted_mask