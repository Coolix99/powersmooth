import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import warnings
from typing import Dict, Tuple

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
    """
    Construct sparse finite-difference matrix approximating the
    `order`-th derivative on possibly non-uniform grid `x`.

    Uses:
        - 3-point stencil for 1st and 2nd derivatives
        - 5-point stencil for 3rd derivative

    One-sided stencils are automatically used at boundaries.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Strictly increasing grid points.
    order : int
        Derivative order (1, 2, or 3).

    Returns
    -------
    csr_matrix of shape (n, n)
        Sparse derivative operator.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if order not in (1, 2, 3):
        raise ValueError("Only 1st, 2nd, 3rd derivatives supported.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    # ---- stencil size selection ----
    if order in (1, 2):
        stencil_size = 3
    else:  # order == 3
        stencil_size = 5

    if n < stencil_size:
        raise ValueError("Not enough grid points for requested derivative.")

    rows, cols, data = [], [], []

    for i in range(n):
        # determine stencil indices
        left = max(0, i - stencil_size // 2)
        right = min(n, left + stencil_size)

        if right - left < stencil_size:
            left = max(0, right - stencil_size)

        idx = np.arange(left, right)
        xi = x[idx]

        w = _fd_weights_vandermonde(xi, x[i], order)

        rows.extend([i] * len(idx))
        cols.extend(idx.tolist())
        data.extend(w.tolist())

    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))

def powersmooth_general(
    x: np.ndarray,
    y: np.ndarray,
    weights: Dict[int, float],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Solve the regularized least-squares problem

        min_u || M (u - y) ||_2^2 + sum_k w_k || D_k u ||_2^2

    on a possibly non-uniform grid.

    Parameters
    ----------
    x : ndarray of shape (n,)
        Strictly increasing sample positions.
    y : ndarray of shape (n,)
        Observed data values.
    weights : dict[int, float]
        Mapping derivative order to regularization weight.
    mask : ndarray of shape (n,), optional
        Data fidelity mask (1 = enforce data, 0 = free).

    Returns
    -------
    ndarray of shape (n,)
        Smoothed signal.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    assert len(x) == len(y), "x and y must have same length"

    if not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    for order, w in weights.items():
        if order < 1:
            raise ValueError("Derivative order must be >= 1.")
        if w < 0:
            raise ValueError("Regularization weights must be non-negative.")

    n = len(x)

    if mask is None:
        mask = np.ones(n)
    else:
        mask = np.asarray(mask, dtype=float).flatten()
        if len(mask) != n:
            raise ValueError("mask must have same length as x and y.")

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

def upsample_with_mask(
    x: np.ndarray,
    y: np.ndarray,
    dx: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

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

def upsample_with_exact_data_inclusion(
    x: np.ndarray,
    y: np.ndarray,
    dx: float,
    atol: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def powersmooth_upsample(
    x: np.ndarray,
    y: np.ndarray,
    weights: Dict[int, float],
    dx: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth data by first inserting intermediate points between original x values.
    Original values are retained via a binary mask, and intermediate values are
    filled in by solving a regularized smoothing system.

    Parameters
    ----------
    x : array_like
        Original sample positions (not necessarily uniform).
    y : array_like
        Data values at positions x.
    weights : dict
        Regularization weights for derivatives (e.g. {2: 1e-3, 3: 1e-3}).
    dx : float
        Approximate spacing between inserted (upsampled) grid points.

    Returns
    -------
    x_up : ndarray
        Densified x grid including original and inserted points.
    smooth_y : ndarray
        Smoothed y values on the densified grid.
    """
    x_up, y_up, mask_up = upsample_with_mask(x, y, dx)
    smooth_y = powersmooth_general(x_up, y_up, weights=weights, mask=mask_up)
    return x_up, smooth_y,mask_up

def powersmooth_on_uniform_grid(
    x: np.ndarray,
    y: np.ndarray,
    weights: Dict[int, float],
    dx: float = 0.1,
    return_dense: bool = False
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth non-uniform data onto a uniform grid by embedding the original
    points into a regular grid (with insertion if necessary), applying a
    regularized smoothing solver, and removing extra points afterward.

    Parameters
    ----------
    x : array_like
        Original sample positions (non-uniform).
    y : array_like
        Data values at positions x.
    weights : dict
        Regularization weights for derivatives (e.g. {2: 1e-3, 3: 1e-3}).
    dx : float
        Approximate spacing of the uniform grid.
    return_dense : bool
        If True, return the full dense solution with inserted points;
        if False (default), return only the regularly spaced part.

    Returns
    -------
    x_out : ndarray
        Uniform grid (or dense grid if `return_dense=True`).
    y_smooth : ndarray
        Smoothed y values on the returned grid.
    """
    x_dense, y_dense, mask, inserted_mask = upsample_with_exact_data_inclusion(x, y, dx)
    y_smooth = powersmooth_general(x_dense, y_dense, weights=weights, mask=mask)

    if return_dense:
        return x_dense, y_smooth, mask
    else:
        return x_dense[~inserted_mask], y_smooth[~inserted_mask]