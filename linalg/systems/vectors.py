import numpy as np
import scipy.sparse as sp
from zlib import crc32
random = np.random.RandomState(crc32(str.encode(__file__)))

def get_exact_solution(A, b):
    print("# WARNING: Use of `get_exact_solution` is not recommended.")
    # Get the corresponding least-squares least-norm solution.
    if sp.issparse(A):
        return sp.linalg.lsqr(A, b)[0]
    return np.linalg.lstsq(A, b, rcond=None)[0]

def normalize(orig_vec, *, norm_mat=None, vec_norm=1):
    """
    Normalize the vector `orig_vec` in the `norm_mat` inner product norm to have
    exactly `vec_norm` as its norm. If `norm_mat` isn't given, use the
    standard Euclidean inner product.
    """

    if norm_mat is None:
        orig_norm = np.linalg.norm(orig_vec)
    else:
        orig_norm = np.sqrt((orig_vec.T @ norm_mat @ orig_vec)[0,0])

    return vec_norm * orig_vec / orig_norm

def inconsistent(A, *, sln_norm_mat=None, sln_norm=1, res_norm=1):
    """
    Given a matrix `A`, this function returns a vector `b` such that the
    least-squares solution `x` of the system

        A x = b

    will have the norm `sln_norm`, and the least-squares residual b - A x will
    have the norm `res_norm`.
    """

    # Generate a preliminary RHS vector.
    prelim_b = random.normal(0, 1.0, (A.shape[0], 1))

    # Get the corresponding least-squares least-norm solution.
    prelim_x = get_exact_solution(A, prelim_b)

    # Compute the residual. This vector will be in Null(A.T).
    prelim_r = prelim_b - A @ prelim_x

    # Scale solution and residual to have desired norms.
    x = normalize(prelim_x, norm_mat=sln_norm_mat, vec_norm=sln_norm)
    r = normalize(prelim_r, vec_norm=res_norm)

    # TODO: implement in a way that does not use a LS solver.

    # Compute the final RHS vector.
    b = A @ x + r

    return b, x

def consistent(A, *, sln_norm_mat=None, sln_norm=1):
    """
    Given a matrix `A`, this function returns a vector `b` such that the system

        A x = b

    has an exact solution `x` with norm of exactly `sln_norm`.
    """
    n_rows, n_cols = A.shape

    # Generate an exact solution in the row space of A.
    prelim_x = A.T @ random.normal(0, 1.0, (n_rows, 1))

    # Scale solution to have desired norm.
    x = normalize(prelim_x, norm_mat=sln_norm_mat, vec_norm=sln_norm)

    # Compute the final RHS vector.
    b = A @ x

    return b, x
