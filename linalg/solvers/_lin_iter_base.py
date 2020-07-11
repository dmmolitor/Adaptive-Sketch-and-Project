import numpy as np
import scipy.sparse as sp
import itertools # itertools.count() used in place of range(infinity)
from ..systems.vectors import get_exact_solution

class LinearIterativeSolverBase():
    def update(self, xk, b):
        """Compute x^(k+1) and return it"""
        raise NotImplementedError("Please implement this method")

    def precompute(self, x0, x_exact, b):
        pass

    def bound(self, n_iters):
        """
        Return a 1d array of upper bound on expected error sq for each iteration
        up to n_iters
        """
        raise NotImplementedError("Please implement this method")

    def __init__(self, *, A, seed=0, **kwargs):
        self.A = A
        self.n_rows, self.n_cols = A.shape

        # For randomized methods, it will be useful for the solvers to each have
        # their own personal random number generator for reproducability.
        self.random = np.random.RandomState(seed)

        super().__init__(**kwargs)

    def errsq(self, xk, x_exact):
        """
        Computes the squared error in whatever norm is most meaningful. This
        function can be overwritten for example in coordinate descent where
        the norm should be the A.T @ A norm or something like that. Or in the
        general sketch and project framework where it should be the B norm.
        """
        print("Not using B-norm")
        return np.linalg.norm(xk - x_exact)**2

    def gen_iterates(self, *, b, x_exact=None, x0=None, min_iters=None,
                     max_iters=None, tol=None):
        """
        Solve the problem Ax=b by iterating x^(k+1) = self.update(x^k, b),
        starting from x^0 = x0.
        """
        assert (max_iters is not None) or (tol is not None)


        # Compute the exact solution so we can calculate error and residual at
        # each step of the algorithm.
        if x_exact is None: x_exact = get_exact_solution(self.A, b)

        # Initialize the initial guess with x0 = 0 unless x0 is given.
        if x0 is None: x0 = np.zeros((self.n_cols, 1))

        self.precompute(x0, x_exact, b)

        # Start off with the iterate as the initial guess.
        xk = x0
        yield xk

        # Start at iteration `k=1`, and go until a termination condition is met.
        for k in itertools.count(1):
            xk = self.update(xk, x_exact, b)
            yield xk

            # If we have not done the minimum required iterations, don't stop.
            if (min_iters is not None) and (k < min_iters): continue

            # If a tolerance was given, and the current residual norm is under
            # the specified tolerance, halt.
            if (tol is not None):
                err_norm_sq = self.errsq(xk, x_exact)
                if (err_norm_sq < tol): break

            # If a max number of iterations was given check if we've done
            # enough iterations yet. If so, halt.
            if (max_iters is not None) and (k >= max_iters): break
