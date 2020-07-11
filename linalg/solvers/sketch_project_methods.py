import numpy as np
import scipy.sparse as sp
from ._lin_iter_base import LinearIterativeSolverBase
from .utils import get_partition
from collections import namedtuple
from linalg.systems.vectors import consistent

Result = namedtuple("result", ["iterate", "err_norm_sq", "exp_loss"])

# TODO: rename and reorganize.
class SmallSketchSetBase(LinearIterativeSolverBase):
    """
    """

    @classmethod
    def with_rule(cls, SelectionRule):
        class Solver(SelectionRule, cls): pass
        return Solver

    def __init__(self, *, B, **kwargs):
        """
        """
        super().__init__(**kwargs)

        self.B = B

        self.sketch_losses = None
        self.sketch_probs = None
        self.sketch_residuals = None
        self.err_norm_sq = None

    def precompute(self, x0, x_exact, b):
        self.sketch_residuals = self.compute_sketch_residuals(x0, b)
        self.sketch_losses = self.get_sketch_losses(self.sketch_residuals)
        self.sketch_probs = self.get_sketch_probs(self.sketch_losses)
        self.err_norm_sq = self.errsq(x0, x_exact)

    def errsq(self, xk, x_exact):
        """
        Use the B norm to compute the squared error
        """
        err = xk - x_exact
        print("writing a class specific errsq method is recommended")
        return (err.T @ self.B @ err)[0,0]

    def get_sketch_losses(self, sketch_residuals):
        return np.square(sketch_residuals).sum(axis=1)

    def get_sketch_probs(self, sketch_losses):
        raise NotImplementedError("Forget to attach a selection rule?")

    @property
    def n_sketches(self):
        raise NotImplementedError("Subclass should implement this method.")

    def compute_sketch_residuals(self, x, b):
        raise NotImplementedError("Subclass should implement this method.")

    def update_iterate(self, xk, sketch_idx, sketch_residuals):
        raise NotImplementedError("Subclass should implement this method.")

    def update_sketch_residuals(self, sketch_idx, sketch_residuals):
        raise NotImplementedError("Subclass should implement this method.")

    def update(self, xk, x_exact, b):
        self.err_norm_sq = self.errsq(xk, x_exact)
        self.sketch_losses = self.get_sketch_losses(self.sketch_residuals)
        self.sketch_probs = self.get_sketch_probs(self.sketch_losses)
        sketch_idx = self.random.choice(self.n_sketches, p=self.sketch_probs)
        next_iterate = self.update_iterate(
            xk, sketch_idx, self.sketch_residuals)
        self.sketch_residuals = self.update_sketch_residuals(
            sketch_idx, self.sketch_residuals)

        return next_iterate

    def gen_solve_results(self, *, b=None, x_exact=None, **solve_kwargs):
        """
        yields the iterate xk, err_norm_sq and exp_loss (expected convergence
        factor) for each iterate xk
        """
        # # TODO: Combine this function with gen_iterates.

        # TODO: add functionality for changing b for inconsistent systems.
        if b is None:
            b, x_exact = consistent(A=self.A, sln_norm_mat=self.B)

        # Generates iterates, err_norm_sq, and expected losses.
        for xk in self.gen_iterates(b=b, x_exact=x_exact, **solve_kwargs):

            exp_loss = np.dot(self.sketch_losses.flat, self.sketch_probs)
            exp_step_factor = exp_loss / self.err_norm_sq

            if exp_step_factor > 1 or exp_step_factor < 0:
                print("exp_step_factor > 1 or exp_step_factor < 0")
                raise Exception("This should not happen")
            yield Result(xk, self.err_norm_sq, exp_step_factor)

class Kaczmarz(SmallSketchSetBase):
    def __init__(self, *, A, **kwargs):
        super().__init__(A=A, B=np.eye(A.shape[1]), **kwargs)
        self.row_norms = np.linalg.norm(A, axis=1, keepdims=True)
        self.DA = A / self.row_norms
        self.DAAD = self.DA @ self.DA.T

    @property
    def n_sketches(self):
        return self.n_rows

    def compute_sketch_residuals(self, x, b):
        return (self.DA @ x) - (b / self.row_norms)

    def update_iterate(self, xk, sketch_idx, sketch_residuals):
        selected_residual = sketch_residuals[sketch_idx, 0]
        return xk - selected_residual * self.DA[[sketch_idx], :].T

    def update_sketch_residuals(self, sketch_idx, sketch_residuals):
        selected_residual = sketch_residuals[sketch_idx, 0]
        return sketch_residuals - selected_residual * self.DAAD[:, [sketch_idx]]

    def errsq(self, xk, x_exact):
        """
        Use the B norm to compute the squared error
        """
        err = xk - x_exact
        return (err.T @ err)[0,0]


class CoordinateDescent(SmallSketchSetBase):
    def __init__(self, *, A, **kwargs):
        # if sp.issparse(A):
        #     A = A.todense()
        self.ATA = A.T @ A
        super().__init__(A=A, B=self.ATA, **kwargs)
        self.col_norms = np.linalg.norm(A, axis=0, keepdims=True)
        self.AD = A / self.col_norms
        self.DAAD = (self.ATA / self.col_norms) / self.col_norms.T

    @property
    def n_sketches(self):
        return self.n_cols

    def compute_sketch_residuals(self, x, b):
        return self.AD.T @ (self.A @ x - b)

    def update_iterate(self, xk, sketch_idx, sketch_residuals):
        idx_residual = sketch_residuals[sketch_idx, 0]
        xk[sketch_idx] -= idx_residual / self.col_norms[0,sketch_idx]
        return xk

    def update_sketch_residuals(self, sketch_idx, sketch_residuals):
        idx_residual = sketch_residuals[sketch_idx, 0]
        return sketch_residuals - idx_residual * self.DAAD[:, [sketch_idx]]

    def errsq(self, xk, x_exact):
        """
        Use the B norm to compute the squared error
        """
        err = self.A @ xk - self.A @ x_exact
        return (err.T @ err)[0,0]
