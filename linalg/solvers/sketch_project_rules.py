from .sketch_project_methods import SmallSketchSetBase
import numpy as np

class CyclicRule(SmallSketchSetBase):
    """Cycle through the sketches in the default order"""

    def __init__(self, *, permute_sketches=False, **kwargs):
        super().__init__(**kwargs)
        self.next_sketch_idx = 0
        self.permute_sketches = permute_sketches # at end of each cycle
        self.permutation = np.arange(self.n_sketches, dtype=np.int)
        if permute_sketches:
            self.permutation = self.random.permutation(self.permutation)

    def get_sketch_idx(self, xk, b):
        sketch_idx = self.permutation[self.next_sketch_idx]
        if self.permute_sketches:
            self.permutation = self.random.permutation(self.permutation)
        self.next_sketch_idx += 1
        self.next_sketch_idx %= self.n_sketches
        return sketch_idx

class CyclicPermutationRule(CyclicRule):
    """Cycle through the sketches in the default order"""

    def __init__(self, **kwargs):
        super().__init__(permute_sketches=True, **kwargs)

class FixedProbsRuleBase(SmallSketchSetBase):
    """For selection rules whose probabilities don't change with iteration"""
    def __init__(self, *, probs, **kwargs):
        self.probs = probs
        super().__init__(**kwargs)

    def get_sketch_probs(self, sketch_losses):
        return self.probs

    def bound(self, *, n_iters, x_exact):
        emat = np.zeros((self.n_cols, self.n_cols))
        for prob, At_S, Cinv in zip(self.probs, self.At_S_list, self.Cinv_list):
            emat += prob * self.Binv @ (At_S @ Cinv @ At_S.T)
        eigvals = np.abs(np.linalg.eigvals(emat))

        # 1 - smallest nonzero eigenvalue
        conv_rate = 1 - np.min(eigvals[~np.isclose(eigvals,0)])

        initial_err_sq = self.errsq(np.zeros((self.n_cols,1)), x_exact)
        return initial_err_sq * (conv_rate ** np.arange(n_iters+1))

class UniformRule(FixedProbsRuleBase):
    """Samples uniformly amongst the available sketches"""

    def __init__(self, **kwargs):
        super().__init__(probs=None, **kwargs)
        self.probs = np.full((self.n_sketches,), 1/self.n_sketches)


class ConvenientProbsRule(FixedProbsRuleBase):
    """
    Use the 'convenient probabilities' to sample sketches as described in
    equation () of []
    """

    def __init__(self, **kwargs):
        super().__init__(probs=None, **kwargs)
        traces = [np.trace(S.T @ self.A @ self.Binv @ self.A.T @ S)
                  for S in self.S_list]
        self.probs = np.array(traces) / sum(traces)

class NoReplacementRule(SmallSketchSetBase):
    """
    Avoid sampling the same sketch repeatedly by skipping sketches whose
    corresponding loss is near zero
    """

    def get_sketch_probs(self, sketch_losses):
        # Make a copy of the probs to avoid modifying a mutable member variable.
        probs = super().get_sketch_probs(sketch_losses).copy()

        # The smallest loss corresponds to the most recently used sketch.
        min_loss_idx = np.argmin(sketch_losses)

        # If the corresponding loss is too big, likely this is the first iter.
        if sketch_losses[min_loss_idx] < np.finfo(np.float).eps:
            probs[min_loss_idx] = 0

        # Renormalize
        probs /= np.sum(probs)
        return probs

class CappedRule(SmallSketchSetBase):
    """
    A generalization of Bai and Wu's RGRK method
    """

    def __init__(self, *, greediness=0.5, **kwargs):
        super().__init__(**kwargs)
        self.greediness = greediness

    def get_sketch_probs(self, sketch_losses):
        loss = sketch_losses.copy()
        maxloss = np.max(loss)

        # TODO: allow for weighted average.
        avgloss = np.mean(loss)
        loss_thresh = self.greediness * maxloss + (1-self.greediness) * avgloss
        loss[loss < loss_thresh] = 0

        return loss / np.sum(loss)

class MonomialRule(SmallSketchSetBase):
    """
    probabilities proportional to loss values to some exponent
    """

    def __init__(self, *, exponent=1, probs=None, **kwargs):
        super().__init__(**kwargs)
        self.exponent = exponent

    def get_sketch_probs(self, sketch_losses):
        scaled_loss = sketch_losses ** self.exponent
        return scaled_loss / np.sum(scaled_loss)

    def bound(self, *, n_iters, x_exact):
        # TODO: this is a hack, please don't commit it.
        prob = 1/self.n_sketches
        emat = np.zeros((self.n_cols, self.n_cols))
        for At_S, Cinv in zip(self.At_S_list, self.Cinv_list):
            emat += prob * self.Binv @ (At_S @ Cinv @ At_S.T)
        eigvals = np.abs(np.linalg.eigvals(emat))

        # 1 - 2*smallest nonzero eigenvalue.
        conv_rate = 1 - 2*np.min(eigvals[~np.isclose(eigvals,0)])

        initial_err_sq = self.errsq(np.zeros((self.n_cols,1)), x_exact)
        return initial_err_sq * (conv_rate ** np.arange(n_iters+1))

class MaxDistanceRule(SmallSketchSetBase):
    """Choose the sketch with the largest corresponding sketched loss"""

    def get_sketch_probs(self, sketch_losses):
        maxloss = np.max(sketch_losses)
        wheremax = sketch_losses == maxloss
        return wheremax / np.sum(wheremax)

    def additional_flops(self, n_iters):
        # TODO: fill in.
        return np.arange(n_iters+1)
