from math import ceil
import numpy as np

def get_partition(idxs, *, block_size=1):
    n_idxs = len(idxs)
    idx_lists = []
    idxs_left = n_idxs

    for i in range(ceil(n_idxs/block_size)):
        # Make blocks of size block_size-1 until the remaining can be made
        # all of size block_size.
        take = block_size - 1
        if idxs_left % block_size == 0:
            take = block_size

        idxs_left -= take
        idx_lists.append(idxs[idxs_left : idxs_left + take])

    return idx_lists


def get_flops(*, iters, shape, vector_set, rule_label):
    """CURRENTLY ONLY WORKS FOR RK AND CD."""
    if vector_set != "rk" and vector_set != "cd":
        raise NotImplementedError("Only implemented for rk and cd")

    x = np.arange(iters)
    m, n = shape

    rule_label_flop_dict = {
        "Uniform" : {"rk" : 2 * min(m,n) + 2 * n, "cd" : 2 * n},
        "Max Distance" : {"rk" : 3 * m + 2 * n, "cd" : 3 * n},
        r"$p_i^k \propto r_i^k$" : {"rk" : 5 * m + 2 * n, "cd" : 5 * n},
        r"Capped $(\theta=0.5)$" : {"rk" : 9 * m + 2 * n, "cd" : 9 * n}
    }

    x = x * rule_label_flop_dict[rule_label][vector_set]

    return x
