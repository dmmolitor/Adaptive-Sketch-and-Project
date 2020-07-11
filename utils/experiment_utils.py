import numpy as np
from collections import namedtuple

def get_experiment_data(*, solver, n_solves, **solve_kwargs):
    """Generates results of solver.solve for n_solves times."""
    results_dict = {
        "err_norms_sq":[],
        "exp_losses":[]
    }

    for i in range(n_solves):
        print('solve: ',i)
        _, err_norms_sq, exp_losses = zip(*solver.gen_solve_results(**solve_kwargs))
        if max(exp_losses) > 1 or min(exp_losses) < 0:
            raise Exception("This should not happen")
        results_dict["err_norms_sq"].append(np.array(err_norms_sq))
        results_dict["exp_losses"].append(np.array(exp_losses))
    return results_dict

def stack_data(data_list):
    """If data is not same length, chop to shortest data, then stack."""
    truncate_at = min(len(data) for data in data_list)
    return np.stack([data[:truncate_at] for data in data_list])
