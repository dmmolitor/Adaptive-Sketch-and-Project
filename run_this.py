import linalg.solvers.sketch_project as snp
from linalg.systems import matrices, vectors
from utils import plotting
import numpy as np
import os
import itertools

VectorSketchMethods = [snp.Kaczmarz, snp.CoordinateDescent]
vector_set_names = ["rk", "cd"]

SelectionRules = [snp.UniformRule, snp.MonomialRule, snp.CappedRule, snp.MaxDistanceRule]
rule_labels = ["Uniform", r"$p_i^k \propto r_i^k$", r"Capped $(\theta=0.5)$", "Max Distance"]

mat_name = "gaussian"

n_solves = 50
n_iters = 5000
tol = 1e-14

shapes = [(100, 1000), (1000, 100)]
plot_types = ["error", "flops"]

solve_kwargs = {"max_iters": n_iters, "tol": tol}

# Make directory for figures if it doesn't exist.
os.makedirs('figures', exist_ok=True)

for shape, (SketchMethod, set_name) in itertools.product(
    shapes, zip(VectorSketchMethods, vector_set_names)
):
    print(set_name, shape)

    results_dict = {rule_label: [] for rule_label in rule_labels}

    matrices.random.seed(0)

    # Solve system n_solves times.
    for solve in range(n_solves):
        print('Solve {}'.format(solve))
        A = matrices.gaussian(shape=shape)

        # temp solver so that x_exact is normalized wrt B norm
        temp_solver = SketchMethod(A=A)
        b, x_exact = vectors.consistent(A=A, sln_norm_mat=temp_solver.B)
        solve_kwargs["b"] = b
        solve_kwargs["x_exact"] = x_exact

        # Run experiment with different sampling strategies.
        for Rule, rule_label in zip(SelectionRules, rule_labels):
            Solver = SketchMethod.with_rule(Rule)
            solver = Solver(A=A)

            _, err_norms_sq, _ = zip(*solver.gen_solve_results(**solve_kwargs))
            results_dict[rule_label].append(np.array(err_norms_sq))

    # Plot errors and flops per iteration.
    for plot_type in plot_types:
        # Get filename for the saved figure.
        fig_name = "{}_{}-vs-iter-{}x{}-{}_{}_solves".format(
            mat_name, plot_type, shape[0], shape[1], set_name, n_solves
        )
        filename = os.path.join('figures', fig_name)
        print(filename)

        fig_kwargs = {
            "show": False,
            "save": True,
            "figsize": [10,6],
            "ylabel": r"$\|x^k - x^\star\|_{\mathbf{B}}^2$"
        }
        if plot_type == "error":
            fig_kwargs["xlabel"] = "Iteration $k$"
        elif plot_type == "flops":
            fig_kwargs["xlabel"] = "Approximate flops"

        # Plot the results.
        with plotting.experiment_figure(filepath=filename, **fig_kwargs):
            plotting.plot_exper(
                results_dict,
                plot_type,
                mat_shape=shape,
                rules=rule_labels,
                vector_set=set_name,
            )
