import numpy as np
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from utils.experiment_utils import stack_data
from linalg.solvers.utils import get_flops

sns.set(style="whitegrid", font_scale=1.5, context="talk")

"""
For details on the params below, see the matplotlib docs:
https://matplotlib.org/users/customizing.html
"""

plt.rcParams["axes.edgecolor"] = "0.6"
plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.family'] = 'serif'
plt.rcParams["grid.color"] = "0.85"
plt.rcParams['savefig.dpi'] = 300
plt.rcParams["legend.columnspacing"] *= 0.8
plt.rcParams["legend.edgecolor"] = "0.6"
plt.rcParams["legend.framealpha"] = "1"
plt.rcParams["legend.handlelength"] *= 1.5
plt.rcParams['legend.numpoints'] = 2
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.major.pad'] = -3
plt.rcParams['ytick.major.pad'] = -2
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']

def reset_plot_kwargs():
    global recent_color, recent_linestyle, recent_marker
    recent_color = None
    recent_linestyle = None
    recent_marker = None

    global offset, colors, linestyles, markers
    offset = 0
    colors = cycle(sns.color_palette())

    global linestyles
    linestyles = cycle(["-", "--", ":", "-."])

    global linewidth
    linewidth = 3.0

    global markers
    markers = cycle(['D', 'o', 'X', '*', '<', 'd', 'S', '>', 's', 'v'])

    global markersize
    markersize = 15

def get_fig_kwargs(plot_type):
    """Get dictionary of figure arguments depending on the plot type."""
    fig_kwargs = {
        "show" : False, "save" : True, "transparent" : True,
        "legend_kwargs" : {"frameon" : False, "loc" : "upper right", "bbox_to_anchor" : (1,1)},
        "figsize" : [10,6]
    }

    if plot_type == "error":
        fig_kwargs["xlabel"] = "Iteration $k$"
        fig_kwargs["ylabel"] = r"$\|x^k - x^\star\|_{\mathbf{B}}^2$"

    if plot_type == "flops":
        fig_kwargs["xlabel"] = "Approximate flops"
        fig_kwargs["ylabel"] = r"$\|x^k - x^\star\|_{\mathbf{B}}^2$"

    return fig_kwargs

def markevery(*, n_points, n_markers=10):
    """disperse n_markers evenly amongst n_points in a plot"""
    global offset
    if n_points > n_markers:
        markevery = int(n_points / n_markers)
        to_return = (offset, markevery)
        # golden ratio trick
        offset = (offset + int(.61803398875 * markevery)) % markevery
        return to_return
    else:
        return 1

def plot_line(*, x=None, data, color=None, linestyle=None, marker=None,
              **plot_kwargs):
    """plot data against x as a line with markers"""

    # Optionally use the most recent style variables.
    global recent_color, recent_linestyle, recent_marker
    if color == "recent": color = recent_color
    if linestyle == "recent": linestyle = recent_linestyle
    if marker == "recent": marker = recent_marker

    # If a param was provided, don't churn the corresponding generator.
    plot_kwargs["color"] = color or next(colors)
    plot_kwargs["linestyle"] = linestyle or next(linestyles)
    plot_kwargs["marker"] = marker or next(markers)

    # Update the most recently used style variables.
    recent_color = plot_kwargs["color"]
    recent_linestyle = plot_kwargs["linestyle"]
    recent_marker = plot_kwargs["marker"]

    # This can be safely recomputed every time since there is no generator.
    plot_kwargs["markevery"] = markevery(n_points=len(data))

    # x defaults to 0, 1, ... len(data)-1
    if x is None:
        x = np.arange(len(data))

    plt.plot(x, data, **plot_kwargs)


def plot_means_and_ci(*, x=None, data, ci, axis=0, **plot_kwargs):
    """plot column-wise mean of `data` and optional confidence interval"""
    data = stack_data(data)
    means = np.mean(data, axis=axis)

    # This is utilized for example when plotting errsqs vs iteration.
    if x is None:
        x = np.arange(len(means))

    plot_line(x=x, data=means, **plot_kwargs)

    if ci != 0:
        # Compute the boundaries of the middle ci% of the data.
        lower_ci = np.percentile(data, 50-ci/2, axis=axis)
        upper_ci = np.percentile(data, 50+ci/2, axis=axis)

        # Use the color that was used to plot the means.
        global recent_color
        plt.fill_between(x, lower_ci, upper_ci,
                         color=recent_color,
                         alpha=0.25)

def plot_exper(data_dict, plot_type, mat_shape=None, rules=[], vector_set=None):
    if mat_shape is None or vector_set is None:
        raise Exception("Missing required keyword argument.")

    # Plot error vs iteration.
    if plot_type == "error":
        for rule in rules:
            plot_means_and_ci(data=data_dict[rule], label=rule, ci=95)

    # Plot error vs flops.
    if plot_type == "flops":
        for rule in rules:
            data = stack_data(data_dict[rule])
            flop_counts = get_flops(iters=len(data[0]),
                                    shape=mat_shape,
                                    vector_set=vector_set,
                                    rule_label=rule
                               )
            plot_means_and_ci(x=flop_counts, data=data, label=rule, ci=95)


class experiment_figure:
    """
    Example usage:

        with experiment_figure("image.png"):
            # do some plotting
            ...

    the figure will automatically be saved as "image.png" at the end
    """
    def __init__(self, *,
            filepath,
            title="",
            xlabel="",
            ylabel="",
            ylog_scale=True,
            legend_kwargs={
                "frameon": False, "loc": "upper right", "bbox_to_anchor": (1,1)
            },
            xlims=None,
            show=False,
            save=True,
            transparent=True,
            figsize=[8.0, 6.0]
        ):
        self.fig = plt.figure(figsize=figsize)

        self.ylog_scale = ylog_scale
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if xlims:
            plt.xlim(xlims)

        # Store things needed later for cleaning up.
        self.legend_kwargs = legend_kwargs or {}
        self.show = show
        self.save = save
        self.filepath = filepath

        # Ensure that figures use the same colors etc in the same orders.
        reset_plot_kwargs()

    def __enter__(self):
        return self.fig

    def __exit__(self, *args):

        # If the y axis is err_sq, a log scale is recommended.
        if self.ylog_scale:
            self.fig.axes[0].set_yscale("log", nonposy="clip")

        ylim = plt.ylim()
        if ylim[0] < (10 ** -14):
            plt.ylim(10 ** -14)

        # Can set number of cols (ncol) or location (loc) of legend.
        plt.legend(**self.legend_kwargs)

        # get rid of excessive whitespace
        plt.tight_layout(pad=0.25)

        # Optionally save the figure as a .png.
        if self.save:
            self.fig.savefig(self.filepath+".png")

        # Optionally show the figure.
        if self.show:
            plt.show()

        plt.close()
