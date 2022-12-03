"""Several plotting tools for the notebooks."""
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import JointGrid

from .common import QuantitiesOfInterest


def heatmap(data, title=None, xlabel=None, ylabel=None, **sns_options):
    """
    Plot a 2-d heat map out of the data.

    A convenience wrapper around sns heatmap that adds titles and axes names. Note that data can be
    either a 2d array or a pandas DataFrame
    """
    ax = sns.heatmap(data, **sns_options)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


class CustomJointGrid:
    available_joint_plots = ["contour", "heatmap"]

    def __init__(self, dataframe: pd.DataFrame, joint_plot: str = "contour") -> None:
        self.df = dataframe
        self.g = self._init_joint_grid()
        self.x, self.y = dataframe.columns.values, dataframe.index.values

        assert joint_plot in self.available_joint_plots
        self.joint_plot_is_contour = joint_plot == self.available_joint_plots[0]
        self.marginal_x = None
        self.marginal_y = None

    def _init_joint_grid(self) -> JointGrid:
        x_label = self.df.columns.name
        y_label = self.df.index.name

        x = "x" if x_label is None else x_label
        y = "y" if y_label is None else y_label

        return JointGrid(x=x, y=y, data=pd.DataFrame({x: [0], y: [0]}))

    def _compute_marginals(self) -> None:
        marginal_x = self.df.sum(axis=0)
        marginal_y = self.df.sum(axis=1)
        self.marginal_x = marginal_x / marginal_x.sum()
        self.marginal_y = marginal_y / marginal_y.sum()

    def _replace_joint_plot(self) -> None:
        x, y = np.meshgrid(self.x, self.y)
        if self.joint_plot_is_contour:
            self.g.ax_joint.contour(x, y, self.df, cmap="viridis")
        else:
            self.g.ax_joint.pcolormesh(x, y, self.df, cmap="viridis")

    def _replace_marginals(self):
        self.g.ax_marg_x.plot(self.marginal_x.index, self.marginal_x)
        self.g.ax_marg_y.plot(self.marginal_y, self.marginal_y.index)

    def _output_marginal_stats(self):
        print(f"Marginal {self.marginal_x.index.name} stats:")
        QuantitiesOfInterest(self.marginal_x).run()
        print(25 * "*")
        print(f"Marginal {self.marginal_y.index.name} stats:")
        QuantitiesOfInterest(self.marginal_y).run()

    def run(self, output_marginal_stats=True):
        self._compute_marginals()
        self._replace_joint_plot()
        self._replace_marginals()
        if output_marginal_stats:
            self._output_marginal_stats()
