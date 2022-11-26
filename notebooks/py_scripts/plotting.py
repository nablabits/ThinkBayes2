"""Several plotting tools for the notebooks."""
import seaborn as sns
from matplotlib import pyplot as plt


def heatmap(data, title, xlabel=None, ylabel=None):
    """
    Plot a 2-d heat map out of the data.

    data can be either a 2d array or a pandas DataFrame
    """
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, ax=ax)
    ax.set_title(title)

    # If we pass a df as data we can use the names of the columns and index to label the axes.
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
