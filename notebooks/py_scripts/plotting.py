"""Several plotting tools for the notebooks."""
import seaborn as sns
from matplotlib import pyplot as plt


def heatmap(data, title, xlabel, ylabel):
    """Plot a 2-d heat map out of the data."""
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, ax=ax)
    ax.set_title = title
    ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
