"""Common tools for estimators."""
import numpy as np


class QuantitiesOfInterest:
    """
    Compute some quantities of interest of a probability distribution.

    When solving bayesian problems, one is constantly computing certain amounts, so it
    could be a good idea to display them all at once.
    This could be thought of a tailor made version of pd.describe()
    """

    def __init__(self, probability_distribution):
        """
        Class constructor.

        In principle, we are expecting a pandas Series whose indices are the quantities,
        say for instance, some population, and the values are probability of each
        quantity.
        """
        self.qs = probability_distribution.index
        self.ps = probability_distribution.values
        self.dist = probability_distribution

    def expected_value(self):
        """Compute the expectation of the distribution."""
        return np.sum(self.qs * self.ps)

    def map_(self):
        """Compute the MAP of the distribution."""
        return self.dist.idxmax()

    def _get_quantile(self, p):
        """Get the index where the cdf overcomes some prob p."""
        cdf = self.dist.cumsum()
        quantile = self.dist[cdf > p].index.min()
        return quantile

    def credible_interval_boundary(self):
        """Return the boundary values that limit the 90% of the distribution."""
        return [self._get_quantile(p) for p in (0.05, 0.95)]

    def credible_interval_values(self):
        """Return the values that contain the 90% of the distribution."""
        lo, hi = [self._get_quantile(p) for p in (0.05, 0.95)]
        return self.dist.iloc[lo:hi]

    def run(self):
        print(f"The expected value is: {self.expected_value()}")
        print(f"The MAP is: {self.map_()}")
        print(f"The CI is: {self.credible_interval_boundary()}")
