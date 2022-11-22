import numpy as np
import pandas as pd
from scipy.stats import multinomial

from .common import QuantitiesOfInterest
from .plotting import heatmap


class TwoParamMultinomialEstimator:
    """
    Estimate a population and the probability of spotting an individual within that population.

    We can use this estimation whenever we have 2 observations where all the individuals in a given
    population share the probability of being observed. This can be thought of having an n-face die
    whose faces represent the different combinations of observations.

    This way, with 2 observations, we would have:
        - spotted in the first
        - Spotted in the last
        - spotted in both.

    This was built for two observations because, if we had 3, then we would need 8 cols for counts
    and probs. This number very quickly becomes unmanageable because the number of combinations
    (sides of the die) grows exponentially.
    """

    phs = np.linspace(0.01, 0.99, 101)
    nhs = np.arange(50, 501)
    counts_cols = [
        "k11",
        "k10",
        "k01",
        "k00",
    ]
    probs_cols = [
        "pp",
        "pq",
        "qp",
        "qq",
    ]

    def __init__(self, first=23, last=19, both=4, nhs=None):
        self.first = first
        self.last = last
        self.both = both

        if nhs is not None:
            self.nhs = nhs

        # Let's initialise the likelihood dataframe
        self.likes_df = self._initialise_likes_df()
        self.likes_2d = None
        self.marginal_n = None
        self.marginal_p = None

    def _initialise_likes_df(self):
        """Start the df that will capture all the information needed to compute the likelihood."""
        ps, ns = np.meshgrid(self.phs, self.nhs)
        return pd.DataFrame(
            {
                "ps": ps.ravel(),
                "ns": ns.ravel(),
            }
        )

    def _add_counts_to_the_likes_df(self):
        """Compute the counts for the multinomial distribution."""
        values = [self.both, self.first - self.both, self.last - self.both, np.nan]
        for k, v in zip(self.counts_cols, values):
            self.likes_df[k] = v

        # Now fill the k00. Note that in the first and the last we are adding twice the both
        # hence the both*2
        self.likes_df["k00"] = (
            self.likes_df.ns - self.first - self.last - self.both + self.both * 2
        )

    def _add_probs_to_the_likes_df(self):
        """Compute the probabilities for the multinomial distribution-"""
        seen = self.likes_df.ps.values
        not_seen = 1 - seen
        values = (
            seen * seen,
            seen * not_seen,
            not_seen * seen,
            not_seen * not_seen,
        )
        for k, v in zip(self.probs_cols, values):
            self.likes_df[k] = v

    def _compute_likelihood(self):
        self._add_counts_to_the_likes_df()
        self._add_probs_to_the_likes_df()

        counts = self.likes_df[self.counts_cols].values
        probs = self.likes_df[self.probs_cols].values

        self.likes_df["likes"] = multinomial.pmf(counts, self.likes_df.ns, probs)

    def _extract_marginals(self):
        self.likes_2d = self.likes_df.pivot_table(
            index="ns", columns="ps", values="likes"
        )
        total_prob = self.likes_2d.sum().sum()
        posterior = self.likes_2d.copy() / total_prob

        self.marginal_n = posterior.sum(axis=1)
        self.marginal_p = posterior.sum()

    def run(self):
        self._compute_likelihood()
        self._extract_marginals()

    def print_quantities(self):
        QuantitiesOfInterest(self.marginal_n).run()
        QuantitiesOfInterest(self.marginal_p).run()

    def show_likelihood_heatmap(self):
        title = "Likelihood of bears population N"
        heatmap(self.likes_2d, title=title, xlabel="p", ylabel="N")


class ThreeParamMultinomialEstimator(TwoParamMultinomialEstimator):
    """
    Estimate a population and the probabilities of spotting an individual within that population.

    In TwoParamMultinomialEstimator we assumed that both observations shared the probability of
    spotting an individual. However, sometimes we might want to estimate different probabilities
    for each observation. Therefore, we need to infer three parameters
    """

    nhs = np.arange(32, 350, step=5)

    def __init__(self, first=20, last=15, both=3, nhs=None):
        super().__init__(first=first, last=last, both=both, nhs=nhs)
        self.marginal_p1 = None
        self.marginal_p2 = None

    def _initialise_likes_df(self):
        """Start the df that will capture all the information needed to compute the likelihood."""
        p1, p2, ns = np.meshgrid(self.phs, self.phs, self.nhs)
        return pd.DataFrame(
            {
                "ns": ns.ravel(),
                "p1": p1.ravel(),
                "p2": p2.ravel(),
            }
        )

    def _add_probs_to_the_likes_df(self):
        q1 = 1 - self.likes_df.p1.values
        q2 = 1 - self.likes_df.p2.values
        values = (
            self.likes_df.p1.values * self.likes_df.p2.values,
            self.likes_df.p1.values * q2,
            q1 * self.likes_df.p2.values,
            q1 * q2,
        )
        for k, v in zip(self.probs_cols, values):
            self.likes_df[k] = v

    def _extract_marginals(self):
        self.likes_df["posterior"] = (
            self.likes_df.likes.values / self.likes_df.likes.sum()
        )
        self.marginal_n = self.likes_df.groupby("ns").posterior.sum()
        self.marginal_p1 = self.likes_df.groupby("p1").posterior.sum()
        self.marginal_p2 = self.likes_df.groupby("p2").posterior.sum()

    def print_quantities(self):
        QuantitiesOfInterest(self.marginal_n).run()
        QuantitiesOfInterest(self.marginal_p1).run()
        QuantitiesOfInterest(self.marginal_p2).run()
