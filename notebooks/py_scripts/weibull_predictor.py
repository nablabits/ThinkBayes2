"""Predict using Weibull distribution."""
import numpy as np
from scipy.stats import weibull_min


class WeibullEstimator:

    def __init__(
        self,
        lifetime: np.ndarray = None,
        arrival_date: np.ndarray = None,
        lbd_range: np.ndarray = None,
        k_range: np.ndarray = None
    ):
        self.lifetime = lifetime
        self.arrival_date = arrival_date
        self.lbd_range = lbd_range
        self.k_range = k_range

    def auto_generate(self):
        sample = 10
        hs_size = 101
        dist = weibull_min(.8, scale=3)
        self.lifetime = dist.rvs(sample)
        self.arrival_date = np.random.uniform(8, size=sample)
        self.lbd_range = np.linspace(0.1, 10.1, hs_size)
        self.k_range = np.linspace(0.1, 5.1, hs_size)

    def create_df(self):
        pass

    def likes_with_pdf(self):
        pass

    def likes_with_sf(self):
        pass



    def run(self):
        if self.lifetime is None:
            self.auto_generate()
