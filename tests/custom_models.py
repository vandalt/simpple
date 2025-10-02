import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

import simpple.distributions as sdist
from simpple.model import ForwardModel, Model


class Normal2DModel(Model):
    def __init__(
        self, parameters: dict[str, sdist.Distribution], mu: ArrayLike, sigma: ArrayLike
    ):
        super().__init__(parameters)
        self.norm_dist = norm(mu, sigma)

    def _log_likelihood(self, p):
        return self.norm_dist.logpdf([p["mu1"], p["mu2"]]).sum()


class PolyModel(ForwardModel):
    def __init__(self, parameters: dict[str, sdist.Distribution], order: int):
        super().__init__(parameters)
        self.order = order
        for i in range(self.order + 1):
            k = "a" + str(i)
            if k not in self.parameters:
                raise KeyError(
                    f"Parameters should have keys from a0 to a{self.order} for polynomial of order {self.order}. Key {k} not found."
                )

    def _forward(self, p, x):
        parr = np.array([p[f"a{i}"] for i in range(self.order + 1)])
        return np.vander(x, self.order + 1, increasing=True) @ parr

    def _log_likelihood(self, p, x, y, yerr):
        ymod = self.forward(p, x)
        var = yerr**2 + p["sigma"] ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (y - ymod) ** 2 / var)
