from simpple.model import ForwardModel
import simpple.distributions as sdist
import numpy as np


class PolyModel(ForwardModel):
    def __init__(self, parameters: dict[str, sdist.Distribution], order: int):
        self.order = order
        self.parameters = parameters
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
