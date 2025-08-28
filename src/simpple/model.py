from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from nautilus import Prior


class Model:
    """Simpple model

    :param parameters: dictionary of parameters mapping parameter names to a
                      prior (`simpple.Distribution` object).
    :param log_likelihood: log-likelihood function that accepts a dictionary of parameters.
    """

    def __init__(self, parameters: dict, log_likelihood: Callable):
        self.parameters = parameters
        self._log_likelihood = log_likelihood
        # Attempt to make log-likelihood inherit docstring.
        # Works at runtime but not for static (LSP) tools
        self.log_likelihood.__func__.__doc__ = self._log_likelihood.__doc__

    def _log_likelihood(self, parameters, *args, **kwargs) -> float:
        raise NotImplementedError(
            "log_likelihood must be passed to init or _log_likelihood must be "
            "implemented by subclasses."
        )

    def keys(self) -> list[str]:
        """Get a list of parameter names"""
        return list(self.parameters.keys())

    def log_likelihood(self, parameters, *args, **kwargs):
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        return self._log_likelihood(parameters, *args, **kwargs)

    def log_prior(self, parameters: dict | ArrayLike) -> float:
        """Log of the prior probability for the model

        :param parameters: Dictionary or array of parameters. If an array is used,
                           the order must be the same as `Model.keys()`
        :return: Log-prior probability
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = 0.0
        for pname, pval in parameters.items():
            pdist = self.parameters[pname]
            lp += pdist.log_prob(pval)
        return lp

    def prior_transform(self, u: ArrayLike | dict) -> np.ndarray | dict:
        """Prior transform of the model

        Takes samples from a uniform distribution between 0 and 1
        for all parameters and returns samples transformed according to the prior.

        :param u: Samples from the uniform distribution. Can be a dict or an array
                  ordered as `Model.keys()`
        :return: Prior samples, as a dict or an array depending on the input type.
        """
        is_dict = isinstance(u, dict)
        if is_dict:
            u = np.array(list(u.values()))
        x = np.array(u)
        for i, pdist in enumerate(self.parameters.values()):
            x[i] = pdist.prior_transform(u[i])
        if is_dict:
            x = dict(zip(self.keys(), x, strict=True))
        return x

    def nautilus_prior(self) -> "Prior":
        """Builds and return a `nautilus.Prior` for the model.

        :return: Nautilus Prior object.
        """
        from nautilus import Prior

        prior = Prior()
        for pname, pdist in self.parameters.items():
            prior.add_parameter(pname, pdist.dist)
        return prior

    def log_prob(self, parameters: dict | ArrayLike, *args, **kwargs) -> float:
        """Log posterior probability for the model

        :param parameters: Parameters as a dict or an array ordered as `Model.keys()`
        :return: Log posterior probability
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = self.log_prior(parameters)
        if np.isfinite(lp):
            return lp + self.log_likelihood(parameters, *args, **kwargs)
        return lp

    def get_prior_samples(self, n_samples: int, fmt: str = "dict") -> dict:
        """Generate prior samples for all parameters

        :param n_samples: Number of samples
        :param fmt: Format of the samples (dict or array)
        :return: Dictionary of prior samples
        """
        rng = np.random.default_rng()
        u = rng.uniform(size=(len(self.parameters), n_samples))
        if fmt == "dict":
            u = dict(zip(self.keys(), u, strict=True))
        elif fmt != "array":
            raise ValueError(f"Invalid format: {fmt}. Use 'dict' or 'array'.")
        return self.prior_transform(u)


class ForwardModel(Model):
    """A model whose likelihood calls a forward model as the mean."""

    forward: Callable

    def __init__(self, parameters: dict, log_likelihood: Callable, forward: Callable):
        super().__init__(parameters, log_likelihood)
        self._forward = forward
        self.forward.__func__.doc__ = self._forward.__doc__

    def forward(self, parameters: dict | ArrayLike, *args, **kwargs) -> np.ndarray:
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        return self._forward(parameters, *args, **kwargs)

    def get_prior_pred(self, n_samples: int, *args, **kwargs) -> np.ndarray:
        prior_params = self.get_prior_samples(n_samples, fmt="array")
        pred = []
        for p in prior_params.T:
            pred.append(self.forward(p, *args, **kwargs))
        return np.array(pred)


    def get_posterior_pred(self, chains: dict | ArrayLike, n_samples: int, *args, **kwargs) -> np.ndarray:
        if isinstance(chains, dict):
            chains = np.array([a for a in chains.values()])
        elif chains.ndim != 2 or chains.shape[0] != len(self.keys()):
            raise ValueError("chains must have shape (ndim, nsamples)")

        rng = np.random.default_rng()
        show_idx = rng.choice(chains.shape[1], n_samples, replace=False)

        pred = []
        for i in show_idx:
            pred.append(self.forward(chains[:, i], *args, **kwargs))
        return np.array(pred)
