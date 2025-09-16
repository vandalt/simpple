from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike

from simpple.distributions import Distribution, Fixed

if TYPE_CHECKING:
    from nautilus import Prior


class Model:
    """Simpple model

    :param parameters: dictionary of parameters mapping parameter names to a
                      prior (``simpple.Distribution`` object).
    :param log_likelihood: log-likelihood function that accepts a dictionary of parameters.
    """

    def __init__(
        self,
        parameters: dict[str, Distribution],
        log_likelihood: Callable | None = None,
    ):
        self.parameters = parameters
        if log_likelihood is not None:
            self._log_likelihood = log_likelihood
            # Attempt to make log-likelihood inherit docstring.
            # Works at runtime but not for static (LSP) tools
            self.log_likelihood.__func__.__doc__ = self._log_likelihood.__doc__
        self.fixed_p = {}
        self.fixed_p_vals = {}
        self.vary_p = {}
        for pname, pdist in self.parameters.items():
            if isinstance(pdist, Fixed):
                self.fixed_p[pname] = pdist
                self.fixed_p_vals[pname] = pdist.value
            else:
                self.vary_p[pname] = pdist

    def _log_likelihood(self, parameters, *args, **kwargs) -> float:
        raise NotImplementedError(
            "log_likelihood must be passed to init or _log_likelihood must be "
            "implemented by subclasses."
        )

    @property
    def ndim(self):
        """Number of dimensions (**variable** parameters) in the model"""
        return len(self.keys())

    def keys(self, fixed: bool = False) -> list[str]:
        """Get the ordered list of parameter names

        :param fixed: Whether fixed parameters should be included. Defaults to ``False``.
        :return: List of parameter names
        """
        if fixed:
            return list(self.parameters.keys())
        else:
            return list(self.vary_p.keys())

    def log_likelihood(self, parameters: dict | ArrayLike, *args, **kwargs) -> float:
        """Calculate the log-likelihood of the model

        Internally, this wraps the ``log_likelihood`` function that was given at initialization
        and passes all arguments to it.

        :param parameters: Dictionary or array of parameters. If an array is used,
                           the order must be the same as ``Model.keys()``, without the fixed parameters.
                           Dictionaries can optionally include fixed parameter.s
        :return: The log-likelihood value at ``parameters``.
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        # RHS has precedence
        parameters = self.fixed_p_vals | parameters
        return self._log_likelihood(parameters, *args, **kwargs)

    def log_prior(self, parameters: dict | ArrayLike) -> float:
        """Log of the prior probability for the model

        :param parameters: Dictionary or array of parameters. If an array is used,
                           the order must be the same as ``Model.keys()``, without the fixed parameters.
                           Fixed parameters are always ignored even in dictionaries.
        :return: Log-prior probability
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        lp = 0.0
        for pname, pval in parameters.items():
            # Skip fixed parameters
            if pname in self.fixed_p:
                continue
            pdist = self.parameters[pname]
            lp += pdist.log_prob(pval)
        return lp

    def prior_transform(
        self, u: ArrayLike | dict, fixed: bool = False
    ) -> np.ndarray | dict:
        """Prior transform of the model

        Takes samples from a uniform distribution between 0 and 1
        for all parameters and returns samples transformed according to the prior.

        :param u: Samples from the uniform distribution. Can be a dict or an array
                  ordered as ``Model.keys()``.
        :param fixed: Whether fixed parameters should be included. Defaults to False.
        :return: Prior samples, as a dict or an array depending on the input type.
        """
        is_dict = isinstance(u, dict)
        if is_dict:
            # Using loop over keys instead of list ensures keys are correct
            u = np.array([u[k] for k in self.keys(fixed=fixed)])
        x = np.array(u)
        if x.shape[0] != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} elements for the transform, got {x.shape[0]}"
            )
        pdist_list = self.parameters.values() if fixed else self.vary_p.values()
        for i, pdist in enumerate(pdist_list):
            x[i] = pdist.prior_transform(u[i])
        if is_dict:
            x = dict(zip(self.keys(), x, strict=True))
        return x

    def nautilus_prior(self) -> "Prior":
        """Builds and return a ``nautilus.Prior`` for the model.

        Fixed parameters are not included.

        :return: Nautilus Prior object.
        """
        from nautilus import Prior

        prior = Prior()
        for pname, pdist in self.parameters.items():
            if isinstance(pdist, Fixed):
                continue
            if not hasattr(pdist, "dist"):
                raise AttributeError(
                    f"distribution {pdist} for parameter {pname} has no scipy distribution. "
                    f"This is required to build a Nautilus prior. "
                    "Either use the prior_transform() with nautilus or add a 'dist' attribute with the scipy distribution."
                )
            prior.add_parameter(pname, pdist.dist)
        return prior

    def log_prob(self, parameters: dict | ArrayLike, *args, **kwargs) -> float:
        """Log posterior probability for the model

        All extra arguments are passed to the ``self.log_likelihood()``.

        :param parameters: Dictionary or array of parameters. If an array is used,
                           the order must be the same as ``Model.keys()``, without the fixed parameters.
                           Dictionaries can optionally include fixed parameter.s
        :return: Log posterior probability
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        # RHS has precedence
        parameters = self.fixed_p_vals | parameters

        lp = self.log_prior(parameters)
        if np.isfinite(lp):
            return lp + self.log_likelihood(parameters, *args, **kwargs)
        return lp

    def get_prior_samples(
        self,
        n_samples: int,
        fmt: str = "dict",
        seed: int | Generator | np.ndarray[int] | None = None,
        fixed: bool = False,
    ) -> dict | np.ndarray:
        """Generate prior samples for all parameters

        :param n_samples: Number of samples
        :param fmt: Format of the samples (dict or array)
        :param fixed: Whether fixed parameters should be included. Defaults to False.
        :return: Dictionary of prior samples
        """
        rng = np.random.default_rng(seed=seed)
        u = rng.uniform(size=(len(self.parameters), n_samples))
        if fmt == "dict":
            u = dict(zip(self.keys(), u, strict=True))
        elif fmt != "array":
            raise ValueError(f"Invalid format: {fmt}. Use 'dict' or 'array'.")
        return self.prior_transform(u, fixed=fixed)


class ForwardModel(Model):
    """A model whose likelihood calls a forward model as the mean.

    :param parameters: dictionary of parameters mapping parameter names to a
                      prior (``simpple.Distribution`` object).
    :param log_likelihood: log-likelihood function that accepts a dictionary of parameters. This function should call ``forward``.
    :param forward: Forward model function that accepts a dictionary of parameters as first argument.
    """

    def __init__(
        self,
        parameters: dict,
        log_likelihood: Callable | None = None,
        forward: Callable | None = None,
    ):
        super().__init__(parameters, log_likelihood=log_likelihood)
        if forward is not None:
            self._forward = forward
            self.forward.__func__.doc__ = self._forward.__doc__

    def _forward(self, parameters, *args, **kwargs) -> float:
        raise NotImplementedError(
            "forward must be passed to init or _forward must be "
            "implemented by subclasses."
        )

    def forward(self, parameters: dict | ArrayLike, *args, **kwargs) -> np.ndarray:
        """Evaluate the forward model

        Internally, this wraps the ``forward`` function that was given at initialization
        and passes all arguments to it.

        :param parameters: Dictionary or array of parameters. If an array is used,
                           the order must be the same as ``Model.keys()``, without the fixed parameters.
                           Dictionaries can optionally include fixed parameter.s
        :return: The forward model evalulated at ``parameters``.
        """
        if not isinstance(parameters, dict):
            parameters = dict(zip(self.keys(), parameters, strict=True))
        # RHS has precedence
        parameters = self.fixed_p_vals | parameters
        return self._forward(parameters, *args, **kwargs)

    def get_prior_pred(self, n_samples: int, *args, **kwargs) -> np.ndarray:
        """Get prior predictive samples

        :param n_samples: Number of samples to generate
        :return: Forward model realizations corresponding to prior samples
        """
        prior_params = self.get_prior_samples(n_samples, fmt="array")
        pred = []
        for p in prior_params.T:
            pred.append(self.forward(p, *args, **kwargs))
        return np.array(pred)

    def get_posterior_pred(
        self, chains: dict | ArrayLike, n_samples: int, *args, **kwargs
    ) -> np.ndarray:
        """Get posterior predictive samples

        :param chains: Posterior chains organized as a dictionary or an array.
                       If an array, should have shape ``(ndim, nsamples)`` where
                       nsamples is the total number of samples in the chain and
                       unrelated to the ``n_samples`` argument below.
        :param n_samples: Number of samples to pick from the chains.
        :return: Forward model realizations corresponding to posterior samples
        """
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
