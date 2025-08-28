# Required so ArrayLike does not look terrible
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike
from scipy.stats import rv_continuous, norm


class Distribution(ABC):
    """Abstract base class for distributions.

    Defines the interface distributions must implement.

    All distributions should have a `__repr__()` showing initialization parameters,
    as well as log_prob(), `prior_transform()` and `sample()` methods.
    """

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def log_prob(self, x: float | ArrayLike) -> float | np.ndarray:
        """Log probability density of the distribution

        This should be implemented by subclasses.
        """
        pass

    @abstractmethod
    def prior_transform(self, u: float | ArrayLike) -> float | np.ndarray:
        """Prior transform of the distribution

        This should be implemented by subclasses.
        The prior transform should take values from the uniform unit interval and project them to the prior space.
        This is usually the inverse CDF, or percent-point function, of the distribution.
        """
        pass

    def sample(
        self,
        size: int | None = None,
        seed: int | np.ndarray[int] | None = None,
    ) -> np.ndarray:
        """Draw samples from the distribution

        Generates random uniform samples and calls the prior transform.

        :param size: Shape of the samples
        :param seed: Random seed to use
        :return: Random samples with shape `size`
        """
        rng = np.random.default_rng(seed=seed)
        u = rng.uniform(low=0.0, high=1.0, size=size)
        p = self.prior_transform(u)
        return p


class ScipyDistribution(Distribution):
    """Distribution based on a scipy random variable.

    :param dist: Scipy random variable (RV), either already instantiated or not.
                 If the RV is not instantiated, it will be during init and all args
                 and kwargs are passed to it.
    """

    def __init__(self, dist: rv_continuous | Callable, *args, **kwargs):
        if hasattr(dist, "dist") and hasattr(dist, "args"):
            if len(kwargs) > 0 or len(args) > 0:
                warnings.warn(
                    "The scipy distribution is 'frozen' (already instantiated). "
                    "Extra arguments will be ignored",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            self._dist = dist
        elif isinstance(dist, rv_continuous):
            self._dist = dist(*args, **kwargs)
        else:
            raise TypeError(f"Invalid type {type(dist)} for the scipy distribution.")

    @property
    def dist(self):
        """Underlying scipy distribution"""
        return self._dist

    def __repr__(self):
        args_tuple = self.dist.args
        kwargs_tuple = tuple(f"{k}={v}" for k, v in self.dist.kwds.items())
        signature_tuple = args_tuple + kwargs_tuple
        return f"ScipyDistribution({self.dist.dist.name}{signature_tuple})"

    def log_prob(self, x: float | ArrayLike, *args, **kwargs) -> np.ndarray:
        """Log probability density function.

        Calls the scipy distribution's `logpdf()`.

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        return self.dist.logpdf(x, *args, **kwargs)

    def prior_transform(self, u: float | ArrayLike, **kwargs) -> float | np.ndarray:
        """Prior transform (inverse CDF) for nested sampling

        Calls the scipy distributiion's `ppf()` (percent point function).

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples
        """
        return self.dist.ppf(u, **kwargs)

    def sample(
        self,
        size: int | None = None,
        seed: int | Generator | np.ndarray[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Draw samples from the distribution

        Calls the scipy distribution's `rvs()` method.

        :param size: Shape of the samples
        :param seed: Random seed to use
        :return: Random samples with shape `size`
        """
        if not isinstance(seed, Generator):
            seed = np.random.default_rng(seed=seed)
        return self.dist.rvs(size=size, random_state=seed, **kwargs)


class Uniform(Distribution):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        if self.low > self.high:
            raise ValueError(
                "lower bound should be lower than upper bound for uniform distribution"
            )

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"

    def log_prob(self, x: float | ArrayLike) -> float | np.ndarray:
        """Log-probability for the uniform distribution

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        lp = np.where(
            np.logical_and(x >= self.low, x < self.high),
            -np.log(self.high - self.low),
            -np.inf,
        )
        if lp.ndim == 0:
            return lp.item()
        return lp

    def prior_transform(self, u: float | ArrayLike) -> float | np.ndarray:
        """Prior transform (inverse CDF) of the uniform distribution.

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples between `self.low` and `self.high`
        """
        if np.any(np.logical_or(u < 0.0, u > 1.0)):
            raise ValueError("Prior transform expects values between 0 and 1.")
        return self.low + (self.high - self.low) * u

class Normal(ScipyDistribution):
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        if self.sigma <= 0.0:
            raise ValueError(
                "Standard deviation sigma must be positive for normal distribution"
            )
        super().__init__(norm(self.mu, self.sigma))

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma})"
