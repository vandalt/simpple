# Required so ArrayLike does not look terrible in docs
from __future__ import annotations

import inspect
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
from numpy.random import Generator
from numpy.typing import ArrayLike
from scipy.stats import loguniform, rv_continuous, norm, truncnorm, uniform
from scipy.special import erfinv, erf


class Distribution(ABC):
    """Abstract base class for distributions.

    Defines the interface distributions must implement.

    All distributions should have a `__repr__()` showing initialization parameters,
    as well as log_prob(), `prior_transform()` and `sample()` methods.
    """

    @abstractmethod
    def __repr__(self) -> str:
        pass

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))

    @property
    def required_args(self) -> list[str]:
        sig = inspect.signature(self.__class__.__init__)
        ignored_args = ["self", "args", "kwargs"]
        required_args = [
            pname
            for pname, pval in sig.parameters.items()
            if pname not in ignored_args and pval.default is pval.empty
        ]
        return required_args

    @property
    def optional_args(self) -> list[str]:
        sig = inspect.signature(self.__class__.__init__)
        # NOTE: Even if args and kwargs were not here, they would be ignored as their pval.default is pval.empty
        ignored_args = ["self", "args", "kwargs"]
        required_args = [
            pname
            for pname, pval in sig.parameters.items()
            if pname not in ignored_args and pval.default is not pval.empty
        ]
        return required_args

    def to_yaml_dict(self) -> dict:
        yaml_dict = {}
        yaml_dict["dist"] = self.__class__.__name__
        yaml_dict["args"] = [getattr(self, arg) for arg in self.required_args]
        # TODO: Maybe we should replace inf with None? Depends whether they work well with YAML... Test that
        yaml_dict["kwargs"] = {
            kwarg: getattr(self, kwarg) for kwarg in self.optional_args
        }
        return yaml_dict

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

    .. note::

        While this class can be convenient, custom ``log_prob()`` and ``prior_transform()`` functions written with Numpy are usually faster.

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
            if len(args) == 0 and len(kwargs) == 0:
                warnings.warn(
                    "The scipy distribution has no arguments and will use the default parameters."
                    "Make sure this is what you want.",
                    category=RuntimeWarning,
                    stacklevel=2,
                )
            self._dist = dist(*args, **kwargs)
        else:
            raise TypeError(f"Invalid type {type(dist)} for the scipy distribution.")

    @property
    def dist(self):
        """Underlying scipy distribution"""
        return self._dist

    @property
    def _comparison_dict(self):
        full_dict = self.__dict__
        comp_dict = {}
        for k in full_dict:
            if k != "_dist":
                comp_dict[k] = full_dict[k]
                continue
            dist_dict = full_dict[k].__dict__
            for k in dist_dict:
                if k == "dist":
                    comp_dict["scipy_dist_type"] = type(dist_dict[k])
                    continue
                elif isinstance(dist_dict[k], dict):
                    for kwd in dist_dict[k]:
                        comp_dict[f"scipy_dist_{k}_{kwd}"] = dist_dict[k][kwd]
                else:
                    comp_dict[f"scipy_dist_{k}"] = dist_dict[k]
        return comp_dict

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self._comparison_dict == other._comparison_dict

    def __hash__(self):
        return hash(tuple(sorted(self._comparison_dict.items())))

    def __repr__(self):
        args_tuple = self.dist.args
        kwargs_tuple = tuple(f"{k}={v}" for k, v in self.dist.kwds.items())
        signature_tuple = args_tuple + kwargs_tuple
        return f"ScipyDistribution({self.dist.dist.name}{signature_tuple})"

    def to_yaml_dict(self) -> dict:
        if type(self) is not ScipyDistribution:
            return Distribution.to_yaml_dict(self)

        yaml_dict = {}
        yaml_dict["dist"] = self.__class__.__name__
        dist_name = self.dist.dist.name
        dist_args = self.dist.args
        dist_kwargs = self.dist.kwds
        yaml_dict["args"] = [dist_name] + list(dist_args)
        yaml_dict["kwargs"] = dist_kwargs

        return yaml_dict

    def log_prob(self, x: float | ArrayLike, *args, **kwargs) -> np.ndarray:
        """Log probability density function.

        Calls the scipy distribution's `logpdf()`.

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        return self.dist.logpdf(x, *args, **kwargs)

    def prior_transform(self, u: float | ArrayLike, **kwargs) -> float | np.ndarray:
        """Prior transform (inverse CDF) for nested sampling

        Calls the scipy distributiion's ``ppf()`` (percent point function).

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples
        """
        if np.any(np.logical_or(u < 0.0, u > 1.0)):
            raise ValueError("Prior transform expects values between 0 and 1.")
        return self.dist.ppf(u, **kwargs)

    def sample(
        self,
        size: int | None = None,
        seed: int | Generator | np.ndarray[int] | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Draw samples from the distribution

        Calls the scipy distribution's ``rvs()`` method.

        :param size: Shape of the samples
        :param seed: Random seed to use (int or numpy generator)
        :return: Random samples with shape `size`
        """
        if not isinstance(seed, Generator):
            seed = np.random.default_rng(seed=seed)
        return self.dist.rvs(size=size, random_state=seed, **kwargs)


# TODO: Consider not making most distribution scipy sublcasses but just using dist attribute?
# One downside is that workarounds implemented here (example to_yaml being different for sicpy subclasses) can be unexpected if users subclass distribution.
# So regardless we need to work around that.
# Then would need to ignore or handle _dist in comparison.
class Uniform(ScipyDistribution):
    r"""Uniform distribution

    .. math::

        p(x) =
            \begin{cases}
                \frac{1}{b - a} & \text{if } a \leq x < b \\
                0 & \text{otherwise}
            \end{cases}

    :param low: Lower bound (b)
    :param high: Upper bound (a)
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        if self.low > self.high:
            raise ValueError(
                "lower bound should be lower than upper bound for uniform distribution"
            )
        super().__init__(uniform(self.low, self.high - self.low))

    def __repr__(self) -> str:
        return f"Uniform(low={self.low}, high={self.high})"

    def log_prob(self, x: float | ArrayLike) -> float | np.ndarray:
        """Log-probability for the uniform distribution

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        lp = np.where(
            np.logical_and(np.greater_equal(x, self.low), np.less(x, self.high)),
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
    r"""Normal distribution

    .. math::

        p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)

    :param mu: Mean :math:`\mu`
    :param sigma: Standard deviation :math:`\sigma`
    """

    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma
        if self.sigma <= 0.0:
            raise ValueError(
                "Standard deviation sigma must be positive for normal distribution"
            )
        super().__init__(norm(self.mu, self.sigma))

    def log_prob(self, x: float | ArrayLike, *args, **kwargs) -> np.ndarray:
        """Log probability density function of the normal distribution

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        return -0.5 * (
            np.log(2 * np.pi * self.sigma**2) + ((x - self.mu) / self.sigma) ** 2
        )

    def prior_transform(self, u: float | ArrayLike) -> np.ndarray:
        r"""Prior transform (inverse CDF) of the normal distribution.

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples centered around :math:`\mu` with standard deviation :math:`\simga`.
        """
        if np.any(np.logical_or(u < 0.0, u > 1.0)):
            raise ValueError("Prior transform expects values between 0 and 1.")
        return self.mu + self.sigma * np.sqrt(2) * erfinv(2 * u - 1)

    def __repr__(self) -> str:
        return f"Normal(mu={self.mu}, sigma={self.sigma})"


class LogUniform(ScipyDistribution):
    r"""Log-uniform distribution

    .. math::

        p(x) =
            \begin{cases}
                \frac{1}{x \ln(b/a)} & \text{if } a \leq x < b \\
                0 & \text{otherwise}
            \end{cases}

    :param low: Lower bound (b)
    :param high: Upper bound (a)
    """

    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high
        if self.low > self.high:
            raise ValueError(
                "lower bound should be lower than upper bound for log-uniform distribution"
            )
        if self.low <= 0 or self.high <= 0:
            raise ValueError("Bounds of the log-uniform distribution must be positive.")
        super().__init__(loguniform(self.low, self.high))

    def log_prob(self, x: float | ArrayLike) -> np.ndarray:
        """Log-probability for the log-uniform distribution

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        lp = np.where(
            np.logical_and(np.greater_equal(x, self.low), np.less(x, self.high)),
            -np.log(x * np.log(self.high / self.low)),
            -np.inf,
        )
        if lp.ndim == 0:
            return lp.item()
        return lp

    def prior_transform(self, u: float | ArrayLike) -> np.ndarray:
        """Prior transform (inverse CDF) of the log-uniform distribution.

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples on a logarithmic scale between `self.low` and `self.high`
        """
        if np.any(np.logical_or(u < 0.0, u > 1.0)):
            raise ValueError("Prior transform expects values between 0 and 1.")
        return self.low * np.exp(u * np.log(self.high / self.low))

    def __repr__(self) -> str:
        return f"LogUniform(low={self.low}, high={self.high})"


class TruncatedNormal(ScipyDistribution):
    r"""
    .. math::

        f(x; \mu, \sigma, a, b) =
            \begin{cases}
                \dfrac{1}{\sigma} \dfrac{\phi\left(\frac{x - \mu}{\sigma}\right)}
                {\Phi\left(\frac{b - \mu}{\sigma}\right) - \Phi\left(\frac{a - \mu}{\sigma}\right)}
                & \text{if } a \leq x \leq b \\
                0 & \text{otherwise}
            \end{cases}

    where :math:`\phi` is the standard normal distribution and :math:`\Phi` the standard normal CDF [1]_.

    .. rubric:: Footnotes

    .. [1] https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        low: float | None = None,
        high: float | None = None,
    ):
        self.mu = mu
        self.sigma = sigma
        self.low = -np.inf if low is None else low
        self.high = np.inf if high is None else high
        if self.sigma <= 0.0:
            raise ValueError(
                "Standard deviation sigma must be positive for truncated normal distribution"
            )
        if self.low > self.high:
            raise ValueError(
                "lower bound should be lower than upper bound for uniform distribution"
            )
        a, b = (self.low - self.mu) / self.sigma, (self.high - self.mu) / self.sigma
        super().__init__(truncnorm(a=a, b=b, loc=self.mu, scale=self.sigma))

    def __repr__(self) -> str:
        # TODO: Include low and high
        arg_signature = f"mu={self.mu}, sigma={self.sigma}"
        if np.isfinite(self.low):
            arg_signature += f", low={self.low}"
        if np.isfinite(self.high):
            arg_signature += f", high={self.high}"
        return f"TruncatedNormal({arg_signature})"

    def log_prob(self, x: float | ArrayLike, *args, **kwargs) -> np.ndarray:
        """Log probability density function.

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        phi_low = 0.5 * (1 + erf((self.low - self.mu) / self.sigma / np.sqrt(2)))
        phi_high = 0.5 * (1 + erf((self.high - self.mu) / self.sigma / np.sqrt(2)))
        log_norm = -0.5 * (
            np.log(2 * np.pi * self.sigma**2) + ((x - self.mu) / self.sigma) ** 2
        )
        lp = np.where(
            np.logical_and(np.greater_equal(x, self.low), np.less(x, self.high)),
            log_norm - np.log(phi_high - phi_low),
            -np.inf,
        )
        if lp.ndim == 0:
            return lp.item()
        return lp

    def prior_transform(self, u: float | ArrayLike) -> float | np.ndarray:
        """Prior transform (inverse CDF) of the truncated distribution.

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples between `self.low` and `self.high` following a normal distribution
        """
        if np.any(np.logical_or(u < 0.0, u > 1.0)):
            raise ValueError("Prior transform expects values between 0 and 1.")
        a = erf((self.low - self.mu) / (self.sigma * np.sqrt(2)))
        b = erf((self.high - self.mu) / (self.sigma * np.sqrt(2)))
        return self.mu + self.sigma * np.sqrt(2) * erfinv((1 - u) * a + u * b)


class Fixed(Distribution):
    """Fixed distribution

    This is essentially a delta function centered on ``value``.
    Is handled as a special case in :class:`simpple.model.Model` to avoid inefficient sampling.

    :param value: Value to which the parameter is fixed.
    """

    def __init__(self, value: float):
        self.value = value

    def __repr__(self) -> str:
        return f"Fixed(value={self.value})"

    def log_prob(self, x: float | ArrayLike) -> float | np.narray:
        """Log probability of a fixed variable.

        Returns ``np.inf`` at ``value`` and ``-np.inf`` elsewhere.

        :param x: Value(s) at which to evaluate the log probability.
        :return: Log probability value(s)
        """
        lp = np.where(
            x == self.value,
            np.inf,
            -np.inf,
        )
        if lp.ndim == 0:
            return lp.item()
        return lp

    def prior_transform(self, u: float | ArrayLike) -> float | np.ndarray:
        """Prior transform (inverse CDF) of the fixed distribution

        :param u: Uniform samples between 0 and 1
        :return: Transformed samples, all at ``value``.
        """
        if np.any(np.logical_or(u < 0.0, u > 1.0)):
            raise ValueError("Prior transform expects values between 0 and 1.")
        return np.full_like(u, self.value)
