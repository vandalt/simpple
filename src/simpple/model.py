from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import yaml
from numpy.random import Generator
from numpy.typing import ArrayLike

from simpple.distributions import Distribution, Fixed
from simpple.load import (
    get_func_str,
    parse_parameters,
    resolve,
    unparse_parameters,
)
import simpple.utils as ut

if TYPE_CHECKING:
    from nautilus import Prior


class Model:
    """Simpple model

    .. note::

        Subclasses should call ``super().__init__()`` to ensure that the ``fixed_p`` and ``vary_p`` attributes are properly set.
        See :doc:`Writing Model Classes <../tutorials/writing-model-classes>`.

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

    def __repr__(self):
        class_name = self.__class__.__name__
        prior_str = f"parameters={self.parameters}"
        try:
            likelihood_name = getattr(
                self._log_likelihood, "__name__", "_log_likelihood"
            )
            likelihood_str = f", log_likelihood={likelihood_name}"
        except RecursionError:
            likelihood_str = ""
        return f"{class_name}({prior_str}{likelihood_str})"

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return ut.make_hashable(self.__dict__) == ut.make_hashable(other.__dict__)

    def __hash__(self):
        return hash(ut.make_hashable(self.__dict__))

    @property
    def required_args(self) -> list[str]:
        """List of required arguments at initialization, generated with :func:`simpple.utils.find_args`"""
        return ut.find_args(self, argtype="args")

    @property
    def optional_args(self) -> list[str]:
        """List of optional (keyword) arguments at initialization, generated with :func:`simpple.utils.find_args`"""
        return ut.find_args(self, argtype="kwargs")

    @classmethod
    def from_yaml(cls, path: Path | str, *args, **kwargs) -> Model:
        """Create a model from a YAML file

        The YAML file should have:

        - A ``class`` field mapping to the name of the Model class
        - A ``parameters`` field mapping parameter names to YAML distribution specs.
          See also :meth:`simpple.distributions.Distribution.from_yaml_dict()`.
        - An ``args`` field listing the required arguments for initialization.
          Required only for model subclasses that accept arguments.
        - A ``kwargs`` field listing the keyword arguments for initialization.
          Accepted only for model subclasses that accept keyword arguments.
          ``log_likelihood`` and ``forward`` are treated as special keyword arguments
          that refer to functions and we try to resolve them with :func:`simpple.load.resolve`.

        **Note**: All extra arguments (``args``) and keyword arguments (``kwargs``)
        are passed to this function are passed to the model initialization,
        and they override their YAML counterpart.

        See also: :doc:`Writing Models to and from YAML Files <../tutorials/yaml>`.

        :param path: Path of the YAML file
        :return: Model object
        """
        with open(path) as f:
            mdict = yaml.safe_load(f)
        parameters = parse_parameters(mdict["parameters"])
        yaml_args = mdict.get("args", [])
        if len(args) == 0 and len(yaml_args) > 0:
            args = yaml_args
        kwargs = mdict.get("kwargs", {}) | kwargs
        func_kwargs = ["log_likelihood", "forward"]
        for kwarg in func_kwargs:
            if kwarg in kwargs and isinstance(kwargs[kwarg], str):
                kwargs[kwarg] = resolve(kwargs[kwarg])
        model_classes = ut.get_subclasses(Model)
        class_name = mdict.get("class", None)
        if class_name is not None and class_name != "Model":
            model_cls = model_classes[class_name]
        else:
            model_cls = cls
        return model_cls(parameters, *args, **kwargs)

    def to_yaml(self, path: Path | str, overwrite: bool = False):
        model_dict = {}
        model_dict["class"] = self.__class__.__name__
        model_dict["parameters"] = unparse_parameters(self.parameters)
        func_kwargs = ["log_likelihood", "forward"]
        args_list = []
        for arg in self.required_args:
            arg_attr = getattr(self, arg)
            if arg in func_kwargs:
                arg_attr = get_func_str(arg_attr)
            args_list.append(arg_attr)
        kwargs_dict = {}
        for kwarg in self.optional_args:
            if kwarg in func_kwargs:
                kwarg_attr = getattr(self, f"_{kwarg}")
                kwarg_attr = get_func_str(kwarg_attr)
            else:
                kwarg_attr = getattr(self, kwarg)
            kwargs_dict[kwarg] = kwarg_attr
        if len(args_list) > 0:
            model_dict["args"] = args_list
        if len(kwargs_dict) > 0:
            model_dict["kwargs"] = kwargs_dict

        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"The file {path} already exists. Use overwrite=True to overwrite it."
            )
        with open(path, mode="w") as f:
            yaml.dump(model_dict, f)

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
        n_expect = len(self.keys(fixed=fixed))
        if x.shape[0] != n_expect:
            raise ValueError(
                f"Expected {n_expect} elements for the transform, got {x.shape[0]}"
            )
        pdist_list = self.parameters.values() if fixed else self.vary_p.values()
        for i, pdist in enumerate(pdist_list):
            x[i] = pdist.prior_transform(u[i])
        if is_dict:
            x = dict(zip(self.keys(fixed=fixed), x, strict=True))
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
        u = rng.uniform(size=(len(self.keys(fixed=fixed)), n_samples))
        if fmt == "dict":
            u = dict(zip(self.keys(fixed=fixed), u, strict=True))
        elif fmt != "array":
            raise ValueError(f"Invalid format: {fmt}. Use 'dict' or 'array'.")
        return self.prior_transform(u, fixed=fixed)


class ForwardModel(Model):
    """A model whose likelihood calls a forward model as the mean.

    .. note::

        Subclasses should call ``super().__init__()`` to ensure that the ``fixed_p`` and ``vary_p`` attributes are properly set.
        See :doc:`Writing Model Classes <../tutorials/writing-model-classes>`.


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

    def __repr__(self):
        class_name = self.__class__.__name__
        prior_str = f"parameters={self.parameters}"
        try:
            likelihood_name = getattr(
                self._log_likelihood, "__name__", "_log_likelihood"
            )
            likelihood_str = f", log_likelihood={likelihood_name}"
        except RecursionError:
            likelihood_str = ""
        try:
            forward_name = getattr(self._forward, "__name__", "_forward")
            forward_str = f", forward={forward_name}"
        except RecursionError:
            forward_str = ""
        return f"{class_name}({prior_str}{likelihood_str}{forward_str})"

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
