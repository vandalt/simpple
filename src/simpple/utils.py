import inspect
from typing import Any
import numpy as np
from scipy.stats._distn_infrastructure import rv_continuous_frozen


def get_subclasses(cls: type) -> dict[str, type]:
    """Get all subclasses of a class

    This is used internally when reading models and distributions from YAML.

    :param cls: Class for which we want the subclasses
    :return: Dictionary mapping subclass names to the actual subclasses
    """
    subclasses = cls.__subclasses__()
    results = {s.__name__: s for s in subclasses}
    if len(subclasses) == 0:
        return {}
    for subclass in subclasses:
        results |= get_subclasses(subclass)
    return results


def scipy_dist_to_dict(dist) -> dict:
    """Convert a scipy distribution to a dictionary

    This first calls the ``__dict__`` method and then tries to unroll
    arguments and keyword arguments.
    Used internally by :func:`make_hashable`.

    :param dist: Any scipy distribution creted through scipy.stats.
    :return: Dictionary with the distribution attributes
    """
    dist_dict = dist.__dict__
    comp_dict = {}
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


def make_hashable(obj: Any) -> tuple | bytes | Any:
    """Return an hashable version of an object

    Used internally to hash and compare models and distributions.

    Handles the following objects:

    - dict is converted to a tuple recursively
    - list and tuples are converted to tuples recursively
    - numpy arrays are converted to bytes with ``.tobytes()``
    - Scipy distribution are converted to dictionaries with :func:`scipy_dist_to_dict`
      and then made hashable.
    - Any other object is returned as is

    :param obj: Object to be made hashable
    :return: Hashable version of the object
    """
    # vibe-coded
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, (list, tuple)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tobytes()
    elif isinstance(obj, rv_continuous_frozen):
        return make_hashable(scipy_dist_to_dict(obj))
    else:
        return obj


def find_args(obj: Any, argtype: str = "args") -> list[str]:
    """Find arguments that an object requires at initialization

    Inspects the ``__init__`` method of the object class.
    Used internally by models and distributions.

    :param obj: Any object
    :param argtype: Type of argument desired (``"args"`` or ``"kwars"``)
    :return: List of argument names
    """

    def check_pval_type(pval, argtype: str):
        if argtype == "args":
            return pval.default is pval.empty
        elif argtype == "kwargs":
            return pval.default is not pval.empty
        else:
            raise ValueError("argtype must be one of 'args' or 'kwargs'")

    sig = inspect.signature(obj.__class__.__init__)
    ignored_args = ["self", "args", "kwargs", "parameters"]
    required_args = [
        pname
        for pname, pval in sig.parameters.items()
        if pname not in ignored_args and check_pval_type(pval, argtype)
    ]
    return required_args
