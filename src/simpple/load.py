# TODO: Unit tests for this module
from collections.abc import Callable
import sys
from pathlib import Path
from importlib import import_module

import scipy.stats
import yaml

from simpple.distributions import Distribution


def get_subclasses(cls):
    subclasses = cls.__subclasses__()
    results = {s.__name__: s for s in subclasses}
    if len(subclasses) == 0:
        return {}
    for subclass in subclasses:
        results |= get_subclasses(subclass)
    return results


DISTRIBUTIONS = get_subclasses(Distribution)


def load_parameters(path: Path | str) -> dict[str, Distribution]:
    with open(path) as f:
        mdict = yaml.safe_load(f)
    if "parameters" in mdict:
        pdict = mdict["parameters"]
    else:
        pdict = mdict
    return parse_parameters(pdict)


def parse_parameters(pdict: dict) -> dict[str, Distribution]:
    parameters = {}
    # TODO: Support kwargs
    for name, spec in pdict.items():
        dist = spec["dist"]
        args = spec["args"]
        if dist == "ScipyDistribution":
            if isinstance(args, list):
                k = 0
            elif isinstance(args, dict):
                k = "dist"
            args[k] = getattr(scipy.stats, args[k])
        if isinstance(args, list):
            parameters[name] = DISTRIBUTIONS[spec["dist"]](*spec["args"])
        elif isinstance(args, dict):
            parameters[name] = DISTRIBUTIONS[spec["dist"]](**spec["args"])
    return parameters


def write_parameters(parameters: dict[str, Distribution]) -> dict:
    out_dict = {}
    for pname, pdist in parameters.items():
        out_dict[pname] = pdist.to_yaml_dict()
    return out_dict


def resolve(func_str):
    if "." in func_str:
        module_str, func_str = func_str.rsplit(".", 1)
        module = import_module(module_str)
        func = getattr(module, func_str)
    elif func_str in globals():
        func = globals()["func_str"]
    elif func_str in sys.modules.get("__main__", {}).__dict__:
        func = sys.modules["__main__"].__dict__[func_str]
    else:
        raise ValueError(f"Could not find function {func_str}")
    return func


def get_func_str(func: Callable) -> str:
    mod = func.__module__
    name = func.__qualname__
    if mod == "__main__":
        return name
    return f"{mod}.{name}"
