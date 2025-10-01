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


# TODO: Test implementing this in the distribution class?
# That way symmetric with to_yaml_dict and scipy mess is isolated
# Would need to bypass for scipy subclasses though
def parse_parameters(pdict: dict) -> dict[str, Distribution]:
    parameters = {}
    for name, spec in pdict.items():
        if "dist" not in spec:
            raise KeyError(
                "Distribution dictionaries should have a 'dist' key for the distribution name"
            )
        dist = spec["dist"]
        args = spec.get("args", [])
        kwargs = spec.get("kwargs", {})
        if dist == "ScipyDistribution":
            if len(args) > 0:
                if isinstance(args, list):
                    k = 0
                elif isinstance(args, tuple):
                    k = 0
                    # Cannot assign to tuples so convert to list
                    args = list(args)
                elif isinstance(args, dict):
                    k = "dist"
                else:
                    raise TypeError(
                        "args for ScipyDistribution should be a list, tuple or a dict"
                    )
                args[k] = getattr(scipy.stats, args[k])
            elif "dist" in kwargs:
                kwargs["dist"] = getattr(scipy.stats, kwargs["dist"])
            else:
                raise ValueError(
                    "ScipyDistribution should have a distribution specified in 'args' or 'kwargs'"
                )
        if dist not in DISTRIBUTIONS:
            raise KeyError(
                f"Distribution name '{dist}' not found. Available distributions are {DISTRIBUTIONS.keys()}"
            )
        dist_cls = DISTRIBUTIONS[dist]
        if isinstance(args, (list, tuple)):
            parameters[name] = dist_cls(*args, **kwargs)
        elif isinstance(args, dict):
            parameters[name] = dist_cls(**args, **kwargs)
    return parameters


def load_parameters(path: Path | str) -> dict[str, Distribution]:
    with open(path) as f:
        mdict = yaml.safe_load(f)
    if "parameters" in mdict:
        pdict = mdict["parameters"]
    else:
        pdict = mdict
    return parse_parameters(pdict)


def unparse_parameters(parameters: dict[str, Distribution]) -> dict:
    out_dict = {}
    for pname, pdist in parameters.items():
        out_dict[pname] = pdist.to_yaml_dict()
    return out_dict


def write_parameters(
    path: Path | str, parameters: dict[str, Distribution], overwrite: bool = False
):
    yaml_dict = unparse_parameters(parameters)

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"The file {path} already exists. Use overwrite=True to overwrite it."
        )
    with open(path, mode="w") as f:
        yaml.dump(yaml_dict, f)


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
