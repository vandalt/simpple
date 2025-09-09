from pathlib import Path
from importlib import import_module

import scipy.stats
import yaml

from simpple.distributions import Distribution
from simpple.model import Model


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
        pdict = yaml.safe_load(f)
    return parse_parameters(pdict)


def parse_parameters(pdict: dict) -> dict[str, Distribution]:
    parameters = {}
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


def resolve(func_str):
    if "." in func_str:
        module_str, func_str = func_str.rsplit(".", 1)
        module = import_module(module_str)
        func = getattr(module, func_str)
    elif func_str in globals():
        func = globals()["func_str"]
    else:
        raise ValueError(f"Could not find function {func_str}")
    return func


def load_model(mdict: dict) -> Model:
    pass
