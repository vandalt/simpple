import copy
import inspect
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import yaml

from simpple.distributions import Distribution


def parse_parameters(pdict: dict) -> dict[str, Distribution]:
    pdict = copy.deepcopy(pdict)
    parameters = {}
    for name, spec in pdict.items():
        parameters[name] = Distribution.from_yaml_dict(spec)
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
    func = None
    if "<locals>" in func_str:
        func_str = func_str.split(".")[-1]
    if "." in func_str and "<locals>" not in func_str:
        module_str, func_str = func_str.rsplit(".", 1)
        module = import_module(module_str)
        func = getattr(module, func_str)
    elif func_str in globals():
        func = globals()[func_str]
    elif func_str in sys.modules.get("__main__", {}).__dict__:
        func = sys.modules["__main__"].__dict__[func_str]
    else:
        # this loop was 100% vibe-coded
        for frame_info in inspect.stack():
            frame = frame_info.frame
            if func_str in frame.f_locals:
                func = frame.f_locals[func_str]
            if func_str in frame.f_globals:
                func = frame.f_globals[func_str]
    if not callable(func):
        raise ValueError(f"Could not find function {func_str}")
    return func


def get_func_str(func: Callable) -> str:
    if not callable(func):
        raise TypeError("func should be a callable")
    mod = func.__module__
    name = func.__qualname__
    if mod == "__main__":
        return name
    return f"{mod}.{name}"
