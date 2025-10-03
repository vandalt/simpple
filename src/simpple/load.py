import copy
import inspect
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path

import yaml

from simpple.distributions import Distribution


def parse_parameters(pdict: dict) -> dict[str, Distribution]:
    """Parse parameter distributions from a YAML dictionary

    Each parameter spec is read with :meth:`simpple.distributions.Distribution.from_yaml_dict`

    :param pdict: Dictionary mapping parameter names to distribution specs
    :return: Dictionary mapping parameter names to :class:`simpple.distributions.Distribution` objects
    """
    pdict = copy.deepcopy(pdict)
    parameters = {}
    for name, spec in pdict.items():
        parameters[name] = Distribution.from_yaml_dict(spec)
    return parameters


def load_parameters(path: Path | str) -> dict[str, Distribution]:
    """Load parameter dictionary from YAML file

    The YAML file should contain a parameter specification consistent with :func:`parse_parameters`,
    either under the ``parameters`` key or at the top level.
    This allows users to read only the parmeters from a full model YAML file.

    :param path: Path to a YAML file.
    :return: Dictionary mapping parameter names to :class:`simpple.distributions.Distribution` objects
    """
    with open(path) as f:
        mdict = yaml.safe_load(f)
    if "parameters" in mdict:
        pdict = mdict["parameters"]
    else:
        pdict = mdict
    return parse_parameters(pdict)


def unparse_parameters(parameters: dict[str, Distribution]) -> dict:
    """Convert parameter dictionary to a YAML-compatible dictionary

    Does the exact opposite from :func:`parse_parameters`.
    Calls :meth:`simpple.distributions.Distribution.to_yaml_dict` for each parameter.
    (Note: this link is to the default implementation, see the docs for each class to see if it overrides it).

    :param parameters: Dictionary mapping parameter names to :class:`simpple.distributions.Distribution` objects
    :return: Dictionary mapping parameter names to YAML specifications
    """
    out_dict = {}
    for pname, pdist in parameters.items():
        out_dict[pname] = pdist.to_yaml_dict()
    return out_dict


def write_parameters(
    path: Path | str, parameters: dict[str, Distribution], overwrite: bool = False
):
    """Write parameters to a YAML file

    Calls :func:`unparse_parameters` and dumps it to the YAML file.

    :param path: Path of the YAML file
    :param parameters: Dictionary mapping parameter names to :class:`simpple.distributions.Distribution` objects
    :param overwrite: Overwrite the YAML file if ``True``
    """
    yaml_dict = unparse_parameters(parameters)

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"The file {path} already exists. Use overwrite=True to overwrite it."
        )
    with open(path, mode="w") as f:
        yaml.dump(yaml_dict, f)


def resolve(func_str: str) -> Callable:
    """Resolve a function based on its name

    This function tries to resolve a function based on its name.
    It is used by :meth:`simpple.model.Model.from_yaml` to
    resolve the likelihood and forward model functions based on the name given in a YAML file.

    If there is a dot (``.``) in the string, it will treat everything before the last dot as the
    module name and will try to import the function.

    Otherwise, it loops through the ``globals()`` dictionary, the ``__main__`` namespace,
    and then goes up the stack of contexts to find one with the function.

    If nothing is found, a ``ValueError`` is raised.

    :param func_str: Name of the function
    :return: The function object that the name refers to
    """
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
    """Get a string representing the function

    :param func: Function object
    :return: String representing the function
    """
    if not callable(func):
        raise TypeError("func should be a callable")
    mod = func.__module__
    name = func.__qualname__
    if mod == "__main__":
        return name
    return f"{mod}.{name}"
