import inspect
from pathlib import Path

import yaml
import pytest

import simpple.distributions as sdist
from simpple.distributions import Distribution
from simpple.load import (
    DISTRIBUTIONS,
    get_subclasses,
    get_func_str,
    load_parameters,
    parse_parameters,
    resolve,
    unparse_parameters,
    write_parameters,
)
from simpple.model import ForwardModel, Model


# TODO: Test for custom class implemented in a different module
# TODO: Test for custom class implemented in the same module
# TODO: Test for custom class implemented in the current module
def test_get_subclasses():
    model_subclasses = get_subclasses(Model)
    assert len(model_subclasses) == 1
    assert "ForwardModel" in model_subclasses
    assert model_subclasses["ForwardModel"] == ForwardModel

    assert DISTRIBUTIONS == get_subclasses(Distribution)
    all_distribution_classes = [
        cls
        for _name, cls in inspect.getmembers(sdist, inspect.isclass)
        if cls.__module__ == sdist.__name__ and cls.__name__ != "Distribution"
    ]
    for cls in all_distribution_classes:
        assert cls.__name__ in DISTRIBUTIONS
        assert DISTRIBUTIONS[cls.__name__] == cls


yaml_parameter_dicts = [
    {"a": {"dist": "Uniform", "args": [-10, 10]}},  # Basic example
    {"a": {"dist": "Uniform", "args": (-10, 10)}},  # One with a tuple
    {"a": {"dist": "Uniform", "args": {"low": -10, "high": 10}}},  # args dict
    {"a": {"dist": "ScipyDistribution", "args": ["uniform", -10, 20]}},  # Scipy
    {"a": {"dist": "ScipyDistribution", "args": ("uniform", -10, 20)}},  # Scipy tuple
    {
        "a": {
            "dist": "ScipyDistribution",
            "args": {"dist": "uniform", "loc": -10, "scale": 20},
        }
    },  # Scipy dict
    {
        "a": {
            "dist": "ScipyDistribution",
            "kwargs": {"dist": "uniform", "loc": -10, "scale": 20},
        }
    },  # Scipy kwargs
    {"a": {"dist": "Normal", "args": [10, 5]}},  # Normal
    {"a": {"dist": "LogUniform", "args": [1e-5, 1e5]}},  # LogUniform
    {"a": {"dist": "Fixed", "args": [1.0]}},  # Fixed
    {"a": {"dist": "TruncatedNormal", "args": [10, 5]}},  # Trunc no bounds
    {"a": {"dist": "TruncatedNormal", "args": [10, 5, 4]}},  # Trunc lower
    {"a": {"dist": "TruncatedNormal", "args": [10, 5, None, 15]}},  # Trunc upper
    {"a": {"dist": "TruncatedNormal", "args": [10, 5, 4, 15]}},  # Trunc both
    {  # More than one parameter
        "a": {"dist": "Normal", "args": [10, 5]},
        "b": {"dist": "LogUniform", "args": [1e-5, 1e5]},
    },
    {"a": {"dist": "Uniform", "kwargs": {"low": -10, "high": 10}}},  # kwargs only
    {  # args+kwargs
        "a": {
            "dist": "TruncatedNormal",
            "args": [10, 5],
            "kwargs": {"low": 5, "high": 15},
        }
    },
    {  # args dict + kwargs
        "a": {
            "dist": "TruncatedNormal",
            "args": {"mu": 10, "sigma": 5},
            "kwargs": {"low": 5, "high": 15},
        }
    },
]


@pytest.mark.parametrize("yaml_dict", yaml_parameter_dicts)
def test_parse_parameters(yaml_dict):
    pdict = parse_parameters(yaml_dict)
    assert yaml_dict.keys() == pdict.keys()

    for pname, pdist in pdict.items():
        yaml_param = yaml_dict[pname]
        yaml_dist = yaml_param["dist"]
        yaml_args = yaml_param.get("args", [])
        yaml_kwargs = yaml_param.get("kwargs", {})
        yaml_args = (
            list(yaml_args.values()) if isinstance(yaml_args, dict) else list(yaml_args)
        )
        for i, arg in enumerate(pdist.required_args):
            if yaml_dist == "ScipyDistribution" and arg == "dist":
                continue
            elif len(yaml_args) > i:
                assert yaml_args[i] == getattr(pdist, arg)
            else:
                assert arg in yaml_kwargs
        for kwarg in yaml_kwargs:
            if yaml_dist == "ScipyDistribution":
                continue
            assert yaml_kwargs[kwarg] == getattr(pdist, kwarg)


def test_parse_parameters_errors():
    with pytest.raises(KeyError, match="Distribution dictionaries"):
        parse_parameters({"a": {}})

    with pytest.raises(TypeError, match="Uniform.__init__"):
        parse_parameters({"a": {"dist": "Uniform"}})

    with pytest.raises(TypeError, match="Uniform.__init__"):
        parse_parameters({"a": {"dist": "Uniform", "args": (-1,)}})

    with pytest.raises(KeyError, match="Distribution name"):
        parse_parameters({"a": {"dist": "Skibidi"}})

    with pytest.raises(
        ValueError, match="ScipyDistribution should have a distribution"
    ):
        parse_parameters({"a": {"dist": "ScipyDistribution", "args": []}})

    with pytest.raises(TypeError, match="args for ScipyDistribution"):
        parse_parameters({"a": {"dist": "ScipyDistribution", "args": "0, 10"}})

    with pytest.raises(TypeError, match="Distribution arguments"):
        parse_parameters({"a": {"dist": "Uniform", "args": "0, 10"}})


YAML_DIR = Path(__file__).parent / "data"
yaml_files = list(YAML_DIR.glob("*.yaml"))


@pytest.mark.parametrize("yaml_path", yaml_files)
def test_load_parameters(yaml_path: Path):
    pdict = load_parameters(yaml_path)
    with open(yaml_path, "r") as f:
        yaml_dict = yaml.safe_load(f)
    if "parameters" in yaml_dict:
        parsed_dict = parse_parameters(yaml_dict["parameters"])
    else:
        parsed_dict = parse_parameters(yaml_dict)
    assert pdict == parsed_dict


@pytest.mark.parametrize("yaml_dict", yaml_parameter_dicts)
def test_unparse_parameters(yaml_dict):
    pdict = parse_parameters(yaml_dict)
    yaml_results = unparse_parameters(pdict)
    pdict_v2 = parse_parameters(yaml_results)
    assert pdict == pdict_v2


@pytest.mark.parametrize("yaml_dict", yaml_parameter_dicts)
def test_write_parameters_dicts(yaml_dict, tmp_path):
    # Test round-trip parameter dict -> yaml -> parameter dict
    pdict = parse_parameters(yaml_dict)

    test_yaml_path = tmp_path / "test.yaml"
    write_parameters(test_yaml_path, pdict)
    pdict_read = load_parameters(test_yaml_path)
    assert pdict == pdict_read


@pytest.mark.parametrize("yaml_path", yaml_files)
def test_write_parameters_files(yaml_path: Path, tmp_path):
    pdict = load_parameters(yaml_path)

    test_yaml_path = tmp_path / "test.yaml"
    write_parameters(test_yaml_path, pdict)
    pdict_read = load_parameters(test_yaml_path)

    # We still test the parameter dicts
    # because the files will be slightly different (args vs kwargs, etc.)
    assert pdict == pdict_read


def test_write_actual_file(tmp_path):
    # This yaml file was machine-generated so the output should be the same
    yaml_path = Path(__file__).parent / "data/line_params_machine.yaml"
    pdict = load_parameters(yaml_path)
    test_yaml_path = tmp_path / "test.yaml"
    write_parameters(test_yaml_path, pdict)

    with open(yaml_path) as f_true, open(test_yaml_path) as f_test:
        dict_true = yaml.safe_load(f_true)
        dict_test = yaml.safe_load(f_test)
    assert dict_true == dict_test


def dummy_function():
    pass


@pytest.mark.parametrize(
    "func_str,expect",
    [
        ("numpy", False),  # Not available
        ("yaml", False),  # Not a function
        ("numpy.linspace", True),  # Can be imported
        ("write_parameters", True),  # In globals
        ("dummy_function", True),  # In parent module
        ("local_dummy_function", True),  # In parent local function scope
        ("simpple.plot.chainplot", True),  # Not already imported
    ],
)
def test_resolve(func_str, expect):
    def local_dummy_function():
        pass

    if not expect:
        with pytest.raises(ValueError, match="Could not find function"):
            resolve(func_str)
        return
    resolve(func_str)


@pytest.mark.parametrize(
    "func,expect",
    [
        (None, "test_load.test_get_func_str.<locals>.local_dummy_function"),
        (dummy_function, "test_load.dummy_function"),
        (write_parameters, "simpple.load.write_parameters"),
        (yaml.safe_load, "yaml.safe_load"),
        (yaml, False),
    ],
)
def test_get_func_str(func, expect):
    def local_dummy_function():
        pass

    if func is None:
        func = local_dummy_function
    if not expect:
        with pytest.raises(TypeError, match="func should be a callable"):
            get_func_str(func)
        return
    func_str = get_func_str(func)
    assert func_str == expect
