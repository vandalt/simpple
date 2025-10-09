import inspect

import pytest
from scipy.stats import uniform

import simpple.utils as ut
from simpple.distributions import Distribution
import simpple.distributions as sdist
from simpple.model import ForwardModel, Model


def test_get_subclasses():
    from custom_models import PolyModel, Normal2DModel

    model_subclasses = ut.get_subclasses(Model)
    assert len(model_subclasses) == 3
    assert model_subclasses["ForwardModel"] == ForwardModel
    assert model_subclasses["PolyModel"] == PolyModel
    assert model_subclasses["Normal2DModel"] == Normal2DModel

    class DummyModel(Model):
        pass

    model_dict = ut.get_subclasses(Model)
    assert sorted(list(model_dict)) == [
        "DummyModel",
        "ForwardModel",
        "Normal2DModel",
        "PolyModel",
    ]

    all_distributions = ut.get_subclasses(Distribution)
    all_distribution_classes = [
        cls
        for _name, cls in inspect.getmembers(sdist, inspect.isclass)
        if cls.__module__ == sdist.__name__ and cls.__name__ != "Distribution"
    ]
    for cls in all_distribution_classes:
        assert cls.__name__ in all_distributions
        assert all_distributions[cls.__name__] == cls


def test_scipy_dist_to_dict():
    # If it work on these two simple cases, it should work most of the time...
    uniform_args = uniform(0, 10)
    dict_args = ut.scipy_dist_to_dict(uniform_args)
    assert dict_args["scipy_dist_a"] == 0
    assert dict_args["scipy_dist_b"] == 1.0
    assert dict_args["scipy_dist_args"] == (0, 10)

    uniform_kwargs = uniform(loc=0, scale=10)
    dict_kwargs = ut.scipy_dist_to_dict(uniform_kwargs)
    assert dict_kwargs["scipy_dist_a"] == 0
    assert dict_kwargs["scipy_dist_b"] == 1.0
    assert dict_kwargs["scipy_dist_kwds_loc"] == 0.0
    assert dict_kwargs["scipy_dist_kwds_scale"] == 10.0


@pytest.mark.parametrize(
    "obj,hashed",
    [
        # Test some objects that are not handled
        (1, 1),
        ("allo", "allo"),
        (set(range(3)), set(range(3))),
        # Test cases that are handled
        ({"b": 1, "a": 2}, (("a", 2), ("b", 1))),
        ([1, 2, 3], (1, 2, 3)),
        ((1, 2, 3), (1, 2, 3)),
        # We already know make_hashable works on dicts so no need to write it out by hand
        (uniform(-50, 100), ut.make_hashable(ut.scipy_dist_to_dict(uniform(-50, 100)))),
    ],
)
def test_make_hashable(obj, hashed):
    assert ut.make_hashable(obj) == hashed


def test_find_args():
    def _myfun(arg1, arg2, parameters, *args, kwarg1=None, kwarg2=100.0, **kwargs):
        pass

    assert ut.find_args(_myfun) == ["arg1", "arg2"]
    assert ut.find_args(_myfun, argtype="args") == ["arg1", "arg2"]
    assert ut.find_args(_myfun, argtype="kwargs") == ["kwarg1", "kwarg2"]

    class MyClass:
        def __init__(
            self, arg1, arg2, parameters, *args, kwarg1=None, kwarg2=100.0, **kwargs
        ):
            pass

    assert ut.find_args(MyClass) == ["arg1", "arg2"]
    assert ut.find_args(MyClass, argtype="args") == ["arg1", "arg2"]
    assert ut.find_args(MyClass, argtype="kwargs") == ["kwarg1", "kwarg2"]

    myobj = MyClass(1, 2, 3)
    assert ut.find_args(myobj) == ["arg1", "arg2"]
    assert ut.find_args(myobj, argtype="args") == ["arg1", "arg2"]
    assert ut.find_args(myobj, argtype="kwargs") == ["kwarg1", "kwarg2"]
