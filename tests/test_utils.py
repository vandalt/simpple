import inspect

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
