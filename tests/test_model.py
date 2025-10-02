import copy
from pathlib import Path

import numpy as np
import pytest
from custom_models import Normal2DModel, PolyModel
from scipy.stats import norm

from simpple import distributions as sd
from simpple.load import load_parameters
from simpple.model import ForwardModel, Model


@pytest.fixture
def data_path():
    return Path(__file__).parent / "data"


def build_2d_normal():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Normal(10.0, 50.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def log_likelihood_norm(p):
        return norm_dist.logpdf([p["mu1"], p["mu2"]]).sum()

    return Model(params, log_likelihood_norm)


def build_2d_normal_forward():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Normal(10.0, 50.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def forward(p):
        return np.array([p["mu1"], p["mu2"]])

    def log_likelihood_norm(p):
        return norm_dist.logpdf(forward(p)).sum()

    return ForwardModel(params, log_likelihood_norm, forward)


def build_2d_normal_fixed():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Fixed(6.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def forward(p):
        return np.array([p["mu1"], p["mu2"]])

    def log_likelihood_norm(p):
        return norm_dist.logpdf(forward(p)).sum()

    return ForwardModel(params, log_likelihood_norm, forward)


def get_line_data():
    rng = np.random.default_rng(123)
    x = np.sort(10 * rng.random(100))
    m_true = 1.338
    b_true = -0.45
    y_true = m_true * x + b_true
    yerr = 0.1 + 0.5 * rng.random(x.size)
    y = y_true + 2 * yerr * rng.normal(size=x.size)
    return x, y, yerr


@pytest.fixture
def line_data():
    return get_line_data()


models = {
    "2d_normal": {
        "builder": build_2d_normal,
        "expect": {"keys": ["mu1", "mu2"], "fixed_keys": []},
        "test_point": {"mu1": 3.0, "mu2": 30.0},
        "test_point_outside": {"mu1": 888.0, "mu2": 30.0},
        "test_u": {"mu1": 0.5, "mu2": 0.8},
    },
    "2d_normal_forward": {
        "builder": build_2d_normal_forward,
        "expect": {"keys": ["mu1", "mu2"], "fixed_keys": []},
        "test_point": {"mu1": 3.0, "mu2": 30.0},
        "test_point_outside": {"mu1": 888.0, "mu2": 30.0},
        "test_u": {"mu1": 0.5, "mu2": 0.8},
    },
    "2d_normal_fixed": {
        "builder": build_2d_normal_fixed,
        "expect": {"keys": ["mu1"], "fixed_keys": ["mu2"]},
        "test_point": {"mu1": 3.0, "mu2": 6.0},
        "test_point_outside": {"mu1": 888.0, "mu2": 30.0},
        "test_u": {"mu1": 0.5, "mu2": 0.8},
    },
}


@pytest.mark.parametrize("model_name", models)
def test_keys_ndim(model_name):
    model_dict = models[model_name]
    model = model_dict["builder"]()
    expect = model_dict["expect"]

    assert model.keys() == expect["keys"]
    assert model.keys() == list(model.vary_p.keys())
    assert list(model.fixed_p.keys()) == expect["fixed_keys"]
    assert model.keys(fixed=True) == expect["keys"] + expect["fixed_keys"]
    assert model.ndim == len(model.keys())
    assert model.ndim == len(expect["keys"])


@pytest.mark.parametrize("model_name", models)
def test_log_likelihood(model_name):
    model_dict = models[model_name]
    model = model_dict["builder"]()

    p_dict = model_dict["test_point"]
    p_list = [v for k, v in p_dict.items() if k in model.keys()]
    print(p_list)
    assert model.log_likelihood(p_dict) == model.log_likelihood(p_list)

    if "test_point_oustide" in model_dict:
        model.log_likelihood(model_dict["test_point_outside"])


@pytest.mark.parametrize("model_name", models)
def test_log_prior(model_name):
    model_dict = models[model_name]
    model = model_dict["builder"]()

    p_dict = model_dict["test_point"]
    p_list = [v for k, v in p_dict.items() if k in model.keys()]
    assert model.log_prior(p_dict) == model.log_prior(p_list)
    if "test_point_outside" in model_dict:
        assert model.log_prior(model_dict["test_point_outside"]) == -np.inf


@pytest.mark.parametrize("model_name", models)
def test_prior_transform(model_name):
    model_dict = models[model_name]
    model: Model = model_dict["builder"]()

    u_dict = model_dict["test_u"]
    u_list = [v for k, v in u_dict.items() if k in model.keys()]
    u_list_fix = list(u_dict.values())
    p_dict = model.prior_transform(u_dict)
    p_dict_fix = model.prior_transform(u_dict, fixed=True)
    p_list = model.prior_transform(u_list)
    p_list_fix = model.prior_transform(u_list_fix, fixed=True)
    assert isinstance(p_dict, dict)
    assert isinstance(p_list, np.ndarray)
    assert list(p_dict), model.keys()
    np.testing.assert_equal(list(p_dict.values()), p_list)
    np.testing.assert_equal(list(p_dict_fix.values()), p_list_fix)


@pytest.mark.parametrize("model_name", models)
def test_nautilus_prior(model_name):
    model_dict = models[model_name]
    model = model_dict["builder"]()

    # This test should be more exhaustive when more distributions are implemented
    # with expected errors or OK for each distribution
    nautilus_prior = model.nautilus_prior()
    assert nautilus_prior.keys == model.keys()


@pytest.mark.parametrize("model_name", models)
def test_log_prob(model_name):
    model_dict = models[model_name]
    model = model_dict["builder"]()

    p_dict = model_dict["test_point"]
    p_list = [v for k, v in p_dict.items() if k in model.keys()]
    assert model.log_prob(p_dict) == model.log_prob(p_list)
    assert model.log_prob(p_dict) == model.log_prior(p_dict) + model.log_likelihood(
        p_dict
    )
    if "test_point_outside" in model_dict:
        assert model.log_prior(model_dict["test_point_outside"]) == -np.inf


@pytest.mark.parametrize("model_name", models)
def test_prior_samples(model_name):
    model_dict = models[model_name]
    model = model_dict["builder"]()

    n_samples = 100
    np.testing.assert_equal(
        model.get_prior_samples(n_samples, fmt="dict", seed=8),
        model.get_prior_samples(n_samples, seed=8),
    )
    np.testing.assert_equal(
        model.get_prior_samples(n_samples, fmt="dict", seed=42),
        dict(
            zip(model.keys(), model.get_prior_samples(n_samples, fmt="array", seed=42))
        ),
    )
    assert model.get_prior_samples(n_samples, fmt="array", seed=42).shape == (
        model.ndim,
        n_samples,
    )
    with pytest.raises(ValueError, match="Invalid format"):
        (model.get_prior_samples(n_samples, fmt="list"),)


def test_forward():
    model = build_2d_normal_forward()

    p_dict = {"mu1": 3.0, "mu2": 30.0}
    p_list = list(p_dict.values())
    np.testing.assert_equal(model.forward(p_dict), model.forward(p_list))


def test_prior_pred():
    model = build_2d_normal_forward()

    p_dict = {"mu1": 3.0, "mu2": 30.0}
    forward_output = model.forward(p_dict)

    n_samples = 100
    prior_pred_samples = model.get_prior_pred(n_samples)

    prior_pred_samples.shape == (n_samples,) + forward_output.shape


def test_posterior_pred():
    model = build_2d_normal_forward()

    p_dict = {"mu1": 3.0, "mu2": 30.0}
    forward_output = model.forward(p_dict)
    n_samples = 100

    samples = model.get_prior_samples(n_samples * 10)

    pred_samples = model.get_posterior_pred(samples, n_samples)

    pred_samples.shape == (n_samples,) + forward_output.shape


methods = [
    "log_likelihood",
    "log_prob",
    "log_prior",
    "prior_transform",
]


@pytest.mark.parametrize("model_name", models)
@pytest.mark.parametrize("method", methods)
def test_bad_params(model_name, method):
    model = models[model_name]["builder"]()

    p_dict = {"muuuuuuu": 0.5, "mu2": 1.0}
    p_list = [0.5, 1.0, 0.1]
    with pytest.raises(KeyError):
        getattr(model, method)(p_dict)
    with pytest.raises(ValueError):
        getattr(model, method)(p_list)


def test_model_with_args():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Normal(10.0, 50.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def forward(p):
        return np.array([p["mu1"], p["mu2"]])

    def log_likelihood_norm(p, like_dist, do_sum=False):
        if do_sum:
            return like_dist.logpdf(forward(p)).sum()
        return like_dist.logpdf(forward(p))

    model = ForwardModel(params, log_likelihood_norm, forward)

    p_dict = {"mu1": 3.0, "mu2": 30.0}
    # Test that args are required
    with pytest.raises(TypeError, match=".*missing 1 required.*"):
        model.log_prob(p_dict)
    with pytest.raises(TypeError, match=".*missing 1 required.*"):
        model.log_likelihood(p_dict)
    # Test that args are accepted
    model.log_prob(p_dict, norm_dist)
    model.log_likelihood(p_dict, norm_dist)
    # Test that kwarg has an impact
    assert model.log_likelihood(p_dict, norm_dist).ndim == 1
    assert model.log_likelihood(p_dict, norm_dist, do_sum=True).ndim == 0


def test_model_hash():
    model = build_2d_normal()
    hash(model)


def test_model_equal():
    model2d = build_2d_normal()
    model2d_cp = copy.deepcopy(model2d)
    model2d_fixed = build_2d_normal_fixed()
    assert model2d == model2d_cp
    assert id(model2d) != id(model2d_cp)
    assert model2d != model2d_fixed


def test_model_from_yaml(data_path):
    yaml_path = data_path / "normal2d.yaml"
    model_nolike = Model.from_yaml(yaml_path)
    with pytest.raises(NotImplementedError):
        model_nolike.log_likelihood({})

    # TODO: De-duplicate all the log_likelihoods
    norm_dist = norm([1.0, 5.0], 0.5)

    def log_likelihood_norm(p):
        return norm_dist.logpdf([p["mu1"], p["mu2"]]).sum()

    model = Model.from_yaml(yaml_path, log_likelihood_norm)
    test_p = {"mu1": 0.0, "mu2": 0.0}
    model.log_likelihood(test_p)
    model.log_likelihood([0.0, 0.0])

    yaml_path_like = data_path / "normal2d_likelihood.yaml"
    model_like = Model.from_yaml(yaml_path_like)
    model_like.log_likelihood(test_p)
    assert model.log_likelihood(test_p) == model_like.log_likelihood(test_p)

    assert model == model_like

    model_like_zero = Model.from_yaml(yaml_path_like, log_likelihood=lambda p: 0.0)
    assert model_like_zero.log_likelihood(test_p) == 0.0
    assert model_like_zero.log_likelihood(test_p) != model_like.log_likelihood(test_p)
    assert model_like_zero != model_like


def test_forward_from_yaml(data_path, line_data):
    # Model with functions should fail if functions not found
    yaml_path = data_path / "line.yaml"
    with pytest.raises(ValueError, match="Could not find function"):
        ForwardModel.from_yaml(yaml_path)

    test_p = {"m": 1.0, "b": 2.0, "sigma": 3.0}
    x, y, yerr = line_data
    test_args = (test_p, x, y, yerr)

    # Model without functions should raise NotImplemented for both
    yaml_path_noargs = data_path / "line_noargs.yaml"
    model_noargs = ForwardModel.from_yaml(yaml_path_noargs)
    with pytest.raises(NotImplementedError):
        model_noargs.log_likelihood({})
    with pytest.raises(NotImplementedError):
        model_noargs.forward({})

    def linear_model(p, x):
        return p["m"] * x + p["b"]

    def log_likelihood_line(p, x, y, yerr):
        ymod = linear_model(p, x)
        var = yerr**2 + p["sigma"] ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (y - ymod) ** 2 / var)

    model = ForwardModel.from_yaml(yaml_path)

    model_kwargs = ForwardModel.from_yaml(
        yaml_path_noargs, log_likelihood=log_likelihood_line, forward=linear_model
    )
    assert model.log_likelihood(*test_args) == model_kwargs.log_likelihood(*test_args)
    assert model_kwargs == model

    model_zero_one = ForwardModel.from_yaml(
        yaml_path, forward=lambda p: 0.0, log_likelihood=lambda p: 1.0
    )
    assert model_zero_one.log_likelihood(test_p) == 1.0
    assert model_zero_one.forward(test_p) == 0.0
    assert model_zero_one != model


@pytest.mark.parametrize(
    "model_dict",
    [
        {
            "file": "custom_normal2d.yaml",
            "cls": Normal2DModel,
            "test_p": {"mu1": 0.0, "mu2": 0.0},
            "alt_args": ([1.0, 5.0], 0.5),
        },
        {
            "file": "custom_poly.yaml",
            "cls": PolyModel,
            "test_p": {"a1": 5.0, "a0": 1.0, "sigma": 2.0},
            "alt_args": (0,),
            "extra_args": get_line_data(),
        },
    ],
)
def test_custom_normal_yaml(data_path, model_dict):
    yaml_path = data_path / model_dict["file"]
    cls = model_dict["cls"]
    test_p = model_dict["test_p"]
    alt_args = model_dict["alt_args"]
    model_custom = cls.from_yaml(yaml_path)
    extra_args = model_dict.get("extra_args", ())

    model_custom_kwargs = cls.from_yaml(yaml_path, *alt_args)
    pdict = load_parameters(yaml_path)
    model_custom_noyaml = cls(pdict, *alt_args)
    assert model_custom_noyaml.log_likelihood(
        test_p, *extra_args
    ) == model_custom_kwargs.log_likelihood(test_p, *extra_args)
    assert model_custom.log_likelihood(
        test_p, *extra_args
    ) != model_custom_kwargs.log_likelihood(test_p, *extra_args)
    assert model_custom_noyaml == model_custom_kwargs


norm_dist = norm([1.0, 5.0], 0.5)


def log_likelihood_norm(p):
    return norm_dist.logpdf([p["mu1"], p["mu2"]]).sum()


def test_models_roundtrip(tmp_path):
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Normal(10.0, 50.0)}

    model = Model(params, log_likelihood_norm)

    test_yaml_path = tmp_path / "test.yaml"
    model.to_yaml(test_yaml_path)
    model_read = model.__class__.from_yaml(test_yaml_path)
    assert model == model_read


@pytest.mark.parametrize(
    "yaml_file,cls",
    [
        ("normal2d.yaml", Model),
        ("normal2d_likelihood.yaml", Model),
        ("line.yaml", ForwardModel),
        ("line_noargs.yaml", ForwardModel),
        ("custom_normal2d.yaml", Normal2DModel),
        ("custom_poly.yaml", PolyModel),
    ],
)
def test_models_yaml_roundtrip(yaml_file: str, cls, data_path, tmp_path):
    norm_dist = norm([1.0, 5.0], 0.5)

    def log_likelihood_norm(p):
        return norm_dist.logpdf([p["mu1"], p["mu2"]]).sum()

    def linear_model(p, x):
        return p["m"] * x + p["b"]

    def log_likelihood_line(p, x, y, yerr):
        ymod = linear_model(p, x)
        var = yerr**2 + p["sigma"] ** 2
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (y - ymod) ** 2 / var)

    # Test that Model.from_yaml finds subclasses
    yaml_path = data_path / yaml_file

    if yaml_path.stem == "normal2d":
        kwargs = {"log_likelihood": log_likelihood_norm}
    elif yaml_path.stem == "line_noargs":
        kwargs = {"log_likelihood": log_likelihood_line, "forward": linear_model}
    else:
        kwargs = {}

    model = cls.from_yaml(yaml_path, **kwargs)

    model_generic = Model.from_yaml(yaml_path, **kwargs)
    assert model == model_generic

    test_yaml_path = tmp_path / "test.yaml"
    model.to_yaml(test_yaml_path)
    model_read = cls.from_yaml(test_yaml_path, **kwargs)
    assert model == model_read
