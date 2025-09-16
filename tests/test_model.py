import numpy as np
import pytest

from simpple import distributions as sd
from scipy.stats import norm
from simpple.model import Model, ForwardModel


def build_2d_normal():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Normal(10.0, 50.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def log_likelihood(p):
        return norm_dist.logpdf([p["mu1"], p["mu2"]]).sum()

    return Model(params, log_likelihood)


def build_2d_normal_forward():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Normal(10.0, 50.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def forward(p):
        return np.array([p["mu1"], p["mu2"]])

    def log_likelihood(p):
        return norm_dist.logpdf(forward(p)).sum()

    return ForwardModel(params, log_likelihood, forward)


def build_2d_normal_fixed():
    params = {"mu1": sd.Uniform(-5, 5), "mu2": sd.Fixed(6.0)}
    norm_dist = norm([1.0, 5.0], 0.5)

    def forward(p):
        return np.array([p["mu1"], p["mu2"]])

    def log_likelihood(p):
        return norm_dist.logpdf(forward(p)).sum()

    return ForwardModel(params, log_likelihood, forward)


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

    def log_likelihood(p, like_dist, do_sum=False):
        if do_sum:
            return like_dist.logpdf(forward(p)).sum()
        return like_dist.logpdf(forward(p))

    model = ForwardModel(params, log_likelihood, forward)

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
