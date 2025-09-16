import pytest
import numpy as np

from simpple import distributions as sd
from scipy.stats import loguniform, truncnorm, uniform, norm


def x2z(x, loc, scale):
    return (x - loc) / scale


# Map distributions to scipy distributions for comparison
DISTRIBUTIONS_EXACT = {
    sd.ScipyDistribution(uniform, -5, 10): uniform(-5, 10),
    sd.ScipyDistribution(uniform(-1000, 1000)): uniform(-1000, 1000),
    sd.ScipyDistribution(norm(0, 1)): norm(0, 1),
    sd.ScipyDistribution(norm(5, 0.001)): norm(5, 0.001),
    sd.Uniform(10, 25): uniform(10.0, 15.0),
    sd.Normal(10, 25): norm(10.0, 25.0),
    sd.LogUniform(1e-5, 1e5): loguniform(1e-5, 1e5),
    sd.TruncatedNormal(1, 5): truncnorm(-np.inf, np.inf, 1.0, 5.0),
    sd.TruncatedNormal(1, 5, low=0.0, high=10.0): truncnorm(
        x2z(0, 1, 5), x2z(10, 1, 5), 1, 5
    ),
    sd.TruncatedNormal(1, 5, high=10.0): truncnorm(-np.inf, x2z(10, 1, 5), 1, 5),
    sd.TruncatedNormal(1, 5, low=0.0): truncnorm(x2z(0, 1, 5), np.inf, 1, 5),
}
# Some distributions will not yield the same samples even when seed is fixed
# We store them separately for some tests
DISTRIBUTIONS_NOT_EXACT = {
    sd.TruncatedNormal(1, 5): norm(1.0, 5.0),
}
DISTRIBUTIONS = DISTRIBUTIONS_EXACT | DISTRIBUTIONS_NOT_EXACT
# Some tests apply only to distributions with bounds
DISTRIBUTIONS_BOUNDED = {k: v for k, v in DISTRIBUTIONS.items() if hasattr(k, "low")}


@pytest.mark.parametrize("dist,sp_dist", list(DISTRIBUTIONS.items()))
def test_log_prob(dist, sp_dist):
    n_samples = 100
    samples = dist.sample(size=n_samples)
    lp = dist.log_prob(samples)

    assert isinstance(lp, np.ndarray)
    lp_single = dist.log_prob(samples[0])
    assert isinstance(lp_single, (float))

    assert lp.shape == samples.shape
    np.testing.assert_allclose(lp, sp_dist.logpdf(samples))


@pytest.mark.parametrize("dist,sp_dist", list(DISTRIBUTIONS.items()))
def test_prior_transform(dist, sp_dist):
    n_samples = 100
    rng = np.random.default_rng()
    u = rng.uniform(size=n_samples)
    p = dist.prior_transform(u)

    assert isinstance(p, np.ndarray)
    assert isinstance(dist.prior_transform(u[0]), float)

    assert p.shape == u.shape
    np.testing.assert_allclose(p, sp_dist.ppf(u))

    with pytest.raises(ValueError):
        dist.prior_transform(1.2)


@pytest.mark.parametrize("dist,sp_dist", list(DISTRIBUTIONS_EXACT.items()))
def test_sample_exact(dist, sp_dist):
    n_samples = 1_000_000
    seed = 100

    samples_fix = dist.sample(size=n_samples, seed=seed)
    scipy_samples_fix = sp_dist.rvs(
        size=n_samples, random_state=np.random.default_rng(seed=seed)
    )
    np.testing.assert_allclose(samples_fix, scipy_samples_fix)


@pytest.mark.parametrize("dist,sp_dist", list(DISTRIBUTIONS.items()))
def test_sample_stats(dist, sp_dist):
    n_samples = 1_000_000

    samples = dist.sample(size=n_samples)
    scipy_samples = sp_dist.rvs(size=n_samples)
    std_err = samples.std() / np.sqrt(n_samples)
    se_var = np.sqrt(2 * samples.std() ** 4 / (n_samples - 1))
    np.testing.assert_allclose(samples.mean(), scipy_samples.mean(), atol=5 * std_err)
    np.testing.assert_allclose(
        samples.var(), scipy_samples.var(), atol=5 * se_var, rtol=1e-2
    )


@pytest.mark.parametrize("dist", list(DISTRIBUTIONS_BOUNDED))
def test_sample_bounds(dist):
    n_samples = 1_00_000
    samples = dist.sample(size=n_samples)
    np.testing.assert_array_less(samples, dist.high)
    np.testing.assert_array_less(dist.low, samples)


@pytest.mark.parametrize(
    "dist_class,bad_args",
    [
        (sd.Uniform, (10.0, 0.0)),
        (sd.Normal, (10.0, -50.0)),
        (sd.Normal, (10.0, -50.0)),
    ],
)
def test_bad_inits(dist_class, bad_args):
    with pytest.raises(ValueError):
        dist_class(*bad_args)


def test_out_of_bounds():
    assert sd.Uniform(100, 200).log_prob(300) == -np.inf


def test_fixed_distribution():
    val = 1.0
    fd = sd.Fixed(val)

    assert fd.value == val

    assert fd.log_prob(val) == np.inf
    assert fd.log_prob(1.2) == -np.inf

    rng = np.random.default_rng()

    random_vals = rng.uniform(-100, 100, size=10_000)
    assert np.all(~np.isfinite(fd.log_prob(random_vals)))

    u_vals = rng.uniform(size=10_000)
    p_vals = fd.prior_transform(u_vals)
    np.testing.assert_equal(p_vals, val)
