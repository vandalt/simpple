import pytest
import numpy as np

from simpple import distributions as sd
from scipy.stats import uniform, norm

# Map distributions to scipy distributions for comparison
DISTRIBUTIONS = {
    sd.ScipyDistribution(uniform, -5, 10): uniform(-5, 10),
    sd.ScipyDistribution(uniform(-1000, 1000)): uniform(-1000, 1000),
    sd.ScipyDistribution(norm(0, 1)): norm(0, 1),
    sd.ScipyDistribution(norm(5, 0.001)): norm(5, 0.001),
    sd.Uniform(10, 25): uniform(10.0, 15.0),
}

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

@pytest.mark.parametrize("dist,sp_dist", list(DISTRIBUTIONS.items()))
def test_sample(dist, sp_dist):
    n_samples = 1_000_000
    seed = 100

    samples_fix = dist.sample(size=n_samples, seed=seed)
    scipy_samples_fix = sp_dist.rvs(size=n_samples, random_state=np.random.default_rng(seed=seed))
    np.testing.assert_allclose(samples_fix, scipy_samples_fix)

    samples = dist.sample(size=n_samples)
    scipy_samples = sp_dist.rvs(size=n_samples)
    std_err = samples.std() / np.sqrt(n_samples)
    se_var = np.sqrt(2 * samples.std()**4 / (n_samples - 1))
    np.testing.assert_allclose(samples.mean(), scipy_samples.mean(), atol=3 * std_err)
    np.testing.assert_allclose(samples.var(), scipy_samples.var(), atol=3 * se_var)
