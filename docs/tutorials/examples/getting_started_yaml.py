from scipy.stats import multivariate_normal


def log_likelihood(params):
    """Log-likelihood function for a 3D normal distribution."""
    p = [params["x1"], params["x2"], params["x3"]]
    mean = [0.0, 3.0, 2.0]
    cov = [[1, 0.5, 0], [0.5, 1, 0], [0, 0, 1.0]]
    return multivariate_normal.logpdf(p, mean=mean, cov=cov)


# model = Model(parameters, log_likelihood)
