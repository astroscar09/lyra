# evaluation.py
import numpy as np
from scipy.special import erf, erfinv
from scipy.stats import gaussian_kde
from lyra import config


def sample_from_posterior(posterior, x_obs_tensor, num_samples=1000):
    samples = posterior.sample_batched((num_samples,), x=x_obs_tensor)
    samples = samples.squeeze(-1).T
    return samples


def grab_percentiles_from_posterior(samples):
    l2pt5, l16, med, u84, u97pt5 = np.percentile(samples, q=config.PERCENTILES, axis=1)
    return l2pt5, l16, med, u84, u97pt5


def posterior_mean(posterior, x, n=1000):
    samples = sample_from_posterior(posterior, x, num_samples=n)
    return samples.mean(dim=1)


def cdf_transform(x, bounds):
    """Map Gaussian-space values x to physical EW space defined by bounds."""
    return _gaussian_cdf(x, 0, 1) * (bounds[1] - bounds[0]) + bounds[0]


def inv_cdf_transform(x, bounds):
    """Map physical EW values x (within bounds) to Gaussian space."""
    return _inv_gaussian_cdf((x - bounds[0]) / (bounds[1] - bounds[0]), 0, 1)


def _gaussian_cdf(x, mu, sigma):
    return 0.5 * (1 + erf((x - mu) / (np.sqrt(2) * sigma)))


def _inv_gaussian_cdf(x, mu, sigma):
    return mu + sigma * np.sqrt(2) * erfinv(2 * x - 1)


def make_kde(dist, bounds, resolution=1000):
    """Compute KDE over dist on a grid within bounds. Returns (density, mode)."""
    x = np.linspace(bounds[0], bounds[1], resolution)
    g = gaussian_kde(dist)
    y = g(x)
    return y, x[np.argmax(y)]
