# evaluation.py
import numpy as np
from lyra import config

def sample_from_posterior(posterior, x_obs_tensor, num_samples = 1000):

    samples = posterior.sample_batched((num_samples,), x=x_obs_tensor)

    samples= samples.squeeze(-1).T

    return samples

def grab_percentiles_from_posterior(samples):

    l2pt5, l16, med, u84, u97pt5 = np.percentile(samples, q=config.PERCENTILES, axis=1)

    return l2pt5, l16, med, u84, u97pt5


def posterior_mean(posterior, x, n=1000):
    samples = sample_from_posterior(posterior, x, num_samples=n)
    return samples.mean(dim=1)
