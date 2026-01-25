# evaluation.py
import numpy as np
import torch

def to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return x

def build_posterior(inference):
    return inference.build_posterior()


def sample_from_posterior(posterior, x_obs_tensor, num_samples = 1000):

    samples = posterior.sample_batched((num_samples,), x=x_obs_tensor)

    samples= samples.squeeze(-1).T

    return samples

def grab_percentiles_from_posterior(samples):

    l2pt5, l16, med, u84, u97pt5 = np.percentile(samples, q = (2.5, 16, 50, 84, 97.5), axis = 1)

    return l2pt5, l16, med, u84, u97pt5


def posterior_mean(posterior, x, n=1000):
    samples = sample_from_posterior(posterior, x, num_samples=n)
    return samples.mean(dim=0)
