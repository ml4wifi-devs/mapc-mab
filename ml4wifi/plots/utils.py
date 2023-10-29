from typing import Tuple

import jax.numpy as jnp
from chex import Array, Scalar
from scipy.stats import t, ttest_ind


def confidence_interval(data: Array, ci: Scalar = 0.99) -> Tuple:
    measurements = data.shape[0]
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    alpha = 1 - ci
    z = t.ppf(1 - alpha / 2, measurements - 1)

    ci_low = mean - z * std / jnp.sqrt(measurements)
    ci_high = mean + z * std / jnp.sqrt(measurements)

    return mean, ci_low, ci_high


def ttest(data: Array) -> Array:
    n = data.shape[0]
    results = jnp.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            _, pval = ttest_ind(data[i], data[j], equal_var=False)
            results[i, j] = pval

    return results
