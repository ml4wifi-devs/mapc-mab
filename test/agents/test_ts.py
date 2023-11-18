import unittest

import jax
import numpy as np
from tensorflow_probability.substrates.numpy import distributions as npd

from ml4wifi.agents.thompson_sampling import NormalThompsonSampling, LogNormalThompsonSampling


class TSTestCase(unittest.TestCase):
    def test_cycle(self):
        ts = NormalThompsonSampling(8)
        k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
        state = ts.init(k1)
        next_state = ts.update(state, k2, action=3, reward=3.0)
        a = ts.sample(next_state, k3)

    def test_cycle(self):
        ts = LogNormalThompsonSampling(8)
        k1, k2, k3 = jax.random.split(jax.random.key(4), 3)
        state = ts.init(k1)
        next_state = ts.update(state, k2, action=3, reward=3.0)
        a = ts.sample(next_state, k3)

    def test_agent(self):
        env = npd.Normal(loc=np.asarray([5, 20.]), scale=3)
        env.sample()

        ts = NormalThompsonSampling(2)
        k, k2, k3 = jax.random.split(jax.random.key(4), 3)
        state = ts.init(k2)

        for i in range(500):
            k1, k2, k = jax.random.split(k, 3)
            a = ts.sample(state, k1)
            r = env.sample()[a]
            state = ts.update(state, k2, action=a, reward=r)
        self.assertTrue(state.mu[1] > state.mu[0])
        ...


if __name__ == '__main__':
    unittest.main()
