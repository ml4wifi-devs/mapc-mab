import os
import unittest

import jax
import matplotlib.pyplot as plt

from ml4wifi.envs.scenarios.static import *


class ScenarioClassTestCase(unittest.TestCase):
    def test_simple_plotting(self):
        scenario = simple_scenario_3()
        scenario.plot("test_simple_scenario.pdf")
        assert os.path.exists("test_simple_scenario.pdf")

    def test_random_plotting(self):
        scenario = random_scenario_1(seed=88)
        scenario.plot("test_random_scenario.png")
        assert os.path.exists("test_random_scenario.png")

    def test_simple_sim(self):
        # Define test-case key and scenario
        key = jax.random.PRNGKey(42)
        scenario = simple_scenario_3()

        # Transmission matrices indicating which node is transmitting to which node:
        # - in this example, STA 1 is transmitting to AP A
        tx1 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # - in this example, STA 2 is transmitting to AP A and STA 3 is transmitting to AP B
        tx2 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

        # - in this example, STA 1 is transmitting to AP A and STA 4 is transmitting to AP B
        tx3 = jnp.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Simulate the network for 150 steps
        thr1, thr2, thr3 = [], [], []

        for _ in range(150):
            key, k1, k2, k3 = jax.random.split(key, 4)
            thr1.append(scenario.thr_fn(k1, tx1))
            thr2.append(scenario.thr_fn(k2, tx2))
            thr3.append(scenario.thr_fn(k3, tx3))

        # Plot the approximate throughput
        xs = jnp.arange(150)
        plt.scatter(xs, thr1, label='STA 1 -> AP A', alpha=0.5, s=10, edgecolor='none')
        plt.scatter(xs, thr2, label='STA 2 -> AP A and STA 3 -> AP B', alpha=1., s=10, edgecolor='none')
        plt.scatter(xs, thr3, label='STA 1 -> AP A and STA 4 -> AP B', alpha=1., s=10, edgecolor='none')
        plt.xlim(0, 150)
        plt.ylim(0, 100)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_3_thr.pdf', bbox_inches='tight')
        plt.clf()

    def test_cca_threshold(self):
        assert simple_scenario_1(d=40.).is_cca_single_tx()
        assert simple_scenario_2(d_ap=50.).is_cca_single_tx()
        assert not simple_scenario_1(d=100.).is_cca_single_tx()
        assert not simple_scenario_2(d_ap=100.).is_cca_single_tx()
