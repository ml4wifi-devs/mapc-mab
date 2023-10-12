import unittest

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ml4wifi.envs.sim import DEFAULT_TX_POWER, init_static_network


class SimTestCase(unittest.TestCase):
    def test_simple_network(self):
        # Position of the nodes given by X and Y coordinates
        pos = jnp.array([
            [10., 10.],  # AP A
            [40., 10.],  # AP B
            [10., 20.],  # STA 1
            [ 5., 10.],  # STA 2
            [25., 10.],  # STA 3
            [45., 10.]   # STA 4
        ])

        n_nodes = pos.shape[0]

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

        # Modulation and coding scheme of the nodes (here, all nodes use MCS 4)
        mcs = jnp.ones(n_nodes, dtype=jnp.int32) * 4

        # Transmission power of the nodes (all nodes use the default transmission power)
        tx_power = jnp.ones(n_nodes) * DEFAULT_TX_POWER

        # Standard deviation of the additive white Gaussian noise
        noise = 1.

        # JAX random number generator key
        key = jax.random.PRNGKey(42)

        # Initialize the approximate network throughput function
        throughput_fn = init_static_network(pos, mcs, tx_power, noise)

        # Simulate the network for 150 steps
        thr1, thr2, thr3 = [], [], []

        for _ in range(150):
            key, k1, k2, k3 = jax.random.split(key, 4)
            thr1.append(throughput_fn(k1, tx1))
            thr2.append(throughput_fn(k2, tx2))
            thr3.append(throughput_fn(k3, tx3))

        # Plot the positions of the nodes
        plt.figure(figsize=(7.5, 4.5))
        plt.scatter(pos[:, 0], pos[:, 1], marker='x')

        for p, name in zip(pos, ['AP A', 'AP B', 'STA 1', 'STA 2', 'STA 3', 'STA 4']):
            plt.text(p[0] - 1, p[1] + 1, name)

        plt.xlim(0, 50)
        plt.ylim(0, 30)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.title('Location of nodes')
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_loc.pdf', bbox_inches='tight')
        plt.clf()

        # Plot the approximate throughput
        plt.plot(thr1, label='STA 1 -> AP A')
        plt.plot(thr2, label='STA 2 -> AP A and STA 3 -> AP B')
        plt.plot(thr3, label='STA 1 -> AP A and STA 4 -> AP B')
        plt.xlim(0, 150)
        plt.ylim(0, 100)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_thr.pdf', bbox_inches='tight')
        plt.clf()
