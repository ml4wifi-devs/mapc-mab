import unittest

import jax
import matplotlib.pyplot as plt
import numpy as np
from reinforced_lib.agents.mab import UCB

from ml4wifi.envs.scenarios.static import simple_scenario_2
from ml4wifi.agents import MapcAgentFactory


class ScenarioClassTestCase(unittest.TestCase):
    def test_simple_sim(self):
        # Define test-case key and scenario
        key = jax.random.PRNGKey(42)
        np.random.seed(42)

        scenario = simple_scenario_2()
        scenario.plot("scenario_2.pdf")

        # Define the agent factory and create MAPC agent
        agent_factory = MapcAgentFactory(
            scenario.associations,
            agent_type=UCB,
            agent_params={'c': 500.0}
        )
        agent = agent_factory.create_mapc_agent()

        # Simulate the network for 150 steps
        n_steps = 150
        data_rate = []
        reward = 0.

        for step in range(n_steps + 1):
            # Sample the agent
            key, tx_key = jax.random.split(key, 2)
            tx = agent.sample(reward)

            # Simulate the network
            reward = scenario(tx_key, tx)
            data_rate.append(reward)

        # Plot the effective data rate
        plt.plot(data_rate)
        plt.xlim(0, n_steps)
        plt.ylim(0, 300)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated data_rateoughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_2_data_rate.pdf', bbox_inches='tight')
        plt.clf()
