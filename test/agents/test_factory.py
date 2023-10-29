import unittest

import jax
import matplotlib.pyplot as plt
from reinforced_lib.agents.mab import UCB

from ml4wifi.envs.scenarios.static import simple_scenario_2
from ml4wifi.agents import MapcAgentFactory


class ScenarioClassTestCase(unittest.TestCase):
    def test_simple_sim(self):
        # Define test-case key and scenario
        key = jax.random.PRNGKey(42)
        scenario = simple_scenario_2()
        scenario.plot("scenario_3.pdf")

        # Define the agent factory and create MAPC agent
        agent_factory = MapcAgentFactory(
            scenario.associations,
            agent_type=UCB,
            agent_params={'c': 500.0}
        )
        agent = agent_factory.create_mapc_agent()

        # Simulate the network for 150 steps
        n_steps = 150
        thr = []
        reward = 0.

        for step in range(n_steps + 1):
            # Sample the agent
            key, agent_key, tx_key = jax.random.split(key, 3)
            tx = agent.sample(agent_key, reward)

            # Simulate the network
            reward = scenario(tx_key, tx)
            thr.append(reward)

        # Plot the approximate throughput
        plt.plot(thr)
        plt.xlim(0, n_steps)
        plt.ylim(0, 300)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_3_thr.pdf', bbox_inches='tight')
        plt.clf()
