import unittest

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from reinforced_lib.agents.mab import UCB

from ml4wifi.envs.scenarios.static import *
from ml4wifi.agents import MAPCAgentFactory


class ScenarioClasssTestCase(unittest.TestCase):

    key = jax.random.PRNGKey(42)
    
    def test_simple_sim(self):
        
        # Define test-case key and scenario
        self.key, subkey = jax.random.split(self.key)
        scenario = simple_scenario_3()
        scenario.plot("scenario_3.png")

        # Define the agent factory and create MAPC agent
        agent_factory = MAPCAgentFactory(
            scenario.associations,
            agent_type=UCB,
            agent_params={
                'c': 5.0
            }
        )
        agent = agent_factory.create_mapc_agent()

        # Simulate the network for 50 steps
        n_steps = 50
        thr = []
        thr_reward = 0.
        for step in range(n_steps):
            subkey, key_tx = jax.random.split(subkey)

            # Step I: Round robin to select the sharing AP
            sharinng_ap = agent_factory.access_points[step % agent_factory.n_ap]

            # Step II: Round robin to select the designated station
            designated_station = agent_factory.stations[step % agent_factory.n_sta]

            # Step III and IV: Sample the agent to get the transmission matrix
            tx = agent.sample(reward=thr_reward, sharinng_ap=sharinng_ap, designated_station=designated_station)
            
            # Simulate the network
            thr_reward = scenario.thr_fn(key_tx, tx)
            thr.append(thr_reward)
            
        
        # Plot the approximate throughput
        xs = jnp.arange(n_steps)
        plt.plot(xs, thr)
        plt.xlim(0, n_steps)
        plt.ylim(0, n_steps)
        plt.xlabel('Timestep')
        plt.ylabel('Approximated throughput [Mb/s]')
        plt.title('Simulation of MAPC')
        plt.grid()
        plt.tight_layout()
        plt.savefig('scenario_3_thr.pdf', bbox_inches='tight')
        plt.clf()