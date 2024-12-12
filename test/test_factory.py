import unittest

from reinforced_lib.agents.mab import UCB

from mapc_mab import MapcAgentFactory


class MapcAgentFactoryTestCase(unittest.TestCase):

    def test_hierarchical_agent(self):
        associations = {
            0: [4, 5, 6, 7],
            1: [8, 9, 10, 11],
            2: [12, 13, 14, 15],
            3: [16, 17, 18, 19]
        }

        agent_factory = MapcAgentFactory(
            associations,
            agent_type=UCB,
            agent_params_lvl1={'c': 500.0},
            agent_params_lvl2={'c': 500.0},
            agent_params_lvl3={'c': 500.0},
            hierarchical=True
        )
        agent = agent_factory.create_mapc_agent()

        for _ in range(200):
            tx, tx_power = agent.sample(0.)

    def test_flat_agent(self):
        associations = {
            0: [4, 5, 6, 7],
            1: [8, 9, 10, 11],
            2: [12, 13, 14, 15],
            3: [16, 17, 18, 19]
        }

        agent_factory = MapcAgentFactory(
            associations,
            agent_type=UCB,
            agent_params_lvl1={'c': 500.0},
            hierarchical=False
        )
        agent = agent_factory.create_mapc_agent()

        for _ in range(200):
            tx, tx_power = agent.sample(0.)
