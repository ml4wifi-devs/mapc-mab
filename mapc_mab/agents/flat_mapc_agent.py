from collections import defaultdict
from typing import Callable

import numpy as np
from chex import Array, Shape, Scalar
from reinforced_lib import RLib

from mapc_mab.agents.mapc_agent import MapcAgent


class FlatMapcAgent(MapcAgent):
    """
    The classic MAB agent responsible for the selection of the transmission power
    as well as AP and station pairs in one step.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    agents : dict[int, RLib]
        The dictionary of agents for each AP-station pair.
    agent_action_to_pairs : Callable
        The function which translates the action of the agent to the list of AP-station pairs.
    tx_matrix_shape : Shape
        The shape of the transmission matrix.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            agents: dict[int, RLib],
            agent_action_to_pairs: Callable,
            tx_matrix_shape: Shape
    ) -> None:
        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.array(list(associations.keys()))

        self.agents = agents
        self.last_step = defaultdict(int)
        self.agent_action_to_pairs = agent_action_to_pairs
        self.tx_matrix_shape = tx_matrix_shape

        self.step = 0
        self.rewards = []

    def sample(self, reward: Scalar) -> tuple[Array, Array]:
        """
        Samples the agent to get the transmission matrix.

        Returns
        -------
        tuple
            The transmission matrix and the tx power vector.
        """

        self.step += 1
        self.rewards.append(reward)

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points).item()
        designated_station = np.random.choice(self.associations[sharing_ap]).item()

        # Sample the appropriate agent
        reward = self.rewards[self.last_step[designated_station]]
        self.last_step[designated_station] = self.step

        action = self.agents[designated_station].sample(reward).item()
        pairs, tx_power = self.agent_action_to_pairs(designated_station, action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in pairs:
            tx_matrix[ap, sta] = 1

        return tx_matrix, tx_power
