from collections import defaultdict
from typing import Callable

import numpy as np
from chex import Array, Scalar, Shape
from mapc_mab.agents.offline_wrapper import OfflineWrapper as RLib

from mapc_mab.agents.mapc_agent import MapcAgent


class FlatMapcAgent(MapcAgent):
    """
    The classic MAB agent responsible for the selection of the AP and station pairs in one step.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    agent_dict : dict[int, RLib]
        The dictionary of agents for each AP-station pair.
    agent_action_to_pairs : Callable
        The function which translates the action of the agent to the list of AP-station pairs.
    tx_matrix_shape : Shape
        The shape of the transmission matrix.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            agent_dict: dict[int, RLib],
            agent_action_to_pairs: Callable,
            tx_matrix_shape: Shape
    ) -> None:
        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.array(list(associations.keys()))

        self.agent_dict = agent_dict
        self.agent_action_to_pairs = agent_action_to_pairs
        self.tx_matrix_shape = tx_matrix_shape

        self.step = 0
        self.rewards = []

        self.find_last_step = defaultdict(int)

    def sample(self, reward: Scalar) -> Array:
        """
        Samples the agent to get the transmission matrix.

        Parameters
        ----------
        reward: float
            The reward obtained in the previous step.

        Returns
        -------
        Array
            The transmission matrix.
        """

        self.step += 1
        self.rewards.append(reward)

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points)
        designated_station = np.random.choice(self.associations[sharing_ap])

        # Sample the appropriate agent
        reward_id = self.find_last_step[designated_station]
        self.find_last_step[designated_station] = self.step

        action = self.agent_dict[designated_station].sample(self.rewards[reward_id])
        pairs = self.agent_action_to_pairs(designated_station, action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in pairs:
            tx_matrix[ap, sta] = 1

        return tx_matrix
    
    def sample_offline(self, reward: Scalar) -> Array:
        """
        Samples the agent to get the transmission matrix in offline mode, meaning that the internal agent state is not updated.

        Parameters
        ----------
        reward: float
            The reward obtained in the previous step.

        Returns
        -------
        Array
            The transmission matrix.
        """

        self.step += 1
        self.rewards.append(reward)

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points)
        designated_station = np.random.choice(self.associations[sharing_ap])

        # Sample the appropriate agent
        reward_id = self.find_last_step[designated_station]
        self.find_last_step[designated_station] = self.step

        action = self.agent_dict[designated_station].sample_offline()
        pairs = self.agent_action_to_pairs(designated_station, action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in pairs:
            tx_matrix[ap, sta] = 1

        return tx_matrix
