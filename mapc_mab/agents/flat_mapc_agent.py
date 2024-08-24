from typing import Callable

import numpy as np
from chex import Shape, Array
from mapc_mab.agents.offline_wrapper import OfflineWrapper as RLib

from mapc_mab.agents.mapc_agent import MapcAgent


class FlatMapcAgent(MapcAgent):
    """
    The classic MAB agent responsible for the selection of the transmission power
    as well as AP and station pairs in one step.

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
        self.buffer = {}
    
    def update(self, rewards: Array) -> None:
        """
        Updates the agent with the rewards obtained in the previous steps.

        Parameters
        ----------
        rewards : Array
            The buffer of rewards obtained in the previous steps.
        """
        
        for reward, (_, step_buffer) in zip(rewards, self.buffer.items()):
            designated_station, action = step_buffer

            # Update the agent
            self.agent_dict[designated_station].update(action, reward)
        
        # Reset buffer
        self.buffer = {}

    def sample(self) -> tuple:
        """
        Samples the agent to get the transmission matrix.

        Returns
        -------
        tuple
            The transmission matrix and the tx power vector.
        """

        self.step += 1

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points)
        designated_station = np.random.choice(self.associations[sharing_ap])        # Save the designated station

        # Sample the appropriate agent
        action = self.agent_dict[designated_station].sample()                       # Save the action
        pairs, tx_power = self.agent_action_to_pairs(designated_station, action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in pairs:
            tx_matrix[ap, sta] = 1

        # Save step info to buffer
        self.buffer[self.step] = (designated_station, action)

        return tx_matrix, tx_power
