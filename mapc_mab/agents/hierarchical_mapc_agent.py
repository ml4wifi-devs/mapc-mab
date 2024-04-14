from collections import defaultdict
from typing import Callable
from copy import copy

import numpy as np
from chex import Array, Scalar, Shape
from mapc_mab.agents.offline_wrapper import OfflineWrapper as RLib

from mapc_mab.agents.mapc_agent import MapcAgent


class HierarchicalMapcAgent(MapcAgent):
    """
    The hierarchical MAB agent responsible for the selection of the AP and station pairs.
    The agent consists of two phases:

      1. Selection of the group of APs which are sharing the channel.
      2. Selection of the stations which are served simultaneously by the APs in the group.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    find_groups_dict : dict[int, RLib]
        The dictionary of agents responsible for the selection of the APs group.
    assign_stations_dict : dict[tuple[int], dict[int, RLib]]
        The dictionary of agents responsible for the selection of the associated stations.
    ap_group_action_to_ap_group : Callable
        The function which translates the action of the agent to the tuple of APs sharing the channel.
    sta_group_action_to_sta_group : Callable
        The function which translates the action of the agent to the list of served stations.
    tx_matrix_shape : Shape
        The shape of the transmission matrix.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            find_groups_dict: dict[int, RLib],
            assign_stations_dict: dict[tuple[int], dict[int, RLib]],
            ap_group_action_to_ap_group: Callable,
            sta_group_action_to_sta_group: Callable,
            tx_matrix_shape: Shape
    ) -> None:
        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.array(list(associations.keys()))

        self.find_groups_dict = find_groups_dict
        self.assign_stations_dict = assign_stations_dict
        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.sta_group_action_to_sta_group = sta_group_action_to_sta_group
        self.tx_matrix_shape = tx_matrix_shape

        self.step = 0
        self.buffer = {}

        self.find_groups_last_step = defaultdict(int)
        self.assign_stations_last_step = defaultdict(lambda: defaultdict(int))
    
    def update(self, rewards: Array) -> None:
        """
        Updates the agent with the rewards obtained in the previous steps.

        Parameters
        ----------
        rewards : Array
            The buffer of rewards obtained in the previous steps.
        """
        
        for reward, (_, step_info) in zip(rewards, self.buffer.items()):
            sharing_ap, designated_station, ap_group_action, sta_group_action = step_info

            # Update the agent that finds groups of APs
            self.find_groups_dict[designated_station].update(ap_group_action, reward)

            # Update the agent which assigns stations to APs
            ap_group = self.ap_group_action_to_ap_group(
                ap_group_action,
                sharing_ap
            )
            all_aps = tuple(sorted(ap_group + (sharing_ap,)))
            for ap in ap_group:
                self.assign_stations_dict[all_aps][ap].update(sta_group_action[ap], reward)
        
        # Reset buffer
        self.buffer = {}
    

    def sample(self) -> Array:
        """
        Samples the hierarchical agent to get the transmission matrix.

        Returns
        -------
        Array
            The transmission matrix.
        """

        self.step += 1

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points)                       # Save the sharing AP
        designated_station = np.random.choice(self.associations[sharing_ap])    # Save the designated station

        # Sample the agent that finds groups of APs
        ap_group_action = self.find_groups_dict[designated_station].sample()    # Save the ap_group_action
        ap_group = self.ap_group_action_to_ap_group(
            ap_group_action,
            sharing_ap
        )
        all_aps = tuple(sorted(ap_group + (sharing_ap,)))

        # Sample the agent which assigns stations to APs
        sta_group_action = {}

        for ap in ap_group:
            self.assign_stations_last_step[all_aps][ap] = self.step
            sta_group_action[ap] = self.assign_stations_dict[all_aps][ap].sample()

        sta_group = self.sta_group_action_to_sta_group(sta_group_action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in zip(ap_group, sta_group):
            tx_matrix[ap, sta] = 1
        
        # Save step info to buffer
        self.buffer[self.step] = (sharing_ap, designated_station, ap_group_action, copy(sta_group_action))

        return tx_matrix
