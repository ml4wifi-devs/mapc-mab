from typing import Callable, Dict, List, Tuple

import numpy as np
from chex import Array, Scalar, Shape
from reinforced_lib import RLib


class MapcAgent:
    """
    The MAPC agent which is responsible for the selection of the APs and stations pairs. The agent
    is hierarchical and consists of two phases:

      1. The agents selecting the group of APs which are sharing the channel.
      2. The agents selecting the stations which are served simultaneously by the APs in the group.

    Parameters
    ----------
    associations : Dict[int, List[int]]
        The dictionary of associations between APs and stations.
    find_groups_dict : Dict[int, RLib]
        The dictionary of agents responsible for the selection of the APs group.
    assign_stations_dict : Dict[Tuple[int], Dict[int, RLib]]
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
            associations: Dict[int, List[int]],
            find_groups_dict: Dict[int, RLib],
            assign_stations_dict: Dict[Tuple[int], Dict[int, RLib]],
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
        self.rewards = []

        self.find_groups_last_step = {sta: 0 for sta in find_groups_dict}
        self.assign_stations_last_step = {group: {ap: 0 for ap in group} for group in assign_stations_dict}

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

        # Sample the agent which find groups of APs
        ap_reward_id = self.find_groups_last_step[designated_station]
        self.find_groups_last_step[designated_station] = self.step

        ap_group = self.ap_group_action_to_ap_group(
            self.find_groups_dict[designated_station].sample(self.rewards[ap_reward_id]),
            sharing_ap
        )
        all_aps = tuple(sorted(ap_group + (sharing_ap,)))

        # Sample the agent which assigns stations to APs
        sta_group_action = {}

        for ap in ap_group:
            sta_reward_id = self.assign_stations_last_step[all_aps][ap]
            self.assign_stations_last_step[all_aps][ap] = self.step
            sta_group_action[ap] = self.assign_stations_dict[all_aps][ap].sample(self.rewards[sta_reward_id])

        sta_group = self.sta_group_action_to_sta_group(sta_group_action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in zip(ap_group, sta_group):
            tx_matrix[ap, sta] = 1

        return tx_matrix