from itertools import chain, combinations
from typing import Dict, List, Callable, Iterable, Tuple

import numpy as np
from chex import Array, Scalar, Shape
from reinforced_lib import RLib

from ml4wifi.envs.sim import MapcSimExt


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

        # Sample sharing AP and designated station
        sharing_ap = np.random.choice(self.access_points)
        designated_station = np.random.choice(self.associations[sharing_ap])

        # Sample the agent which find groups of APs
        ap_group = self.ap_group_action_to_ap_group(
            self.find_groups_dict[designated_station].sample(reward),
            sharing_ap
        )

        # Sample the agent which assigns stations to APs
        sta_group = self.sta_group_action_to_sta_group(
            {ap: self.assign_stations_dict[ap_group][ap].sample(reward) for ap in ap_group if ap != sharing_ap}
        )

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in zip(ap_group, sta_group):
            tx_matrix[ap, sta] = 1

        return tx_matrix


class MapcAgentFactory:
    """
    The factory which creates the MAPC agent with the given agent type and parameters.

    Parameters
    ----------
    associations : Dict[int, List[int]]
        The dictionary of associations between APs and stations.
    agent_type : type
        The type of the agent.
    agent_params : Dict
        The parameters of the agent.
    """

    def __init__(
            self,
            associations: Dict[int, List[int]],
            agent_type: type,
            agent_params: Dict
    ) -> None:
        self.associations = associations
        self.agent_type = agent_type
        self.agent_params = agent_params

        # Retrieve stations and access points from associations
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)

    def create_mapc_agent(self) -> MapcAgent:
        """
        Initializes the MAPC agent.

        Returns
        -------
        MapcAgent
            The hierarchical MAPC agent.
        """

        # Define dictionary of agents selecting groups
        find_groups: Dict = {
            sta: RLib(
                agent_type=self.agent_type,
                agent_params=self.agent_params.copy(),
                ext_type=MapcSimExt,
                ext_params={'n_arms': 2 ** (self.n_ap - 1)}
            ) for sta in self.stations
        }

        # Define dictionary of agents selecting stations
        assign_stations: Dict = {
            group: {
                ap: RLib(
                    agent_type=self.agent_type,
                    agent_params=self.agent_params.copy(),
                    ext_type=MapcSimExt,
                    ext_params={'n_arms': len(self.associations[ap])}
                ) for ap in group
            } for group in self._powerset(self.access_points)
        }

        return MapcAgent(
            associations=self.associations,
            find_groups_dict=find_groups,
            assign_stations_dict=assign_stations,
            ap_group_action_to_ap_group=self._ap_group_action_to_ap_group,
            sta_group_action_to_sta_group=self._sta_group_action_to_sta_group,
            tx_matrix_shape=(self.n_ap + self.n_sta, self.n_ap + self.n_sta)
        )

    @staticmethod
    def _powerset(iterable: Iterable) -> Iterable:
        """
        Returns the powerset of the given iterable. For example, the powerset of [1, 2, 3] is:
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)].

        Parameters
        ----------
        iterable: Iterable
            The iterable to compute the powerset.

        Returns
        -------
        Iterable
            The powerset of the given iterable.
        """

        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def _ap_group_action_to_ap_group(self, ap_group_action: int, sharing_ap: int) -> Tuple[int]:
        """
        Translates the action of the agent to the list of APs which are sharing the channel.

        Parameters
        ----------
        ap_group_action : int
            The action of agent responsible for the selection of the access points group.
        sharing_ap : int
            The designated access point which has won the DCF contention and is sharing the channel.

        Returns
        -------
        List[int]
            The list of all APs sharing the channel.
        """

        ap_set = set(self.access_points).difference({sharing_ap})
        ap_group = list(self._powerset(ap_set))[ap_group_action]
        ap_group = sorted(ap_group + (sharing_ap,))

        return tuple(ap_group)

    def _sta_group_action_to_sta_group(self, sta_group_action: Dict[int, int]) -> List[int]:
        """
        Translates the action of the agent to the list of stations which are served simultaneously.

        Parameters
        ----------
        sta_group_action : Dict[int, int]
            The action of agent responsible for the selection of the stations group.

        Returns
        -------
        List[int]
            The list of stations which are served.
        """

        return [self.associations[ap][sta_id] for ap, sta_id in sta_group_action.items()]
