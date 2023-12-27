from itertools import chain, combinations
from typing import Dict, List, Iterable, Tuple

from reinforced_lib import RLib
from reinforced_lib.exts import BasicMab

from mapc_mab.agents.mapc_agent import MapcAgent


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
                ext_type=BasicMab,
                ext_params={'n_arms': 2 ** (self.n_ap - 1)}
            ) for sta in self.stations
        }

        # Define dictionary of agents selecting stations
        assign_stations: Dict = {
            group: {
                ap: RLib(
                    agent_type=self.agent_type,
                    agent_params=self.agent_params.copy(),
                    ext_type=BasicMab,
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

        s = sorted(list(iterable))
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
        return tuple(self._powerset(ap_set))[ap_group_action]

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
