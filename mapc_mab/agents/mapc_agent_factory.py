from itertools import chain, combinations
from typing import Iterable, Iterator

import numpy as np
from reinforced_lib import RLib
from reinforced_lib.exts import BasicMab

from mapc_mab.agents.flat_mapc_agent import FlatMapcAgent
from mapc_mab.agents.hierarchical_mapc_agent import HierarchicalMapcAgent
from mapc_mab.agents.mapc_agent import MapcAgent
from mapc_mab.agents.utils import iter_tx


class MapcAgentFactory:
    """
    The factory which creates the MAPC agent with the given agent type and parameters.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    agent_type : type
        The type of the agent.
    agent_params : dict
        The parameters of the agent.
    hierarchical : bool
        The flag indicating whether the hierarchical or flat MAPC agent should be created.
    seed : int
        The seed for the random number generator.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            agent_type: type,
            agent_params: dict,
            hierarchical: bool = True,
            tx_power_levels: int = 4,
            seed: int = 42
    ) -> None:
        self.associations = associations
        self.agent_type = agent_type
        self.agent_params = agent_params
        self.hierarchical = hierarchical
        self.tx_power_levels = tx_power_levels
        self.seed = seed

        np.random.seed(seed)

        # Retrieve stations and access points from associations
        self.inv_associations = {sta: ap for ap, stas in self.associations.items() for sta in stas}
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)
        self.n_nodes = self.n_ap + self.n_sta

    def create_mapc_agent(self) -> MapcAgent:
        """
        Initializes the MAPC agent.

        Returns
        -------
        MapcAgent
            The MAPC agent.
        """

        if self.hierarchical:
            return self.create_hierarchical_mapc_agent()
        else:
            return self.create_flat_mapc_agent()

    def create_hierarchical_mapc_agent(self) -> MapcAgent:
        """
        Initializes the hierarchical MAPC agent.

        Returns
        -------
        HierarchicalMapcAgent
            The hierarchical MAPC agent.
        """

        # Define dictionary of agents selecting groups
        find_groups: dict = {
            sta: RLib(
                agent_type=self.agent_type,
                agent_params=self.agent_params.copy(),
                ext_type=BasicMab,
                ext_params={'n_arms': 2 ** (self.n_ap - 1)}
            ) for sta in self.stations
        }

        # Define dictionary of agents selecting stations
        assign_stations: dict = {
            group: {
                ap: RLib(
                    agent_type=self.agent_type,
                    agent_params=self.agent_params.copy(),
                    ext_type=BasicMab,
                    ext_params={'n_arms': len(self.associations[ap])}
                ) for ap in group
            } for group in self._powerset(self.access_points)
        }

        # Define dictionary of agents selecting tx power
        select_tx_power = {}

        for bcoo in iter_tx(self.associations):
            mask, conf, *_ = bcoo.tree_flatten()[0]
            conf = tuple(tuple(c.tolist()) for m, c in zip(mask, conf) if m)

            select_tx_power[conf] = {
                sta: RLib(
                    agent_type=self.agent_type,
                    agent_params=self.agent_params.copy(),
                    ext_type=BasicMab,
                    ext_params={'n_arms': self.tx_power_levels}
                ) for _, sta in conf
            }

        for agent in chain(
                find_groups.values(),
                (ap for group in assign_stations.values() for ap in group.values()),
                (sta for conf in select_tx_power.values() for sta in conf.values())
        ):
            agent.init(self.seed)
            self.seed += 1

        return HierarchicalMapcAgent(
            associations=self.associations,
            find_groups_dict=find_groups,
            assign_stations_dict=assign_stations,
            select_tx_power_dict=select_tx_power,
            ap_group_action_to_ap_group=self._ap_group_action_to_ap_group,
            sta_group_action_to_sta_group=self._sta_group_action_to_sta_group,
            tx_matrix_shape=(self.n_nodes, self.n_nodes)
        )

    def create_flat_mapc_agent(self) -> MapcAgent:
        """
        Initializes the flat MAPC agent.

        Returns
        -------
        FlatMapcAgent
            The flat MAPC agent.
        """

        agents: dict = {
            sta: RLib(
                agent_type=self.agent_type,
                agent_params=self.agent_params.copy(),
                ext_type=BasicMab,
                ext_params={'n_arms': sum(map(lambda x: x[1], self._list_pairs_num(sta)))}
            ) for sta in self.stations
        }

        for agent in agents.values():
            agent.init(self.seed)
            self.seed += 1

        return FlatMapcAgent(
            associations=self.associations,
            agent_dict=agents,
            agent_action_to_pairs=self._agent_action_to_pairs,
            tx_matrix_shape=(self.n_nodes, self.n_nodes)
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

    def _ap_group_action_to_ap_group(self, ap_group_action: int, sharing_ap: int) -> tuple[int]:
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
        list[int]
            The list of all APs sharing the channel.
        """

        ap_set = set(self.access_points).difference({sharing_ap})
        return tuple(self._powerset(ap_set))[ap_group_action]

    def _sta_group_action_to_sta_group(self, sta_group_action: dict[int, int]) -> list[int]:
        """
        Translates the action of the agent to the list of stations which are served simultaneously.

        Parameters
        ----------
        sta_group_action : dict[int, int]
            The action of agent responsible for the selection of the stations group.

        Returns
        -------
        list[int]
            The list of stations which are served.
        """

        return [self.associations[ap][sta_id] for ap, sta_id in sta_group_action.items()]

    def _list_pairs_num(self, designated_station: int) -> Iterator[tuple[tuple, int]]:
        """
        Iteratively return the number of possible AP-station pairs for parallel transmission alongside the designated
        station. The number of possible pairs is computed as the product of the number of associated stations for each
        group of sharing APs.

        Parameters
        ----------
        designated_station : int
            The station selected by the winner of the DCF contention.

        Returns
        -------
        Iterator[tuple[tuple, int]]
            The group of sharing APs and the number of possible pairs within the group.
        """

        for aps in self._powerset(set(self.access_points).difference({self.inv_associations[designated_station]})):
            yield aps, np.prod([self.tx_power_levels * len(self.associations[ap]) for ap in aps]).astype(int).item()

    def _agent_action_to_pairs(self, designated_station: int, action: int) -> list[tuple[int, int]]:
        """
        Translates the action of the flat agent to the list of AP-station pairs which are served simultaneously.

        Parameters
        ----------
        designated_station : int
            The station selected by the winner of the DCF contention.
        action : int
            The action of the agent.

        Returns
        -------
        tuple[list, list]
            The list of AP-station pairs and tx power levels.
        """

        sharing_aps = tuple()

        for aps, n in self._list_pairs_num(designated_station):
            if action < n:
                sharing_aps = aps
                break
            action -= n

        pairs = []
        tx_power = []

        # TODO
        raise NotImplementedError
