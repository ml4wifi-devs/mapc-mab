from collections import defaultdict
from copy import deepcopy
from itertools import chain, combinations, product
from typing import Iterable, Iterator

import numpy as np
from reinforced_lib import RLib
from reinforced_lib.agents import BaseAgent
from reinforced_lib.exts import BasicMab

from mapc_mab.agents.flat_mapc_agent import FlatMapcAgent
from mapc_mab.agents.hierarchical_mapc_agent import HierarchicalMapcAgent
from mapc_mab.agents.mapc_agent import MapcAgent


class MapcAgentFactory:
    """
    The factory which creates the MAPC agent with the given agent type and parameters.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    agent_type : BaseAgent
        The type of the agent.
    agent_params_lvl1 : dict
        The parameters of the first level agent.
    agent_params_lvl2 : dict
        The parameters of the second level agent.
    agent_params_lvl3 : dict
        The parameters of the third level agent.
    hierarchical : bool
        The flag indicating whether the hierarchical or flat MAPC agent should be created.
    tx_power_levels : int
        The number of transmission power levels.
    seed : int
        The seed for the random number generator.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            agent_type: BaseAgent,
            agent_params_lvl1: dict,
            agent_params_lvl2: dict = None,
            agent_params_lvl3: dict = None,
            hierarchical: bool = True,
            tx_power_levels: int = 4,
            seed: int = 42
    ) -> None:
        self.associations = associations
        self.agent_type = agent_type
        self.agent_params_lvl1 = agent_params_lvl1
        self.agent_params_lvl2 = agent_params_lvl2
        self.agent_params_lvl3 = agent_params_lvl3
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

        # Agent selecting groups
        find_groups = RLib(
            agent_type=self.agent_type,
            agent_params=self.agent_params_lvl1.copy(),
            ext_type=BasicMab,
            ext_params={'n_arms': 2 ** (self.n_ap - 1)}
        )
        find_groups_dict = {}

        for i, sta in enumerate(self.stations):
            find_groups_dict[sta] = i
            find_groups.init(self.seed)
            self.seed += 1

        # Define dictionary of agents selecting stations
        assign_stations = {
            n: RLib(
                agent_type=self.agent_type,
                agent_params=self.agent_params_lvl2.copy(),
                ext_type=BasicMab,
                ext_params={'n_arms': n}
            ) for n in np.unique([len(stas) for stas in self.associations.values()])
        }
        assign_stations_dict = {}
        assign_stations_idxs = defaultdict(int)

        for group in self._powerset(self.access_points):
            for ap in group:
                n = len(self.associations[ap])
                idx = assign_stations_idxs[n]
                assign_stations_idxs[n] += 1
                assign_stations_dict[group, ap] = (n, idx)
                assign_stations[n].init(self.seed)
                self.seed += 1

        # Agent selecting tx power
        select_tx_power = RLib(
            agent_type=self.agent_type,
            agent_params=self.agent_params_lvl3.copy(),
            ext_type=BasicMab,
            ext_params={'n_arms': self.tx_power_levels}
        )
        select_tx_power_dict = {}
        idx = 0

        for group in self._powerset(self.access_points):
            for sta in chain.from_iterable(self.associations[ap] for ap in group):
                select_tx_power_dict[group, sta] = idx
                idx += 1
                select_tx_power.init(self.seed)
                self.seed += 1

        return HierarchicalMapcAgent(
            associations=self.associations,
            find_groups_agent=find_groups,
            find_groups_dict=find_groups_dict,
            assign_stations_agents=assign_stations,
            assign_stations_dict=assign_stations_dict,
            select_tx_power_agent=select_tx_power,
            select_tx_power_dict=select_tx_power_dict,
            ap_group_action_to_ap_group=self._ap_group_action_to_ap_group,
            sta_group_action_to_sta_group=self._sta_group_action_to_sta_group,
            tx_matrix_shape=(self.n_nodes, self.n_nodes),
            tx_power_levels=self.tx_power_levels
        )

    def create_flat_mapc_agent(self) -> MapcAgent:
        """
        Initializes the flat MAPC agent.

        Returns
        -------
        FlatMapcAgent
            The flat MAPC agent.
        """

        agents = {
            sta: RLib(
                agent_type=self.agent_type,
                agent_params=self.agent_params_lvl1.copy(),
                ext_type=BasicMab,
                ext_params={'n_arms': sum(map(lambda x: x[1], self._list_pairs_num(sta)))}
            ) for sta in self.stations
        }

        for agent in agents.values():
            agent.init(self.seed)
            self.seed += 1

        return FlatMapcAgent(
            associations=self.associations,
            agents=agents,
            agent_action_to_pairs=self._agent_action_to_pairs,
            tx_matrix_shape=(self.n_nodes, self.n_nodes)
        )

    @staticmethod
    def _iter_tx(associations: dict) -> Iterable:
        """
        Iterate through all possible actions: combinations of active APs and associated stations.

        Parameters
        ----------
        associations : dict
            A dictionary mapping APs to a list of stations associated with each AP.

        Returns
        -------
        iter
            An iterator over all possible actions.
        """

        aps = set(associations)

        for active in chain.from_iterable(combinations(aps, r) for r in range(1, len(aps) + 1)):
            for stations in product(*((s for s in associations[a]) for a in active)):
                yield tuple(zip(active, stations))

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
        Iteratively return the number of possible configurations (AP-station pairs and tx power levels) for parallel
        transmission alongside the designated station.

        Parameters
        ----------
        designated_station : int
            The station selected by the winner of the DCF contention.

        Returns
        -------
        Iterator[tuple[tuple, int]]
            The transmission pairs and the number of possible configurations.
        """

        associations = deepcopy(self.associations)
        associations.pop(self.inv_associations[designated_station])

        for conf in self._iter_tx(associations):
            yield conf, self.tx_power_levels ** (len(conf) + 1)

    def _agent_action_to_pairs(self, designated_station: int, action: int) -> tuple[tuple, list]:
        """
        Translates the action of the flat agent to the list of AP-station pairs and tx power levels
        which are served simultaneously.

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

        conf = tuple()

        for c, n in self._list_pairs_num(designated_station):
            if action < n:
                conf = c
                break
            action -= n

        tx_power = np.zeros(self.n_nodes, dtype=int)

        for ap, _ in conf:
            tx_power[ap] = action % self.tx_power_levels
            action //= self.tx_power_levels

        sharing_ap = self.inv_associations[designated_station]
        tx_power[sharing_ap] = action

        return conf, tx_power
