from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List

import jax
import jax.numpy as jnp
from itertools import chain, combinations
import matplotlib.pyplot as plt
from chex import Array, Scalar, PRNGKey
from reinforced_lib import RLib
from reinforced_lib.agents.mab import UCB
from envs.sim_ext import MapcSimExt


class FindAPGroupAgent():

    def __init__(self, n_actions) -> None:
        pass


class AssignStationsAgent():

    def __init__(self) -> None:
        pass


class AgentFactory():

    def __init__(self, associations: Dict[int, List[int]]) -> None:
        self.associations = associations
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)

        # Define possible groups
        self.non_trivial_groups = self._non_trivial_groups()

    def hierarchical_agent(self) -> None:
        
        # Define dictionary of FindAPGroupAgents
        no_groups = jnp.power(2, self.n_ap - 1)
        find_friends_agents : Dict = {
            sta: FindAPGroupAgent(no_groups) \
                for sta in self.stations
        }

        # Define dictionary of AssignStationsAgents
        assign_stations_agents : Dict = {
            group: AssignStationsAgent(self._assign_stations_agents_no_actions(group)) \
                for group in self.non_trivial_groups
        }
    
    def _non_trivial_groups(self) -> List:
        return list(chain.from_iterable(combinations(self.n_ap, r) for r in range(2, self.n_ap + 1)))
    
    def _assign_stations_agents_no_actions(self, group: List) -> int:
        pass


        