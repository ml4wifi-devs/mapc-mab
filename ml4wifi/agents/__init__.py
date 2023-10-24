from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List

import copy
import jax
import jax.numpy as jnp
from itertools import chain, combinations
import matplotlib.pyplot as plt
from chex import Array, Scalar, PRNGKey
from reinforced_lib import RLib
from reinforced_lib.agents import BaseAgent
from envs.sim_ext import MapcSimExt


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# TODO The agent created by the Agent Factory
class MAPCAgent():
    pass


class AgentFactory():

    def __init__(self, associations: Dict[int, List[int]], agent_type: BaseAgent, agent_params: Dict) -> None:
        self.associations = associations
        self.agent_type = agent_type
        self.agent_params = agent_params

        # Retrieve stations and access points from associations
        self.access_points = list(associations.keys())
        self.stations = list(chain.from_iterable(associations.values()))
        self.n_ap = len(self.access_points)
        self.n_sta = len(self.stations)

        # Define number of groups each sharing AP can create
        self.no_groups = jnp.power(2, self.n_ap - 1)

    def hierarchical_agent(self) -> None:
        
        # Define dictionary of Agent who are selecting groups
        find_groups_params = copy.deepcopy(self.agent_params)
        find_groups_params['n_arms'] = self.no_groups
        find_groups : Dict = {
            sta: RLib(
                agent_type=self.agent_type,
                agent_params=find_groups_params,
                ext_type=MapcSimExt,
            ) for sta in self.stations
        }

        # Define dictionary of AssignStationsAgents
        assign_stations : Dict = {
            # Tutaj mona na spokojnie wszystkie podzbiory, bez jebania się które są trywialne
            group: RLib(
                agent_type=self.agent_type,
                agent_params=self._assign_stations_params(group, self.agent_params),
                ext_type=MapcSimExt,
            ) for group in powerset(self.access_points)
        }

        # TODO Create and return the MAPCAgent
    
    def _assign_stations_params(self, group: List[int], agent_params: Dict) -> Dict:
        params = copy.deepcopy(agent_params)
        
        # TODO Define number of actions, and assign to params

        return params


        