from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Callable

import copy
import jax
import jax.numpy as jnp
from itertools import chain, combinations
import matplotlib.pyplot as plt
from chex import Array, Scalar, Shape, PRNGKey
from reinforced_lib import RLib
from reinforced_lib.agents import BaseAgent
from ml4wifi.envs.sim_ext import MapcSimExt


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


class MAPCAgent():
    
    def __init__(
            self,
            find_groups_dict: Dict,
            assign_stations_dict: Dict,
            ap_group_action_to_ap_group: Callable,
            sta_group_action_to_sta_group: Callable,
            tx_matrix_shape: Shape
        ) -> None:

        self.find_groups_dict = find_groups_dict
        self.assign_stations_dict = assign_stations_dict
        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.sta_group_action_to_sta_group = sta_group_action_to_sta_group
        self.tx_matrix_shape = tx_matrix_shape
    

    def sample(self, reward: Scalar, sharinng_ap: int, designated_station: int) -> Array:
        
        # Sample the agent which groups access points
        ap_group = self.ap_group_action_to_ap_group(
            self.find_groups_dict[designated_station].sample(reward=reward),
            sharinng_ap
        )

        # Sample the agent which assigns stations to access points groups
        sta_group = self.sta_group_action_to_sta_group(
            self.assign_stations_dict[ap_group].sample(reward=reward),
            ap_group
        )

        # Create the transmission matrix based on the sampled groups
        tx_matrix = jnp.zeros(self.tx_matrix_shape)
        tx_matrix = tx_matrix.at[sharinng_ap, designated_station].set(1)
        for ap, sta in zip(ap_group, sta_group):
            tx_matrix = tx_matrix.at[ap, sta].set(1)

        return tx_matrix


class MAPCAgentFactory():

    def __init__(
            self,
            associations: Dict[int, List[int]],
            agent_type: BaseAgent, agent_params: Dict
        ) -> None:

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


    def create_mapc_agent(self) -> MAPCAgent:
        
        # Define dictionary of Agent who are selecting groups
        find_groups_params = copy.deepcopy(self.agent_params)
        find_groups_params['n_arms'] = int(self.no_groups)
        find_groups : Dict = {
            sta: RLib(
                agent_type=self.agent_type,
                agent_params=find_groups_params,
                ext_type=MapcSimExt,
            ) for sta in self.stations
        }

        # Define dictionary of AssignStationsAgents
        assign_stations : Dict = {
            group: RLib(
                agent_type=self.agent_type,
                agent_params=self._assign_stations_params(group, self.agent_params),
                ext_type=MapcSimExt,
            ) for group in powerset(self.access_points)
        }

        # Define the MAPCAgent
        return MAPCAgent(
            find_groups_dict=find_groups,
            assign_stations_dict=assign_stations,
            ap_group_action_to_ap_group=self._ap_group_action_to_ap_group,
            sta_group_action_to_sta_group=self._sta_group_action_to_sta_group,
            tx_matrix_shape=(self.n_ap + self.n_sta, self.n_ap + self.n_sta)
        )
    

    def _assign_stations_params(self, group: List[int], agent_params: Dict) -> Dict:

        params = copy.deepcopy(agent_params)
        n_actions = 1
        for ap in group:
            n_actions *= len(self.associations[ap])
        params['n_arms'] = n_actions

        return params
    

    def _ap_group_action_to_ap_group(self, ap_group_action: int, sharing_ap: int) -> List[int]:
        """
        Translates the action of the agent to the list of access points which are sharing the channel.

        Parameters
        ----------
        ap_group_action : int
            The action of agent responsible for the selection of the access points group.
        sharing_ap : int
            The designated access point which has won the DCF contention and is sharing the channel.

        Returns
        -------
        List[int]
            The list of access points which are sharing the channel with ``sharing_ap``.
        """

        ap_set = set(self.access_points).difference({sharing_ap})
        ap_group = list(powerset(ap_set))[ap_group_action]

        return ap_group
        

    def _sta_group_action_to_sta_group(self, sta_group_action: int, ap_group: List[int]) -> List[int]:
        """
        Translates the action of the agent to the list of stations which are served simoultaneously.

        Parameters
        ----------
        sta_group_action : int
            The action of agent responsible for the selection of the stations group.
        ap_group : List[int]
            The list of access points which are sharing the same channel. These access points choose one
            station each to transmit.

        Returns
        -------
        List[int]
            The list os stations which are served simoultaneously by the access points in ``ap_group``.
        """
        
        sta_group_id = sta_group_action + 1
        sta_group = []
        divider = 1.0
        for ap in ap_group:
            sta_id = jnp.ceil(sta_group_id / divider) % len(self.associations[ap])
            sta_group.append(self.associations[ap][sta_id])
            divider *= len(self.associations[ap])
        
        return sta_group



        