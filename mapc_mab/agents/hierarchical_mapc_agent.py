from itertools import chain
from typing import Callable
from copy import copy

import numpy as np
from chex import Shape, Array
from mapc_mab.agents.offline_wrapper import OfflineWrapper as RLib

from mapc_mab.agents.mapc_agent import MapcAgent


class HierarchicalMapcAgent(MapcAgent):
    """
    The hierarchical MAB agent responsible for the selection of the AP and station pairs.
    The agent consists of three phases:

      1. Selection of the group of APs which are sharing the channel.
      2. Selection of the transmission power for each AP in the group.
      3. Selection of the stations which are served simultaneously by the APs in the group.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    find_groups_dict : dict[int, int]
        The dictionary of agent ids responsible for the selection of the APs group.
    find_groups_agent : RLib
        The agent which selects the group of APs sharing the channel.
    assign_stations_dict : dict[int, dict[tuple, int]]
        The dictionary of agent ids responsible for the selection of the associated stations.
    assign_stations_agents : dict[int, RLib]
        The agents which select the stations served by the APs.
    select_tx_power_dict : dict[tuple, int]
        The dictionary of agent ids responsible for the selection of the transmission power.
    select_tx_power_agents : dict[int, RLib]
        The agents which selects the transmission power.
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
            find_groups_dict: dict[int, int],
            find_groups_agent: RLib,
            assign_stations_dict: dict[int, dict[tuple, int]],
            assign_stations_agents: dict[int, RLib],
            select_tx_power_dict: dict[tuple, int],
            select_tx_power_agents: dict[int, RLib],
            ap_group_action_to_ap_group: Callable,
            sta_group_action_to_sta_group: Callable,
            tx_matrix_shape: Shape,
            tx_power_levels: int
    ) -> None:
        self.associations = {ap: np.array(stations) for ap, stations in associations.items()}
        self.access_points = np.array(list(associations.keys()))
        self.n_nodes = len(self.access_points) + len(list(chain.from_iterable(associations.values())))

        self.find_groups_dict = find_groups_dict
        self.find_groups_agent = find_groups_agent
        self.assign_stations_dict = assign_stations_dict
        self.assign_stations_agents = assign_stations_agents
        self.select_tx_power_dict = select_tx_power_dict
        self.select_tx_power_agents = select_tx_power_agents
        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.sta_group_action_to_sta_group = sta_group_action_to_sta_group
        self.tx_matrix_shape = tx_matrix_shape
        self.tx_power_levels = tx_power_levels

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
            sharing_ap, designated_station, ap_group_action, tx_power, tx_power_action, sta_group_action = step_buffer

            # Recover the AP group, all APs and all tx power
            ap_group = self.ap_group_action_to_ap_group(
                ap_group_action,
                sharing_ap
            )
            all_aps = tuple(sorted(ap_group + (sharing_ap,)))
            all_tx_power = tuple((ap, tx_power[ap]) for ap in all_aps)

            # Update the agent that finds groups of APs
            find_groups_agent_id = self.find_groups_dict[designated_station]
            self.find_groups_agent.update(ap_group_action, reward, find_groups_agent_id)

            # Update the agent which assigns tx power
            tx_power_agent_id = self.select_tx_power_dict[all_aps]
            self.select_tx_power_agents[len(all_aps)].update(tx_power_action, reward, tx_power_agent_id)

            # Update the agent which assigns stations to APs
            for ap in ap_group:
                assign_stations_agent_id = self.assign_stations_dict[ap][all_tx_power]
                self.assign_stations_agents[ap].update(sta_group_action[ap], reward, assign_stations_agent_id)
        
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
        designated_station = np.random.choice(self.associations[sharing_ap])

        # Sample the agent that finds groups of APs
        agent_id = self.find_groups_dict[designated_station]
        ap_group_action = self.find_groups_agent.sample(agent_id)
        ap_group = self.ap_group_action_to_ap_group(
            ap_group_action,
            sharing_ap
        )
        all_aps = tuple(sorted(ap_group + (sharing_ap,)))

        # Sample the agent which assigns tx power
        agent_id = self.select_tx_power_dict[all_aps]
        tx_power_action = self.select_tx_power_agents[len(all_aps)].sample(agent_id)
        tx_power = np.zeros(self.n_nodes, dtype=np.int32)

        for i, ap in enumerate(all_aps):
            tx_power[ap] = tx_power_action % self.tx_power_levels
            tx_power_action //= self.tx_power_levels

        all_tx_power = tuple((ap, tx_power[ap]) for ap in all_aps)

        # Sample the agents which assign stations to APs
        sta_group_action = {}

        for ap in ap_group:
            agent_id = self.assign_stations_dict[ap][all_tx_power]
            sta_group_action[ap] = self.assign_stations_agents[ap].sample(agent_id)

        sta_group = self.sta_group_action_to_sta_group(sta_group_action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in zip(ap_group, sta_group):
            tx_matrix[ap, sta] = 1

        # Save step info to buffer
        self.buffer[self.step] = (sharing_ap, designated_station, ap_group_action, copy(tx_power), tx_power_action, copy(sta_group_action))

        return tx_matrix, tx_power
