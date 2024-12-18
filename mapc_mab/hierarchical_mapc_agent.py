from collections import defaultdict
from typing import Callable

import jax
import numpy as np
from chex import Array, PRNGKey, Scalar
from reinforced_lib.agents import BaseAgent
from reinforced_lib.agents.mab.scheduler import RandomScheduler

from mapc_mab.mapc_agent import MapcAgent


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
    find_groups_agent : BaseAgent
        The agent which selects the group of APs sharing the channel.
    find_groups_dict : dict[int, tuple]
        The dictionary which maps the station to the agent state.
    assign_stations_agents : dict[int, BaseAgent]
        The agents which select the stations served by the APs.
    assign_stations_dict : dict[tuple[tuple, int], tuple]
        The dictionary which maps the station and the group of APs to the agent state.
    select_tx_power_agent : BaseAgent
        The agent which selects the transmission power.
    select_tx_power_dict : dict[tuple[tuple, int], tuple]
        The dictionary which maps the group of APs and the served station to the agent state.
    ap_group_action_to_ap_group : Callable
        The function which translates the action of the agent to the tuple of APs sharing the channel.
    sta_group_action_to_sta_group : Callable
        The function which translates the action of the agent to the list of served stations.
    n_nodes : int
        The number of nodes in the network.
    key : PRNGKey
        The key for the random number generator.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            find_groups_agent: BaseAgent,
            find_groups_dict: dict[int, tuple],
            assign_stations_agents: dict[int, BaseAgent],
            assign_stations_dict: dict[tuple[tuple, int], tuple],
            select_tx_power_agent: BaseAgent,
            select_tx_power_dict: dict[tuple[tuple, int], tuple],
            ap_group_action_to_ap_group: Callable,
            sta_group_action_to_sta_group: Callable,
            n_nodes: int,
            key: PRNGKey
    ) -> None:
        self.associations = associations
        self.aps = dict(enumerate(associations.keys()))
        self.find_groups_agent = find_groups_agent
        self.find_groups_dict = find_groups_dict
        self.find_groups_last_step = defaultdict(int)
        self.assign_stations_agents = assign_stations_agents
        self.assign_stations_dict = assign_stations_dict
        self.assign_stations_last_step = defaultdict(int)
        self.select_tx_power_agent = select_tx_power_agent
        self.select_tx_power_dict = select_tx_power_dict
        self.select_tx_power_last_step = defaultdict(int)

        self.ap_group_action_to_ap_group = ap_group_action_to_ap_group
        self.sta_group_action_to_sta_group = sta_group_action_to_sta_group
        self.tx_matrix_shape = (n_nodes, n_nodes)
        self.tx_power_shape = (n_nodes,)

        self.ap_scheduler = RandomScheduler(len(associations))
        self.sta_schedulers = {ap: RandomScheduler(len(stations)) for ap, stations in associations.items()}
        self.key = key

        self.step = 0
        self.rewards = []

    def sample(self, reward: Scalar) -> tuple[Array, Array]:
        """
        Samples the agent to get the transmission matrix.

        Returns
        -------
        tuple
            The transmission matrix and the tx power vector.
        """

        self.step += 1
        self.rewards.append(reward)

        # Sample sharing AP and designated station
        self.key, ap_key, sta_key = jax.random.split(self.key, 3)
        sharing_ap = self.ap_scheduler.sample(None, ap_key).item()
        sharing_ap = self.aps[sharing_ap]
        designated_station = self.sta_schedulers[sharing_ap].sample(None, sta_key).item()
        designated_station = self.associations[sharing_ap][designated_station]

        # Sample the agent that finds groups of APs
        reward = self.rewards[self.find_groups_last_step[designated_station]]
        self.find_groups_last_step[designated_station] = self.step

        self.key, update_key, sample_key = jax.random.split(self.key, 3)
        last_action, state = self.find_groups_dict[designated_station]

        if last_action is not None:
            state = self.find_groups_agent.update(state, update_key, last_action, reward)

        ap_group_action = self.find_groups_agent.sample(state, sample_key).item()
        self.find_groups_dict[designated_station] = (ap_group_action, state)

        ap_group = self.ap_group_action_to_ap_group(ap_group_action, sharing_ap)
        all_aps = tuple(sorted(ap_group + (sharing_ap,)))

        # Sample the agents which assign stations to APs
        sta_group_action = {}

        for ap in ap_group:
            reward = self.rewards[self.assign_stations_last_step[ap_group, ap]]
            self.assign_stations_last_step[ap_group, ap] = self.step

            self.key, update_key, sample_key = jax.random.split(self.key, 3)
            n, last_action, state = self.assign_stations_dict[ap_group, ap]

            if last_action is not None:
                state = self.assign_stations_agents[n].update(state, update_key, last_action, reward)

            sta_group_action[ap] = self.assign_stations_agents[n].sample(state, sample_key).item()
            self.assign_stations_dict[ap_group, ap] = (n, sta_group_action[ap], state)

        sta_group = self.sta_group_action_to_sta_group(sta_group_action)
        all_aps_sort = np.argsort(np.asarray(tuple(ap_group) + (sharing_ap,)))
        all_stas = tuple(np.asarray(tuple(sta_group) + (designated_station,))[all_aps_sort].tolist())

        # Sample the agent which assigns tx power
        tx_power = np.zeros(self.tx_power_shape, dtype=np.int32)

        for i, (ap, sta) in enumerate(zip(all_aps, all_stas)):
            reward = self.rewards[self.select_tx_power_last_step[all_aps, sta]]
            self.select_tx_power_last_step[all_aps, sta] = self.step

            self.key, update_key, sample_key = jax.random.split(self.key, 3)
            last_action, state = self.select_tx_power_dict[all_aps, sta]

            if last_action is not None:
                state = self.select_tx_power_agent.update(state, update_key, last_action, reward)

            tx_power[ap] = self.select_tx_power_agent.sample(state, sample_key).item()
            self.select_tx_power_dict[all_aps, sta] = (tx_power[ap], state)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in zip(ap_group, sta_group):
            tx_matrix[ap, sta] = 1

        return tx_matrix, tx_power
