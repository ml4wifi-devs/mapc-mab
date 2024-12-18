from collections import defaultdict
from typing import Callable

import jax
import numpy as np
from chex import Array, PRNGKey, Scalar
from reinforced_lib.agents import BaseAgent
from reinforced_lib.agents.mab.scheduler import RandomScheduler

from mapc_mab.mapc_agent import MapcAgent


class FlatMapcAgent(MapcAgent):
    """
    The classic MAB agent responsible for the selection of the transmission power
    as well as AP and station pairs in one step.

    Parameters
    ----------
    associations : dict[int, list[int]]
        The dictionary of associations between APs and stations.
    agents : dict[int, BaseAgent]
        The dictionary of agents.
    agents_dict : dict[int, tuple]
        The dictionary which maps the station to the agent state.
    agent_action_to_pairs : Callable
        The function which translates the action of the agent to the list of AP-station pairs.
    n_nodes : int
        The number of nodes in the network.
    key : PRNGKey
        The key for the random number generator.
    """

    def __init__(
            self,
            associations: dict[int, list[int]],
            agents: dict[int, BaseAgent],
            agents_dict: dict[int, tuple],
            agent_action_to_pairs: Callable,
            n_nodes: int,
            key: PRNGKey
    ) -> None:
        self.associations = associations
        self.aps = dict(enumerate(associations.keys()))
        self.agents = agents
        self.agents_dict = agents_dict
        self.last_step = defaultdict(int)

        self.agent_action_to_pairs = agent_action_to_pairs
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

        # Sample the appropriate agent
        reward = self.rewards[self.last_step[designated_station]]
        self.last_step[designated_station] = self.step

        self.key, update_key, sample_key = jax.random.split(self.key, 3)
        n, last_action, state = self.agents_dict[designated_station]

        if last_action is not None:
            state = self.agents[n].update(state, update_key, last_action, reward)

        action = self.agents[n].sample(state, sample_key).item()
        self.agents_dict[designated_station] = (n, action, state)

        pairs, tx_power = self.agent_action_to_pairs(designated_station, action)

        # Create the transmission matrix based on the sampled pairs
        tx_matrix = np.zeros(self.tx_matrix_shape, dtype=np.int32)
        tx_matrix[sharing_ap, designated_station] = 1

        for ap, sta in pairs:
            tx_matrix[ap, sta] = 1

        return tx_matrix, tx_power
