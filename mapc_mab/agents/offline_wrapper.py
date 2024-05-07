from typing import Any, Dict

import jax

from reinforced_lib.agents import BaseAgent, AgentState
from reinforced_lib.exts import BaseExt

class OfflineWrapper():

    def __init__(
            self,
            agent_type: BaseAgent,
            agent_params: dict[str, Any],
            ext_type: BaseExt,
            ext_params: dict[str, Any]
        ) -> None:

        self.agent_type = agent_type
        self.agent_params = agent_params
        self.ext_type = ext_type
        self.ext_params = ext_params
        self.agent: BaseAgent = agent_type(
            n_arms=ext_params['n_arms'],
            **agent_params
        )
        self.next_id = 0
        self.agent_states: Dict[int, AgentState] = {}

    def init(self, seed: int = 42):
        self.key = jax.random.PRNGKey(seed)
        self.key, key_init = jax.random.split(self.key)
        self.agent_states[self.next_id] = self.agent.init(key_init)
        self.next_id += 1

    def update(self, action: int, reward: float, agent_id: int = 0) -> None:
        self.key, key_update = jax.random.split(self.key)
        self.agent_states[agent_id] = self.agent.update(self.agent_states[agent_id], key_update, action, reward)

    def sample(self, agent_id: int = 0) -> int:
        self.key, key_sample = jax.random.split(self.key)
        return self.agent.sample(self.agent_states[agent_id], key_sample)
