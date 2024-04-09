from typing import Any

import jax

from reinforced_lib.agents import BaseAgent
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
    
    def init(self, seed: int = 42):
        self.key = jax.random.PRNGKey(seed)
        self.key, key_init, key_sample = jax.random.split(self.key, 3)
        self.agent_state = self.agent.init(key_init)
        self.last_action = self.agent.sample(self.agent_state, key_sample)
    
    def sample(self, reward: float) -> int:
        self.key, key_update, key_sample = jax.random.split(self.key, 3)
        self.agent_state = self.agent.update(self.agent_state, key_update, self.last_action, reward)
        self.last_action = self.agent.sample(self.agent_state, key_sample)
        return self.last_action
    
    def sample_offline(self) -> int:
        self.key, key_sample = jax.random.split(self.key)
        self.last_action = self.agent.sample(self.agent_state, key_sample)
        return self.last_action
        