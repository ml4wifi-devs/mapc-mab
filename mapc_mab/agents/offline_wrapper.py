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
        self.key, key_init = jax.random.split(self.key)
        self.agent_state = self.agent.init(key_init)
    
    def update(self, action: int, reward: float) -> None:
        self.key, key_update = jax.random.split(self.key)
        self.agent_state = self.agent.update(self.agent_state, key_update, action, reward)
    
    def sample(self) -> int:
        self.key, key_sample = jax.random.split(self.key)
        return self.agent.sample(self.agent_state, key_sample)
        