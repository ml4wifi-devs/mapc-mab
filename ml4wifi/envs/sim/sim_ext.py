import gymnasium as gym
import jax.numpy as jnp

from reinforced_lib.exts import BaseExt, observation, parameter


class MapcSimExt(BaseExt):
    """
    Reinforced-lib extension for the Multi-Acces Point Coordination (MAPC) simulator. This extension
    can be used with Multi-Armed Bandit (MAB) algorithms.
    """

    def __init__(self, n_arms: int) -> None:
        super().__init__()
        self.n = n_arms

    observation_space = gym.spaces.Dict({})

    @parameter(parameter_type=gym.spaces.Box(1, jnp.inf, (1,), jnp.int32))
    def n_arms(self) -> int:
        return self.n

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,)))
    def reward(self, reward, *args, **kwargs) -> float:
        return reward