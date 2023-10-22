import gymnasium as gym
import jax.numpy as jnp

from reinforced_lib.exts import BaseExt, observation


class MapcSimExt(BaseExt):
    """
    Reinforced-lib extension for the Multi-Acces Point Coordination (MAPC) simulator. This extension
    can be used with Multi-Armed Bandit (MAB) algorithms.
    """

    observation_space = gym.spaces.Dict({})

    @observation(observation_type=gym.spaces.Box(-jnp.inf, jnp.inf, (1,)))
    def reward(self, reward, *args, **kwargs) -> float:
        return reward
