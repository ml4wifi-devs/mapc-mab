from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from chex import dataclass, Array, PRNGKey, Scalar
from reinforced_lib.agents import BaseAgent, AgentState
from tensorflow_probability.substrates.jax import bijectors as tfb
from tensorflow_probability.substrates.jax import distributions as tfd


@dataclass
class NormalThompsonSamplingState(AgentState):
    """
    Parameters of the Normal-inverse-gamma distribution a conjugate prior the Thompson sampling agent.

    Interpretaion of hyoperparameters
    ---------------------------------

    Mean was estimated from  :math:`\lambda` observations with sample mean  :math:`\mu`; variance was estimated from
    :math:`2\alpha` observations with sample mean :math:`\mu` and sum of squared deviations :math:`2\beta`

    Attributes
    ----------
    alpha Floating point array, the concentration params of the distribution(s). Must contain only positive values.
    beta Floating point tensor, the scale params of the distribution(s). Must contain only positive values.
    lam
    mu location
    """

    alpha: Array
    beta: Array
    lam: Array
    mu: Array


class NormalThompsonSampling(BaseAgent):
    r"""
    Normal Thompson sampling agent [1]_ , [2]_.

    Parameters
    ----------
    n_arms : int
        Number of bandit arms. :math:`N \in \mathbb{N}_{+}` .

    References
    ----------

    .. [1] https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
    .. [2] https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    """

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms

        self.init = jax.jit(partial(self.init, n_arms=self.n_arms))
        self.update = jax.jit(self.update)
        self.sample = jax.jit(self.sample)

    @staticmethod
    def parameter_space() -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {'n_arms': gym.spaces.Box(1, jnp.inf, (1,), jnp.int32)})

    @property
    def update_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {'action': gym.spaces.Discrete(self.n_arms),
             'reward': gym.spaces.Box(-jnp.inf, jnp.inf, (1,), jnp.int32),
             })

    @property
    def sample_observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict({})

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Space(self.n_arms)

    @staticmethod
    def init(key: PRNGKey, n_arms: int) -> NormalThompsonSamplingState:
        r"""
        Creates and initializes an instance of the Thompson sampling agent for ``n_arms`` arms. .

        Parameters
        ----------
        key : PRNGKey
            A PRNG key used as the random key.
        n_arms : int
            Number of bandit arms.

        Returns
        -------
        NormalThompsonSamplingState
            Initial state of the Thompson sampling agent.
        """

        return NormalThompsonSamplingState(alpha=jnp.ones((n_arms)), beta=jnp.ones((n_arms)),
                                           lam=2 * jnp.ones((n_arms)),
                                           mu=jnp.zeros((n_arms)))

    @staticmethod
    def update(state: NormalThompsonSamplingState, key: PRNGKey, action: jnp.int32,
               reward: Scalar, ) -> NormalThompsonSamplingState:
        r"""
        Thompson sampling according to [2]_.

        Parameters
        ----------
        state : NormalThompsonSamplingState
            Current state of the agent.
        key : PRNGKey
            A PRNG key used as the random key.
        action : int
            Previously selected action.
        reward : Float
            Reward obtained upon execution of action.


        Returns
        -------
        NormalThompsonSamplingState
            Updated agent state.
        """

        x = jnp.atleast_1d(reward)
        n = jnp.asarray(x.shape[0])
        xbar = jnp.mean(x)

        st = jtu.tree_map(lambda x: x[action], state)

        lam = st.lam + n
        mu = (st.mu * st.lam + n * xbar) / lam
        alpha = st.alpha + n / 2
        beta = st.beta + 0.5 * jnp.sum(jnp.square(x - xbar)) + 0.5 * n * st.lam / (n + st.lam) * jnp.square(
            xbar - st.mu)
        up = NormalThompsonSamplingState(alpha=alpha, beta=beta, mu=mu, lam=lam)
        ret = jtu.tree_map(lambda s, u: s.at[action].set(u), state, up)

        return ret

    @staticmethod
    def sample(state: NormalThompsonSamplingState, key: PRNGKey) -> jnp.int32:
        r"""
        The Thompson sampling policy is stochastic. The algorithm draws :math:`q_a` from the distribution

        """

        dist = tfd.JointDistributionNamedAutoBatched(
            dict(scale=tfb.Invert(tfb.Square())(tfd.InverseGamma(concentration=state.alpha, scale=state.beta)),
                 loc=lambda scale: tfd.Normal(loc=state.mu, scale=scale / jnp.sqrt(state.lam))))
        theta = dist.sample(seed=key)
        action = jnp.argmax(tfd.Normal(**theta).mean())
        return action


class LogNormalThompsonSampling(NormalThompsonSampling):
    """
    Thompson Sampling based on lognormal distribution. This algorithm is designed to handle positive rewards.

    For more details, refer to the documentation on :ref:`NormalThompsonSampling`.


    """
    @staticmethod
    def sample(state: NormalThompsonSamplingState, key: PRNGKey) -> jnp.int32:
        r"""
        Abstract distribution into base class

        """

        dist = tfd.JointDistributionNamedAutoBatched(
            dict(scale=tfb.Invert(tfb.Square())(tfd.InverseGamma(concentration=state.alpha, scale=state.beta)),
                 loc=lambda scale: tfd.Normal(loc=state.mu, scale=scale / jnp.sqrt(state.lam))))
        theta = dist.sample(seed=key)
        action = jnp.argmax(tfd.LogNormal(**theta).mean())
        return action

    @staticmethod
    def update(state: NormalThompsonSamplingState, key: PRNGKey, action: jnp.int32,
               reward: Scalar, ) -> NormalThompsonSamplingState:
        return NormalThompsonSampling.update(state, key, action, jnp.log(reward))
