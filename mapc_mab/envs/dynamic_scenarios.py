from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from chex import Array, Scalar, PRNGKey
from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA, DATA_RATES
from mapc_sim.sim import network_data_rate

from mapc_mab.envs.scenario import Scenario
from mapc_mab.envs.static_scenarios import StaticScenario

DEFAULT_MCS = 11


class DynamicScenario(Scenario):
    """
    Dynamic scenario with possibility to change the configuration of the scenario at runtime.

    Parameters
    ----------
    associations: dict
        Dictionary of associations between access points and stations.
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    mcs: int
        Modulation and coding scheme of the nodes. Each entry corresponds to a node.
    tx_power: Scalar
        Default transmission power of the nodes.
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    pos_sec: Optional[Array]
        Array of node positions after the change.
    mcs_sec: Optional[int]
        Modulation and coding scheme of the nodes after the change.
    tx_power_sec: Optional[Scalar]
        Default transmission power of the nodes after the change.
    sigma_sec: Optional[Scalar]
        Standard deviation of the noise after the change.
    walls_sec: Optional[Array]
        Adjacency matrix of walls after the change.
    walls_pos_sec: Optional[Array]
        Array of wall positions after the change.
    tx_power_delta: Scalar
        Difference in transmission power between the tx power levels.
    """

    def __init__(
            self,
            associations: dict,
            pos: Array,
            mcs: int,
            tx_power: Scalar,
            sigma: Scalar,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            pos_sec: Optional[Array] = None,
            mcs_sec: Optional[int] = None,
            tx_power_sec: Optional[Scalar] = None,
            sigma_sec: Optional[Scalar] = None,
            walls_sec: Optional[Array] = None,
            walls_pos_sec: Optional[Array] = None,
            switch_steps: Optional[list] = None,
            tx_power_delta: Scalar = 3.0
    ) -> None:
        super().__init__(associations)

        if walls is None:
            walls = jnp.zeros((pos.shape[0], pos.shape[0]))
        if switch_steps is None:
            switch_steps = []

        self.data_rate_fn_first = jax.jit(partial(
            network_data_rate,
            pos=pos,
            mcs=jnp.full(pos.shape[0], mcs, dtype=jnp.int32),
            sigma=sigma,
            walls=walls
        ))
        self.tx_power_first = jnp.full(pos.shape[0], tx_power)
        self.mcs_first = mcs

        if pos_sec is None:
            pos_sec = pos.copy()
        if mcs_sec is None:
            mcs_sec = mcs
        if tx_power_sec is None:
            tx_power_sec = tx_power
        if sigma_sec is None:
            sigma_sec = sigma
        if walls_sec is None:
            walls_sec = walls.copy()

        self.data_rate_fn_sec = jax.jit(partial(
            network_data_rate,
            pos=pos_sec,
            mcs=jnp.full(pos_sec.shape[0], mcs_sec, dtype=jnp.int32),
            sigma=sigma_sec,
            walls=walls_sec
        ))
        self.tx_power_sec = jnp.full(pos_sec.shape[0], tx_power_sec)
        self.mcs_sec = mcs_sec

        self.data_rate_fn = self.data_rate_fn_first
        self.tx_power = self.tx_power_first
        self.mcs = self.mcs_first
        self.switch_steps = switch_steps
        self.step = 0
        self.tx_power_delta = tx_power_delta

    def __call__(self, key: PRNGKey, tx: Array, tx_power: Array) -> tuple[Scalar, Scalar]:
        if tx_power is None:
            tx_power = jnp.zeros(self.pos.shape[0])

        if self.step in self.switch_steps:
            self.switch()

        self.step += 1

        thr = self.data_rate_fn(key, tx, tx_power=self.tx_power - self.tx_power_delta * tx_power)
        reward = thr / DATA_RATES[self.mcs]
        return thr, reward

    def reset(self) -> None:
        self.data_rate_fn = self.data_rate_fn_first
        self.step = 0

    def switch(self) -> None:
        if self.data_rate_fn is self.data_rate_fn_first:
            self.data_rate_fn = self.data_rate_fn_sec
            self.tx_power = self.tx_power_sec
            self.mcs = self.mcs_sec
        else:
            self.data_rate_fn = self.data_rate_fn_first
            self.tx_power = self.tx_power_first
            self.mcs = self.mcs_first

    @staticmethod
    def from_static_params(
            scenario: StaticScenario,
            pos_sec: Optional[Array] = None,
            mcs_sec: Optional[int] = None,
            tx_power_sec: Optional[Scalar] = None,
            sigma_sec: Optional[Scalar] = None,
            walls_sec: Optional[Array] = None,
            walls_pos_sec: Optional[Array] = None,
            switch_steps: Optional[list] = None
    ) -> 'DynamicScenario':
        return DynamicScenario(
            scenario.associations,
            scenario.pos,
            scenario.mcs,
            scenario.tx_power,
            scenario.sigma,
            scenario.walls,
            scenario.walls_pos,
            pos_sec,
            mcs_sec,
            tx_power_sec,
            sigma_sec,
            walls_sec,
            walls_pos_sec,
            switch_steps
        )

    @staticmethod
    def from_static_scenarios(
            scenario: StaticScenario,
            scenario_sec: StaticScenario,
            switch_steps: list
    ) -> 'DynamicScenario':
        return DynamicScenario(
            scenario.associations,
            scenario.pos,
            scenario.mcs,
            scenario.tx_power,
            scenario.sigma,
            scenario.walls,
            scenario.walls_pos,
            scenario_sec.pos,
            scenario_sec.mcs,
            scenario_sec.tx_power,
            scenario_sec.sigma,
            scenario_sec.walls,
            scenario_sec.walls_pos,
            switch_steps
        )


def random_scenario(
        seed: int,
        d_ap: Scalar = 100.,
        n_ap: int = 4,
        d_sta: Scalar = 1.,
        n_sta_per_ap: int = 4,
        max_steps: int = 600,
        mcs: int = DEFAULT_MCS,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> DynamicScenario:
    def _draw_positions(key: PRNGKey) -> Array:
        ap_key, key = jax.random.split(key)
        ap_pos = jax.random.uniform(ap_key, (n_ap, 2)) * d_ap
        sta_pos = []

        for pos in ap_pos:
            sta_key, key = jax.random.split(key)
            center = jnp.repeat(pos[None, :], n_sta_per_ap, axis=0)
            stations = center + jax.random.normal(sta_key, (n_sta_per_ap, 2)) * d_sta
            sta_pos += stations.tolist()

        pos = jnp.array(ap_pos.tolist() + sta_pos)
        return pos

    associations = {i: list(range(n_ap + i * n_sta_per_ap, n_ap + (i + 1) * n_sta_per_ap)) for i in range(n_ap)}

    key_first, key_sec = jax.random.split(jax.random.PRNGKey(seed), 2)
    pos_first = _draw_positions(key_first)
    pos_sec = _draw_positions(key_sec)

    return DynamicScenario(associations, pos_first, mcs, tx_power, sigma, pos_sec=pos_sec, switch_steps=[max_steps // 2])
