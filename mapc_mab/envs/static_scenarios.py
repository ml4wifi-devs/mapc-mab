from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from chex import Array, Scalar, PRNGKey
from mapc_sim.constants import DEFAULT_TX_POWER, DEFAULT_SIGMA
from mapc_sim.sim import network_data_rate

from mapc_mab.envs.scenario import Scenario


DEFAULT_MCS = 11


class StaticScenario(Scenario):
    """
    Static scenario with fixed node positions, MCS, tx power, and noise standard deviation.
    The configuration of parallel transmissions is variable.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    mcs: int
        Modulation and coding scheme of the nodes. Each entry corresponds to a node.
    tx_power: Scalar
        Transmission power of the nodes. Each entry corresponds to a node.
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.
    associations: dict
        Dictionary of associations between access points and stations.
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    tx_power_delta: Scalar
        Difference in transmission power between the tx power levels.
    """

    def __init__(
            self,
            pos: Array,
            mcs: int,
            tx_power: Scalar,
            sigma: Scalar,
            associations: dict,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None,
            tx_power_delta: Scalar = 3.0
    ) -> None:
        super().__init__(associations)

        self.pos = pos
        self.mcs = mcs
        self.tx_power = jnp.full(pos.shape[0], tx_power)
        self.tx_power_delta = tx_power_delta
        self.sigma = sigma
        self.walls = walls if walls is not None else jnp.zeros((pos.shape[0], pos.shape[0]))
        self.walls_pos = walls_pos

        self.data_rate_fn = jax.jit(partial(
            network_data_rate,
            pos=self.pos,
            mcs=jnp.full(pos.shape[0], mcs, dtype=jnp.int32),
            sigma=self.sigma,
            walls=self.walls
        ))

    def __call__(self, key: PRNGKey, tx: Array, tx_power: Optional[Array] = None) -> tuple[Scalar, Scalar]:
        if tx_power is None:
            tx_power = jnp.zeros(self.pos.shape[0])

        thr = self.data_rate_fn(key, tx, tx_power=self.tx_power - self.tx_power_delta * tx_power)
        reward = thr / DATA_RATES[self.mcs]
        return thr, reward

    def plot(self, filename: str = None) -> None:
        super().plot(self.pos, filename, self.walls_pos)

    def is_cca_single_tx(self) -> bool:
        return super().is_cca_single_tx(self.pos, self.tx_power, self.walls)


def simple_scenario_1(
        d: Scalar = 40.,
        mcs: int = DEFAULT_MCS,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 1     AP A     STA 2     STA 3     AP B     STA 4
    """

    pos = jnp.array([
        [0 * d, 0.],  # STA 1
        [1 * d, 0.],  # AP A
        [2 * d, 0.],  # STA 2
        [3 * d, 0.],  # STA 3
        [4 * d, 0.],  # AP B
        [5 * d, 0.]   # STA 4
    ])

    associations = {
        1: [0, 2],
        4: [3, 5]
    }

    return StaticScenario(pos, mcs, tx_power, sigma, associations)


def simple_scenario_2(
        d_ap: Scalar = 50.,
        d_sta: Scalar = 1.,
        mcs: int = DEFAULT_MCS,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 16   STA 15                  STA 12   STA 11

         AP D                             AP C

    STA 13   STA 14                  STA 9    STA 10



    STA 4    STA 3                   STA 8    STA 7

         AP A                             AP B

    STA 1    STA 2                   STA 5    STA 6
    """

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    return StaticScenario(pos, mcs, tx_power, sigma, associations)


def simple_scenario_3(
        d: Scalar = 30.,
        mcs: int = DEFAULT_MCS,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
              STA 1


    STA 2     AP A               STA 3               AP B     STA 4
    """

    pos = jnp.array([
        [d, 0],      # AP A
        [d, d],      # STA 1
        [0, 0],      # STA 2
        [5 * d, 0],  # AP B
        [3 * d, 0],  # STA 3
        [6 * d, 0]   # STA 4
    ])

    associations = {
        0: [1, 2],
        3: [4, 5]
    }

    return StaticScenario(pos, mcs, tx_power, sigma, associations)


def simple_scenario_4(
        d_ap: Scalar = 10.,
        d_sta: Scalar = 1.,
        mcs: int = DEFAULT_MCS,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 16   STA 15         |        STA 12   STA 11
                            |
         AP D               |              AP C
                            |
    STA 13   STA 14         |         STA 9    STA 10
                            |
    ------------------------+------------------------
                            |
    STA 4    STA 3          |         STA 8    STA 7
                            |
         AP A               |              AP B
                            |
    STA 1    STA 2          |         STA 5    STA 6
    """

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    aps = associations.keys()

    # Setup walls in between each BSS
    walls = jnp.zeros((20, 20))
    walls = walls.at[4:, 4:].set(True)
    for i in range(20):
        for j in range(20):

            # If both are APs
            if i in aps and j in aps:
                walls = walls.at[i, j].set(i != j)

            # If i is an AP
            elif i in aps:
                for ap_j in set(aps) - {i}:
                    for sta in associations[ap_j]:
                        walls = walls.at[i, sta].set(True)

            # If j is an AP
            elif j in aps:
                for ap_i in set(aps) - {j}:
                    for sta in associations[ap_i]:
                        walls = walls.at[sta, j].set(True)

            # If both are STAs
            else:
                for ap in aps:
                    if i in associations[ap] and j in associations[ap]:
                        walls = walls.at[i, j].set(False)

    # Walls positions
    walls_pos = jnp.array([
        [-d_ap / 2, d_ap / 2, d_ap + d_ap / 2, d_ap / 2],
        [d_ap / 2, -d_ap / 2, d_ap / 2, d_ap + d_ap / 2],
    ])

    return StaticScenario(pos, mcs, tx_power, sigma, associations, walls, walls_pos)


def simple_scenario_5(
        d_ap: Scalar = 25.,
        d_sta: Scalar = 2.,
        mcs: int = DEFAULT_MCS,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    """
    STA 16   STA 15         |        STA 12   STA 11
                            |
         AP D               |              AP C
                            |
    STA 13   STA 14         |         STA 9    STA 10
                            |
    ------------------------+------------------------

    STA 4    STA 3                    STA 8    STA 7

         AP A                              AP B

    STA 1    STA 2                    STA 5    STA 6
    """

    ap_pos = [
        [0 * d_ap, 0 * d_ap],  # AP A
        [1 * d_ap, 0 * d_ap],  # AP B
        [1 * d_ap, 1 * d_ap],  # AP C
        [0 * d_ap, 1 * d_ap],  # AP D
    ]

    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)

    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    associations = {
        0: [4, 5, 6, 7],
        1: [8, 9, 10, 11],
        2: [12, 13, 14, 15],
        3: [16, 17, 18, 19]
    }

    aps = associations.keys()

    # Setup walls in between each BSS
    walls = jnp.zeros((20, 20))
    walls = walls.at[4:, 4:].set(True)
    for i in range(20):
        for j in range(20):

            # If both are APs
            if i in aps and j in aps:
                walls = walls.at[i, j].set(i != j)

            # If i is an AP
            elif i in aps:
                for ap_j in set(aps) - {i}:
                    for sta in associations[ap_j]:
                        walls = walls.at[i, sta].set(True)

            # If j is an AP
            elif j in aps:
                for ap_i in set(aps) - {j}:
                    for sta in associations[ap_i]:
                        walls = walls.at[sta, j].set(True)

            # If both are STAs
            else:
                for ap in aps:
                    if i in associations[ap] and j in associations[ap]:
                        walls = walls.at[i, j].set(False)

    # - Remove wall between AP A and AP B
    walls = walls.at[:2, :2].set(False)
    walls = walls.at[1, 4:8].set(False)
    walls = walls.at[4:8, 1].set(False)
    walls = walls.at[0, 8:12].set(False)
    walls = walls.at[8:12, 0].set(False)
    walls = walls.at[4:12, 4:12].set(False)

    # Walls positions
    walls_pos = jnp.array([
        [-d_ap / 2, d_ap / 2, d_ap + d_ap / 2, d_ap / 2],
        [d_ap / 2, d_ap / 2, d_ap / 2, d_ap + d_ap / 2],
    ])

    return StaticScenario(pos, mcs, tx_power, sigma, associations, walls, walls_pos)
