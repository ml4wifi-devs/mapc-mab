import jax.numpy as jnp
import jax.random
from chex import Scalar

from ml4wifi.envs.scenarios import StaticScenario
from ml4wifi.envs.sim import DEFAULT_TX_POWER, DEFAULT_SIGMA


def simple_scenario_1(
        d: Scalar = 40.,
        mcs: int = 4,
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
        mcs: Scalar = 11,
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
        mcs: int = 4,
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


def random_scenario_1(
        seed: int,
        d_ap: Scalar = 100.,
        n_ap: int = 4,
        d_sta: Scalar = 1.,
        n_sta_per_ap: int = 4,
        mcs: int = 11,
        tx_power: Scalar = DEFAULT_TX_POWER,
        sigma: Scalar = DEFAULT_SIGMA
) -> StaticScenario:
    ap_key, key = jax.random.split(jax.random.PRNGKey(seed))
    ap_pos = jax.random.uniform(ap_key, (n_ap, 2)) * d_ap
    sta_pos = []

    for pos in ap_pos:
        sta_key, key = jax.random.split(key)
        center = jnp.repeat(pos[None, :], n_sta_per_ap, axis=0)
        stations = center + jax.random.normal(sta_key, (n_sta_per_ap, 2)) * d_sta
        sta_pos += stations.tolist()

    pos = jnp.array(ap_pos.tolist() + sta_pos)
    associations = {i: list(range(n_ap + i * n_sta_per_ap, n_ap + (i + 1) * n_sta_per_ap)) for i in range(n_ap)}

    return StaticScenario(pos, mcs, tx_power, sigma, associations)
