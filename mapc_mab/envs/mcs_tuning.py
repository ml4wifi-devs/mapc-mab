from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array
from mapc_sim.constants import NOISE_FLOOR, MEAN_SNRS, DATA_RATES, REFERENCE_DISTANCE
from mapc_sim.utils import logsumexp_db, tgax_path_loss
from tensorflow_probability.substrates import jax as tfp
from tqdm import tqdm

from mapc_mab.agents.utils import iter_tx
from mapc_mab.envs.static_scenarios import *
from mapc_mab.plots.config import set_style

tfd = tfp.distributions


def ideal_mcs(tx: Array, pos: Array, tx_power: Array, walls: Array) -> tuple:
    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

    signal_power = tx_power[:, None] - tgax_path_loss(distance, walls)

    interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=1, keepdims=True) * (1 - tx)
    a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
    b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
    interference = jax.vmap(logsumexp_db, in_axes=(1, 1))(a, b)

    sinr = ((signal_power - interference) * tx).sum(axis=1)
    data_rate = tfd.Normal(loc=MEAN_SNRS[:, None], scale=2.).cdf(sinr) * DATA_RATES[:, None]

    return jnp.argmax(data_rate, axis=0), jnp.max(data_rate, axis=0).sum()


def ideal_tx(scenario: StaticScenario) -> tuple:
    best_rate, best_mcs, best_tx = 0., 0, None
    ideal_mcs_fn = jax.jit(partial(ideal_mcs, pos=scenario.pos, tx_power=scenario.tx_power, walls=scenario.walls))

    for tx in iter_tx(scenario.get_associations()):
        tx = tx.todense()
        mcs, rate = ideal_mcs_fn(tx)

        if rate > best_rate:
            best_rate, best_mcs, best_tx = rate, mcs, tx

    return best_rate, best_mcs, best_tx


if __name__ == '__main__':
    set_style()

    colors = plt.cm.viridis([0.4, 0.75])
    distances = jnp.logspace(0, 2, 100)
    rates, tx_num, mcses = [], [], []

    for d in tqdm(distances):
        scenario = simple_scenario_5(d)
        rate, mcs, tx = ideal_tx(scenario)

        rates.append(rate)
        tx_num.append(tx.sum())
        mcses.append(list(map(lambda i: mcs[i] if tx[i].sum() > 0 else jnp.nan, scenario.get_associations().keys())))

    fig, ax1 = plt.subplots()
    ax1.plot(distances, rates, label='Best possible data rate', color=colors[0])
    ax1.set_xscale('log')
    ax1.set_xlim(distances.min(), distances.max())
    ax1.set_ylim(0, 600)
    ax1.set_ylabel('Effective data rate [Mb/s]')
    ax1.set_xlabel('Distance [m]')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(distances, tx_num, label='Num of TXs', color=colors[1])
    ax2.set_ylim(0, 5)
    ax2.set_ylabel('Number of TXs')
    ax2.legend(loc='lower right')

    plt.grid(which='major')
    plt.tight_layout()
    plt.savefig('scenario_5.pdf', bbox_inches='tight')
    plt.show()

    mcses = jnp.array(mcses)
    colors = plt.cm.viridis(jnp.linspace(0., 0.8, mcses.shape[1]))

    for i in range(mcses.shape[1]):
        plt.plot(distances, mcses[:, i], label=f'AP{i}', color=colors[i])

    plt.xscale('log')
    plt.xlim(distances.min(), distances.max())
    plt.xlabel('Distance [m]')
    plt.ylim(0, 12)
    plt.ylabel('MCS')
    plt.yticks(range(0, 13, 2))
    plt.legend()
    plt.grid(which='major')
    plt.tight_layout()
    plt.savefig('scenario_5_mcs.pdf', bbox_inches='tight')
    plt.show()

