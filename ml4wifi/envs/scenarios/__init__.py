from abc import ABC, abstractmethod
from functools import partial
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from chex import Array, Scalar, PRNGKey

from ml4wifi.envs.sim import network_throughput


class Scenario(ABC):
    """
    Base class for scenarios.

    Parameters
    ----------
    associations: Dict
        Dictionary of associations between access points and stations.
    """

    def __init__(self, name: str, associations: Dict) -> None:
        self.name = name
        self.associations = associations

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Scalar:
        pass

    def __str__(self) -> str:
        return self.name

    def get_associations(self) -> Dict:
        return self.associations

    def plot(self, pos: Array, associations: Dict, filename: str = None) -> None:
        colors = plt.colormaps['viridis'](jnp.linspace(0, 1, len(associations)))
        _, ax = plt.subplots()

        for i, (ap, stations) in enumerate(associations.items()):
            ax.scatter(pos[ap, 0], pos[ap, 1], marker='x', color=colors[i], label=f'AP {ap}')
            ax.scatter(pos[stations, 0], pos[stations, 1], marker='.', color=colors[i])
            ax.annotate(f'AP {ap + 1}', (pos[ap, 0], pos[ap, 1]), color=colors[i], va='bottom', ha='center')

            radius = jnp.max(jnp.sqrt(jnp.sum((pos[stations, :] - pos[ap, :]) ** 2, axis=-1)))
            circle = plt.Circle((pos[ap, 0], pos[ap, 1]), radius * 1.1, fill=False, linewidth=0.5)
            ax.add_patch(circle)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Location of nodes')
        ax.grid()

        if filename:
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight')
            plt.clf()
        else:
            plt.show()


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
    associations: Dict
        Dictionary of associations between access points and stations.
    """

    def __init__(self, name: str, pos: Array, mcs: int, tx_power: Scalar, sigma: Scalar, associations: Dict) -> None:
        super().__init__(name, associations)

        self.pos = pos
        mcs = jnp.ones(pos.shape[0], dtype=jnp.int32) * mcs
        tx_power = jnp.ones(pos.shape[0]) * tx_power

        self.thr_fn = jax.jit(partial(network_throughput, pos=pos, mcs=mcs, tx_power=tx_power, sigma=sigma))

    def __call__(self, key: PRNGKey, tx: Array) -> Scalar:
        return self.thr_fn(key, tx)

    def plot(self, filename: str = None) -> None:
        super().plot(self.pos, self.associations, filename)


class DynamicScenario(Scenario):
    """
    Dynamic scenario with fixed noise standard deviation. The configuration of node positions, MCS,
    and tx power is variable.

    Parameters
    ----------
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.
    associations: Dict
        Dictionary of associations between access points and stations.
    """

    def __init__(self, name: str, sigma: Scalar, associations: Dict) -> None:
        super().__init__(name, associations)
        self.thr_fn = jax.jit(partial(network_throughput, sigma=sigma))

    def __call__(self, key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array) -> Scalar:
        return self.thr_fn(key, tx, pos, mcs, tx_power)

    def plot(self, pos: Array, filename: str = None) -> None:
        super().plot(pos, self.associations, filename)
