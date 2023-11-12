import string
from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Optional

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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

    def __init__(self, associations: Dict) -> None:
        self.associations = associations

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Scalar:
        pass

    def get_associations(self) -> Dict:
        return self.associations

    def plot(self, pos: Array, associations: Dict, filename: str = None, wall_pos: Optional[Array] = None) -> None:
        colors = plt.colormaps['viridis'](np.linspace(0, 1, len(associations)))
        ap_labels = string.ascii_uppercase

        _, ax = plt.subplots()

        for i, (ap, stations) in enumerate(associations.items()):
            ax.scatter(pos[ap, 0], pos[ap, 1], marker='x', color=colors[i])
            ax.scatter(pos[stations, 0], pos[stations, 1], marker='.', color=colors[i])
            ax.annotate(f'AP {ap_labels[i]}', (pos[ap, 0], pos[ap, 1] + 2), color=colors[i], va='bottom', ha='center')

            radius = np.max(np.sqrt(np.sum((pos[stations, :] - pos[ap, :]) ** 2, axis=-1)))
            circle = plt.Circle((pos[ap, 0], pos[ap, 1]), radius * 1.2, fill=False, linewidth=0.5)
            ax.add_patch(circle)
        
        # Plot walls
        if wall_pos is not None:
            for wall in wall_pos:
                ax.plot([wall[0], wall[2]], [wall[1], wall[3]], color='black', linewidth=1)


        ax.set_axisbelow(True)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
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

    def __init__(
            self,
            pos: Array,
            mcs: int,
            tx_power: Scalar,
            sigma: Scalar,
            associations: Dict,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None
        ) -> None:
        super().__init__(associations)

        self.pos = pos
        self.walls = walls if walls is not None else jnp.zeros((pos.shape[0], pos.shape[0]))
        self.walls_pos = walls_pos
        mcs = jnp.ones(pos.shape[0], dtype=jnp.int32) * mcs
        tx_power = jnp.ones(pos.shape[0]) * tx_power

        self.thr_fn = jax.jit(partial(network_throughput, pos=pos, mcs=mcs, tx_power=tx_power, sigma=sigma, walls=self.walls))

    def __call__(self, key: PRNGKey, tx: Array) -> Scalar:
        return self.thr_fn(key, tx)

    def plot(self, filename: str = None) -> None:
        super().plot(self.pos, self.associations, filename, self.walls_pos)


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

    def __init__(self, sigma: Scalar, associations: Dict) -> None:
        super().__init__(associations)
        self.thr_fn = jax.jit(partial(network_throughput, sigma=sigma))

    def __call__(self, key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array, walls: Optional[Array]) -> Scalar:
        walls = walls if walls is not None else jnp.zeros((pos.shape[0], pos.shape[0]))
        return self.thr_fn(key, tx, pos, mcs, tx_power, walls)

    def plot(self, pos: Array, filename: str = None) -> None:
        super().plot(pos, self.associations, filename)
