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
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    """

    def __init__(self, associations: Dict, walls: Optional[Array] = None, walls_pos: Optional[Array] = None) -> None:
        n_nodes = len(associations) + sum([len(n) for n in associations.values()])
        self.associations = associations
        self.walls = walls if walls is not None else jnp.zeros((n_nodes, n_nodes))
        self.walls_pos = walls_pos

    @abstractmethod
    def __call__(self, *args, **kwargs) -> Scalar:
        pass

    def get_associations(self) -> Dict:
        return self.associations

    def plot(self, pos: Array, filename: str = None) -> None:
        colors = plt.colormaps['viridis'](np.linspace(0, 1, len(self.associations)))
        ap_labels = string.ascii_uppercase

        _, ax = plt.subplots()

        for i, (ap, stations) in enumerate(self.associations.items()):
            ax.scatter(pos[ap, 0], pos[ap, 1], marker='x', color=colors[i])
            ax.scatter(pos[stations, 0], pos[stations, 1], marker='.', color=colors[i])
            ax.annotate(f'AP {ap_labels[i]}', (pos[ap, 0], pos[ap, 1] + 2), color=colors[i], va='bottom', ha='center')

            radius = np.max(np.sqrt(np.sum((pos[stations, :] - pos[ap, :]) ** 2, axis=-1)))
            circle = plt.Circle((pos[ap, 0], pos[ap, 1]), radius * 1.2, fill=False, linewidth=0.5)
            ax.add_patch(circle)

        # Plot walls
        if self.walls_pos is not None:
            for wall in self.walls_pos:
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
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
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
        super().__init__(associations, walls, walls_pos)

        self.pos = pos
        mcs = jnp.ones(pos.shape[0], dtype=jnp.int32) * mcs
        tx_power = jnp.ones(pos.shape[0]) * tx_power

        self.thr_fn = jax.jit(partial(network_throughput, pos=pos, mcs=mcs, tx_power=tx_power, sigma=sigma, walls=self.walls))

    def __call__(self, key: PRNGKey, tx: Array) -> Scalar:
        return self.thr_fn(key, tx)

    def plot(self, filename: str = None) -> None:
        super().plot(self.pos, filename)


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
    walls: Optional[Array]
        Adjacency matrix of walls. Each entry corresponds to a node.
    walls_pos: Optional[Array]
        Two dimensional array of wall positions. Each row corresponds to X and Y coordinates of a wall.
    """

    def __init__(
            self,
            sigma: Scalar,
            associations: Dict,
            walls: Optional[Array] = None,
            walls_pos: Optional[Array] = None
    ) -> None:
        super().__init__(associations, walls, walls_pos)
        self.thr_fn = jax.jit(partial(network_throughput, sigma=sigma))

    def __call__(self, key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array) -> Scalar:
        return self.thr_fn(key, tx, pos, mcs, tx_power, self.walls)

    def plot(self, pos: Array, filename: str = None) -> None:
        super().plot(pos, filename)
