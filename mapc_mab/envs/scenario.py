import string
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from chex import Array, Scalar
from jax import numpy as jnp
from mapc_sim.utils import tgax_path_loss
from matplotlib import pyplot as plt

from mapc_mab.plots.config import get_cmap


CCA_THRESHOLD = -82.0  # IEEE Std 802.11-2020 (Revision of IEEE Std 802.11-2016), 17.3.10.6: CCA requirements


class Scenario(ABC):
    """
    Base class for scenarios.

    Parameters
    ----------
    associations: dict
        Dictionary of associations between access points and stations.
    """

    def __init__(self, associations: dict) -> None:
        self.associations = associations

    @abstractmethod
    def __call__(self, *args, **kwargs) -> tuple[Scalar, Scalar]:
        pass

    def reset(self) -> None:
        pass

    def plot(self, pos: Array, filename: str = None, walls_pos: Optional[Array] = None) -> None:
        """
        Plot the current state of the scenario.

        Parameters
        ----------
        pos : Array
            Two dimensional array of node positions.
        filename : str
            Name of the file to save the plot. If None, the plot is shown.
        walls_pos : Optional[Array]
            Two dimensional array of wall positions.
        """

        colors = get_cmap(len(self.associations))
        ap_labels = string.ascii_uppercase

        _, ax = plt.subplots()

        for i, (ap, stations) in enumerate(self.associations.items()):
            ax.scatter(pos[ap, 0], pos[ap, 1], marker='x', color=colors[i])
            ax.scatter(pos[stations, 0], pos[stations, 1], marker='.', color=colors[i])
            ax.annotate(f'AP {ap_labels[i]}', (pos[ap, 0], pos[ap, 1] + 2), color=colors[i], va='bottom', ha='center')

            radius = np.max(np.sqrt(np.sum((pos[stations, :] - pos[ap, :]) ** 2, axis=-1)))
            circle = plt.Circle((pos[ap, 0], pos[ap, 1]), radius * 1.2, fill=False, linewidth=0.5)
            ax.add_patch(circle)

        if walls_pos is not None:
            for wall in walls_pos:
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

    def is_cca_single_tx(self, pos: Array, tx_power: Array, walls: Array) -> bool:
        """
        Check if the scenario is a CSMA single transmission scenario, i.e., if there is only one transmission
        possible at a time due to the CCA threshold. **Note**: This function assumes that the scenario
        contains downlink transmissions only.

        Parameters
        ----------
        pos : Array
            Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
        tx_power : Array
            Transmission power of the nodes. Each entry corresponds to a node.
        walls: Optional[Array]
            Adjacency matrix of walls. Each entry indicates if there is a wall between two nodes.

        Returns
        -------
        bool
            True if the scenario is a CSMA single transmission scenario, False otherwise.
        """

        ap_ids = np.array(list(self.associations.keys()))

        ap_pos = pos[ap_ids]
        ap_tx_power = tx_power[ap_ids]
        ap_walls = walls[ap_ids][:, ap_ids]

        distance = np.sqrt(np.sum((ap_pos[:, None, :] - ap_pos[None, ...]) ** 2, axis=-1))
        signal_power = ap_tx_power - tgax_path_loss(distance, ap_walls)
        signal_power = np.where(np.isnan(signal_power), np.inf, signal_power)

        return np.all(signal_power > CCA_THRESHOLD)

    def tx_matrix_to_action(self, tx_matrix: Array) -> dict:
        """
        Convert a transmission matrix to a list of transmissions. Assumes downlink.

        Parameters
        ----------
        tx_matrix: Array
            Transmission matrix. Each entry corresponds to a node.

        Returns
        -------
        dict
            A dictionary where the keys are APs, and the values are the stations associated.
        """

        aps = list(self.associations.keys())
        action = {}

        for ap in aps:
            assert jnp.sum(tx_matrix[ap, :]) <= 1, 'Multiple transmissions at AP'
            for sta in self.associations[ap]:
                if tx_matrix[ap, sta]:
                    action[ap] = sta

        return action
