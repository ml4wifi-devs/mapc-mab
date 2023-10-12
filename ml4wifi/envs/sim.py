from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, Scalar


# LogDistance channel model
# https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
# https://www.nsnam.org/docs/models/html/propagation.html#logdistancepropagationlossmodel
DEFAULT_TX_POWER = 16.0206
DEFAULT_NOISE = -93.97
REFERENCE_LOSS = 46.6777
EXPONENT = 3.0

# Data rates for IEEE 802.11ax standard, 20 MHz channel width, 1 spatial stream, and 3200 ns GI
DATA_RATES = jnp.array([7.3, 14.6, 21.9, 29.3, 43.9, 58.5, 65.8, 73.1, 87.8, 97.5, 109.7, 121.9])

# Based on ns-3 simulations with LogDistance channel model
MIN_SNRS = jnp.array([
    10.613624240405125, 10.647249582547907, 10.660723984151614, 10.682584060100158,
    11.151267538857537, 15.413200906170632, 16.735812667249125, 18.091175930406580,
    21.806290592040960, 23.331824973610920, 29.788906076547470, 31.750234694079595
])


def network_throughput(key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array, noise: Scalar) -> Scalar:
    """
    Calculates the approximate network throughput based on the nodes' positions, MCS, and tx power.
    Channel is modeled using log-distance path loss model with additive white Gaussian noise.
    Network throughput is calculated as the sum of expected data rates of all transmitting nodes.
    Expected data rate is calculated as the product of data rate and success probability which depends on the SINR.
    SINR is calculated as the difference between the SNR of the transmitting node and the sum of SNRs of all
    interfering nodes.

    Parameters
    ----------
    key: PRNGKey
        JAX random number generator key.
    tx: Array
        Two dimensional array of booleans indicating whether a node is transmitting to another node.
        If node i is transmitting to node j, then tx[i, j] = 1, otherwise tx[i, j] = 0.
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    mcs: Array
        Modulation and coding scheme of the nodes. Each entry corresponds to a node.
    tx_power: Array
        Transmission power of the nodes. Each entry corresponds to a node.
    noise: Scalar
        Standard deviation of the additive white Gaussian noise.

    Returns
    -------
    Scalar
        Approximated network throughput in Mb/s.
    """

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))

    path_loss = REFERENCE_LOSS + 10 * EXPONENT * jnp.log10(distance)
    snr = tx_power - path_loss - DEFAULT_NOISE
    snr = jnp.where(jnp.isinf(snr), 0., snr)

    tx_nodes = tx.any(axis=-1)[..., None]
    snr_plus_interference = jnp.where(tx_nodes, snr, 0.).sum(axis=0)
    sinr = 2 * snr - snr_plus_interference
    sinr = sinr + noise * jax.random.normal(key, shape=path_loss.shape)
    sinr = (sinr * tx).sum(axis=-1)

    success_probability = jax.scipy.stats.norm.cdf(sinr, loc=MIN_SNRS[mcs], scale=2.)
    expected_data_rate = DATA_RATES[mcs] * success_probability

    return expected_data_rate.sum()


def init_static_network(pos: Array, mcs: Array, tx_power: Array, noise: Scalar) -> Callable:
    """
    Returns a function that calculates the approximate network throughput initialized with the given parameters.

    Parameters
    ----------
    pos: Array
        Two dimensional array of node positions. Each row corresponds to X and Y coordinates of a node.
    mcs: Array
        Modulation and coding scheme of the nodes. Each entry corresponds to a node.
    tx_power: Array
        Transmission power of the nodes. Each entry corresponds to a node.
    noise: Scalar
        Standard deviation of the additive white Gaussian noise.

    Returns
    -------
    Callable
        Function that calculates the network throughput.
    """

    return jax.jit(partial(network_throughput, pos=pos, mcs=mcs, tx_power=tx_power, noise=noise))
