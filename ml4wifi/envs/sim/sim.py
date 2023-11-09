import jax
import jax.numpy as jnp
from chex import Array, PRNGKey, Scalar


# LogDistance channel model
# https://www.nsnam.org/docs/models/html/propagation.html#logdistancepropagationlossmodel
DEFAULT_TX_POWER = 16.0206  # (40 mW) https://www.nsnam.org/docs/release/3.40/doxygen/d0/d7d/wifi-phy_8cc_source.html#l00171
REFERENCE_LOSS = 46.6777    # https://www.nsnam.org/docs/release/3.40/doxygen/d5/d74/propagation-loss-model_8cc_source.html#l00493
REFERENCE_DISTANCE = 1.0    # https://www.nsnam.org/docs/release/3.40/doxygen/d5/d74/propagation-loss-model_8cc_source.html#l00488
EXPONENT = 2.0              # https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=908165
NOISE_FLOOR = -93.97        # https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
NOISE_FLOOR_LIN = jnp.power(10, NOISE_FLOOR / 10)
DEFAULT_SIGMA = 2.          # https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=908165

# Data rates for IEEE 802.11ax standard, 20 MHz channel width, 1 spatial stream, and 800 ns GI
DATA_RATES = jnp.array([8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.2])

# Based on ns-3 simulations with LogDistance channel model
MEAN_SNRS = jnp.array([
    10.613624240405125, 10.647249582547907, 10.660723984151614, 10.682584060100158,
    11.151267538857537, 15.413200906170632, 16.735812667249125, 18.091175930406580,
    21.806290592040960, 23.331824973610920, 29.788906076547470, 31.750234694079595
])


def network_throughput(key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array, sigma: Scalar) -> Scalar:
    """
    Calculates the approximate network throughput based on the nodes' positions, MCS, and tx power.
    Channel is modeled using log-distance path loss model with additive white Gaussian noise. Network
    throughput is calculated as the sum of data rates of all successful transmissions. Success of
    a transmission is  Bernoulli random variable with success probability depending on the SINR. SINR is
    calculated as the difference between the signal power and the interference level which is calculated
    as the sum of the signal powers of all interfering nodes and the noise floor. **Attention:** This
    simulation does not support multiple simultaneous transmissions to the same node.

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
    sigma: Scalar
        Standard deviation of the additive white Gaussian noise.

    Returns
    -------
    Scalar
        Approximated network throughput in Mb/s.
    """

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

    path_loss = REFERENCE_LOSS + 10 * EXPONENT * jnp.log10(distance)
    signal_power = tx_power - path_loss
    signal_power = jnp.where(jnp.isinf(signal_power), 0., signal_power)

    interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=-1, keepdims=True) * (1 - tx)
    interference_lin = jnp.power(10, signal_power / 10)
    interference_lin = (interference_matrix * interference_lin).sum(axis=0)
    interference = 10 * jnp.log10(interference_lin + NOISE_FLOOR_LIN)

    sinr = signal_power - interference
    sinr = sinr + sigma * jax.random.normal(key, shape=path_loss.shape)
    sinr = (sinr * tx).sum(axis=0)

    success_probability = jax.scipy.stats.norm.cdf(sinr, loc=MEAN_SNRS[mcs], scale=2.)
    frame_transmitted = jax.random.bernoulli(key, success_probability * (sinr > 0))
    expected_data_rate = DATA_RATES[mcs] * frame_transmitted

    return expected_data_rate.sum()
