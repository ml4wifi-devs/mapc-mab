import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from chex import Array, PRNGKey, Scalar

tfd = tfp.distributions


# TGax channel model
# https://www.ieee802.org/11/Reports/tgax_update.htm#:~:text=TGax%20Selection%20Procedure-,11%2D14%2D0980,-TGax%20Simulation%20Scenarios
DEFAULT_TX_POWER = 16.0206  # (40 mW) https://www.nsnam.org/docs/release/3.40/doxygen/d0/d7d/wifi-phy_8cc_source.html#l00171
NOISE_FLOOR = -93.97        # https://www.nsnam.org/docs/models/html/wifi-testing.html#packet-error-rate-performance
NOISE_FLOOR_LIN = jnp.power(10, NOISE_FLOOR / 10)
DEFAULT_SIGMA = 2.          # https://tsapps.nist.gov/publication/get_pdf.cfm?pub_id=908165
BREAKING_POINT = 10.        # https://mentor.ieee.org/802.11/dcn/14/11-14-0980-16-00ax-simulation-scenarios.docx (p. 19)
CENTRAL_FREQUENCY = 5.160   # (GHz) https://en.wikipedia.org/wiki/List_of_WLAN_channels#5_GHz_(802.11a/h/n/ac/ax)
REFERENCE_DISTANCE = 1.0
WALL_LOSS = 7.

# Data rates for IEEE 802.11ax standard, 20 MHz channel width, 1 spatial stream, and 800 ns GI
DATA_RATES = jnp.array([8.6, 17.2, 25.8, 34.4, 51.6, 68.8, 77.4, 86.0, 103.2, 114.7, 129.0, 143.2])

# Tx slot duration
TAU = 5.484 * 1e-3          # (s) https://ieeexplore.ieee.org/document/8930559
FRAME_LEN = jnp.asarray(1500 * 8)

# Parameters of the success probability curves - mean of the normal distribution with standard deviation of 2
# (derived from ns-3 simulations)
MEAN_SNRS = jnp.array([
    10.613624240405125, 10.647249582547907, 10.660723984151614, 10.682584060100158,
    11.151267538857537, 15.413200906170632, 16.735812667249125, 18.091175930406580,
    21.806290592040960, 23.331824973610920, 29.788906076547470, 31.750234694079595
])


def path_loss(distance: Array, walls: Array) -> Array:
    return (40.05 + 20 * jnp.log10((jnp.minimum(distance, BREAKING_POINT) * CENTRAL_FREQUENCY) / 2.4) +
            (distance > BREAKING_POINT) * 35 * jnp.log10(distance / BREAKING_POINT) + WALL_LOSS * walls)


def _logsumexp_db(a:Array, b:Array)->Array:
    r"""
    Computes :ref:`jax.nn.logsumexp` for dB i.e. :math:`10log_10(\sum_i b_i 10^{a_i/10})`

    This function is equivalent to

    .. code-block:: python

        interference_lin = jnp.power(10, a / 10)
        interference_lin = (b * interference_lin).sum()
        interference = 10 * jnp.log10(interference_lin )


    Parameters
    ----------
    a: Array
        Parameters are the same as for :ref:`jax.nn.logsumexp`
    b: Array
        Parameters are the same as for :ref:`jax.nn.logsumexp`

    Returns
    -------
    Array
        `logsumexp` for dB
    """

    LOG10DIV10 = jnp.log(10.) / 10.
    return jax.nn.logsumexp(a=LOG10DIV10 * a, b=b) / LOG10DIV10


def network_data_rate(key: PRNGKey, tx: Array, pos: Array, mcs: Array, tx_power: Array, sigma: Scalar, walls: Array) -> Scalar:
    """
    Calculates the aggregated effective data rate based on the nodes' positions, MCS, and tx power.
    Channel is modeled using TGax channel model with additive white Gaussian noise. Effective
    data rate is calculated as the sum of data rates of all successful transmissions. Success of
    a transmission is a Binomial random variable with success probability depending on the SINR and
    number of trials equal to the number of frames in the slot. SINR is calculated as the difference
    between the signal power and the interference level. Interference level is calculated as the sum
    of the signal powers of all interfering nodes and the noise floor in the linear scale. **Attention:**
    This simulation does not support multiple simultaneous transmissions to the same node.

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
    walls: Array
        Adjacency matrix of walls. Each entry corresponds to a node.

    Returns
    -------
    Scalar
        Aggregated effective data rate in Mb/s.
    """

    normal_key, binomial_key = jax.random.split(key)

    distance = jnp.sqrt(jnp.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
    distance = jnp.clip(distance, REFERENCE_DISTANCE, None)

    signal_power = tx_power - path_loss(distance, walls)
    signal_power = jnp.where(jnp.isinf(signal_power), 0., signal_power)

    interference_matrix = jnp.ones_like(tx) * tx.sum(axis=0) * tx.sum(axis=-1, keepdims=True) * (1 - tx)
    a = jnp.concatenate([signal_power, jnp.full((1, signal_power.shape[1]), fill_value=NOISE_FLOOR)], axis=0)
    b = jnp.concatenate([interference_matrix, jnp.ones((1, interference_matrix.shape[1]))], axis=0)
    interference = jax.vmap(_logsumexp_db, in_axes=(1, 1))(a, b)

    sinr = signal_power - interference
    sinr = sinr + tfd.Normal(loc=jnp.zeros_like(signal_power), scale=sigma).sample(seed=normal_key)
    sinr = (sinr * tx).sum(axis=0)

    success_probability = tfd.Normal(loc=MEAN_SNRS[mcs], scale=2.).cdf(sinr) * (sinr > 0)
    n = jnp.round(DATA_RATES[mcs] * 1e6 * TAU / FRAME_LEN)
    frames_transmitted = tfd.Binomial(total_count=n, probs=success_probability).sample(seed=binomial_key)

    average_data_rate = FRAME_LEN * (frames_transmitted / TAU)
    return average_data_rate.sum() / float(1e6)  # (Mbps)
