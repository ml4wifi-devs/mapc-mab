# MAPC-MAB Repository

This repository contains the implementation of a Multi-Armed Bandit (MAB) algorithm for Multi-Access Point Coordination (MAPC). The MAB algorithm aims to solve the scheduling problem in coordinated spatial reuse (C-SR) by suggesting valid and fair AP-station pairs for simultanous transmissions. Its detailed operation and performance analysis can be found in:

- Maksymilian Wojnar, Wojciech Ciężobka, Artur Tomaszewski, Piotr Chołda, Krzysztof Rusek, Katarzyna Kosek-Szott, Jetmir Haxhibeqiri, Jeroen Hoebeke, Boris Bellalta, Anatolij Zubow, Falko Dressler, and Szymon Szott. "Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks", 2025.

## Repository Structure

The repository has the following structure:

- `mapc_mab/`: Main package containing the MAPC MAB agent. 
  - `mapc_agent.py`: Base class for the MAPC agent.
  - `flat_mapc_agent.py`: The classic MAB MAPC agent
  - `hierarchical_mapc_agent.py`: The hierarchical MAB MAPC agent.
  - `mapc_agent_factory.py`: The factory which creates the MAPC agent with the given agent type and parameters.
- `test/`: Unit tests and benchmarking scripts.

## Installation

To install the mapc-mab, you need to first clone the repository and then install it using `pip`.

```bash
# Clone the repository
git clone https://github.com/ml4wifi-devs/mapc-optimal-research.git

# Install the package
pip install -e ./mapc-mab
```

The `-e` flag installs the package in editable mode, so you can change the code and test it without reinstalling the package.

## Usage
We provide a simple example on how to use the mapc-mab agent in the C-SR experiments. We assume a user already have a network environment to test the agents.

```python
from reinforced_lib.agents.mab import UCB

from mapc_mab import MapcAgentFactory


# Define a network simulator:
# - it must specify the associations dictionary where the key, value pair represents the AP, list of STAs associated with that AP, i.e.:
"""
associations = {
    0: [4, 5, 6, 7],
    1: [8, 9, 10, 11],
    2: [12, 13, 14, 15],
    3: [16, 17, 18, 19]
}
"""
# - it must output a reward metric that evaluates the proposed TX configuration.
network_simulator = "YOUR CODE"

# Instantiate the agent factory
agent_factory = MapcAgentFactory(
    network_simulator.associations,

    # The agent type is shared across levels,
    agent_type=UCB,

    # but you can set the hyperparameters for each level separately.
    agent_params_lvl1={'c': 500.0},
    agent_params_lvl2={'c': 500.0},
    agent_params_lvl3={'c': 500.0},

    # Hierarchical mode exploits the inductive bias and uses the original and efficient hierarchical MAB approach (more details in the paper).
    hierarchical=True
)

# Use the agent factory to create an MAB agent
agent = agent_factory.create_mapc_agent()

# The classic RL loop of acting and observing the reward
reward = 0
for _ in range(200):
    tx, tx_power = agent.sample(reward)
    reward = network_simulator.get_reward(tx, tx_power)
```

## Additional Notes

-   The MAB agents are written in JAX, which is an autodiff library similar to PyTorch or TensorFlow. This means that it may require additional dependencies or configurations to run properly, especially with GPU acceleration. For more information on JAX, please refer to the official [JAX repository](https://jax.readthedocs.io/en/latest/).

## How to reference MAPC-MAB?

```
@article{wojnar2025coordinated,
  author={Wojnar, Maksymilian and Ciężobka, Wojciech and Tomaszewski, Artur and Chołda, Piotr and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Haxhibeqiri, Jetmir and Hoebeke, Jeroen and Bellalta, Boris and Zubow, Anatolij and Dressler, Falko and Szott, Szymon},
  title={{Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks}}, 
  year={2025},
}
```
