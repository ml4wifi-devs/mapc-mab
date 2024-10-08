# MAPC-MAB Repository

This repository contains the implementation of a Multi-Armed Bandit (MAB) algorithm for Multi-Access Point Coordination (MAPC). The MAB algorithm aims to solve the scheduling problem in coordinated spatial reuse (C-SR) by suggesting valid and fair AP-station pairs for simultanous transmissions. Its detailed operation and performance analysis can be found in:

- Maksymilian Wojnar, Wojciech Ciezobka, Katarzyna Kosek-Szott, Krzysztof Rusek, Szymon Szott, David Nunez, and Boris Bellalta. "IEEE 802.11bn Multi-AP Coordinated Spatial Reuse with Hierarchical Multi-Armed Bandits", $JOURNAL_NAME_TODO, 2024. [[TODO_PREPRINT_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main), [TODO_PUBLICATION_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main)]

## Concept

Our proposal is described in detail in the paper referenced above. Here, we provide some additional information which will hopefully facilitate understanding.

### Graphical Algorithm

The figure below provides a graphical representation of our hierarchical MAB algorithm. At first, the sharing AP is selected. Then, one of its associated stations is chosen for downlink C-SR transmission. After that, the 1<sup>st</sup> level agent is selected based on the <em>context</em> (i.e., the sharing AP and its selected station) and the set of the sharing APs is sampled. Additionally, the 2<sup>nd</sup> level agent is selected based on the <em>context</em> and an associated station is sampled for each of the shared APs. Finally, the chosen set of AP-station pairs is tested by running downlink C-SR transmission simulations and obtaining the reward. The resulting reward is uploaded to the 1<sup>st</sup> and 2<sup>nd</sup> level agents. The whole process is then repeated to find the best C-SR AP-station pairs.

<img src="https://github.com/user-attachments/assets/ec166c78-cf1b-4a77-9b31-45732ec5d6ee" width="100%" height="100%">


### MAB Algorithms

We assume that all MAB agents (on both levels in the hierarchy) are of the same type, i.e., use the same MAB algorithm.
The table below provides a brief description of the algorithms that we use as well as their advantages and drawbacks.

| Algorithm   | Description                                                                     | Pros & cons                                                                     |
|-------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| ε-greedy    | Selects best option with probability 1-ε otherwise explore                      | It quickly finds the best option but is not suitable for a dynamic environment  |
| UCB         | Choose best option according to estimated average and confidence                | Asymptotically optimal and deterministic but at the expense of convergence time |
| Softmax     | Assess options using stochastic gradient ascent                                 | Slowly convergent but the gradient can find the best option                     |
| Thompson Sampling (normal) | Directly models the reward distribution (assuming normally-distributed rewards) | Quickly converges and models uncertainty, but is inflexible in the long run     |

## Installation

All the requirements can be installed automatically with the following commands:

```bash
cd mapc-mab   # Project root where the .toml file is located
pip install .
```

A complete list of dependencies is also provided in [pyproject.toml](https://github.com/ml4wifi-devs/mapc-mab/blob/main/pyproject.toml).

### MAPC Simulator

`mapc-mab` uses a dedicated network simulator for the evaluation of the agents. [The simulator](https://github.com/ml4wifi-devs/mapc-sim) is published as a [pip package](https://pypi.org/project/mapc-sim/) and is installed automatically with `mapc-mab`. 

However, you can also install the simulator from the source code to allow for changes without the need to reinstall the package. To do so, follow these steps:

```bash
git clone git@github.com:ml4wifi-devs/mapc-sim.git
cd $PATH_TO_MAPC_MAB
pip install -e $PATH_TO_MAPC_SIM
```

## Usage

The standard workflow for running the simulation is as follows:

1.  Define a simulation configuration in a JSON file (e.g.,  `default_config.json`):

```json
{
  "n_reps": 10,
  "seed": 42,
  "scenarios": [
    {
      "scenario": "simple_scenario_5",
      "name": "scenario_10m",
      "n_steps": 600,
      "params": {
        "d_ap": 10.0,
        "d_sta": 2.0,
        "mcs": 11
      }
    }
  ],
  "agents": [
    {
      "name": "EGreedy",
      "params": {
        "e": 9.0e-05,
        "optimistic_start": 556.0
      }
    }
  ]
}
```
    
2.  Run the simulation with the following command:

```bash
python mapc_mab/envs/run.py -c $PATH_TO_CONFIG_FILE
```

3.  This will generate an `all_results.json` file in the project root.
    
4.  Plot the experiment results with the following command (`-f` is the option to provide the results file path):

```bash
python mapc_mab/plots/article/data_rate_plot_combined.py -f all_results.json
```

## Repository Structure

The repository is organized into three main directories:

-   **envs:** This directory contains the environment part, namely the simulation scenarios, JSON configuration files and the running script.

    -   `scenarios.py` contains an abstract class for simulation scenario definition and `static_scenarios.py` contains example scenarios with different network topologies (including non-line-of-sight topologies)
    -   `default_config.json`: Example configuration file
    -   `run.py`: Starts the simulation with the configuration given in a JSON file.
    
-   **agents:** This directory contains the implementation of a MAB algorithm.
    
    -   `mapc_agent_factory.py`: Factory pattern for creating a hierarchical MAPC agent with freedom of choosing the base bandit.
    -   `mapc_agent.py`: Base class for MAPC agents responsible for the selection of AP-stations pairs.
    -   `tuning`: Directory with code for optimizing agent hyperparameters.
    
-   **plots:** This directory contains scripts for plotting simulation results.

    - `article`: Scripts for generating figures used in the published article.
    -   `article/data_rate_plot_combined.py`: It is the most important plotting feature that plots the combined results of an experiment.


## Additional Notes

-   The code and the simulator are written in JAX, which is an autodiff library similar to PyTorch or TensorFlow. This means that it may require additional dependencies or configurations to run properly, especially with GPU acceleration. For more information on JAX, please refer to the official [JAX repository](https://jax.readthedocs.io/en/latest/).

# How to reference MAPC-MAB? #TODO

```
@INPROCEEDINGS{TODO,
  author={Wojnar, Maksymilian and Ciezobka, Wojciech and Kosek-Szott, Katarzyna and Rusek, Krzysztof and Szott, Szymon and Nunez, David and Bellalta, Boris},
  booktitle={TODO}, 
  title={{IEEE 802.11bn Multi-AP Coordinated Spatial Reuse with Hierarchical Multi-Armed Bandits}}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={TODO}
}
```

We hope this repository is useful for your research!

