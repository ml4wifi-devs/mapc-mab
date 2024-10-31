# MAPC-MAB Repository

This repository contains the implementation of a Multi-Armed Bandit (MAB) algorithm for Multi-Access Point Coordination (MAPC). The MAB algorithm aims to solve the scheduling problem in coordinated spatial reuse (C-SR) by suggesting valid and fair AP-station pairs for simultanous transmissions. Its detailed operation and performance analysis can be found in:

- Maksymilian Wojnar, Wojciech Ciężobka, Artur Tomaszewski, Piotr Chołda, Krzysztof Rusek, Katarzyna Kosek-Szott, Jetmir Haxhibeqiri, Jeroen Hoebeke, Boris Bellalta, Anatolij Zubow, Falko Dressler, and Szymon Szott. "C-SR Scheduling with Machine Learning in IEEE 802.11 MAPC Networks", $JOURNAL_NAME_TODO, 2024. [[TODO_PREPRINT_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main), [TODO_PUBLICATION_INSERT](https://github.com/ml4wifi-devs/mapc-mab/tree/main)]

## Installation

All the requirements can be installed automatically with the following commands:

```bash
cd mapc-mab   # Project root where the .toml file is located
pip install .
```

A complete list of dependencies is also provided in [pyproject.toml](https://github.com/ml4wifi-devs/mapc-mab/blob/main/pyproject.toml).

## Usage

TODO

## Additional Notes

-   The MAB agents are written in JAX, which is an autodiff library similar to PyTorch or TensorFlow. This means that it may require additional dependencies or configurations to run properly, especially with GPU acceleration. For more information on JAX, please refer to the official [JAX repository](https://jax.readthedocs.io/en/latest/).

## How to reference MAPC-MAB? #TODO

```
@INPROCEEDINGS{TODO,
  author={Wojnar, Maksymilian and Ciężobka, Wojciech and Tomaszewski, Artur and Chołda, Piotr and Rusek, Krzysztof and Kosek-Szott, Katarzyna and Haxhibeqiri, Jetmir and Hoebeke, Jeroen and Bellalta, Boris and Zubow, Anatolij and Dressler, Falko and Szott, Szymon},
  booktitle={TODO}, 
  title={{C-SR Scheduling with Machine Learning in IEEE 802.11 MAPC Networks}}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={TODO}
}
```

We hope this repository is useful for your research!

