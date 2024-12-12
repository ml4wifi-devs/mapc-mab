# MAPC-MAB Repository

This repository contains the implementation of a Multi-Armed Bandit (MAB) algorithm for Multi-Access Point Coordination (MAPC). The MAB algorithm aims to solve the scheduling problem in coordinated spatial reuse (C-SR) by suggesting valid and fair AP-station pairs for simultanous transmissions. Its detailed operation and performance analysis can be found in:

- Maksymilian Wojnar, Wojciech Ciężobka, Artur Tomaszewski, Piotr Chołda, Krzysztof Rusek, Katarzyna Kosek-Szott, Jetmir Haxhibeqiri, Jeroen Hoebeke, Boris Bellalta, Anatolij Zubow, Falko Dressler, and Szymon Szott. "Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks", 2025.

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
