from chex import Array, Scalar


class MapcAgent:
    """
    Base class for the MAPC agent.
    """

    def sample(self, reward: Scalar) -> tuple[Array, Array]:
        raise NotImplementedError
