from chex import Array, Scalar


class MapcAgent:
    """
    Base class for the MAPC agent.
    """

    def update(self, rewards: Array) -> None:

        raise NotImplementedError

    def sample(self) -> Array:

        raise NotImplementedError
