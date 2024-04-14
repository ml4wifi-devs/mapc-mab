from chex import Array, Scalar


class MapcAgent:
    """
    Base class for the MAPC agent.
    """

    def update(self, rewards: Array) -> None:

        raise NotImplementedError

    def sample(self) -> Array:
        """
        Samples the agent to get the transmission matrix.

        Parameters
        ----------
        reward: float
            The reward obtained in the previous step.

        Returns
        -------
        Array
            The transmission matrix.
        """

        raise NotImplementedError
