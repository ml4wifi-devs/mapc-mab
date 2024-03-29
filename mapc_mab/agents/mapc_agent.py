from chex import Array, Scalar


class MapcAgent:
    """
    Base class for the MAPC agent.
    """

    def sample(self, reward: Scalar) -> Array:
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
