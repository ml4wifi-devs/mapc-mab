from chex import Array, Scalar


class MapcAgent:
    """
    Base class for the MAPC agent.
    """

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
    
    def batch_update(self, rewards: Array) -> None:
        """
        Updates the agent with the rewards obtained in the previous step.

        Parameters
        ----------
        rewards: Array
            The rewards obtained in the previous step.
        """

        raise NotImplementedError
