from abc import ABC, abstractmethod


class ThresholdStrategy(ABC):
    """Interface for threshold update strategies."""

    @abstractmethod
    def update_threshold(
        self, current_threshold: float, cr: float = 0.05, **kwargs
    ) -> float:
        """
        Update the current threshold to flexibilize the inclusion of new
        instances.

        Args:
            current_threshold (float): The current threshold.
            cr (float): Change rate parameter. This parameter acts as a
            step to smooth the threshold variation with each change.

        Raises:
            NotImplementedError: If you use superclass method.

        Returns:
            float: New threshold value.
        """
        raise NotImplementedError('implement me!')
