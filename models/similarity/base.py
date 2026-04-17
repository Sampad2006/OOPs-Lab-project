"""
Similarity Metric — Abstract Base Class.

Defines the interface and template for all similarity metrics.
Each concrete metric implements the `compute` method to calculate
a similarity score between two text strings.

Design Pattern: TEMPLATE METHOD PATTERN
    - The ABC defines the skeleton: validate inputs → compute score → clamp.
    - Concrete classes implement the specific similarity algorithm.
"""

from abc import ABC, abstractmethod


class SimilarityMetric(ABC):
    """
    Abstract base class for all text similarity metrics.

    Each concrete metric computes a similarity score between 0 and 1,
    where 0 means completely different and 1 means identical.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this similarity metric."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of what this metric measures."""
        pass

    @abstractmethod
    def _compute_raw(self, text1: str, text2: str) -> float:
        """
        Compute the raw similarity score (to be implemented by subclasses).

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Raw similarity score (ideally between 0 and 1).
        """
        pass

    def compute(self, text1: str, text2: str) -> float:
        """
        Template method: validates inputs, computes score, clamps to [0, 1].

        This is the public interface. Subclasses override _compute_raw().

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Similarity score clamped between 0.0 and 1.0.
        """
        # Handle edge cases
        if not text1 and not text2:
            return 1.0  # Both empty = identical
        if not text1 or not text2:
            return 0.0  # One empty = no similarity

        # Compute and clamp
        raw_score = self._compute_raw(text1, text2)
        return max(0.0, min(1.0, raw_score))

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()
