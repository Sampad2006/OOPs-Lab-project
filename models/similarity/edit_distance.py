"""
Edit Distance Similarity — Levenshtein-based text similarity.

Computes the Levenshtein edit distance between two strings and
converts it to a similarity score:
    similarity = 1 - (edit_distance / max_length)

A score of 1.0 means the texts are identical; 0.0 means completely
different (edit distance equals the length of the longer string).

Design Pattern: TEMPLATE METHOD (Concrete Implementation)
"""

from models.similarity.base import SimilarityMetric


class EditDistanceSimilarity(SimilarityMetric):
    """
    Similarity metric based on Levenshtein edit distance.

    Measures how many single-character edits (insertions, deletions,
    substitutions) are needed to transform one string into another.
    """

    @property
    def name(self) -> str:
        return "Edit Similarity"

    @property
    def description(self) -> str:
        return (
            "Character-level similarity using Levenshtein distance. "
            "Measures the minimum edits needed to transform one text into another."
        )

    def _compute_raw(self, text1: str, text2: str) -> float:
        """
        Compute edit distance similarity.

        Formula: 1 - (levenshtein_distance(t1, t2) / max(len(t1), len(t2)))

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        import Levenshtein

        # Normalize: lowercase and strip extra whitespace
        t1 = " ".join(text1.lower().split())
        t2 = " ".join(text2.lower().split())

        if not t1 and not t2:
            return 1.0

        distance = Levenshtein.distance(t1, t2)
        max_len = max(len(t1), len(t2))

        if max_len == 0:
            return 1.0

        return 1.0 - (distance / max_len)
