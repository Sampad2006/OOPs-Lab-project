"""
Similarity Aggregator — Combines all metric scores into a final score.

Takes a list of SimilarityMetric instances, computes all their
individual scores, and produces a normalized final similarity score
on a scale of 0 to 1.

Final Score = weighted_mean(individual_scores)

By default, all metrics are equally weighted (weight = 1.0).
"""

from typing import Dict, List

from models.similarity.base import SimilarityMetric
import config


class SimilarityAggregator:
    """
    Aggregates multiple similarity metrics into a final normalized score.

    Attributes:
        metrics: List of SimilarityMetric instances to compute.
    """

    def __init__(self, metrics: List[SimilarityMetric]):
        """
        Initialize the aggregator with a list of metrics.

        Args:
            metrics: List of SimilarityMetric instances.

        Raises:
            ValueError: If metrics list is empty.
        """
        if not metrics:
            raise ValueError("At least one similarity metric is required.")
        self._metrics = metrics

    @property
    def metrics(self) -> List[SimilarityMetric]:
        """Return the list of registered metrics."""
        return list(self._metrics)

    def compute_all(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compute all individual similarity scores and the final aggregated score.

        The final score is the weighted mean of all individual scores,
        normalized to the 0–1 range.

        Args:
            text1: First text string (from document 1).
            text2: Second text string (from document 2).

        Returns:
            Dictionary containing:
                - Each metric's name → its score (0–1)
                - "Final Similarity" → normalized aggregate score (0–1)
        """
        results = {}
        weighted_sum = 0.0
        total_weight = 0.0

        # Map metric names to their config weight keys
        weight_keys = {
            "Edit Similarity": "edit",
            "TF-IDF Similarity": "tfidf",
            "Embedding Similarity": "embedding",
        }

        for metric in self._metrics:
            try:
                score = metric.compute(text1, text2)
            except Exception as e:
                # If a metric fails, log it and assign 0
                score = 0.0
                print(f"Warning: {metric.name} failed with error: {e}")

            results[metric.name] = round(score, 4)

            # Get weight from config (default 1.0)
            weight_key = weight_keys.get(metric.name, "default")
            weight = config.SIMILARITY_WEIGHTS.get(weight_key, 1.0)

            weighted_sum += score * weight
            total_weight += weight

        # Compute final normalized score
        if total_weight > 0:
            final_score = weighted_sum / total_weight
        else:
            final_score = 0.0

        results["Final Similarity"] = round(final_score, 4)

        return results

    def __str__(self) -> str:
        metric_names = [m.name for m in self._metrics]
        return f"SimilarityAggregator(metrics={metric_names})"
