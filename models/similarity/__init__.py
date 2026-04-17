"""
Similarity subpackage — Template Method Pattern for similarity metrics.

This package provides multiple similarity metrics for comparing
extracted texts, following the Template Method design pattern.

Classes:
    SimilarityMetric       — Abstract base class (Template)
    EditDistanceSimilarity — Levenshtein-based similarity
    TFIDFSimilarity        — TF-IDF cosine similarity
    EmbeddingSimilarity    — Sentence-BERT embedding similarity
    SimilarityAggregator   — Aggregates and normalizes scores
"""

from models.similarity.base import SimilarityMetric
from models.similarity.edit_distance import EditDistanceSimilarity
from models.similarity.tfidf_similarity import TFIDFSimilarity
from models.similarity.embedding_similarity import EmbeddingSimilarity
from models.similarity.aggregator import SimilarityAggregator

__all__ = [
    "SimilarityMetric",
    "EditDistanceSimilarity",
    "TFIDFSimilarity",
    "EmbeddingSimilarity",
    "SimilarityAggregator",
]
