"""
Embedding Similarity — Sentence-BERT-based semantic similarity.

Encodes both texts using a pre-trained sentence transformer model
and computes cosine similarity between the dense embedding vectors.
This captures semantic meaning beyond simple keyword overlap:
    - "The cat sat on the mat" ≈ "A feline rested on the rug" → high score
    - "The cat sat on the mat" ≈ "Stock prices rose today" → low score

Design Pattern: TEMPLATE METHOD (Concrete Implementation)
"""

from models.similarity.base import SimilarityMetric
import config


class EmbeddingSimilarity(SimilarityMetric):
    """
    Similarity metric based on sentence-transformer embeddings.

    Uses the all-MiniLM-L6-v2 model (~80MB) to generate dense
    vector representations of text, then computes cosine similarity.
    """

    def __init__(self):
        """Initialize with lazy-loaded model."""
        self._model = None

    @property
    def name(self) -> str:
        return "Embedding Similarity"

    @property
    def description(self) -> str:
        return (
            "Semantic similarity using sentence-transformer embeddings. "
            "Captures meaning and context beyond keyword matching."
        )

    def _get_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def _compute_raw(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity using sentence embeddings.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Cosine similarity between embedding vectors (0.0 to 1.0).
        """
        from sentence_transformers import util

        model = self._get_model()

        # Encode both texts into dense vectors
        embedding1 = model.encode(text1, convert_to_tensor=True)
        embedding2 = model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.cos_sim(embedding1, embedding2)

        return float(similarity.item())
