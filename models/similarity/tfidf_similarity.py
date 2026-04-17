"""
TF-IDF Similarity — Term Frequency–Inverse Document Frequency cosine similarity.

Converts both texts into TF-IDF vectors and computes their cosine
similarity. This captures keyword-level overlap:
    - High score → texts share many of the same important words.
    - Low score  → texts use different vocabulary.

Design Pattern: TEMPLATE METHOD (Concrete Implementation)
"""

from models.similarity.base import SimilarityMetric


class TFIDFSimilarity(SimilarityMetric):
    """
    Similarity metric based on TF-IDF vector cosine similarity.

    Uses scikit-learn's TfidfVectorizer to represent texts as
    sparse vectors weighted by term frequency and inverse document
    frequency, then computes cosine similarity.
    """

    @property
    def name(self) -> str:
        return "TF-IDF Similarity"

    @property
    def description(self) -> str:
        return (
            "Word-level similarity using TF-IDF vectors and cosine distance. "
            "Measures keyword overlap weighted by word importance."
        )

    def _compute_raw(self, text1: str, text2: str) -> float:
        """
        Compute TF-IDF cosine similarity.

        Args:
            text1: First text string.
            text2: Second text string.

        Returns:
            Cosine similarity between TF-IDF vectors (0.0 to 1.0).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        # Normalize
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Create TF-IDF matrix from both documents
        vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            max_features=5000
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([t1, t2])
        except ValueError:
            # If vectorizer produces empty vocabulary (no meaningful words)
            return 0.0

        # Compute cosine similarity between the two documents
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        return float(similarity_matrix[0][0])
