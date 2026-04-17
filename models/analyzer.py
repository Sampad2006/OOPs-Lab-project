"""
Document Analyzer — Facade that orchestrates extraction and similarity.

Provides a single, clean entry point for the Streamlit app to:
    1. Extract text from documents using a chosen strategy.
    2. Compare extracted texts using all similarity metrics.

Design Pattern: FACADE PATTERN
    - Hides the complexity of strategy selection, extraction, and
      similarity computation behind a simple interface.
    - The Streamlit app only needs to interact with DocumentAnalyzer.
"""

from typing import Dict, List, Optional

from models.document import Document
from models.extraction.base import ExtractionStrategy
from models.similarity.base import SimilarityMetric
from models.similarity.aggregator import SimilarityAggregator


class DocumentAnalyzer:
    """
    Facade for document text extraction and similarity analysis.

    Orchestrates the interaction between extraction strategies
    and similarity metrics, providing a unified API.
    """

    def __init__(
        self,
        strategy: ExtractionStrategy,
        metrics: Optional[List[SimilarityMetric]] = None
    ):
        """
        Initialize the analyzer with an extraction strategy and metrics.

        Args:
            strategy: The ExtractionStrategy to use for text extraction.
            metrics:  List of SimilarityMetric instances. If None,
                      all default metrics are used.
        """
        self._strategy = strategy
        self._metrics = metrics or self._default_metrics()
        self._aggregator = SimilarityAggregator(self._metrics)

    @staticmethod
    def _default_metrics() -> List[SimilarityMetric]:
        """Create the default set of similarity metrics."""
        from models.similarity.edit_distance import EditDistanceSimilarity
        from models.similarity.tfidf_similarity import TFIDFSimilarity
        from models.similarity.embedding_similarity import EmbeddingSimilarity

        return [
            EditDistanceSimilarity(),
            TFIDFSimilarity(),
            EmbeddingSimilarity(),
        ]

    @property
    def strategy(self) -> ExtractionStrategy:
        """Return the current extraction strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, new_strategy: ExtractionStrategy):
        """
        Swap the extraction strategy at runtime.

        This demonstrates the power of the Strategy Pattern — the
        algorithm can be changed without modifying the Analyzer.
        """
        self._strategy = new_strategy

    @property
    def metrics(self) -> List[SimilarityMetric]:
        """Return the list of similarity metrics."""
        return self._metrics

    def extract(self, document: Document) -> str:
        """
        Extract text from a document using the current strategy.

        Args:
            document: A Document object with loaded images.

        Returns:
            Extracted text string.

        Raises:
            ExtractionError: If extraction fails.
        """
        text = self._strategy.extract_text(
            images=document.images,
            doc_type=document.doc_type.value
        )
        document.extracted_text = text
        return text

    def compare(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Compare two texts using all registered similarity metrics.

        Args:
            text1: Text extracted from document 1.
            text2: Text extracted from document 2.

        Returns:
            Dictionary of metric names → scores, plus "Final Similarity".
        """
        return self._aggregator.compute_all(text1, text2)

    def analyze(
        self, doc1: Document, doc2: Document
    ) -> Dict[str, object]:
        """
        Full pipeline: extract text from both docs and compare.

        This is the main entry point for the Streamlit app.

        Args:
            doc1: First document (e.g., handwritten).
            doc2: Second document (e.g., printed).

        Returns:
            Dictionary containing:
                - "text1": Extracted text from doc1
                - "text2": Extracted text from doc2
                - "scores": Dict of similarity scores
        """
        text1 = self.extract(doc1)
        text2 = self.extract(doc2)

        scores = self.compare(text1, text2)

        return {
            "text1": text1,
            "text2": text2,
            "scores": scores,
        }
