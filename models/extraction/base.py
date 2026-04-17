"""
Extraction Strategy — Abstract Base Class.

Defines the interface that all text extraction strategies must implement.
This is the core of the Strategy Pattern: the client code (DocumentAnalyzer)
depends only on this abstract interface, not on any concrete implementation.

Design Pattern: STRATEGY PATTERN
    - This ABC acts as the "Strategy" interface.
    - Concrete strategies (OCR, API, LocalModel) implement extract_text().
    - The "Context" (DocumentAnalyzer) holds a reference to a strategy
      and delegates extraction to it.
"""

from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class ExtractionStrategy(ABC):
    """
    Abstract base class for all text extraction strategies.

    Each concrete strategy encapsulates a different approach to
    extracting text from document images (OCR engines, cloud APIs,
    or local ML models).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this extraction strategy."""
        pass

    @abstractmethod
    def extract_text(self, images: List[Image.Image], doc_type: str = "printed") -> str:
        """
        Extract text from a list of document page images.

        Args:
            images:   List of PIL Image objects, one per document page.
            doc_type: Either "handwritten" or "printed", used by some
                      strategies to select the appropriate model/engine.

        Returns:
            The extracted text as a single string, with pages separated
            by newlines.

        Raises:
            ExtractionError: If text extraction fails.
        """
        pass

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return self.__str__()


class ExtractionError(Exception):
    """Custom exception raised when text extraction fails."""

    def __init__(self, strategy_name: str, message: str):
        self.strategy_name = strategy_name
        self.message = message
        super().__init__(f"[{strategy_name}] {message}")
