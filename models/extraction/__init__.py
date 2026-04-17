"""
Extraction subpackage — Strategy Pattern for text extraction.

This package provides multiple interchangeable strategies for extracting
text from document images, following the Strategy design pattern.

Classes:
    ExtractionStrategy  — Abstract base class (Strategy interface)
    OCRStrategy         — Concrete strategy using Tesseract + EasyOCR
    APIStrategy         — Concrete strategy using Google Gemini Vision API
    LocalModelStrategy  — Concrete strategy using TrOCR (HuggingFace)
    StrategyFactory     — Factory for creating strategy instances
"""

from models.extraction.base import ExtractionStrategy
from models.extraction.factory import StrategyFactory

__all__ = ["ExtractionStrategy", "StrategyFactory"]
