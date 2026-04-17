"""
Models package — contains the OOP core of the Document Similarity Analyzer.

Subpackages:
    extraction  — Strategy Pattern for text extraction (OCR, API, Local Model)
    similarity  — Template Method Pattern for similarity metrics
"""

from models.document import Document
from models.analyzer import DocumentAnalyzer

__all__ = ["Document", "DocumentAnalyzer"]
