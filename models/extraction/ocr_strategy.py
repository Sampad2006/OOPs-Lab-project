"""
OCR Strategy — Concrete Strategy using Tesseract + EasyOCR.

Mode 1: Traditional OCR Engines
    - Tesseract (pytesseract): Best for printed/typed text.
    - EasyOCR: Better for handwritten text than Tesseract.

The strategy auto-selects the engine based on the doc_type parameter:
    - "printed"     → Tesseract
    - "handwritten" → EasyOCR

Design Pattern: STRATEGY PATTERN (Concrete Strategy)
"""

from typing import List

import numpy as np
from PIL import Image

from models.extraction.base import ExtractionStrategy, ExtractionError
from utils.preprocessor import ImagePreprocessor


class OCRStrategy(ExtractionStrategy):
    """
    Text extraction using traditional OCR engines.

    Uses Tesseract for printed documents and EasyOCR for handwritten
    documents, with image preprocessing to improve accuracy.
    Falls back to EasyOCR for all document types if Tesseract is not installed.
    """

    def __init__(self, languages: List[str] = None):
        """
        Initialize the OCR strategy.

        Args:
            languages: List of language codes for EasyOCR (default: ["en"]).
        """
        self._languages = languages or ["en"]
        self._easyocr_reader = None  # Lazy-loaded
        self._preprocessor = ImagePreprocessor()
        self._tesseract_available = self._check_tesseract()

    @staticmethod
    def _check_tesseract() -> bool:
        """Check if Tesseract is installed and accessible."""
        import shutil
        return shutil.which("tesseract") is not None

    @property
    def name(self) -> str:
        return "OCR (Tesseract + EasyOCR)"

    def _get_easyocr_reader(self):
        """Lazy-load EasyOCR reader to avoid heavy import on startup."""
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(
                self._languages,
                gpu=False,  # CPU for portability
                verbose=False
            )
        return self._easyocr_reader

    def _extract_with_tesseract(self, image: Image.Image) -> str:
        """
        Extract text from a single image using Tesseract.

        Args:
            image: A PIL Image object.

        Returns:
            Extracted text string.
        """
        import pytesseract

        # Preprocess for better Tesseract accuracy
        processed = self._preprocessor.preprocess_for_ocr(image)
        text = pytesseract.image_to_string(processed, lang="eng")
        return text.strip()

    def _extract_with_easyocr(self, image: Image.Image) -> str:
        """
        Extract text from a single image using EasyOCR.

        Args:
            image: A PIL Image object.

        Returns:
            Extracted text string.
        """
        reader = self._get_easyocr_reader()

        # Convert PIL Image to numpy array for EasyOCR
        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0, paragraph=True)
        return "\n".join(results).strip()

    def extract_text(self, images: List[Image.Image], doc_type: str = "printed") -> str:
        """
        Extract text from document images using the appropriate OCR engine.

        Selects Tesseract for printed docs, EasyOCR for handwritten docs.

        Args:
            images:   List of PIL Image objects (one per page).
            doc_type: "handwritten" or "printed".

        Returns:
            Combined extracted text from all pages.

        Raises:
            ExtractionError: If OCR processing fails.
        """
        try:
            all_text = []
            for i, image in enumerate(images):
                if doc_type == "handwritten":
                    page_text = self._extract_with_easyocr(image)
                elif self._tesseract_available:
                    page_text = self._extract_with_tesseract(image)
                else:
                    # Fallback: use EasyOCR when Tesseract is not installed
                    print(
                        "⚠️  Tesseract not found — falling back to EasyOCR "
                        "for printed document. Install tesseract for best results."
                    )
                    page_text = self._extract_with_easyocr(image)

                if page_text:
                    all_text.append(page_text)

            return "\n\n".join(all_text)

        except Exception as e:
            raise ExtractionError(self.name, f"OCR extraction failed: {str(e)}")
