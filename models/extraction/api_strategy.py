"""
API Strategy — Concrete Strategy using Google Gemini Vision API.

Mode 2: Cloud API-based extraction
    - Sends document images to Google's Gemini 2.0 Flash model.
    - Uses vision capabilities to extract text from both handwritten
      and printed documents with high accuracy.

Design Pattern: STRATEGY PATTERN (Concrete Strategy)
"""

import base64
import io
from typing import List

from PIL import Image

from models.extraction.base import ExtractionStrategy, ExtractionError
import config


class APIStrategy(ExtractionStrategy):
    """
    Text extraction using Google Gemini Vision API.

    Sends each page image to the Gemini model with a text extraction
    prompt and aggregates the results.
    """

    def __init__(self, api_key: str):
        """
        Initialize the API strategy with a Gemini API key.

        Args:
            api_key: Google Gemini API key for authentication.

        Raises:
            ValueError: If api_key is empty or None.
        """
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required for API mode.")
        self._api_key = api_key
        self._model = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return "API (Google Gemini Vision)"

    def _get_model(self):
        """Lazy-load the Gemini generative model."""
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(config.GEMINI_MODEL)
        return self._model

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert a PIL Image to a base64-encoded string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _extract_single_page(self, image: Image.Image) -> str:
        """
        Extract text from a single page image using Gemini.

        Args:
            image: A PIL Image object.

        Returns:
            Extracted text from the image.
        """
        model = self._get_model()

        response = model.generate_content(
            [config.GEMINI_EXTRACTION_PROMPT, image],
            generation_config={"temperature": 0.1, "max_output_tokens": 4096}
        )

        if response and response.text:
            return response.text.strip()
        return ""

    def extract_text(self, images: List[Image.Image], doc_type: str = "printed") -> str:
        """
        Extract text from document images using Google Gemini Vision.

        Args:
            images:   List of PIL Image objects (one per page).
            doc_type: "handwritten" or "printed" (Gemini handles both well).

        Returns:
            Combined extracted text from all pages.

        Raises:
            ExtractionError: If API call fails.
        """
        try:
            all_text = []
            for i, image in enumerate(images):
                page_text = self._extract_single_page(image)
                if page_text:
                    all_text.append(page_text)

            return "\n\n".join(all_text)

        except ValueError as e:
            raise ExtractionError(self.name, str(e))
        except Exception as e:
            raise ExtractionError(
                self.name,
                f"Gemini API call failed: {str(e)}"
            )
