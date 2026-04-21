"""
API Strategy — Concrete Strategy using Google Gemini Vision API.

Mode 2: Cloud API-based extraction
    - Sends document images to Google Gemini models.
    - Uses vision capabilities to extract text from both handwritten
      and printed documents with high accuracy.

Design Pattern: STRATEGY PATTERN (Concrete Strategy)
"""

import io
from typing import List

from PIL import Image

from models.extraction.base import ExtractionStrategy, ExtractionError
import config


class APIStrategy(ExtractionStrategy):
    """
    Text extraction using Google Gemini Vision API (new google.genai SDK).
    """

    def __init__(self, api_key: str, model_name: str = config.GEMINI_MODEL):
        if not api_key or not api_key.strip():
            raise ValueError("Gemini API key is required for API mode.")
        self._api_key = api_key
        self._model_name = model_name
        self._client = None  # Lazy-loaded

    @property
    def name(self) -> str:
        return f"API (Google Gemini Vision)"

    def _get_client(self):
        """Lazy-load the Gemini client using the new google.genai SDK."""
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _extract_single_page(self, image: Image.Image) -> str:
        """Extract text from a single PIL Image page using Gemini Vision."""
        from google.genai import types

        client = self._get_client()

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        response = client.models.generate_content(
            model=self._model_name,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                config.GEMINI_EXTRACTION_PROMPT,
            ],
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
            for image in images:
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
