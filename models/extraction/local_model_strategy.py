"""
Local Model Strategy — Concrete Strategy using TrOCR (HuggingFace).

Mode 3: Local transformer-based extraction
    - Uses Microsoft's TrOCR model for text recognition.
    - Separate models for handwritten vs. printed documents.
    - Runs entirely locally — no API calls needed.
    - Requires ~900MB model download on first run.

Design Pattern: STRATEGY PATTERN (Concrete Strategy)

Note: This module is guarded by the ENABLE_LOCAL_MODEL feature flag
in config.py. It can be disabled if PyTorch/Transformers are unavailable.
"""

from typing import List, Optional

from PIL import Image

from models.extraction.base import ExtractionStrategy, ExtractionError
import config


class LocalModelStrategy(ExtractionStrategy):
    """
    Text extraction using Microsoft TrOCR (Transformer-based OCR).

    Uses Vision Encoder-Decoder architecture:
    - Encoder: ViT (Vision Transformer) processes the image
    - Decoder: RoBERTa generates the text output

    Selects model variant based on doc_type:
    - handwritten → microsoft/trocr-large-handwritten
    - printed     → microsoft/trocr-large-printed
    """

    def __init__(self):
        """Initialize with lazy-loaded models."""
        self._processors = {}   # Cache per doc_type
        self._models = {}       # Cache per doc_type

    @property
    def name(self) -> str:
        return "Local Model (TrOCR)"

    def _get_model_and_processor(self, doc_type: str):
        """
        Lazy-load the TrOCR model and processor for the given doc_type.

        Models are cached after first load to avoid re-downloading.

        Args:
            doc_type: "handwritten" or "printed".

        Returns:
            Tuple of (processor, model).
        """
        if doc_type not in self._processors:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            model_name = (
                config.TROCR_HANDWRITTEN_MODEL
                if doc_type == "handwritten"
                else config.TROCR_PRINTED_MODEL
            )

            self._processors[doc_type] = TrOCRProcessor.from_pretrained(model_name)
            self._models[doc_type] = VisionEncoderDecoderModel.from_pretrained(model_name)

        return self._processors[doc_type], self._models[doc_type]

    def _segment_lines(self, image: Image.Image) -> List[Image.Image]:
        """
        Segment a document image into individual text lines.

        TrOCR works best on single-line images, so we attempt to
        split the document into horizontal strips containing text.

        Args:
            image: Full-page document image.

        Returns:
            List of cropped line images. Falls back to [image] if
            segmentation fails.
        """
        import numpy as np

        try:
            # Convert to grayscale numpy array
            gray = np.array(image.convert("L"))

            # Binarize: pixels darker than threshold are "ink"
            threshold = 200
            binary = (gray < threshold).astype(np.uint8)

            # Project horizontally: sum ink pixels per row
            row_sums = binary.sum(axis=1)

            # Find rows with significant ink
            min_ink = image.width * 0.01  # At least 1% of width
            ink_rows = row_sums > min_ink

            # Group consecutive ink rows into line regions
            lines = []
            in_line = False
            start = 0
            padding = 10  # Extra pixels above/below each line

            for i, has_ink in enumerate(ink_rows):
                if has_ink and not in_line:
                    start = max(0, i - padding)
                    in_line = True
                elif not has_ink and in_line:
                    end = min(image.height, i + padding)
                    if end - start > 15:  # Skip very thin regions (noise)
                        lines.append(image.crop((0, start, image.width, end)))
                    in_line = False

            # Handle last line if image ends with text
            if in_line:
                end = min(image.height, len(ink_rows) + padding)
                if end - start > 15:
                    lines.append(image.crop((0, start, image.width, end)))

            return lines if lines else [image]

        except Exception:
            # If segmentation fails, process the whole image
            return [image]

    def _extract_single_line(self, line_image: Image.Image, doc_type: str) -> str:
        """
        Extract text from a single line image using TrOCR.

        Args:
            line_image: Cropped image containing a single line of text.
            doc_type:   "handwritten" or "printed".

        Returns:
            Recognized text string.
        """
        import torch

        processor, model = self._get_model_and_processor(doc_type)

        # Convert to RGB if needed
        if line_image.mode != "RGB":
            line_image = line_image.convert("RGB")

        # Process image
        pixel_values = processor(
            images=line_image,
            return_tensors="pt"
        ).pixel_values

        # Generate text
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=256)

        # Decode
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        return text[0] if text else ""

    def extract_text(self, images: List[Image.Image], doc_type: str = "printed") -> str:
        """
        Extract text from document images using TrOCR.

        For each page: segments into lines → recognizes each line → joins.

        Args:
            images:   List of PIL Image objects (one per page).
            doc_type: "handwritten" or "printed".

        Returns:
            Combined extracted text from all pages.

        Raises:
            ExtractionError: If model loading or inference fails.
        """
        if not config.ENABLE_LOCAL_MODEL:
            raise ExtractionError(
                self.name,
                "Local model mode is disabled. Enable it in config.py."
            )

        try:
            all_text = []

            for page_image in images:
                # Segment page into lines
                line_images = self._segment_lines(page_image)

                # Extract text from each line
                page_lines = []
                for line_img in line_images:
                    line_text = self._extract_single_line(line_img, doc_type)
                    if line_text.strip():
                        page_lines.append(line_text.strip())

                if page_lines:
                    all_text.append("\n".join(page_lines))

            return "\n\n".join(all_text)

        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                self.name,
                f"TrOCR extraction failed: {str(e)}"
            )
