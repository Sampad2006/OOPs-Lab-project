"""
Visual Detector — OpenCV-based text region detection and confidence heatmaps.

Provides utilities for:
    1. Detecting text regions via contour analysis and drawing bounding boxes.
    2. Generating color-coded HTML for confidence-annotated extracted text.

Used by the Microscope view when "Show Detection Visuals" or
"Show Confidence Heatmap" toggles are enabled.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


class VisualDetector:
    """
    Detects text regions in document images and provides
    visual feedback (bounding boxes + confidence heatmaps).
    """

    @staticmethod
    def detect_text_regions(
        image: Image.Image,
        min_area: int = 200
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in an image using OpenCV contour detection.

        Args:
            image:    PIL Image to analyze.
            min_area: Minimum contour area to consider as text.

        Returns:
            List of bounding boxes as (x, y, w, h) tuples.
        """
        import cv2

        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=10
        )

        # Dilate to merge nearby text into regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter by area and collect bounding boxes
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= min_area and w > 10 and h > 5:
                regions.append((x, y, w, h))

        # Sort top-to-bottom, then left-to-right
        regions.sort(key=lambda r: (r[1], r[0]))

        return regions

    @staticmethod
    def draw_bounding_boxes(
        image: Image.Image,
        regions: Optional[List[Tuple[int, int, int, int]]] = None,
        box_color: Tuple[int, int, int] = (92, 252, 180),
        box_width: int = 2
    ) -> Image.Image:
        """
        Draw bounding boxes on a copy of the image.

        Args:
            image:     Original PIL Image.
            regions:   List of (x, y, w, h) bounding boxes.
                       If None, auto-detects regions.
            box_color: RGB color for the bounding boxes.
            box_width: Line width for bounding boxes.

        Returns:
            New PIL Image with bounding boxes drawn.
        """
        if regions is None:
            regions = VisualDetector.detect_text_regions(image)

        # Work on a copy
        annotated = image.copy()
        if annotated.mode != "RGB":
            annotated = annotated.convert("RGB")

        draw = ImageDraw.Draw(annotated)

        for (x, y, w, h) in regions:
            draw.rectangle(
                [x, y, x + w, y + h],
                outline=box_color,
                width=box_width,
            )

        return annotated

    @staticmethod
    def generate_confidence_html(
        text: str,
        word_confidences: Optional[List[float]] = None,
        mock: bool = False
    ) -> str:
        """
        Generate HTML with color-coded words based on confidence scores.

        Green (#5cfcb4) = high confidence (>= 0.8)
        Yellow (#fcbc5c) = medium confidence (0.5 - 0.8)
        Red   (#fc5c7c)  = low confidence (< 0.5)

        Args:
            text:              The extracted text.
            word_confidences:  List of floats (0-1), one per word.
                               If None and mock=True, generates mock values.
            mock:              If True, generate simulated confidence values.

        Returns:
            HTML string with color-coded <span> elements.
        """
        if not text or not text.strip():
            return '<span style="color: #6a6a80;">(No text extracted)</span>'

        words = text.split()

        if word_confidences is None and mock:
            # Generate mock confidence: mostly high, some medium, few low
            word_confidences = []
            for word in words:
                # Longer words tend to have higher confidence (heuristic)
                base = min(0.95, 0.6 + len(word) * 0.04)
                jitter = random.uniform(-0.2, 0.1)
                conf = max(0.1, min(1.0, base + jitter))
                word_confidences.append(round(conf, 2))

        if word_confidences is None:
            # No confidence data — render plain
            return f'<span style="color: #e8e8f0;">{text}</span>'

        # Ensure lengths match
        while len(word_confidences) < len(words):
            word_confidences.append(0.5)

        spans = []
        for i, word in enumerate(words):
            conf = word_confidences[i] if i < len(word_confidences) else 0.5

            if conf >= 0.8:
                color = "#5cfcb4"  # Green — high
            elif conf >= 0.5:
                color = "#fcbc5c"  # Yellow — medium
            else:
                color = "#fc5c7c"  # Red — low

            opacity = max(0.6, conf)
            spans.append(
                f'<span style="color:{color}; opacity:{opacity:.2f};" '
                f'title="Confidence: {conf:.0%}">{word}</span>'
            )

        label = ""
        if mock:
            label = (
                '<div style="font-size:0.7rem; color:#6a6a80; '
                'margin-bottom:0.5rem; font-style:italic;">'
                '⚠ Confidence values are simulated for this model</div>'
            )

        return (
            f'{label}'
            f'<div class="confidence-text" style="font-family: \'JetBrains Mono\','
            f' monospace; font-size:0.85rem; line-height:1.8; '
            f'word-wrap:break-word;">'
            f'{" ".join(spans)}'
            f'</div>'
        )
