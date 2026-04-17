"""
Preprocessor — Image and text preprocessing utilities.

Provides reusable preprocessing functions for improving OCR accuracy
and normalizing extracted text for similarity comparison.
"""

import re

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """
    Preprocesses document images to improve OCR accuracy.

    Applies standard techniques:
    - Grayscale conversion
    - Adaptive thresholding (binarization)
    - Noise removal (morphological operations)
    """

    @staticmethod
    def preprocess_for_ocr(image: Image.Image) -> Image.Image:
        """
        Apply standard preprocessing pipeline for OCR.

        Args:
            image: Input PIL Image.

        Returns:
            Preprocessed PIL Image optimized for OCR.
        """
        import cv2

        # Convert PIL Image to OpenCV format (numpy array)
        img_array = np.array(image)

        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Adaptive thresholding for binarization
        # Works better than global thresholding on uneven lighting
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # Morphological operations to remove small noise
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert back to PIL Image
        return Image.fromarray(cleaned)

    @staticmethod
    def resize_for_model(image: Image.Image, max_size: int = 1024) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.

        Args:
            image:    Input PIL Image.
            max_size: Maximum dimension (width or height).

        Returns:
            Resized PIL Image.
        """
        width, height = image.size
        if max(width, height) <= max_size:
            return image

        ratio = max_size / max(width, height)
        new_size = (int(width * ratio), int(height * ratio))
        return image.resize(new_size, Image.Resampling.LANCZOS)


class TextPreprocessor:
    """
    Preprocesses extracted text for similarity comparison.

    Applies normalization steps to reduce noise from OCR artifacts
    and ensure fair comparison between texts.
    """

    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize text for comparison.

        Steps:
            1. Convert to lowercase
            2. Remove extra whitespace
            3. Remove common OCR artifacts
            4. Strip leading/trailing whitespace

        Args:
            text: Raw extracted text.

        Returns:
            Normalized text string.
        """
        if not text:
            return ""

        # Lowercase
        text = text.lower()

        # Remove common OCR artifacts (isolated single characters, etc.)
        text = re.sub(r'\b[a-z]\b(?!\s[a-z]\b)', '', text)

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def clean_ocr_output(text: str) -> str:
        """
        Clean raw OCR output while preserving meaningful content.

        Less aggressive than normalize() — keeps original casing
        and structure, but removes obvious artifacts.

        Args:
            text: Raw OCR output text.

        Returns:
            Cleaned text string.
        """
        if not text:
            return ""

        # Remove null bytes and control characters (except newlines/tabs)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # Collapse multiple blank lines into one
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove lines that are only whitespace or punctuation
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if line.strip() and not re.match(r'^[\s\W]+$', line.strip())
        ]

        return '\n'.join(cleaned_lines).strip()
