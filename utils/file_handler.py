"""
File Handler — Loads uploaded files and converts them to PIL Images.

Handles both image files (JPG, JPEG, PNG) and PDF documents.
PDFs are converted to a list of page images using pdf2image.
"""

from typing import List

from PIL import Image

from models.document import FileType


class FileHandler:
    """
    Handles file loading and conversion for uploaded documents.

    Converts uploaded file bytes into a list of PIL Image objects
    (one per page for PDFs, or a single-item list for images).
    """

    @staticmethod
    def load_images(file_bytes: bytes, file_type: FileType) -> List[Image.Image]:
        """
        Load an uploaded file and return it as a list of PIL Images.

        Args:
            file_bytes: Raw bytes of the uploaded file.
            file_type:  The FileType enum value.

        Returns:
            List of PIL Image objects (one per page).

        Raises:
            ValueError: If file type is unsupported.
            RuntimeError: If PDF conversion fails.
        """
        if file_type.is_image:
            return FileHandler._load_image(file_bytes)
        elif file_type.is_pdf:
            return FileHandler._load_pdf(file_bytes)
        else:
            raise ValueError(f"Unsupported file type: {file_type.value}")

    @staticmethod
    def _load_image(file_bytes: bytes) -> List[Image.Image]:
        """
        Load a single image from bytes.

        Args:
            file_bytes: Raw image bytes.

        Returns:
            List containing a single PIL Image.
        """
        import io
        image = Image.open(io.BytesIO(file_bytes))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return [image]

    @staticmethod
    def _load_pdf(file_bytes: bytes) -> List[Image.Image]:
        """
        Convert a PDF to a list of page images.

        Uses pdf2image (which requires poppler-utils installed
        on the system) to render each PDF page as a PIL Image.

        Args:
            file_bytes: Raw PDF bytes.

        Returns:
            List of PIL Images, one per PDF page.

        Raises:
            RuntimeError: If poppler is not installed or conversion fails.
        """
        try:
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(
                file_bytes,
                dpi=300,      # High DPI for better OCR accuracy
                fmt="png",
            )

            # Ensure all images are RGB
            rgb_images = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                rgb_images.append(img)

            return rgb_images

        except ImportError:
            raise RuntimeError(
                "pdf2image is not installed. Run: pip install pdf2image"
            )
        except Exception as e:
            if "poppler" in str(e).lower():
                raise RuntimeError(
                    "Poppler is not installed. "
                    "On macOS: brew install poppler | "
                    "On Ubuntu: sudo apt-get install poppler-utils"
                )
            raise RuntimeError(f"PDF conversion failed: {str(e)}")
