"""
Document module — defines the Document dataclass.

This module encapsulates all metadata and content related to an uploaded
document, including its images (converted from PDF or loaded directly)
and the extracted text result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from PIL import Image


class DocType(Enum):
    """Enumeration for document types."""
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"


class FileType(Enum):
    """Enumeration for supported file types."""
    JPG = "jpg"
    JPEG = "jpeg"
    PNG = "png"
    PDF = "pdf"

    @classmethod
    def from_extension(cls, ext: str) -> "FileType":
        """Create a FileType from a file extension string."""
        ext = ext.lower().lstrip(".")
        try:
            return cls(ext)
        except ValueError:
            raise ValueError(
                f"Unsupported file type: '{ext}'. "
                f"Supported types: {[ft.value for ft in cls]}"
            )

    @property
    def is_image(self) -> bool:
        """Check if this file type is an image format."""
        return self in (FileType.JPG, FileType.JPEG, FileType.PNG)

    @property
    def is_pdf(self) -> bool:
        """Check if this file type is PDF."""
        return self == FileType.PDF


@dataclass
class Document:
    """
    Represents an uploaded document with its metadata.

    Attributes:
        file_name:      Original filename of the uploaded document.
        file_type:      The type/format of the file (JPG, PDF, etc.).
        doc_type:       Whether the document is handwritten or printed.
        images:         List of PIL Image objects (one per page).
        extracted_text: The text extracted from the document (populated after OCR).
    """
    file_name: str
    file_type: FileType
    doc_type: DocType
    images: List[Image.Image] = field(default_factory=list)
    extracted_text: Optional[str] = None

    def __str__(self) -> str:
        status = "extracted" if self.extracted_text else "pending"
        return (
            f"Document('{self.file_name}', type={self.doc_type.value}, "
            f"pages={len(self.images)}, status={status})"
        )

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def page_count(self) -> int:
        """Return the number of pages/images in this document."""
        return len(self.images)

    @property
    def has_text(self) -> bool:
        """Check if text has been extracted from this document."""
        return self.extracted_text is not None and len(self.extracted_text.strip()) > 0
