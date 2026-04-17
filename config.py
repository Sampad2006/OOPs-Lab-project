"""
Configuration and constants for the Document Similarity Analyzer.
Centralizes all app-wide settings, feature flags, and model names.
"""


# ──────────────────────────────────────────────
# App Metadata
# ──────────────────────────────────────────────
APP_TITLE = "📄 Document Similarity Analyzer"
APP_DESCRIPTION = (
    "Upload a handwritten document and a printed document, "
    "then compare their textual similarity across multiple metrics."
)
PAGE_ICON = "📄"
LAYOUT = "wide"

# ──────────────────────────────────────────────
# Supported File Types
# ──────────────────────────────────────────────
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_PDF_TYPES = ["pdf"]
SUPPORTED_FILE_TYPES = SUPPORTED_IMAGE_TYPES + SUPPORTED_PDF_TYPES

# ──────────────────────────────────────────────
# Feature Flags
# ──────────────────────────────────────────────
ENABLE_LOCAL_MODEL = True  # Toggle Mode 3 (TrOCR) on/off

# ──────────────────────────────────────────────
# Extraction Modes
# ──────────────────────────────────────────────
MODE_OCR = "🔍 OCR (Tesseract + EasyOCR)"
MODE_API = "🌐 API (Google Gemini Vision)"
MODE_LOCAL_MODEL = "🤖 Local Model (TrOCR)"

EXTRACTION_MODES = [MODE_OCR, MODE_API]
if ENABLE_LOCAL_MODEL:
    EXTRACTION_MODES.append(MODE_LOCAL_MODEL)

# ──────────────────────────────────────────────
# Model Names
# ──────────────────────────────────────────────
# TrOCR models (Mode 3)
TROCR_HANDWRITTEN_MODEL = "microsoft/trocr-large-handwritten"
TROCR_PRINTED_MODEL = "microsoft/trocr-large-printed"

# Gemini API (Mode 2)
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_EXTRACTION_PROMPT = (
    "Extract all text from this document image exactly as written. "
    "Return only the raw text content, no formatting, no markdown, "
    "no explanations. If multiple lines exist, preserve line breaks."
)

# Sentence Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ──────────────────────────────────────────────
# EasyOCR
# ──────────────────────────────────────────────
EASYOCR_LANGUAGES = ["en"]

# ──────────────────────────────────────────────
# Similarity Weights (equal by default)
# ──────────────────────────────────────────────
SIMILARITY_WEIGHTS = {
    "edit": 1.0,
    "tfidf": 1.0,
    "embedding": 1.0,
}
