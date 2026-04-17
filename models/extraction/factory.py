"""
Strategy Factory — Creates extraction strategy instances.

Implements the Factory Pattern to decouple strategy creation from
the client code. The client specifies a mode string, and the factory
returns the appropriate concrete ExtractionStrategy instance.

Design Pattern: FACTORY PATTERN
    - Encapsulates object creation logic.
    - Centralizes the mapping from mode strings to concrete classes.
    - Makes adding new strategies trivial (just add a new elif branch).
"""

from models.extraction.base import ExtractionStrategy
from models.extraction.ocr_strategy import OCRStrategy
from models.extraction.api_strategy import APIStrategy
import config


class StrategyFactory:
    """
    Factory class for creating ExtractionStrategy instances.

    Usage:
        strategy = StrategyFactory.create("OCR")
        strategy = StrategyFactory.create("API", api_key="your-key")
        strategy = StrategyFactory.create("Local Model")
    """

    @staticmethod
    def create(mode: str, **kwargs) -> ExtractionStrategy:
        """
        Create and return an ExtractionStrategy based on the mode.

        Args:
            mode:   The extraction mode string. Accepts both short names
                    and the full display names from config.
            **kwargs:
                api_key (str): Required when mode is "API".
                languages (list): Optional language codes for OCR mode.

        Returns:
            An instance of the appropriate ExtractionStrategy subclass.

        Raises:
            ValueError: If the mode is unrecognized or required args
                        are missing.
        """
        # Normalize: accept both short names and full config names
        mode_lower = mode.lower()

        if "ocr" in mode_lower:
            languages = kwargs.get("languages", config.EASYOCR_LANGUAGES)
            return OCRStrategy(languages=languages)

        elif "api" in mode_lower:
            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError(
                    "API key is required for API mode. "
                    "Please enter your Gemini API key in the sidebar."
                )
            return APIStrategy(api_key=api_key)

        elif "local" in mode_lower or "model" in mode_lower:
            if not config.ENABLE_LOCAL_MODEL:
                raise ValueError(
                    "Local Model mode is disabled. "
                    "Enable ENABLE_LOCAL_MODEL in config.py to use TrOCR."
                )
            from models.extraction.local_model_strategy import LocalModelStrategy
            return LocalModelStrategy()

        else:
            available = ", ".join(config.EXTRACTION_MODES)
            raise ValueError(
                f"Unknown extraction mode: '{mode}'. "
                f"Available modes: {available}"
            )

    @staticmethod
    def available_modes() -> list:
        """Return list of available extraction mode display names."""
        return list(config.EXTRACTION_MODES)
