"""
Strategy Factory — Creates extraction strategy instances.

Implements the Factory Pattern to decouple strategy creation from
the client code. The client specifies a mode string, and the factory
returns the appropriate concrete ExtractionStrategy instance.

Design Pattern: FACTORY PATTERN
    - Encapsulates object creation logic.
    - Centralizes the mapping from mode strings to concrete classes.
    - Makes adding new strategies trivial (just add a new elif branch).

Extended with BYOM Plugin Discovery:
    - On first call, scans the `plugins/` directory for Python scripts
      containing subclasses of ExtractionStrategy.
    - Discovered plugins are automatically registered and available
      alongside built-in strategies.
"""

import importlib
import importlib.util
import inspect
import os
from typing import Dict, List, Optional, Type

from models.extraction.base import ExtractionStrategy
from models.extraction.ocr_strategy import OCRStrategy
from models.extraction.api_strategy import APIStrategy
import config


class StrategyFactory:
    """
    Factory class for creating ExtractionStrategy instances.

    Supports both built-in modes and dynamically discovered plugins.

    Usage:
        strategy = StrategyFactory.create("OCR")
        strategy = StrategyFactory.create("API", api_key="your-key")
        strategy = StrategyFactory.create("Local Model")
        strategy = StrategyFactory.create("🔌 My Custom Model")  # Plugin
    """

    # Class-level cache of discovered plugin classes
    _plugin_registry: Dict[str, Type[ExtractionStrategy]] = {}
    _plugins_scanned: bool = False

    @classmethod
    def scan_plugins(cls) -> None:
        """
        Scan the plugins/ directory for valid ExtractionStrategy subclasses.

        Walks through all .py files in config.PLUGINS_DIR, imports them,
        and registers any class that inherits from ExtractionStrategy.
        Results are cached — subsequent calls are no-ops.
        """
        if cls._plugins_scanned or not config.ENABLE_PLUGINS:
            return

        cls._plugins_scanned = True
        plugins_dir = config.PLUGINS_DIR

        if not os.path.isdir(plugins_dir):
            return

        for filename in os.listdir(plugins_dir):
            if not filename.endswith(".py") or filename.startswith("_"):
                continue

            filepath = os.path.join(plugins_dir, filename)
            module_name = f"plugins.{filename[:-3]}"

            try:
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find all ExtractionStrategy subclasses in the module
                for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(attr_value, ExtractionStrategy)
                        and attr_value is not ExtractionStrategy
                        and not inspect.isabstract(attr_value)
                    ):
                        # Instantiate to get the name, then store the class
                        try:
                            instance = attr_value()
                            plugin_name = instance.name
                            cls._plugin_registry[plugin_name] = attr_value
                        except Exception:
                            # Skip plugins that fail to instantiate
                            pass

            except Exception:
                # Skip files that fail to import
                pass

    @classmethod
    def _ensure_plugins_loaded(cls) -> None:
        """Ensure plugins have been scanned at least once."""
        if not cls._plugins_scanned:
            cls.scan_plugins()

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
        # Ensure plugins are loaded
        StrategyFactory._ensure_plugins_loaded()

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

        # Check plugin registry
        elif mode in StrategyFactory._plugin_registry:
            plugin_class = StrategyFactory._plugin_registry[mode]
            return plugin_class(**kwargs)

        else:
            available = ", ".join(StrategyFactory.available_modes())
            raise ValueError(
                f"Unknown extraction mode: '{mode}'. "
                f"Available modes: {available}"
            )

    @staticmethod
    def available_modes() -> List[str]:
        """
        Return list of all available extraction mode display names.

        Includes both built-in modes and discovered plugins.
        """
        StrategyFactory._ensure_plugins_loaded()
        modes = list(config.EXTRACTION_MODES)

        # Append plugin names
        for plugin_name in StrategyFactory._plugin_registry:
            if plugin_name not in modes:
                modes.append(plugin_name)

        return modes

    @staticmethod
    def get_plugin_names() -> List[str]:
        """Return names of discovered plugin strategies only."""
        StrategyFactory._ensure_plugins_loaded()
        return list(StrategyFactory._plugin_registry.keys())
