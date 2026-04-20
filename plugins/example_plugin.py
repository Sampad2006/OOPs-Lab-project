"""
Example BYOM Plugin — Template for creating custom extraction strategies.

HOW TO CREATE A PLUGIN:
    1. Create a new .py file in this `plugins/` directory.
    2. Import and inherit from `ExtractionStrategy`.
    3. Implement the `name` property and `extract_text()` method.
    4. The StrategyFactory will auto-discover your class on startup.

UNCOMMENT the code below to activate this example plugin.
"""

# from typing import List
# from PIL import Image
# from models.extraction.base import ExtractionStrategy
#
#
# class ExampleCustomStrategy(ExtractionStrategy):
#     """
#     Example plugin strategy — replace with your own logic.
#
#     This strategy will automatically appear in the Streamlit UI
#     dropdown once uncommented.
#     """
#
#     @property
#     def name(self) -> str:
#         return "🔌 My Custom Model"
#
#     def extract_text(
#         self, images: List[Image.Image], doc_type: str = "printed"
#     ) -> str:
#         """
#         Extract text from images using your custom approach.
#
#         Args:
#             images:   List of PIL Image objects (one per page).
#             doc_type: "handwritten" or "printed".
#
#         Returns:
#             Extracted text string.
#         """
#         # --- Your extraction logic here ---
#         # Example: use a custom ONNX model, a different API, etc.
#         all_text = []
#         for image in images:
#             # Replace this with real extraction
#             all_text.append(f"[Placeholder] Extracted from {image.size} image")
#         return "\n".join(all_text)
