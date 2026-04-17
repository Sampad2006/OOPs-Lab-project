# Document Similarity Analyzer

A full-stack, Object-Oriented web application implemented in Python and Streamlit to extract text from documents (handwritten and printed) and compute their semantic and lexical similarity using multiple state-of-the-art metrics.

## Features

- **Multi-Modal Text Extraction:**
  - **OCR Mode:** Uses `pytesseract` for printed documents and `EasyOCR` for handwritten text.
  - **API Mode:** Uses Google Gemini 2.0 Flash Vision for highly accurate extraction via the cloud.
  - **Local Model Mode:** Uses Microsoft `TrOCR` (Vision Encoder-Decoder) for local, private machine learning extraction.
- **Advanced Similarity Metrics:**
  - **Edit Distance:** Character-level Levenshtein similarity.
  - **TF-IDF:** Keyword importance and cosine similarity overlap.
  - **Semantic Embeddings:** Deep learning embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) to capture textual meaning.
- **Dynamic Normalization:** Computes an aggregated, normalized matching score (0-100%).
- **Premium UI:** Glassmorphism design system built on top of Streamlit with CSS injections.

## System Requirements

- Python 3.9+
- On macOS:
  ```bash
  brew install tesseract poppler
  ```
- On Ubuntu/Linux:
  ```bash
  sudo apt-get install tesseract-ocr poppler-utils
  ```
  *(Note: Poppler is required to convert PDF files into images. Tesseract is required for the baseline OCR mode).*

## Setup Instructions

1. **Clone the repository and set up a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
   The app will automatically open in your default browser at `http://localhost:8501`.

## Usage Guide

1. Open the web interface.
2. Select an **Extraction Mode** from the left sidebar.
   - *If using API Mode, enter your Google Gemini API Key in the sidebar.*
   - *If using Local Model Mode, please be aware the first run will download ~900MB of model weights.*
3. Upload a **Handwritten Document** (JPG/PNG/PDF) on the left panel.
4. Upload a **Printed Document** (JPG/PNG/PDF) on the right panel.
5. Click **Analyze & Compare** and view the extracted texts alongside the comprehensive similarity breakdown.

## Troubleshooting Common Warnings

When running the application, specifically the local ML models, you might see several warnings in your terminal. **These are harmless and standard behaviors of underlying libraries:**

- **`UserWarning: 'pin_memory' argument is set as true but not supported on MPS...`**
  - PyTorch warning on Apple Silicon MPS.Should work for other gpu arch. It doesn't affect processing.
- **`Warning: You are sending unauthenticated requests to the HF Hub.`**
  - Expected when downloading the `sentence-transformers` models without a Hugging Face token. The download will still succeed.
- **`BertModel LOAD REPORT...` and `embeddings.position_ids | UNEXPECTED`**
  - Informational logs from the transformer library stating some weights are randomly initialized or ignored. Standard for sentence models.
- **`Accessing __path__ from... Returning __path__ instead...`**
  - Deprecation warnings within the `transformers` library code. They do not impact the application functionality.
