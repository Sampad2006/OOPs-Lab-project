from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os

import config
from models.document import Document, DocType, FileType
from models.extraction.factory import StrategyFactory
from models.analyzer import DocumentAnalyzer
from utils.file_handler import FileHandler
from utils.preprocessor import TextPreprocessor

app = FastAPI(title="Document Similarity Analyzer API")

# Setup CORS to allow requests from our Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/analyze")
async def analyze_documents(
    handwritten_file: UploadFile = File(...),
    printed_file: UploadFile = File(...),
    mode: str = Form(...),
    api_key: str = Form(None)
):
    try:
        hw_bytes = await handwritten_file.read()
        pr_bytes = await printed_file.read()

        hw_ext = handwritten_file.filename.rsplit(".", 1)[-1].lower()
        pr_ext = printed_file.filename.rsplit(".", 1)[-1].lower()

        hw_images = FileHandler.load_images(hw_bytes, FileType.from_extension(hw_ext))
        pr_images = FileHandler.load_images(pr_bytes, FileType.from_extension(pr_ext))

        doc1 = Document(
            file_name=handwritten_file.filename,
            file_type=FileType.from_extension(hw_ext),
            doc_type=DocType.HANDWRITTEN,
            images=hw_images,
        )
        doc2 = Document(
            file_name=printed_file.filename,
            file_type=FileType.from_extension(pr_ext),
            doc_type=DocType.PRINTED,
            images=pr_images,
        )

        kwargs = {}
        if "api" in mode.lower():
            if not api_key:
                raise HTTPException(status_code=400, detail="API Key is required for API Mode")
            kwargs["api_key"] = api_key

        strategy = StrategyFactory.create(mode, **kwargs)
        analyzer = DocumentAnalyzer(strategy)

        text1 = analyzer.extract(doc1)
        text2 = analyzer.extract(doc2)

        text1 = TextPreprocessor.clean_ocr_output(text1)
        text2 = TextPreprocessor.clean_ocr_output(text2)

        scores = analyzer.compare(text1, text2)

        return {
            "text1": text1,
            "text2": text2,
            "scores": scores
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
