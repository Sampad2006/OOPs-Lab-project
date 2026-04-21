from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import json
import threading
import queue
import base64
from io import BytesIO

import config
from models.document import Document, DocType, FileType
from models.extraction.factory import StrategyFactory
from models.analyzer import DocumentAnalyzer
from utils.file_handler import FileHandler
from utils.preprocessor import TextPreprocessor
from utils.visual_detector import VisualDetector

app = FastAPI(title="Document Similarity Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_base64_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_visuals(images):
    images_b64 = []
    if images:
        for img in images:
            regions = VisualDetector.detect_text_regions(img)
            annotated = VisualDetector.draw_bounding_boxes(img, regions)
            images_b64.append(get_base64_image(annotated))
    return images_b64

@app.post("/api/analyze")
async def analyze_documents(
    handwritten_file: UploadFile = File(...),
    printed_file: UploadFile = File(...),
    mode: str = Form(...),
    api_key: str = Form(None),
    provider: str = Form("gemini"),
    api_model: str = Form(None)
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
            kwargs["provider"] = provider
            if api_model:
                kwargs["model_name"] = api_model
            
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

@app.post("/api/analyze/stream")
async def analyze_documents_stream(
    handwritten_file: UploadFile = File(...),
    printed_file: UploadFile = File(...),
    mode: str = Form(...),
    api_key: str = Form(None),
    provider: str = Form("gemini"),
    api_model: str = Form(None)
):
    # Read files in async context before passing to thread
    hw_bytes = await handwritten_file.read()
    pr_bytes = await printed_file.read()
    hw_filename = handwritten_file.filename
    pr_filename = printed_file.filename

    def sse_generator():
        q = queue.Queue()

        def logger_cb(level, message):
            q.put({"type": "log", "level": level, "message": message})

        def worker():
            try:
                logger_cb("INFO", "Loading documents...")
                hw_ext = hw_filename.rsplit(".", 1)[-1].lower()
                pr_ext = pr_filename.rsplit(".", 1)[-1].lower()

                hw_images = FileHandler.load_images(hw_bytes, FileType.from_extension(hw_ext))
                pr_images = FileHandler.load_images(pr_bytes, FileType.from_extension(pr_ext))
                logger_cb("SUCCESS", f"Loaded: {len(hw_images)} + {len(pr_images)} page(s)")

                doc1 = Document(
                    file_name=hw_filename,
                    file_type=FileType.from_extension(hw_ext),
                    doc_type=DocType.HANDWRITTEN,
                    images=hw_images,
                )
                doc2 = Document(
                    file_name=pr_filename,
                    file_type=FileType.from_extension(pr_ext),
                    doc_type=DocType.PRINTED,
                    images=pr_images,
                )

                logger_cb("INFO", "Generating bounding boxes visuals...")
                hw_visuals = generate_visuals(hw_images)
                pr_visuals = generate_visuals(pr_images)

                kwargs = {}
                if "api" in mode.lower():
                    if not api_key:
                        raise Exception("API Key is required for API Mode")
                    kwargs["api_key"] = api_key
                    kwargs["provider"] = provider
                    if api_model:
                        kwargs["model_name"] = api_model

                strategy = StrategyFactory.create(mode, **kwargs)
                analyzer = DocumentAnalyzer(strategy)

                text1 = analyzer.extract(doc1, logger=logger_cb)
                text2 = analyzer.extract(doc2, logger=logger_cb)

                text1 = TextPreprocessor.clean_ocr_output(text1)
                text2 = TextPreprocessor.clean_ocr_output(text2)

                scores = analyzer.compare(text1, text2, logger=logger_cb)
                
                logger_cb("INFO", "Generating confidence heatmaps...")
                heatmap1 = VisualDetector.generate_confidence_html(text1, mock=True)
                heatmap2 = VisualDetector.generate_confidence_html(text2, mock=True)

                logger_cb("SUCCESS", "✅ Analysis complete!")

                result = {
                    "text1": text1,
                    "text2": text2,
                    "scores": scores,
                    "visuals1": hw_visuals,
                    "visuals2": pr_visuals,
                    "heatmap1": heatmap1,
                    "heatmap2": heatmap2
                }
                q.put({"type": "done", "result": result})

            except Exception as e:
                import traceback
                traceback.print_exc()
                q.put({"type": "error", "message": str(e)})

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = q.get()
            if item["type"] == "log":
                yield f"event: log\ndata: {json.dumps({'level': item['level'], 'message': item['message']})}\n\n"
            elif item["type"] == "done":
                yield f"event: result\ndata: {json.dumps(item['result'])}\n\n"
                break
            elif item["type"] == "error":
                yield f"event: error\ndata: {json.dumps({'detail': item['message']})}\n\n"
                break

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.post("/api/arena/stream")
async def analyze_arena_stream(
    dataset_path: str = Form(...),
    strategies: str = Form(...), # JSON string list of strategy names
    api_keys: str = Form("{}") # JSON string dict of provider -> api key
):
    import json
    from utils.arena_runner import ArenaRunner

    try:
        selected_strategies = json.loads(strategies)
        keys_dict = json.loads(api_keys)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format for strategies or api_keys")

    def sse_generator():
        q = queue.Queue()

        def progress_cb(current, total, status):
            q.put({"type": "progress", "current": current, "total": total, "status": status})

        def worker():
            try:
                # Create strategy instances
                strategy_instances = []
                for m in selected_strategies:
                    kwargs = {}
                    if "api" in m.lower():
                        if "Groq" in m:
                            kwargs["provider"] = "groq"
                            kwargs["model_name"] = "meta-llama/llama-4-scout-17b-16e-instruct"
                            k = keys_dict.get("groq")
                        else:
                            kwargs["provider"] = "gemini"
                            kwargs["model_name"] = "gemini-2.5-flash"
                            k = keys_dict.get("gemini")
                            
                        if not k:
                            raise Exception(f"API Key is required for API Mode ({m})")
                        kwargs["api_key"] = k
                            
                    strategy_instances.append(StrategyFactory.create(m, **kwargs))

                runner = ArenaRunner(dataset_path=dataset_path, strategies=strategy_instances)
                
                # Pre-scan dataset to validate
                q.put({"type": "log", "level": "INFO", "message": f"Scanning dataset at {dataset_path}..."})
                pairs = runner.scan_dataset()
                q.put({"type": "log", "level": "SUCCESS", "message": f"Found {len(pairs)} image + ground-truth pairs."})

                # Run benchmark
                results = runner.run(progress_callback=progress_cb, log_callback=lambda lvl, msg: q.put({"type": "log", "level": lvl, "message": msg}))
                
                # Compute summary
                summary = ArenaRunner.compute_arena_scores(results)
                
                q.put({"type": "done", "result": {"results": results, "summary": summary}})

            except Exception as e:
                import traceback
                traceback.print_exc()
                q.put({"type": "error", "message": str(e)})

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = q.get()
            if item["type"] == "log":
                yield f"event: log\ndata: {json.dumps({'level': item['level'], 'message': item['message']})}\n\n"
            elif item["type"] == "progress":
                yield f"event: progress\ndata: {json.dumps({'current': item['current'], 'total': item['total'], 'status': item['status']})}\n\n"
            elif item["type"] == "done":
                yield f"event: result\ndata: {json.dumps(item['result'])}\n\n"
                break
            elif item["type"] == "error":
                yield f"event: error\ndata: {json.dumps({'detail': item['message']})}\n\n"
                break

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

from pydantic import BaseModel
from typing import Dict, Any, Optional

class SaveHistoryRequest(BaseModel):
    view: str
    strategy: str
    file_name: str
    scores: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

@app.post("/api/history/save")
async def save_history(req: SaveHistoryRequest):
    try:
        from utils.storage import ResultsStore
        store = ResultsStore()
        run_id = store.save_run(
            view=req.view,
            strategy=req.strategy,
            file_name=req.file_name,
            scores=req.scores,
            metadata=req.metadata
        )
        return {"success": True, "run_id": run_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
async def get_history(view: str = None, limit: int = 100):
    try:
        from utils.storage import ResultsStore
        store = ResultsStore()
        runs = store.get_runs(view=view, limit=limit)
        return {"runs": runs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
