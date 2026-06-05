
import os
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pdf_craft import transform_markdown, transform_epub, BookMeta


# Directories
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
STATIC_DIR = Path("/app/static")

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading DeepSeek OCR model... (first run downloads ~2GB)")
    from pdf_craft import Transform
    transform = Transform()
    transform.load_models()
    print("Model loaded, server ready")
    yield


app = FastAPI(
    title="PDF Craft Server",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# In-memory task store
tasks = {}


def run_conversion(task_id: str, pdf_path: str, output_format: str, options: dict):
    task = tasks[task_id]
    task["status"] = "running"
    task["progress"] = "Starting conversion..."
    try:
        if output_format == "markdown":
            output_path = OUTPUT_DIR / task_id / f"{os.path.basename(pdf_path).rsplit('.', 1)[0]}.md"
        else:
            output_path = OUTPUT_DIR / task_id / f"{os.path.basename(pdf_path).rsplit('.', 1)[0]}.epub"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        task["progress"] = "Converting..."

        if output_format == "markdown":
            transform_markdown(
                pdf_path=pdf_path,
                markdown_path=str(output_path),
                markdown_assets_path=str(output_path.parent / "assets"),
                analysing_path=options.get("temp_path"),
                ocr_size=options.get("ocr_size", "gundam"),
                models_cache_path=options.get("models_path"),
                dpi=options.get("dpi", 300),
                includes_cover=options.get("include_cover", False),
                includes_footnotes=options.get("include_footnotes", True),
                ignore_pdf_errors=options.get("ignore_pdf_errors", False),
                ignore_ocr_errors=options.get("ignore_ocr_errors", False),
                generate_plot=options.get("generate_plot", False),
                toc_assumed=options.get("toc_assumed", False),
            )
        else:
            transform_epub(
                pdf_path=pdf_path,
                epub_path=str(output_path),
                analysing_path=options.get("temp_path"),
                ocr_size=options.get("ocr_size", "gundam"),
                models_cache_path=options.get("models_path"),
                dpi=options.get("dpi", 300),
                includes_cover=options.get("include_cover", True),
                includes_footnotes=options.get("include_footnotes", True),
                ignore_pdf_errors=options.get("ignore_pdf_errors", False),
                ignore_ocr_errors=options.get("ignore_ocr_errors", False),
                generate_plot=options.get("generate_plot", False),
                toc_assumed=options.get("toc_assumed", True),
                book_meta=BookMeta(
                    title=options.get("book_title", "Untitled"),
                    authors=[options["book_author"]] if options.get("book_author") else [],
                    publisher=options.get("book_publisher", ""),
                ),
                lan=options.get("language", "en"),
            )

        task["status"] = "completed"
        task["progress"] = "Done"
        task["output_path"] = str(output_path)
        task["output_filename"] = output_path.name
    except Exception as e:
        task["status"] = "failed"
        task["progress"] = f"Error: {str(e)}"


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=500)


@app.post("/api/convert")
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = "markdown",
    ocr_size: str = "gundam",
    dpi: int = 300,
    include_cover: bool = False,
    include_footnotes: bool = True,
    ignore_pdf_errors: bool = False,
    ignore_ocr_errors: bool = False,
    generate_plot: bool = False,
    toc_assumed: bool = False,
    book_title: Optional[str] = None,
    book_author: Optional[str] = None,
    book_publisher: Optional[str] = None,
    language: str = "en",
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    with open(pdf_path, "wb") as f:
        content = await file.read()
        f.write(content)

    task_id = uuid.uuid4().hex
    tasks[task_id] = {
        "status": "pending",
        "progress": "Queued",
        "output_path": None,
        "output_filename": None,
        "pdf_path": str(pdf_path),
        "output_format": output_format,
    }

    options = {
        "ocr_size": ocr_size,
        "dpi": dpi,
        "include_cover": include_cover,
        "include_footnotes": include_footnotes,
        "ignore_pdf_errors": ignore_pdf_errors,
        "ignore_ocr_errors": ignore_ocr_errors,
        "generate_plot": generate_plot,
        "toc_assumed": toc_assumed,
        "book_title": book_title,
        "book_author": book_author,
        "book_publisher": book_publisher,
        "language": language,
    }

    background_tasks.add_task(run_conversion, task_id, str(pdf_path), output_format, options)

    return {"task_id": task_id, "status": "pending"}


@app.get("/api/task/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    result = {k: v for k, v in task.items() if k != "pdf_path"}
    return result


@app.get("/api/download/{task_id}")
async def download(task_id: str):
    task = tasks.get(task_id)
    if not task or task["status"] != "completed":
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(
        path=task["output_path"],
        filename=task["output_filename"],
        media_type="application/octet-stream",
    )


@app.get("/api/health")
async def health():
    import torch
    return {
        "status": "ok",
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
