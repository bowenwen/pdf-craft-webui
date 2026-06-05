import json
import os
import sys
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pdf_craft import transform_markdown, transform_epub, BookMeta


# Directories
UPLOAD_DIR = Path("/app/uploads")
OUTPUT_DIR = Path("/app/outputs")
STATIC_DIR = Path("/app/static")
TASKS_FILE = Path("/app/tasks.json")

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_tasks() -> dict:
    if TASKS_FILE.exists():
        try:
            return json.loads(TASKS_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_tasks(tasks: dict):
    TASKS_FILE.write_text(json.dumps(tasks, indent=2))


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


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "type": type(exc).__name__,
            "traceback": tb,
        },
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Persistent task store
tasks = load_tasks()


def run_conversion(task_id: str, pdf_path: str, output_format: str, options: dict):
    task = tasks[task_id]
    task["status"] = "running"
    task["progress"] = "Starting conversion..."
    task["output_format"] = output_format
    save_tasks(tasks)

    try:
        base = os.path.basename(pdf_path).rsplit('.', 1)[0]
        if output_format == "markdown":
            output_path = OUTPUT_DIR / task_id / f"{base}.md"
        else:
            output_path = OUTPUT_DIR / task_id / f"{base}.epub"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        task["progress"] = "Converting..."
        save_tasks(tasks)

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
        task["output_format"] = output_format
        save_tasks(tasks)
    except Exception as e:
        tb = traceback.format_exc()
        task["status"] = "failed"
        task["progress"] = str(e)
        task["error_traceback"] = tb
        task["error_type"] = type(e).__name__
        save_tasks(tasks)


def run_reconvert(task_id: str, target_format: str, options: dict):
    task = tasks[task_id]
    if task["status"] != "completed" or not task.get("pdf_path"):
        task["status"] = "failed"
        task["progress"] = "Cannot reconvert: original file missing or previous task not completed"
        task["error_type"] = "ValueError"
        task["error_traceback"] = "No traceback available"
        save_tasks(tasks)
        return

    task["status"] = "running"
    task["progress"] = f"Reconverting to {target_format}..."
    task["target_format"] = target_format
    save_tasks(tasks)

    try:
        pdf_path = task["pdf_path"]
        base = os.path.basename(pdf_path).rsplit('.', 1)[0]
        if target_format == "markdown":
            output_path = OUTPUT_DIR / task_id / f"{base}.md"
        else:
            output_path = OUTPUT_DIR / task_id / f"{base}.epub"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        task["progress"] = f"Converting to {target_format}..."
        save_tasks(tasks)

        if target_format == "markdown":
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
        task["progress"] = f"Done (converted to {target_format})"
        task["output_path"] = str(output_path)
        task["output_filename"] = output_path.name
        task["output_format"] = target_format
        save_tasks(tasks)
    except Exception as e:
        tb = traceback.format_exc()
        task["status"] = "failed"
        task["progress"] = str(e)
        task["error_traceback"] = tb
        task["error_type"] = type(e).__name__
        save_tasks(tasks)


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
    output_format: str = Form("markdown"),
    ocr_size: str = Form("gundam"),
    dpi: int = Form(300),
    include_cover: bool = Form(False),
    include_footnotes: bool = Form(True),
    ignore_pdf_errors: bool = Form(False),
    ignore_ocr_errors: bool = Form(False),
    generate_plot: bool = Form(False),
    toc_assumed: bool = Form(False),
    book_title: Optional[str] = Form(None),
    book_author: Optional[str] = Form(None),
    book_publisher: Optional[str] = Form(None),
    language: str = Form("en"),
):
    if not file.filename or not file.filename.endswith(".pdf"):
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
        "output_format": None,
        "pdf_path": str(pdf_path),
        "filename": file.filename,
        "created_at": pdf_path.stat().st_mtime if pdf_path.exists() else 0,
        "error_type": None,
        "error_traceback": None,
    }
    save_tasks(tasks)

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


@app.get("/api/tasks")
async def list_tasks():
    result = []
    for tid, task in tasks.items():
        entry = {k: v for k, v in task.items() if k != "pdf_path" and k != "error_traceback"}
        entry["task_id"] = tid
        if entry.get("status") == "failed":
            entry["error"] = {
                "message": entry.get("progress", "Unknown error"),
                "type": entry.get("error_type", "Error"),
                "traceback": task.get("error_traceback", ""),
            }
        result.append(entry)
    result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return result


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    result = {k: v for k, v in task.items() if k != "pdf_path" and k != "error_traceback"}
    result["task_id"] = task_id
    if result.get("status") == "failed":
        result["error"] = {
            "message": result.get("progress", "Unknown error"),
            "type": result.get("error_type", "Error"),
            "traceback": task.get("error_traceback", ""),
        }
    return result


@app.delete("/api/tasks/{task_id}")
async def delete_task(task_id: str):
    task = tasks.pop(task_id, None)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    save_tasks(tasks)

    if task.get("pdf_path") and Path(task["pdf_path"]).exists():
        os.remove(task["pdf_path"])
    if task.get("output_path"):
        output = Path(task["output_path"])
        if output.parent.exists():
            shutil.rmtree(output.parent, ignore_errors=True)

    return {"detail": "Task deleted"}


@app.post("/api/reconvert/{task_id}")
async def reconvert(
    background_tasks: BackgroundTasks,
    task_id: str,
    target_format: str = Form("epub"),
    ocr_size: str = Form("gundam"),
    dpi: int = Form(300),
    include_cover: bool = Form(False),
    include_footnotes: bool = Form(True),
    ignore_pdf_errors: bool = Form(False),
    ignore_ocr_errors: bool = Form(False),
    generate_plot: bool = Form(False),
    toc_assumed: bool = Form(False),
    book_title: Optional[str] = Form(None),
    book_author: Optional[str] = Form(None),
    book_publisher: Optional[str] = Form(None),
    language: str = Form("en"),
):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task["status"] != "completed":
        raise HTTPException(status_code=400, detail="Can only reconvert completed tasks")
    if not task.get("pdf_path"):
        raise HTTPException(status_code=400, detail="Original PDF file not available")

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

    background_tasks.add_task(run_reconvert, task_id, target_format, options)
    return {"task_id": task_id, "status": task["status"]}


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
