import hashlib
import json
import os
import sys
import shutil
import time
import traceback
import uuid
import threading
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
DATA_FILE = Path("/app/data.json")
CONFIG_FILE = Path("/app/config.json")

for d in (UPLOAD_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = {
    "max_concurrent_tasks": 2,
    "default_output_format": "epub",
    "default_ocr_size": "gundam",
    "default_dpi": 300,
    "default_language": "en",
    "default_include_cover": False,
    "default_include_footnotes": True,
    "default_toc_assumed": False,
    "default_ignore_pdf_errors": False,
    "default_ignore_ocr_errors": False,
    "default_generate_plot": False,
}


def load_data() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text())
        except Exception:
            return {"files": {}}
    return {"files": {}}


def save_data(data: dict):
    DATA_FILE.write_text(json.dumps(data, indent=2))


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            user_cfg = json.loads(CONFIG_FILE.read_text())
            cfg = dict(DEFAULT_CONFIG)
            cfg.update(user_cfg)
            return cfg
        except Exception:
            return dict(DEFAULT_CONFIG)
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    # Only persist user changes (diff from defaults)
    user_cfg = {k: v for k, v in cfg.items() if k in DEFAULT_CONFIG and v != DEFAULT_CONFIG.get(k)}
    if user_cfg:
        CONFIG_FILE.write_text(json.dumps(user_cfg, indent=2))
    elif CONFIG_FILE.exists():
        CONFIG_FILE.unlink()


# Persistent data store
data = load_data()
config = load_config()

# Aborted flags: { "file_hash:task_id": bool }
aborted_flags: dict[str, bool] = {}

# Conversion queue lock
queue_lock = threading.Lock()


def compute_task_id(file_hash: str, fmt: str, params: dict) -> str:
    """Deterministic task_id from file_hash + format + key params."""
    canonical = json.dumps({"fh": file_hash, "f": fmt, "p": params}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_params(options: dict) -> dict:
    """Conversion-affecting params only (no metadata)."""
    return {
        "fmt": options.get("output_format", "markdown"),
        "ocr_size": options.get("ocr_size", "gundam"),
        "dpi": options.get("dpi", 300),
        "include_cover": options.get("include_cover", False),
        "include_footnotes": options.get("include_footnotes", True),
        "ignore_pdf_errors": options.get("ignore_pdf_errors", False),
        "ignore_ocr_errors": options.get("ignore_ocr_errors", False),
        "generate_plot": options.get("generate_plot", False),
        "toc_assumed": options.get("toc_assumed", False),
    }


def collect_params(options: dict) -> dict:
    """All params needed to restart a conversion."""
    return {
        "output_format": options.get("output_format", "markdown"),
        "ocr_size": options.get("ocr_size", "gundam"),
        "dpi": options.get("dpi", 300),
        "include_cover": options.get("include_cover", False),
        "include_footnotes": options.get("include_footnotes", True),
        "ignore_pdf_errors": options.get("ignore_pdf_errors", False),
        "ignore_ocr_errors": options.get("ignore_ocr_errors", False),
        "generate_plot": options.get("generate_plot", False),
        "toc_assumed": options.get("toc_assumed", False),
        "book_title": options.get("book_title"),
        "book_author": options.get("book_author"),
        "book_publisher": options.get("book_publisher"),
        "language": options.get("language", "en"),
    }


def make_aborted_check(file_hash: str, task_id: str) -> callable:
    key = f"{file_hash}:{task_id}"
    def check():
        if aborted_flags.get(key, False):
            raise Exception("Task stopped by user")
    return check


def count_running() -> int:
    """Count tasks currently in 'running' state across all files."""
    count = 0
    for fentry in data["files"].values():
        for conv in fentry["conversions"].values():
            if conv.get("status") == "running":
                count += 1
    return count


def get_next_queued() -> tuple | None:
    """Find the oldest queued task (by created_at). Returns (file_hash, task_id) or None."""
    oldest = None
    oldest_time = float('inf')
    for fh, fentry in data["files"].items():
        for tid, conv in fentry["conversions"].items():
            if conv.get("status") == "queued":
                ct = conv.get("created_at", float('inf'))
                if ct < oldest_time:
                    oldest_time = ct
                    oldest = (fh, tid)
    return oldest


def dispatch_queued_tasks():
    """Start queued tasks up to max_concurrent_tasks. Called after each task state change."""
    cfg = load_config()  # Reload in case config changed
    max_concurrent = cfg.get("max_concurrent_tasks", 2)

    with queue_lock:
        running = count_running()
        while running < max_concurrent:
            next_task = get_next_queued()
            if not next_task:
                break
            fh, tid = next_task
            fentry = data["files"].get(fh)
            if not fentry:
                continue
            conv = fentry["conversions"].get(tid)
            if not conv or conv["status"] != "queued":
                continue
            if not fentry.get("pdf_path") or not Path(fentry["pdf_path"]).exists():
                conv["status"] = "failed"
                conv["progress"] = "Source PDF file missing"
                conv["error_type"] = "FileNotFoundError"
                conv["error_traceback"] = "No traceback available"
                save_data(data)
                continue

            params = conv.get("params", {})
            if not params:
                params = conv
            # Start conversion in a thread
            t = threading.Thread(
                target=_do_convert,
                args=(fh, tid, fentry["pdf_path"], params),
                daemon=True,
            )
            t.start()

            # Mark as running
            conv["status"] = "running"
            conv["progress"] = "Starting conversion..."
            save_data(data)
            running += 1


def _do_convert(file_hash: str, task_id: str, pdf_path: str, params: dict):
    """Actual conversion work running in a thread."""
    key = f"{file_hash}:{task_id}"
    aborted_flags[key] = False

    output_format = params.get("output_format", "markdown")
    options = dict(params)

    file_entry = data["files"].get(file_hash)
    if not file_entry:
        aborted_flags.pop(key, None)
        dispatch_queued_tasks()
        return
    conv = file_entry["conversions"].get(task_id)
    if not conv:
        aborted_flags.pop(key, None)
        dispatch_queued_tasks()
        return

    conv["status"] = "running"
    conv["progress"] = "Starting conversion..."
    conv["output_format"] = output_format
    conv["started_at"] = time.time()
    save_data(data)

    aborted = make_aborted_check(file_hash, task_id)

    try:
        aborted()

        base = os.path.basename(pdf_path).rsplit('.', 1)[0]
        if output_format == "markdown":
            output_path = OUTPUT_DIR / task_id / f"{base}.md"
        else:
            output_path = OUTPUT_DIR / task_id / f"{base}.epub"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        conv["progress"] = "Converting..."
        save_data(data)

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
                aborted=aborted,
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
                aborted=aborted,
            )

        conv["status"] = "completed"
        conv["progress"] = "Done"
        conv["output_path"] = str(output_path)
        conv["output_filename"] = output_path.name
        conv["output_format"] = output_format
        conv["completed_at"] = time.time()
        save_data(data)
    except Exception as e:
        tb = traceback.format_exc()
        conv["status"] = "failed"
        conv["progress"] = str(e)
        conv["error_traceback"] = tb
        conv["error_type"] = type(e).__name__
        save_data(data)
    finally:
        aborted_flags.pop(key, None)
        dispatch_queued_tasks()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading DeepSeek OCR model... (first run downloads ~2GB)")
    from pdf_craft import Transform
    transform = Transform()
    transform.load_models()
    print("Model loaded, server ready")
    # Dispatch any queued tasks on startup
    threading.Thread(target=dispatch_queued_tasks, daemon=True).start()
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


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content="<h1>UI not found</h1>", status_code=500)


# ---- Settings Endpoints ----

@app.get("/api/settings")
async def get_settings():
    cfg = load_config()
    return cfg


@app.post("/api/settings")
async def save_user_settings(request: Request):
    global config
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")
    cfg = load_config()
    # Validate and merge
    for key in DEFAULT_CONFIG:
        if key in body:
            cfg[key] = body[key]
    config = cfg
    save_config(cfg)
    return {"detail": "Settings saved"}


# ---- Convert (Queue) ----

@app.post("/api/convert")
async def convert(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    output_format: str = Form("epub"),
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

    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()[:16]

    options = {
        "output_format": output_format,
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

    task_id = compute_task_id(file_hash, output_format, build_params(options))

    file_entry = data["files"].get(file_hash)

    # Check if identical conversion already exists
    if file_entry:
        existing_conv = file_entry["conversions"].get(task_id)
        if existing_conv:
            if existing_conv["status"] in ("queued", "running"):
                raise HTTPException(
                    status_code=409,
                    detail="This conversion is already queued or in progress.",
                    file_hash=file_hash,
                    task_id=task_id,
                )
            # Completed, failed, or cancelled — allow retry by replacing
            del file_entry["conversions"][task_id]
            if existing_conv.get("output_path"):
                old_out = Path(existing_conv["output_path"])
                if old_out.parent.exists():
                    shutil.rmtree(old_out.parent, ignore_errors=True)

    if file_entry:
        pdf_path = file_entry["pdf_path"]
    else:
        pdf_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(content)
        data["files"][file_hash] = {
            "filename": file.filename,
            "pdf_path": str(pdf_path),
            "created_at": time.time(),
            "conversions": {},
        }

    file_entry = data["files"][file_hash]
    params = collect_params(options)
    file_entry["conversions"][task_id] = {
        "status": "queued",
        "progress": "Queued",
        "output_format": output_format,
        "output_path": None,
        "output_filename": None,
        "error_type": None,
        "error_traceback": None,
        "params": params,
        "created_at": time.time(),
    }
    save_data(data)

    # Dispatch: try to start if under concurrency limit
    threading.Thread(target=dispatch_queued_tasks, daemon=True).start()

    return {"file_hash": file_hash, "task_id": task_id, "status": "queued"}


# ---- Task Manager Endpoints ----

@app.get("/api/tasks")
async def list_tasks():
    result = []
    for fh, fentry in data["files"].items():
        for tid, conv in fentry["conversions"].items():
            c = {k: v for k, v in conv.items() if k not in ("error_traceback", "params")}
            c["task_id"] = tid
            c["file_hash"] = fh
            c["filename"] = fentry["filename"]
            if c.get("status") == "failed":
                c["error"] = {
                    "message": c.get("progress", "Unknown error"),
                    "type": c.get("error_type", "Error"),
                    "traceback": conv.get("error_traceback", ""),
                }
            result.append(c)
    result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return result


@app.get("/api/files")
async def list_files():
    result = []
    for fh, fentry in data["files"].items():
        entry = {
            "file_hash": fh,
            "filename": fentry["filename"],
            "created_at": fentry["created_at"],
            "conversions": {},
        }
        for tid, conv in fentry["conversions"].items():
            c = {k: v for k, v in conv.items() if k not in ("error_traceback", "params")}
            c["task_id"] = tid
            if c.get("status") == "failed":
                c["error"] = {
                    "message": c.get("progress", "Unknown error"),
                    "type": c.get("error_type", "Error"),
                    "traceback": conv.get("error_traceback", ""),
                }
            entry["conversions"][tid] = c
        result.append(entry)
    result.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return result


@app.get("/api/files/{file_hash}/tasks/{task_id}")
async def get_task(file_hash: str, task_id: str):
    fentry = data["files"].get(file_hash)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    conv = fentry["conversions"].get(task_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Task not found")
    result = {k: v for k, v in conv.items() if k not in ("error_traceback", "params")}
    result["task_id"] = task_id
    result["file_hash"] = file_hash
    result["filename"] = fentry["filename"]
    if result.get("status") == "failed":
        result["error"] = {
            "message": result.get("progress", "Unknown error"),
            "type": result.get("error_type", "Error"),
            "traceback": conv.get("error_traceback", ""),
        }
    return result


@app.post("/api/files/{file_hash}/tasks/{task_id}/stop")
async def stop_task(file_hash: str, task_id: str):
    fentry = data["files"].get(file_hash)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    conv = fentry["conversions"].get(task_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Task not found")
    if conv["status"] not in ("running", "queued"):
        raise HTTPException(status_code=400, detail=f"Cannot stop a task in '{conv['status']}' state")

    key = f"{file_hash}:{task_id}"
    aborted_flags[key] = True

    if conv["status"] == "queued":
        conv["status"] = "cancelled"
        conv["progress"] = "Cancelled before starting"
        save_data(data)
        # Dispatch next queued task since this one won't run
        threading.Thread(target=dispatch_queued_tasks, daemon=True).start()

    return {"detail": "Stop requested", "status": conv["status"]}


@app.post("/api/files/{file_hash}/tasks/{task_id}/start")
async def start_task(file_hash: str, task_id: str):
    fentry = data["files"].get(file_hash)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    conv = fentry["conversions"].get(task_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Task not found")
    if conv["status"] not in ("cancelled", "failed"):
        raise HTTPException(status_code=400, detail=f"Cannot start a task in '{conv['status']}' state")
    if not fentry.get("pdf_path"):
        raise HTTPException(status_code=400, detail="Original PDF not available")
    if not Path(fentry["pdf_path"]).exists():
        raise HTTPException(status_code=400, detail="Original PDF file missing")

    # Clean up previous output
    if conv.get("output_path"):
        old_out = Path(conv["output_path"])
        if old_out.parent.exists():
            shutil.rmtree(old_out.parent, ignore_errors=True)

    conv["status"] = "queued"
    conv["progress"] = "Queued"
    conv["output_path"] = None
    conv["output_filename"] = None
    conv["error_type"] = None
    conv["error_traceback"] = None
    conv["created_at"] = time.time()
    save_data(data)

    threading.Thread(target=dispatch_queued_tasks, daemon=True).start()
    return {"file_hash": file_hash, "task_id": task_id, "status": "queued"}


@app.delete("/api/files/{file_hash}")
async def delete_file(file_hash: str):
    fentry = data["files"].pop(file_hash, None)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    save_data(data)

    for conv in fentry.get("conversions", {}).values():
        key = f"{file_hash}:{conv.get('task_id', '')}"
        aborted_flags.pop(key, None)

    if fentry.get("pdf_path") and Path(fentry["pdf_path"]).exists():
        os.remove(fentry["pdf_path"])
    for conv in fentry.get("conversions", {}).values():
        if conv.get("output_path"):
            out = Path(conv["output_path"])
            if out.parent.exists():
                shutil.rmtree(out.parent, ignore_errors=True)

    return {"detail": "File and all conversions deleted"}


@app.delete("/api/files/{file_hash}/tasks/{task_id}")
async def delete_task(file_hash: str, task_id: str):
    fentry = data["files"].get(file_hash)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    conv = fentry["conversions"].pop(task_id, None)
    if not conv:
        raise HTTPException(status_code=404, detail="Task not found")
    aborted_flags.pop(f"{file_hash}:{task_id}", None)
    save_data(data)

    if conv.get("output_path"):
        out = Path(conv["output_path"])
        if out.parent.exists():
            shutil.rmtree(out.parent, ignore_errors=True)

    return {"detail": "Conversion deleted"}


@app.post("/api/files/{file_hash}/convert")
async def add_conversion(
    background_tasks: BackgroundTasks,
    file_hash: str,
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
    fentry = data["files"].get(file_hash)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    if not fentry.get("pdf_path"):
        raise HTTPException(status_code=400, detail="Original PDF not available")
    if not Path(fentry["pdf_path"]).exists():
        raise HTTPException(status_code=400, detail="Original PDF file missing")

    options = {
        "output_format": target_format,
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

    new_task_id = compute_task_id(file_hash, target_format, build_params(options))

    existing = fentry["conversions"].get(new_task_id)
    if existing and existing["status"] in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail="This conversion is already queued or in progress.",
            file_hash=file_hash,
            task_id=new_task_id,
        )

    params = collect_params(options)
    fentry["conversions"][new_task_id] = {
        "status": "queued",
        "progress": "Queued",
        "output_format": target_format,
        "output_path": None,
        "output_filename": None,
        "error_type": None,
        "error_traceback": None,
        "params": params,
        "created_at": time.time(),
    }
    save_data(data)

    threading.Thread(target=dispatch_queued_tasks, daemon=True).start()
    return {"file_hash": file_hash, "task_id": new_task_id, "status": "queued"}


@app.get("/api/files/{file_hash}/download/{task_id}")
async def download(file_hash: str, task_id: str):
    fentry = data["files"].get(file_hash)
    if not fentry:
        raise HTTPException(status_code=404, detail="File not found")
    conv = fentry["conversions"].get(task_id)
    if not conv or conv["status"] != "completed":
        raise HTTPException(status_code=404, detail="File not ready")
    return FileResponse(
        path=conv["output_path"],
        filename=conv["output_filename"],
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
