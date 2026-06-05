# PDF Craft Server

Web-based PDF-to-EPUB/Markdown converter powered by [pdf-craft](https://github.com/oomol-lab/pdf-craft) and DeepSeek OCR. Deploy on any Linux server with an NVIDIA GPU via Docker.

## Requirements

- **Linux server** with Ubuntu 22.04+ (or compatible)
- **NVIDIA GPU** with Compute Capability 6.0+ (Pascal or newer) — RTX 3090, A100, etc.
- **NVIDIA drivers** installed on the host
- **Docker** + **NVIDIA Container Toolkit**

> Maxwell GPUs (CC 5.x like Quadro M6000) are **not supported** — the DeepSeek-OCR model requires bfloat16.

## Quick Start

```bash
docker compose up --build -d
```

That's it. The server starts on port **8000**. First launch downloads the OCR model (~2GB).

Open `http://<your-server-ip>:8000` in a browser to use the web UI.

## Web UI

Navigate to `http://<server-ip>:8000`:

1. **Upload** a PDF (drag-and-drop or browse)
2. Choose output format (**EPUB** or Markdown)
3. Fill in book title / author (for EPUB)
4. Click **Start Conversion**
5. Wait for progress to complete, then **Download Result**

## API

All API endpoints are under `/api/`.

### Convert a PDF

```bash
curl -X POST "http://localhost:8000/api/convert" \
  -F "file=@document.pdf" \
  -F "output_format=epub" \
  -F "ocr_size=gundam" \
  -F "book_title=My Book" \
  -F "book_author=Author"
```

Returns: `{"task_id": "...", "status": "pending"}`

### Check Status

```bash
curl http://localhost:8000/api/task/<task_id>
```

### Download Result

```bash
curl -O http://localhost:8000/api/download/<task_id>
```

### Health Check

```bash
curl http://localhost:8000/api/health
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `file` | *(required)* | PDF file |
| `output_format` | `markdown` | `markdown` or `epub` |
| `ocr_size` | `gundam` | `tiny`, `small`, `base`, `large`, `gundam` |
| `dpi` | `300` | Image DPI |
| `include_cover` | `true` (epub) | Include cover page |
| `include_footnotes` | `true` | Preserve footnotes |
| `ignore_pdf_errors` | `false` | Continue on PDF errors |
| `ignore_ocr_errors` | `false` | Continue on OCR errors |
| `toc_assumed` | `true` (epub) | Enable TOC detection |
| `book_title` | `Untitled` | EPUB title |
| `book_author` | *(none)* | EPUB author |
| `book_publisher` | *(none)* | EPUB publisher |
| `language` | `en` | `en` or `zh` |

## Volumes

| Volume | Purpose |
|---|---|
| `hf-cache` (named) | HuggingFace model cache — persists across restarts |
| `./outputs` | Converted output files |

## Stopping and Restarting

```bash
docker compose stop    # stop (keeps container and data)
docker compose start   # restart
docker compose down    # stop and remove container
```

## Troubleshooting

**Container won't start** — ensure NVIDIA Container Toolkit is installed:
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-container-toolkit --configure docker
sudo systemctl restart docker
```

**CUDA not available** — verify with `docker compose run --rm pdf-craft nvidia-smi`

**Model download fails** — check network connectivity to huggingface.co. You can pre-download by mounting a local HF cache.
