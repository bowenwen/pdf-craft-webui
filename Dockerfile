FROM nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    python3-pip \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install uv

RUN uv pip install --system \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126 && \
    uv pip install --system \
    pdf-craft \
    fastapi \
    uvicorn[standard] \
    python-multipart

COPY server.py /app/server.py
COPY static/ /app/static/

RUN mkdir -p /app/uploads /app/outputs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "server.py"]
