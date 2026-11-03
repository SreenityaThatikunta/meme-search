# Serves the FastAPI search API. The dataset, FAISS indexes and MongoDB stay
# outside the image and are mounted / reached at runtime, keeping the build
# small and avoiding baking a 700MB meme dump into a layer.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MEME_DEVICE=cpu

WORKDIR /app

# libGL and libglib are needed by opencv-python (pulled in via easyocr).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py main.py ./
COPY services/ services/

# Indexes, id_map and images are mounted at /app/data; see docker-compose.yml.
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
