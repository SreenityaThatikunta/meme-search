FROM python:3.10-slim

WORKDIR /app

# -----------------------------
# System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Python dependencies
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy ONLY application code
# -----------------------------
COPY main.py .
COPY services/ services/
COPY data/id_map.json data/

# (Optional) if FAISS index is small enough (<200–300MB)
# COPY data/text.index data/

# -----------------------------
# Expose & run
# -----------------------------
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]