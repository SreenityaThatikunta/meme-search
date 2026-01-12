# Meme Search Engine

A semantic meme search engine powered by CLIP (Contrastive Language-Image Pre-training) and FAISS (Facebook AI Similarity Search). Search for memes using natural language queries and get relevant results based on both visual content and text extracted via OCR.

## Features

- **Semantic Search**: Search memes using natural language queries
- **CLIP-Powered**: Uses OpenAI's CLIP model for understanding image-text relationships
- **Fast Similarity Search**: FAISS indexing for efficient similarity search
- **OCR Support**: Extracts and indexes text from memes using EasyOCR
- **Web Interface**: Clean Streamlit-based UI for easy searching
- **REST API**: FastAPI backend for programmatic access
- **MongoDB Storage**: Stores meme metadata and extracted text

## Architecture

- **Frontend**: Streamlit web application ([app.py](app.py))
- **Backend API**: FastAPI server ([main.py](main.py))
- **Search Service**: CLIP + FAISS search implementation ([services/search.py](services/search.py))
- **Data Processing Scripts**: Tools for embedding generation and OCR processing
  - [scripts/embed_memes_fast.py](scripts/embed_memes_fast.py) - Generate CLIP embeddings for memes
  - [scripts/run_ocr_fast.py](scripts/run_ocr_fast.py) - Extract text from memes
  - [scripts/scan_dataset.py](scripts/scan_dataset.py) - Scan and index meme dataset
  - [scripts/store_metadata.py](scripts/store_metadata.py) - Store metadata in MongoDB

## Prerequisites

- Python 3.12+
- MongoDB (for metadata storage)
- CUDA-compatible GPU (optional, for faster processing)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd meme-search
```

2. Create and activate a virtual environment:
```bash
python -m venv memenv
source memenv/bin/activate  # On Windows: memenv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure MongoDB is running:
```bash
# Default connection: mongodb://localhost:27017
```

## Usage

### 1. Prepare Your Dataset

Place your meme images in the `data/` directory.

### 2. Process Memes

Run the preprocessing scripts to extract text and generate embeddings:

```bash
# Scan and index the dataset
python scripts/scan_dataset.py

# Run OCR to extract text
python scripts/run_ocr_fast.py

# Generate CLIP embeddings
python scripts/embed_memes_fast.py

# Store metadata in MongoDB
python scripts/store_metadata.py
```

### 3. Start the Backend API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 4. Launch the Frontend

In a separate terminal:

```bash
streamlit run app.py
```

The web interface will open in your browser (default: `http://localhost:8501`)

## API Endpoints

### POST /search/text

Search for memes using a text query.

**Request Body:**
```json
{
  "query": "funny cat meme",
  "top_k": 5
}
```

**Response:**
```json
[
  {
    "meme_id": "123",
    "image_path": "data/memes/cat_meme.jpg",
    "text": "i can has cheezburger",
    "score": 0.876
  }
]
```

## Configuration

- **API URL**: Update `API_URL` in [app.py](app.py:10) to point to your backend
- **FAISS Index**: Stored at `./data/text.index`
- **ID Mapping**: Stored at `./data/id_map.json`
- **MongoDB**: Default connection to `mongodb://localhost:27017`, database: `meme_search`, collection: `memes`
- **CLIP Model**: Uses `ViT-B/32` by default (configurable in [services/search.py](services/search.py:19))

## Project Structure

```
meme-search/
├── app.py                      # Streamlit frontend
├── main.py                     # FastAPI backend
├── services/
│   └── search.py              # Search logic (CLIP + FAISS)
├── scripts/
│   ├── embed_memes_fast.py    # Embedding generation
│   ├── run_ocr_fast.py        # OCR processing
│   ├── scan_dataset.py        # Dataset scanning
│   └── store_metadata.py      # Metadata storage
├── data/
│   ├── memes/                 # Meme image files
│   ├── text.index             # FAISS index
│   └── id_map.json            # Vector ID to meme ID mapping
└── memenv/                     # Virtual environment
```

## How It Works

1. **Preprocessing**: Memes are processed to extract text via OCR and generate visual embeddings using CLIP
2. **Indexing**: Embeddings are stored in a FAISS index for fast similarity search
3. **Search**: User queries are encoded using CLIP and compared against indexed embeddings
4. **Retrieval**: Top-k most similar memes are retrieved and displayed with similarity scores

## Technologies Used

- **CLIP**: OpenAI's vision-language model
- **FAISS**: Facebook's similarity search library
- **FastAPI**: Modern Python web framework
- **Streamlit**: Interactive web UI framework
- **PyTorch**: Deep learning framework
- **EasyOCR**: Text extraction from images
- **MongoDB**: Document database for metadata
- **Pillow**: Image processing
