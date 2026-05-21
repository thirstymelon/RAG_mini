# RAG Document QA System

Simple Retrieval-Augmented Generation (RAG) document question-answering system built with Flask and Sentence Transformers.

## Features

- Upload and index documents
- Ask questions using natural language
- Hybrid retrieval:
  - Semantic Search
  - BM25
  - TF-IDF
  - CrossEncoder reranking
- AI-generated answers using FLAN-T5
- Source references with relevance scores

## Supported Files

- PDF
- DOCX
- PPTX
- TXT
- MD
- CSV
- JSON
- XML
- HTML

## Installation

```bash
git clone https://github.com/thirstymelon/RAG_mini.git
cd RAG_mini

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python3 app.py
```

Server:

```txt
http://localhost:50005
```

## Dependencies

```txt
flask
sentence-transformers
transformers
torch
numpy
scikit-learn
pypdf
python-docx
python-pptx
```

## API Endpoints

### Upload Files

```http
POST /api/upload
```

### Query Documents

```http
POST /api/query
```

Example:

```json
{
    "question": "What is RAG?",
    "top_k": 6
}
```

### List Documents

```http
GET /api/documents
```

### Delete Document

```http
DELETE /api/documents/<document_name>
```

### Clear All Documents

```http
POST /api/clear
```

### Health Check

```http
GET /api/health
```

## Models

### Embeddings

```txt
sentence-transformers/all-MiniLM-L6-v2
```

### Reranker

```txt
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### LLM

```txt
google/flan-t5-base
```

## License

MIT
