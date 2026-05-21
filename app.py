import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from rag_engine import DocumentStore

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".md", ".rst", ".csv", ".json", ".xml", ".html", ".htm"}
MAX_FILE_SIZE_MB = 50

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.urandom(32)

store = DocumentStore()


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def err(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_files():
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return err("No files provided.")

    results = []
    for f in files:
        if not f or f.filename == "":
            continue
        original_name = f.filename
        if not allowed_file(original_name):
            results.append({
                "name": original_name,
                "success": False,
                "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            })
            continue

        safe_name = secure_filename(original_name)
        # Ensure unique path to avoid collisions
        unique_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
        f.save(str(unique_path))

        result = store.add_document(str(unique_path), original_name)
        results.append(result)

    return jsonify({"results": results})


@app.route("/api/documents", methods=["GET"])
def list_documents():
    docs = store.list_documents()
    return jsonify({"documents": docs, "total": len(docs)})


@app.route("/api/documents/<path:doc_name>", methods=["DELETE"])
def delete_document(doc_name: str):
    if doc_name not in store.documents:
        return err("Document not found.", 404)
    store.remove_document(doc_name)
    return jsonify({"success": True, "message": f"'{doc_name}' removed."})


@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return err("Question cannot be empty.")
    if len(question) > 2000:
        return err("Question too long (max 2000 characters).")

    doc_filter = data.get("doc_filter")  # list of doc names or None
    top_k = min(int(data.get("top_k", 6)), 12)

    result = store.query(question, top_k=top_k, doc_filter=doc_filter or None)
    return jsonify(result)


@app.route("/api/clear", methods=["POST"])
def clear_all():
    store.clear()
    return jsonify({"success": True, "message": "All documents cleared."})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "documents": len(store.documents),
        "chunks": len(store.all_chunks),
    })


if __name__ == "__main__":
    print("=" * 60)
    print("  RAG Document QA System")
    print("  Running at http://localhost:50005")
    print("  Supported: PDF, DOCX, PPTX, TXT, MD, CSV, JSON, HTML")
    print("=" * 60)
    app.run(debug=False, host="0.0.0.0", port=50005)
