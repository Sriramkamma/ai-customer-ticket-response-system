"""Local-only customer support ticket application and HTTP entry point."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import rag
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from database import db as database
from services import ml_service
from services.field_service import validate_dynamic

ROOT = Path(__file__).resolve().parent
DB_PATH = database.DB_PATH
DATASET = ml_service.DATASET
KB_DIR = ROOT / "knowledge_base"
LLM_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
LLM_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
SEMANTIC_MIN_SCORE = rag.MIN_SCORE
SEMANTIC_TOP_K = rag.TOP_K
_embedding_model = None
_faiss_index = None
_semantic_documents = []
_semantic_signature = None

def db():
    """Compatibility wrapper that honors callers overriding server.DB_PATH."""
    database.DB_PATH = DB_PATH
    return database.db()

class Predictor(ml_service.Predictor):
    """Compatibility wrapper using the current server-level dataset path."""
    def __init__(self):
        super().__init__(DATASET)

def now():
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def retrieve(text, limit=SEMANTIC_TOP_K, min_score=SEMANTIC_MIN_SCORE):
    """Return matching documents, preserving the original public interface."""
    documents, _error = retrieve_with_status(text, limit, min_score)
    return documents


def retrieve_with_status(text, limit=SEMANTIC_TOP_K, min_score=SEMANTIC_MIN_SCORE):
    """
    Retrieve documents and distinguish no match from a retrieval failure.

    Use the shared RAG API so ticket creation and the diagnostic use the same
    embedding, scoring, threshold, and top-k behavior. Cache the model and
    index for reuse; rebuild the index when the Markdown files change.
    """
    global _embedding_model, _faiss_index, _semantic_documents
    global _semantic_signature

    if not isinstance(text, str) or not text.strip():
        return [], None

    if not KB_DIR.exists():
        return [], None

    try:
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(rag.MODEL_NAME)

        paths = sorted(KB_DIR.glob("*.md"))
        signature = tuple(
            (path.name, path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
        )

        if _faiss_index is None or signature != _semantic_signature:
            documents = []
            for path in paths:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    documents.append({"source": path.name, "text": content})

            if not documents:
                _semantic_documents = []
                _faiss_index = None
                _semantic_signature = signature
                return [], None

            index = rag.build_index(documents, _embedding_model)

            _semantic_documents = documents
            _faiss_index = index
            _semantic_signature = signature

        results = rag.search_knowledge_base(
            question=text,
            documents=_semantic_documents,
            model=_embedding_model,
            index=_faiss_index,
            top_k=limit,
            min_score=min_score,
        )
        matched_documents = [
            {
                "source": document["source"],
                "content": document["text"],
                "score": document["score"],
            }
            for document in results
        ]
        return matched_documents, None

    except Exception as error:
        # Retrieval failures must not crash ticket creation, but keep their
        # status separate from a successful search with no relevant matches.
        print(f"Semantic retrieval unavailable: {error}")
        return [], error


def build_retrieval_query(ticket, solution):
    """Combine ticket details and the predicted solution for semantic search."""
    parts = [
        ticket.get("subject", ""),
        ticket.get("description", ""),
        solution,
    ]
    return " ".join(
        part.strip()
        for part in parts
        if isinstance(part, str) and part.strip()
    )


# ============================================================
# Customer response generation with local Ollama
# ============================================================

def grounded_response(ticket, solution, docs, retrieval_failed=False):
    """Compatibility wrapper for the shared local RAG response generator."""
    return rag.grounded_response(
        ticket,
        solution,
        docs,
        retrieval_failed=retrieval_failed,
        llm_url=LLM_URL,
        llm_model=LLM_MODEL,
    )


# ============================================================
# Dynamic field validation
# ============================================================



def serialize(row):
    """Convert a SQLite row to a JSON-friendly dictionary."""
    item = dict(row)
    item["dynamic_values"] = json.loads(
        item["dynamic_values"]
    )
    return item


# ============================================================
# HTTP API handler
# ============================================================

class Handler(BaseHTTPRequestHandler):
    """Handle ticket, dynamic-field, and static-file requests."""

    def log_message(self, fmt, *args):
        print(
            "%s - %s"
            % (self.address_string(), fmt % args)
        )

    def send_json(self, code, obj):
        data = json.dumps(
            obj,
            ensure_ascii=False,
        ).encode("utf-8")

        self.send_response(code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"
        return json.loads(raw_body)

    # --------------------------------------------------------
    # GET requests
    # --------------------------------------------------------

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/tickets":
            connection = db()
            try:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM tickets
                    WHERE active = 1
                    ORDER BY updated_at DESC
                    """
                ).fetchall()

                return self.send_json(
                    200,
                    [serialize(row) for row in rows],
                )
            finally:
                connection.close()

        if path == "/api/fields":
            connection = db()
            try:
                rows = connection.execute(
                    """
                    SELECT *
                    FROM field_definitions
                    WHERE active = 1
                    ORDER BY id
                    """
                ).fetchall()

                fields = []

                for row in rows:
                    field = dict(row)
                    field["required"] = bool(field["required"])
                    field["options"] = json.loads(field["options"])
                    fields.append(field)

                return self.send_json(200, fields)
            finally:
                connection.close()

        match = re.fullmatch(r"/api/tickets/(\d+)", path)

        if match:
            connection = db()
            try:
                row = connection.execute(
                    """
                    SELECT *
                    FROM tickets
                    WHERE id = ? AND active = 1
                    """,
                    (match[1],),
                ).fetchone()

                if row:
                    return self.send_json(200, serialize(row))

                return self.send_json(
                    404,
                    {"error": "Ticket not found"},
                )
            finally:
                connection.close()

        if path == "/":
            file_path = ROOT / "static" / "index.html"

            if not file_path.exists():
                return self.send_json(
                    404,
                    {"error": "Frontend file not found"},
                )

            data = file_path.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8",
            )
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        if path in ("/app.js", "/styles.css"):
            file_path = ROOT / "static" / path.lstrip("/")

            if not file_path.exists():
                return self.send_json(
                    404,
                    {"error": "Static file not found"},
                )

            data = file_path.read_bytes()
            content_type = (
                "text/javascript; charset=utf-8"
                if path.endswith(".js")
                else "text/css; charset=utf-8"
            )

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            return self.wfile.write(data)

        return self.send_json(
            404,
            {"error": "Not found"},
        )

    # --------------------------------------------------------
    # POST requests
    # --------------------------------------------------------

    def do_POST(self):
        path = urlparse(self.path).path

        try:
            data = self.body()
        except (ValueError, UnicodeDecodeError):
            return self.send_json(
                400,
                {"error": "Invalid JSON body"},
            )

        if not isinstance(data, dict):
            return self.send_json(
                400,
                {"error": "Request body must be a JSON object"},
            )

        connection = db()

        # Create a dynamic field.
        if path == "/api/fields":
            name = data.get("name")
            label = data.get("label")
            field_type = data.get("type")
            options = data.get("options", [])
            required = data.get("required", False)

            if (
                not isinstance(name, str)
                or not name.strip()
                or not isinstance(label, str)
                or not label.strip()
                or field_type not in (
                    "text",
                    "dropdown",
                    "number",
                    "date",
                    "checkbox",
                )
                or not isinstance(options, list)
                or not isinstance(required, bool)
            ):
                connection.close()
                return self.send_json(
                    400,
                    {
                        "error": (
                            "Field name, label, supported type, "
                            "and valid options are required"
                        )
                    },
                )

            if field_type == "dropdown" and (
                not options
                or not all(
                    isinstance(option, str) and option.strip()
                    for option in options
                )
            ):
                connection.close()
                return self.send_json(
                    400,
                    {"error": "Dropdown fields need non-empty text options"},
                )

            try:
                cursor = connection.execute(
                    """
                    INSERT INTO field_definitions
                        (name, label, type, required, options)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        name.strip(),
                        label.strip(),
                        field_type,
                        int(required),
                        json.dumps(options),
                    ),
                )

                connection.commit()
                field_id = cursor.lastrowid

            except sqlite3.IntegrityError:
                connection.close()
                return self.send_json(
                    409,
                    {"error": "Field name already exists"},
                )

            connection.close()
            return self.send_json(201, {"id": field_id})

        # Create a ticket.
        if path == "/api/tickets":
            required_fields = [
                "customer_name",
                "subject",
                "description",
            ]

            missing = [
                key
                for key in required_fields
                if not isinstance(data.get(key), str)
                or not data[key].strip()
            ]

            if missing:
                connection.close()
                return self.send_json(
                    400,
                    {
                        "error": (
                            "Required ticket fields: "
                            + ", ".join(missing)
                        )
                    },
                )

            field_rows = connection.execute(
                """
                SELECT *
                FROM field_definitions
                WHERE active = 1
                """
            ).fetchall()

            fields = []

            for row in field_rows:
                field = dict(row)
                field["options"] = json.loads(field["options"])
                field["required"] = bool(field["required"])
                fields.append(field)

            dynamic_values = data.get("dynamic_values", {})

            if not isinstance(dynamic_values, dict):
                connection.close()
                return self.send_json(
                    400,
                    {"error": "Dynamic values must be a JSON object"},
                )

            errors = validate_dynamic(dynamic_values, fields)

            if errors:
                connection.close()
                return self.send_json(
                    400,
                    {"error": "; ".join(errors)},
                )

            subject = data["subject"].strip()
            description = data["description"].strip()
            ticket_text = f"{subject} {description}"

            solution = Predictor().predict(ticket_text)

            ticket = {
                "subject": subject,
                "description": description,
            }
            retrieval_query = build_retrieval_query(ticket, solution)
            knowledge_docs, retrieval_error = retrieve_with_status(
                retrieval_query
            )

            response_text, engine, sources = grounded_response(
                ticket,
                solution,
                knowledge_docs,
                retrieval_failed=retrieval_error is not None,
            )

            timestamp = now()

            cursor = connection.execute(
                """
                INSERT INTO tickets (
                    customer_name,
                    subject,
                    description,
                    category,
                    priority,
                    predicted_solution,
                    response,
                    status,
                    created_at,
                    updated_at,
                    dynamic_values
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["customer_name"].strip(),
                    subject,
                    description,
                    data.get("category", "Other"),
                    data.get("priority", "Medium"),
                    solution,
                    response_text,
                    data.get("status", "Open"),
                    timestamp,
                    timestamp,
                    json.dumps(dynamic_values),
                ),
            )

            connection.commit()

            row = connection.execute(
                "SELECT * FROM tickets WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()

            connection.close()

            result = serialize(row)
            result["ai_engine"] = engine
            result["retrieved_sources"] = sources

            return self.send_json(201, result)

        # Update an existing ticket.
        match = re.fullmatch(r"/api/tickets/(\d+)", path)

        if match:
            row = connection.execute(
                """
                SELECT *
                FROM tickets
                WHERE id = ? AND active = 1
                """,
                (match[1],),
            ).fetchone()

            if not row:
                connection.close()
                return self.send_json(
                    404,
                    {"error": "Ticket not found"},
                )

            allowed_fields = [
                "customer_name",
                "subject",
                "description",
                "category",
                "priority",
                "status",
                "response",
            ]

            updates = {
                key: data[key]
                for key in allowed_fields
                if key in data
            }

            if "dynamic_values" in data:
                dynamic_values = data["dynamic_values"]

                if not isinstance(dynamic_values, dict):
                    connection.close()
                    return self.send_json(
                        400,
                        {
                            "error": (
                                "Dynamic values must be a JSON object"
                            )
                        },
                    )

                field_rows = connection.execute(
                    """
                    SELECT *
                    FROM field_definitions
                    WHERE active = 1
                    """
                ).fetchall()

                fields = []

                for field_row in field_rows:
                    field = dict(field_row)
                    field["options"] = json.loads(field["options"])
                    field["required"] = bool(field["required"])
                    fields.append(field)

                errors = validate_dynamic(
                    dynamic_values,
                    fields,
                    require_all=False,
                )

                if errors:
                    connection.close()
                    return self.send_json(
                        400,
                        {"error": "; ".join(errors)},
                    )

                updates["dynamic_values"] = json.dumps(
                    dynamic_values
                )

            subject = updates.get("subject", row["subject"])
            description = updates.get(
                "description",
                row["description"],
            )

            # Regenerate prediction and response when ticket text changes.
            if (
                "subject" in updates or "description" in updates
            ) and "response" not in updates:
                ticket_text = f"{subject} {description}"
                solution = Predictor().predict(ticket_text)
                ticket = {
                    "subject": subject,
                    "description": description,
                }
                retrieval_query = build_retrieval_query(ticket, solution)
                knowledge_docs, retrieval_error = retrieve_with_status(
                    retrieval_query
                )

                response_text, _, _ = grounded_response(
                    ticket,
                    solution,
                    knowledge_docs,
                    retrieval_failed=retrieval_error is not None,
                )

                updates["predicted_solution"] = solution
                updates["response"] = response_text

            if updates:
                updates["updated_at"] = now()

                assignments = ", ".join(
                    f"{key} = ?"
                    for key in updates
                )

                connection.execute(
                    f"UPDATE tickets SET {assignments} WHERE id = ?",
                    (*updates.values(), match[1]),
                )

            connection.commit()

            updated_row = connection.execute(
                "SELECT * FROM tickets WHERE id = ?",
                (match[1],),
            ).fetchone()

            connection.close()

            return self.send_json(
                200,
                serialize(updated_row),
            )

        connection.close()
        return self.send_json(
            404,
            {"error": "Not found"},
        )

    # --------------------------------------------------------
    # DELETE requests
    # --------------------------------------------------------

    def do_DELETE(self):
        match = re.fullmatch(
            r"/api/tickets/(\d+)",
            urlparse(self.path).path,
        )

        if not match:
            return self.send_json(
                404,
                {"error": "Not found"},
            )

        connection = db()

        cursor = connection.execute(
            """
            UPDATE tickets
            SET active = 0, updated_at = ?
            WHERE id = ? AND active = 1
            """,
            (now(), match[1]),
        )

        connection.commit()
        deleted = bool(cursor.rowcount)
        connection.close()

        return self.send_json(
            200,
            {"deleted": deleted},
        )

    def do_PUT(self):
        """Support the existing API's PUT update route."""
        self.do_POST()


# ============================================================
# Application entry point
# ============================================================

def main():
    """Initialize the database and start the local HTTP server."""
    db().close()

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))

    print(
        f"Customer Ticket System running at "
        f"http://{host}:{port} (local AI only)"
    )

    server = ThreadingHTTPServer((host, port), Handler)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
