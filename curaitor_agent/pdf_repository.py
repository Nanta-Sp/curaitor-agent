"""Utilities for parsing PDFs and storing their contents in a local SQLite database.

This module keeps the PDF ingestion logic separate from the MCP server so the
agent file stays focused on tool wiring.  It provides helpers to:

* Parse PDF files into lightweight :class:`ParsedDocument` objects.
* Persist parsed documents in a simple SQLite database for reuse.
* Inspect previously stored documents.

The database location defaults to ``data/tracker/pdf_cache.sqlite`` but can be
overridden when calling the helper functions.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence
from functools import lru_cache
from typing import Optional

import numpy as np
from pypdf import PdfReader

DEFAULT_DB_PATH = Path("data/tracker/pdf_cache.sqlite")


@dataclass(slots=True)
class ParsedDocument:
    """Container for the parsed contents of a single PDF document."""

    arxiv_id: str
    filename: str
    path: str
    text: str
    pages_processed: int
    total_pages: int

    @property
    def text_length(self) -> int:
        """Return the length of the extracted text."""

        return len(self.text)

    def to_dict(self) -> dict[str, object]:
        """Convert the document into a serialisable dictionary."""

        return {
            "arxiv_id": self.arxiv_id,
            "filename": self.filename,
            "path": self.path,
            "text": self.text,
            "pages_processed": self.pages_processed,
            "total_pages": self.total_pages,
            "text_length": self.text_length,
        }


@dataclass(slots=True)
class DocumentChunk:
    """Represents a chunk of a parsed document and its embedding."""

    arxiv_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    embedding: Optional[tuple[float, ...]] = None
    embedding_model: Optional[str] = None
    embedding_prefix: Optional[str] = None

    @property
    def text_length(self) -> int:
        return len(self.text)

    def to_db_params(self) -> tuple:
        """Return a tuple aligned with the ``pdf_chunks`` schema."""

        if self.embedding is not None:
            vector = np.asarray(self.embedding, dtype="float32")
            embedding_blob = sqlite3.Binary(vector.tobytes())
            embedding_dim = vector.shape[0]
        else:
            embedding_blob = None
            embedding_dim = None

        return (
            self.arxiv_id,
            self.chunk_index,
            self.start_char,
            self.end_char,
            self.text,
            self.text_length,
            embedding_blob,
            embedding_dim,
            self.embedding_model,
            self.embedding_prefix,
        )


_ARXIV_ID_PATTERN = re.compile(r"\d{4}\.\d{4,5}")


def _infer_arxiv_id(stem: str) -> str:
    """Attempt to infer an arXiv identifier from a filename stem."""

    cleaned = stem.replace("_", "").lower()
    for suffix in ("v3", "v2", "v1"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
            break
    match = _ARXIV_ID_PATTERN.search(cleaned)
    if match:
        return match.group(0)
    return stem


def parse_pdf_file(
    pdf_file: Path,
    *,
    max_pages: int = 5,
    per_page_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    include_page_headers: bool = False,
) -> Optional[ParsedDocument]:
    """Parse a single PDF file into a :class:`ParsedDocument`.

    Args:
        pdf_file: Path to the PDF file.
        max_pages: Maximum number of pages to attempt to read.
        per_page_chars: Optional limit applied to the extracted text of each
            page.  If ``None`` the full page text is used.
        max_chars: Optional total character limit applied to the concatenated
            document text.
        include_page_headers: When ``True`` each extracted page is prefixed with
            a ``"--- Page N ---"`` marker to preserve page boundaries.

    Returns:
        A :class:`ParsedDocument` when text could be extracted, otherwise
        ``None``.
    """

    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        return None

    try:
        with pdf_path.open("rb") as fh:
            reader = PdfReader(fh)
            total_pages = len(reader.pages)
            text_parts: list[str] = []
            pages_processed = 0

            for page_index, page in enumerate(reader.pages[:max_pages], start=1):
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    # Skip problematic pages but continue parsing others.
                    page_text = ""

                if not page_text.strip():
                    pages_processed += 1
                    continue

                page_text = page_text.strip()
                if per_page_chars is not None and len(page_text) > per_page_chars:
                    page_text = page_text[:per_page_chars]

                if include_page_headers:
                    header = f"--- Page {page_index} ---\n"
                else:
                    header = ""

                text_parts.append(f"{header}{page_text}")
                pages_processed += 1

            if not text_parts:
                return None

            text = "\n\n".join(text_parts).strip()
            if max_chars is not None and len(text) > max_chars:
                text = text[:max_chars]

    except Exception:
        return None

    return ParsedDocument(
        arxiv_id=_infer_arxiv_id(pdf_path.stem),
        filename=pdf_path.name,
        path=str(pdf_path),
        text=text,
        pages_processed=pages_processed,
        total_pages=total_pages,
    )


def collect_documents_from_directory(
    pdf_dir: Path,
    *,
    max_pdfs: int = 3,
    max_pages: int = 5,
    per_page_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> list[ParsedDocument]:
    """Parse multiple PDFs from a directory into ``ParsedDocument`` objects."""

    directory = Path(pdf_dir)
    if not directory.exists():
        return []

    documents: list[ParsedDocument] = []
    for pdf_file in sorted(directory.glob("*.pdf"))[:max_pdfs]:
        doc = parse_pdf_file(
            pdf_file,
            max_pages=max_pages,
            per_page_chars=per_page_chars,
            max_chars=max_chars,
        )
        if doc is not None:
            documents.append(doc)

    return documents


def _chunk_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    arxiv_id: str,
) -> list[DocumentChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")

    step = max(1, chunk_size - overlap)
    chunks: list[DocumentChunk] = []
    start = 0
    index = 0
    text_length = len(text)

    while start < text_length:
        end = min(text_length, start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                DocumentChunk(
                    arxiv_id=arxiv_id,
                    chunk_index=index,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                )
            )
            index += 1
        if end == text_length:
            break
        start = end - overlap if overlap < chunk_size else end
        if start <= 0 and end == text_length:
            break
    return chunks


@lru_cache(maxsize=2)
def _load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, trust_remote_code=True)


def embed_texts(texts: Sequence[str], *, model_name: str) -> list[tuple[float, ...]]:
    """Encode ``texts`` with the configured embedding model."""

    if not texts:
        return []

    model = _load_embedding_model(model_name)
    vectors = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    array = np.asarray(vectors, dtype="float32")
    return [tuple(row.tolist()) for row in array]


def _ensure_connection(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_documents (
            arxiv_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            path TEXT NOT NULL,
            text TEXT NOT NULL,
            text_length INTEGER NOT NULL,
            pages_processed INTEGER NOT NULL,
            total_pages INTEGER NOT NULL,
            stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pdf_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            arxiv_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            text TEXT NOT NULL,
            text_length INTEGER NOT NULL,
            embedding BLOB,
            embedding_dim INTEGER,
            embedding_model TEXT,
            embedding_prefix TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(arxiv_id, chunk_index),
            FOREIGN KEY (arxiv_id) REFERENCES pdf_documents(arxiv_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_pdf_chunks_model
        ON pdf_chunks(embedding_model, arxiv_id, chunk_index)
        """
    )
    return conn


def store_documents(
    documents: Iterable[ParsedDocument],
    *,
    db_path: Path = DEFAULT_DB_PATH,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    embedding_model: Optional[str] = None,
    embedding_prefix: Optional[str] = None,
) -> None:
    """Persist parsed documents in the local SQLite database."""

    docs = list(documents)
    if not docs:
        return

    conn = _ensure_connection(db_path)
    try:
        with conn:
            for doc in docs:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO pdf_documents (
                        arxiv_id, filename, path, text, text_length,
                        pages_processed, total_pages
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.arxiv_id,
                        doc.filename,
                        doc.path,
                        doc.text,
                        doc.text_length,
                        doc.pages_processed,
                        doc.total_pages,
                    ),
                )

                conn.execute(
                    "DELETE FROM pdf_chunks WHERE arxiv_id = ?",
                    (doc.arxiv_id,),
                )

                chunks = _chunk_text(
                    doc.text,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                    arxiv_id=doc.arxiv_id,
                )

                if embedding_model and chunks:
                    texts_for_embedding = [
                        f"{embedding_prefix}{chunk.text}" if embedding_prefix else chunk.text
                        for chunk in chunks
                    ]
                    embeddings = embed_texts(
                        texts_for_embedding,
                        model_name=embedding_model,
                    )
                    for chunk, vector in zip(chunks, embeddings):
                        chunk.embedding = vector
                        chunk.embedding_model = embedding_model
                        chunk.embedding_prefix = embedding_prefix
                else:
                    for chunk in chunks:
                        chunk.embedding_model = embedding_model
                        chunk.embedding_prefix = embedding_prefix

                for chunk in chunks:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO pdf_chunks (
                            arxiv_id, chunk_index, start_char, end_char, text,
                            text_length, embedding, embedding_dim, embedding_model,
                            embedding_prefix
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        chunk.to_db_params(),
                    )
    finally:
        conn.close()


def list_stored_documents(*, db_path: Path = DEFAULT_DB_PATH) -> list[dict[str, object]]:
    """Return metadata for documents stored in the database."""

    db_file = Path(db_path)
    if not db_file.exists():
        return []

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT d.arxiv_id, d.filename, d.path, d.text_length,
                   d.pages_processed, d.total_pages, d.stored_at,
                   COUNT(c.id) AS chunk_count
            FROM pdf_documents AS d
            LEFT JOIN pdf_chunks AS c ON c.arxiv_id = d.arxiv_id
            GROUP BY d.arxiv_id
            ORDER BY d.stored_at DESC
            """
        ).fetchall()
    finally:
        conn.close()

    return [dict(row) for row in rows]


def iter_document_texts(
    *, db_path: Path = DEFAULT_DB_PATH
) -> Iterator[tuple[str, str]]:
    """Yield ``(arxiv_id, text)`` tuples for all stored documents."""

    db_file = Path(db_path)
    if not db_file.exists():
        return iter(())

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT arxiv_id, text FROM pdf_documents").fetchall()
    finally:
        conn.close()

    return ((row["arxiv_id"], row["text"]) for row in rows)


def fetch_document_chunks(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    arxiv_ids: Optional[Iterable[str]] = None,
    embedding_model: Optional[str] = None,
) -> list[DocumentChunk]:
    """Retrieve stored document chunks matching the given filters."""

    db_file = Path(db_path)
    if not db_file.exists():
        return []

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        query = (
            "SELECT arxiv_id, chunk_index, start_char, end_char, text, "
            "text_length, embedding, embedding_dim, embedding_model, embedding_prefix "
            "FROM pdf_chunks"
        )
        conditions: list[str] = []
        params: list[object] = []

        if arxiv_ids:
            arxiv_list = list(arxiv_ids)
            if arxiv_list:
                placeholders = ",".join("?" for _ in arxiv_list)
                conditions.append(f"arxiv_id IN ({placeholders})")
                params.extend(arxiv_list)

        if embedding_model is not None:
            conditions.append("embedding_model = ?")
            params.append(embedding_model)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY arxiv_id, chunk_index"

        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    chunks: list[DocumentChunk] = []
    for row in rows:
        embedding_blob = row["embedding"]
        embedding_dim = row["embedding_dim"]
        if embedding_blob is not None and embedding_dim:
            vector = tuple(
                np.frombuffer(embedding_blob, dtype="float32", count=embedding_dim).tolist()
            )
        else:
            vector = None

        chunks.append(
            DocumentChunk(
                arxiv_id=row["arxiv_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                start_char=row["start_char"],
                end_char=row["end_char"],
                embedding=vector,
                embedding_model=row["embedding_model"],
                embedding_prefix=row["embedding_prefix"],
            )
        )
    return chunks


def fetch_document_by_path(
    path: Path | str,
    *,
    db_path: Path = DEFAULT_DB_PATH,
) -> Optional[ParsedDocument]:
    """Retrieve a stored document by its original file path."""

    db_file = Path(db_path)
    if not db_file.exists():
        return None

    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT arxiv_id, filename, path, text, pages_processed, total_pages
            FROM pdf_documents
            WHERE path = ?
            """,
            (str(path),),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return ParsedDocument(
        arxiv_id=row["arxiv_id"],
        filename=row["filename"],
        path=row["path"],
        text=row["text"],
        pages_processed=row["pages_processed"],
        total_pages=row["total_pages"],
    )


__all__ = [
    "DEFAULT_DB_PATH",
    "DocumentChunk",
    "ParsedDocument",
    "collect_documents_from_directory",
    "embed_texts",
    "fetch_document_chunks",
    "fetch_document_by_path",
    "iter_document_texts",
    "list_stored_documents",
    "parse_pdf_file",
    "store_documents",
]

