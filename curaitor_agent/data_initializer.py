"""Command-line entry point to prime the Curaitor PDF database."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import yaml

try:  # pragma: no cover - import guard for script vs module execution
    from curaitor_agent.pdf_repository import (
        DEFAULT_DB_PATH,
        collect_documents_from_directory,
        store_documents,
    )
except ModuleNotFoundError:  # Running as ``python curaitor_agent/data_initializer.py``
    from pdf_repository import (  # type: ignore[F401]
        DEFAULT_DB_PATH,
        collect_documents_from_directory,
        store_documents,
    )


def _load_config_defaults() -> dict[str, object]:
    """Return default values derived from ``config.yaml`` when available."""

    config_path = Path("config.yaml")
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    defaults: dict[str, object] = {}
    try:
        sources = config.get("source", [])
        if isinstance(sources, list) and sources:
            pdf_path = sources[0].get("pdf_path") if isinstance(sources[0], dict) else None
            if pdf_path:
                defaults["pdf_dir"] = pdf_path
    except AttributeError:
        pass

    rag_cfg = config.get("rag", {}) if isinstance(config, dict) else {}
    if isinstance(rag_cfg, dict):
        if "chunk_size" in rag_cfg:
            defaults["chunk_size"] = rag_cfg["chunk_size"]
        if "overlap" in rag_cfg:
            defaults["chunk_overlap"] = rag_cfg["overlap"]
        if "embedding_model" in rag_cfg:
            defaults["embedding_model"] = rag_cfg["embedding_model"]
        if "embedding_prefix" in rag_cfg:
            defaults["embedding_prefix"] = rag_cfg["embedding_prefix"]

    return defaults


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    defaults = _load_config_defaults()

    parser = argparse.ArgumentParser(
        description="Load PDFs into the Curaitor SQLite cache so they are ready for RAG queries.",
    )
    parser.add_argument(
        "--pdf-dir",
        default=defaults.get("pdf_dir", "data/papers"),
        help="Directory containing PDF files to ingest (defaults to config.yaml source path).",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database file to initialise.",
    )
    parser.add_argument(
        "--max-pdfs",
        type=int,
        default=100,
        help="Maximum number of PDFs to parse from the directory.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=10,
        help="Maximum number of pages to read per PDF.",
    )
    parser.add_argument(
        "--per-page-chars",
        type=int,
        default=None,
        help="Optional character cap applied to each page while parsing.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Optional character cap for the full PDF text.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=defaults.get("chunk_size", 1000),
        help="Token chunk size to use when storing documents.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=defaults.get("chunk_overlap", 100),
        help="Overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        default=defaults.get("embedding_model"),
        help="Optional sentence-transformers model name for embedding storage.",
    )
    parser.add_argument(
        "--embedding-prefix",
        default=defaults.get("embedding_prefix"),
        help="Prompt prefix applied before creating embeddings (ignored when no model).",
    )

    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = _parse_args(argv)

    pdf_dir = Path(args.pdf_dir).expanduser().resolve()
    if not pdf_dir.exists():
        raise SystemExit(f"PDF directory '{pdf_dir}' does not exist")

    documents = collect_documents_from_directory(
        pdf_dir,
        max_pdfs=args.max_pdfs,
        max_pages=args.max_pages,
        per_page_chars=args.per_page_chars,
        max_chars=args.max_chars,
    )

    if not documents:
        print(f"No PDFs parsed from {pdf_dir}. Nothing to store.")
        return

    db_path = Path(args.db_path).expanduser().resolve()

    embedding_prefix = args.embedding_prefix
    if args.embedding_model and not embedding_prefix:
        embedding_prefix = "search_document: "

    store_documents(
        documents,
        db_path=db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        embedding_prefix=embedding_prefix if args.embedding_model else None,
    )

    print("Initialisation complete:")
    print(f"  Database: {db_path}")
    print(f"  Documents stored: {len(documents)}")
    print(f"  Chunk size / overlap: {args.chunk_size} / {args.chunk_overlap}")
    if args.embedding_model:
        print(f"  Embeddings: {args.embedding_model} (prefix='{embedding_prefix}')")
    else:
        print("  Embeddings: disabled")


if __name__ == "__main__":
    main()
