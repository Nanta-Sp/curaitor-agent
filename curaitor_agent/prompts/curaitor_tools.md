# Curaitor MCP Tool Reference

The Curaitor stack exposes the following MCP tools via `curaitor_agent/curaitor_mcp_server.py`. Use them to orchestrate
literature discovery, ingestion, retrieval, and notifications. Unless noted otherwise, parameters are optional and
fall back to `config.yaml` defaults.

## Discovery and ingestion
- **`extract_keywords_only`** — Run the LLM keyword extractor against a natural-language topic and return a ranked list of
  suggested search terms.
- **`search_arxiv_titles_only`** — Query arXiv using explicit keywords and return lightweight metadata (title, link,
  summary) without downloading PDFs.
- **`download_specific_papers`** — Download one or more PDF URLs directly into the configured papers directory and report
  success/failure for each file.
- **`ingest_local_pdfs`** — Parse locally available PDFs (with optional limits on count, pages, or character lengths) and
  persist chunks plus embeddings into the SQLite repository. Use this after manual downloads or when seeding the cache.
- **`refresh_arxiv_feed`** — End-to-end workflow: extract keywords from a topic, search arXiv, download the newest papers,
  store them, and optionally send an email summary via Gmail.

## Retrieval and analysis
- **`chat_with_repository`** — Retrieve the most relevant stored chunks with FAISS (falling back to keyword search when
  no embeddings exist) and answer a question with detailed context metadata.
- **`quick_pdf_search`** — Perform a fast filename-based relevance scan across the local PDF directory.
- **`extract_pdf_text_only`** — Pull cleaned text for a single PDF without performing a full ingestion run.
- **`simple_qa_from_text`** — Answer a question directly against caller-provided text, bypassing the repository entirely.

## Communication
- **`send_email_tool`** — Deliver plain-text or HTML messages through Gmail. Returns structured success metadata or an
  explicit authentication error with an OAuth URL.

## Scheduling
- **`add_daily_job`** — Ensure the APScheduler service is running and schedule a persistent daily refresh job for a given
  topic at `hour:minute`.
- **`delete_job`** — Remove a scheduled job by ID.
- **`jobs`** — List all scheduled jobs currently persisted by the scheduler service.
