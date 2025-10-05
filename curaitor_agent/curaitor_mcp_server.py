
# This is a sample MCP server that provides tools to search for papers on arXiv
# and extract information about specific papers. It uses the `arxiv` library to
# Nanta, Shichuan
# Sep 2025
import arxiv
import json
import os
from typing import Iterable, List
from mcp.server.fastmcp import FastMCP
# from arxiv_search_chunk import download_arxiv_pdfs
# from arxiv_search_chunk import get_keywords_from_llm
import yaml
import subprocess
from pathlib import Path
import importlib.util
import requests
import time as time_module
from dateutil import tz
from datetime import datetime, timedelta, timezone, time
from typing import Optional
import tiktoken
import faiss
from collections import defaultdict
from io import BytesIO
from mcp.server.fastmcp.prompts.base import Message
from mcp.types import TextContent
from gmail_send import AuthRequired, gmail_send
import logging
import numpy as np
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from scheduler_service import (
    schedule_daily_job,
    remove_job,
    list_jobs,
    get_scheduler,  # optional: to ensure scheduler is initialized on demand
)
# Local PDF ingestion helpers
from pdf_repository import (
    DEFAULT_DB_PATH,
    DocumentChunk,
    collect_documents_from_directory,
    embed_texts,
    fetch_document_chunks,
    iter_document_texts,
    list_stored_documents,
    parse_pdf_file,
    store_documents,
)
# import argparse

config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
save_dir = config['source'][0]['pdf_path']
# PAPER_DIR = "data/agent_papers"
PAPER_DIR = save_dir

mcp = FastMCP("curaitor_mcp_server")

config = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
# print(config['llm'][1]['model'])
provider = config['llm'][0]['provider']
raw_model = config['llm'][1]['model']


def _get_config_value(section: str, key: str, default=None):
    data = config.get(section, {})
    if isinstance(data, dict):
        return data.get(key, default)
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and key in item:
                return item[key]
    return default


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _resolve_store_parameters(
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None,
    embedding_prefix: Optional[str] = None,
) -> tuple[int, int, Optional[str], Optional[str]]:
    """Combine explicit overrides with ``config.yaml`` defaults."""

    default_chunk_size = _as_int(_get_config_value("rag", "chunk_size", 1000), 1000)
    default_overlap = _as_int(_get_config_value("rag", "overlap", 100), 100)
    default_model = _get_config_value("rag", "embedding_model", None)
    default_prefix = _get_config_value("rag", "embedding_prefix", None)

    chunk_size_val = chunk_size if chunk_size is not None else default_chunk_size
    overlap_val = chunk_overlap if chunk_overlap is not None else default_overlap
    model_val = embedding_model if embedding_model is not None else default_model
    prefix_val = embedding_prefix if embedding_prefix is not None else default_prefix

    if model_val and not prefix_val:
        prefix_val = "search_document: "

    return chunk_size_val, overlap_val, model_val, prefix_val if model_val else None


def _store_documents_with_defaults(
    documents: Iterable,
    *,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None,
    embedding_prefix: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    docs = list(documents)
    if not docs:
        return None, None

    chunk_size_val, overlap_val, model_val, prefix_val = _resolve_store_parameters(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        embedding_prefix=embedding_prefix,
    )

    store_documents(
        docs,
        chunk_size=chunk_size_val,
        chunk_overlap=overlap_val,
        embedding_model=model_val,
        embedding_prefix=prefix_val,
    )

    return model_val, prefix_val

if provider == 'openrouter':
    api_key = os.getenv("OPENROUTER_API_KEY")
elif provider == "openai":
    api_key = os.getenv("OPENAI_API_KEY")
elif provider == "google":
    api_key = os.getenv("GOOGLE_API_KEY")
else:
    raise ValueError(f"Unsupported LLM provider: {provider}")
if not api_key:
    raise ValueError(f"Missing API key for provider '{provider}'. Set the appropriate env var.")

def _normalize_model_for_provider(p: str, m: str | None) -> str | None:
    if not m:
        return None
    m = m.strip()
    # For direct providers, drop OpenRouter-style prefixes/suffixes
    if p in ("openai", "google"):
        if "/" in m:
            m = m.split("/")[-1]  # e.g., "openai/gpt-4o-mini" -> "gpt-4o-mini"
        if ":" in m:
            m = m.split(":")[0]   # e.g., "...:free" -> base model
    return m

LLM_MODEL = _normalize_model_for_provider(provider, raw_model)
# Reasonable fallbacks if model omitted
if not LLM_MODEL:
    LLM_MODEL = {
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-flash",
        "openrouter": "google/gemini-2.0-flash-exp:free",
    }.get(provider, None)
print(f"[INFO] Provider: {provider} | Model: {LLM_MODEL}")

@mcp.tool()
def get_keywords_from_llm(natural_language_query: str, model: str = None) -> list[str]:
    """
    Uses an LLM (OpenRouter | OpenAI | Google Gemini) to extract keywords and phrases from a natural language query.

    Args:
        natural_language_query: The user's research question or topic.
        model: The model name. For provider='openrouter', pass OpenRouter model id.
               For provider='openai' pass OpenAI model id (e.g., 'gpt-4o-mini').
               For provider='google' pass Gemini model id (e.g., 'gemini-1.5-flash' or 'gemini-2.0-flash-exp').

    Returns:
        A list of keywords and phrases suitable for an arXiv search.
    """
    # Ensure API key is set for the chosen provider
    env_hint = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
    }.get(provider, "API_KEY")
    if not api_key:
        raise ValueError(f"No API key set for provider '{provider}'. Please export {env_hint}.")

    def _normalize_model_for_provider(p: str, m: str | None) -> str | None:
        if not m:
            return None
        m = m.strip()
        # If user accidentally passes OpenRouter-style names to direct providers, trim prefixes/suffixes
        if p in ("openai", "google"):
            if "/" in m:
                m = m.split("/")[-1]   # keep last segment
            if ":" in m:
                m = m.split(":")[0]    # drop qualifiers like ':free'
        return m

    model = _normalize_model_for_provider(provider, model)

    # A detailed prompt telling the LLM exactly what to do.
    prompt = f"""
    You are an expert academic researcher specializing in chemistry, biology, physics, computer science, and condensed matter physics.
    Your task is to analyze the following user query and extract a list of precise, effective keywords and multi-word phrases for searching the arXiv database.
    The goal is to find the most relevant academic papers.

    - Identify core concepts, technical terms, and important named entities.
    - For multi-word concepts (e.g., "machine learning potential"), keep them as a single phrase.
    - Do not include generic words.
    - Return ONLY a comma-separated list of these keywords and phrases. Do not add any explanation or introductory text.

    User Query: "{natural_language_query}"

    Keywords:
    """

    max_retries = 5
    backoff_base = 1.6

    def _sleep(attempt: int, retry_after: str | None):
        if retry_after and retry_after.isdigit():
            wait = float(retry_after)
        else:
            wait = backoff_base ** attempt
        # time_module.sleep(min(wait, 30))
        time_module.sleep(min(wait, 0))

    def _parse_keywords(text: str) -> list[str]:
        return [kw.strip() for kw in (text or "").split(",") if kw.strip()]

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    # optional but recommended by OpenRouter
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                llm_output = data["choices"][0]["message"]["content"]

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model or "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                }
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                llm_output = data["choices"][0]["message"]["content"]

            elif provider == "google":
                # Google Generative Language API (Gemini)
                g_model = model or "gemini-1.5-flash"
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [
                        {"role": "user", "parts": [{"text": prompt}]}
                    ],
                    "generationConfig": {"temperature": 0},
                }
                response = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                if response.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, response.headers.get("Retry-After"))
                    continue
                response.raise_for_status()
                data = response.json()
                # Join all text parts from the top candidate
                parts = data["candidates"][0]["content"]["parts"]
                llm_output = " ".join(p.get("text", "") for p in parts if isinstance(p, dict))

            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")

            keywords = _parse_keywords(llm_output)
            print(f"[INFO] LLM ({provider}:{model}) extracted keywords: {keywords}")
            return keywords

        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "N/A"
            body = e.response.text[:500] if e.response is not None else ""
            print(f"[ERROR] HTTP {status} from {provider}: {body}")
            if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                _sleep(attempt, e.response.headers.get("Retry-After") if e.response else None)
                continue
            return []
        except (KeyError, IndexError, ValueError) as e:
            print(f"[ERROR] Failed to parse {provider} response: {e}")
            try:
                print(f"Raw response: {response.text[:500]}")
            except Exception:
                pass
            return []
        except requests.RequestException as e:
            print(f"[ERROR] API request failed: {e}")
            if attempt < max_retries:
                _sleep(attempt, None)
                continue
            return []

# @mcp.tool()
# def search_and_download_pdfs(
#     keywords: list[str],
#     field: str = "all",
#     max_keywords: int = 5,
#     max_days: int = 30,
#     max_results: int = 50,
#     save_dir_override: str | None = None,
# ) -> dict:
#     """
#     Search arXiv using keywords and download PDFs.

#     Args:
#         keywords: List of keywords/phrases (output of get_keywords_from_llm).
#         field: arXiv field to search (e.g., "all", "ti", "abs", "cat").
#         max_keywords: Max number of keywords to include in the query.
#         max_days: Only include papers published within the last N days.
#         max_results: Max number of papers to download.
#         save_dir_override: If provided, save PDFs here (otherwise uses config.yaml's pdf_path).

#     Returns:
#         Dict with the query used, number saved, and a list of saved items with metadata.
#     """
#     # Lazy-load helpers from arxiv_search_chunk to avoid packaging issues
#     try:
#         tools_dir = Path(__file__).parent
#         mod_path = tools_dir / "arxiv_search_chunk.py"
#         spec = importlib.util.spec_from_file_location("arxiv_search_chunk", mod_path)
#         assert spec and spec.loader
#         mod = importlib.util.module_from_spec(spec)
#         spec.loader.exec_module(mod)  # type: ignore
#         format_arxiv_query = getattr(mod, "format_arxiv_query")
#         download_arxiv_pdfs = getattr(mod, "download_arxiv_pdfs")
#     except Exception as e:
#         raise RuntimeError(f"Failed to load arxiv_search_chunk helpers: {e}")

#     QUERY = format_arxiv_query(keywords or [], field=field, max_keywords=max_keywords)
#     if not QUERY:
#         return {"query": "", "saved_count": 0, "saved": [], "error": "Empty or invalid keyword list."}

#     now_utc = datetime.now(timezone.utc)
#     earliest = now_utc - timedelta(days=max_days)

#     client = arxiv.Client(page_size=100, delay_seconds=3)
#     search = arxiv.Search(
#         query=QUERY,
#         sort_by=arxiv.SortCriterion.SubmittedDate,
#         sort_order=arxiv.SortOrder.Descending,
#     )

#     out_dir = save_dir_override or PAPER_DIR
#     os.makedirs(out_dir, exist_ok=True)

#     saved = []
#     # count = 0
#     for r in client.results(search):
#         published = r.published or r.updated
#     #     if not published:
#     #         continue
#     #     if published < earliest:
#     #         break
#     #     if count >= max_results:
#     #         break

#         pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
#         try:
#             saved_path = download_arxiv_pdfs(pdf_url=pdf_url, output_dir=out_dir)
#             saved.append({
#                 "arxiv_id": r.entry_id.split("/")[-1],
#                 "title": r.title,
#                 "authors": [a.name for a in r.authors],
#                 "published": (published.astimezone(timezone.utc)).isoformat(),
#                 "pdf_url": pdf_url,
#                 "pdf_path": saved_path,
#                 "categories": list(r.categories),
#             })
#             # count += 1
#         except Exception as e:
#             saved.append({
#                 "arxiv_id": r.entry_id.split("/")[-1],
#                 "title": r.title,
#                 "error": f"Download failed: {e}",
#                 "pdf_url": pdf_url,
#             })

#     # return {"query": QUERY, "saved_count": count, "saved": saved}
#     return {"query": QUERY, "saved": saved}

# @mcp.tool()
# async def query_search_and_download(
#     natural_language_query: str,
#     field: str = "all",
#     max_keywords: int = 5,
#     max_days: int = 30,
#     max_results: int = 10,
#     save_dir_override: str | None = None,
# ) -> dict:
#     """
#     One-shot tool: extract keywords from a natural-language query, then search arXiv and download PDFs.

#     Args:
#         natural_language_query: Research question/topic in natural language.
#         field: arXiv field to search (e.g., "all", "ti", "abs", "cat").
#         max_keywords: Max number of extracted keywords to include in the query.
#         max_days: Only include papers published within the last N days.
#         max_results: Max number of papers to download.
#         save_dir_override: Save PDFs here (defaults to config.yaml's pdf_path).

#     Returns:
#         Dict containing extracted keywords, arXiv query, number saved, and saved file metadata.
#     """
#     try:
#         # Extract keywords with the existing tool function
#         print(f"[PROGRESS] Extracting keywords from query...")
#         kws = get_keywords_from_llm(natural_language_query) or []
#         if not kws:
#             return {
#                 "query": "",
#                 "saved_count": 0,
#                 "saved": [],
#                 "keywords": [],
#                 "error": "Keyword extraction returned empty. Provide a clearer query or try again.",
#             }
#         print(f"[PROGRESS] Found keywords: {kws}")

#         def format_arxiv_query(keywords: list[str], field: str = "all", max_keywords: int = 3) -> str:
#             """
#             Formats a list of keywords into a valid arXiv search query string.
#             Uses OR to connect keywords and quotes for multi-word phrases.
#             Limits the number of keywords to avoid hitting URL length limits.
#             """
#             if not keywords:
#                 return ""

#             # Take only the top N most relevant keywords (LLM usually returns them in order)
#             limited_keywords = keywords[:max_keywords]

#             formatted_keywords = []
#             for kw in limited_keywords:
#                 if ' ' in kw:
#                     formatted_keywords.append(f'"{kw}"')  # Add quotes for phrases
#                 else:
#                     formatted_keywords.append(kw)

#             # Use OR to cast a wider net. The RAG pipeline will handle the filtering.
#             return f"{field}:(" + " OR ".join(formatted_keywords) + ")"

#         def download_arxiv_pdfs(pdf_url: str = None, output_dir: str = config['source'][0]['pdf_path']):
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)

#             if pdf_url:
#                 pdf_response = requests.get(pdf_url)
#                 pdf_response.raise_for_status()
#                 pdf_filename = pdf_url.split("/")[-1]
#                 pdf_filename += '.pdf'
#                 pdf_path = Path(output_dir) / pdf_filename
#                 with open(pdf_path, 'wb') as f:
#                     f.write(pdf_response.content)
#                 print(f"Downloaded: {pdf_filename}")
#             else:
#                 print("No PDF URL provided.")

#         QUERY = format_arxiv_query(kws, max_keywords=max_keywords)
#         if not QUERY:
#             print("[ERROR] Could not generate a valid query.")
#             return {}

#         print(f"[PROGRESS] Searching arXiv with query: {QUERY}")

#         # Limit results to avoid timeout
#         if max_results > 20:
#             max_results = 20
#             print(f"[PROGRESS] Limiting results to {max_results} to avoid timeout")
            
#         # -------- Step 2: arXiv search & download --------
#         tz_london = tz.gettz("Europe/London")
#         now_local = datetime.now(tz_london)
#         start_date = (now_local.date() - timedelta(days=max_days - 1))
#         start_dt = datetime.combine(start_date, time(0, 0, tzinfo=tz_london))
#         end_dt = datetime.combine(now_local.date(), time(23, 59, 59, tzinfo=tz_london))
#         USE_UPDATED = False

#         def in_window(dt_utc):
#             dt_local = dt_utc.astimezone(tz_london)
#             return start_dt <= dt_local <= end_dt

#         client = arxiv.Client(page_size=100, delay_seconds=3)
#         search = arxiv.Search(query=QUERY,
#                             sort_by=arxiv.SortCriterion.SubmittedDate,
#                             sort_order=arxiv.SortOrder.Descending)
#         results = []
#         for r in client.results(search):
#             ts = r.updated if USE_UPDATED else r.published
#             if in_window(ts):
#                 results.append(r)
#             else:
#                 if ts.astimezone(tz_london) < start_dt:
#                     break

#         print(f"Found {len(results)} results between {start_dt.date()} and {end_dt.date()} (Europe/London).")
#         pdf_dir = config['source'][0]['pdf_path']
#         os.makedirs(pdf_dir, exist_ok=True)
#         for i, r in enumerate(results, 1):
#             pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
#             abstract = " ".join(r.summary.split())
#             print(f"[{i}] {r.title}\n    PDF: {pdf_url}\n    Abstract: {abstract[:180]}...\n")
#             try:
#                 download_arxiv_pdfs(pdf_url=pdf_url, output_dir=pdf_dir)
#             except Exception as e:
#                 print(f"[WARN] PDF download failed: {e}")

#         # result["keywords"] = kws
#         return {"keywords": kws, "query": QUERY, "saved_count": len(results)}
#     except Exception as e:
#         return {"error": f"Function failed: {str(e)}", "keywords": kws if 'kws' in locals() else []}

# def query_search_and_download(
#     natural_language_query: str,
#     field: str = "all",
#     max_keywords: int = 5,
#     max_days: int = 30,
#     max_results: int = 50,
#     save_dir_override: Optional[str] = None,
#     model: Optional[str] = None,
# ) -> dict:
#     """
#     One-shot tool: extract keywords from a natural-language query, then search arXiv and download PDFs.

#     Args:
#         natural_language_query: Research question/topic in natural language.
#         field: arXiv field to search (e.g., "all", "ti", "abs", "cat").
#         max_keywords: Max number of extracted keywords to include in the query.
#         max_days: Only include papers published within the last N days.
#         max_results: Max number of papers to download.
#         save_dir_override: Save PDFs here (defaults to config.yaml's pdf_path).
#         model: Optional override for the LLM model used for keyword extraction.

#     Returns:
#         Dict containing extracted keywords, arXiv query, number saved, and saved file metadata.
#     """
#     # Extract keywords with the existing tool function
#     kws = get_keywords_from_llm(natural_language_query, model=model) or []
#     if not kws:
#         return {
#             "query": "",
#             "saved_count": 0,
#             "saved": [],
#             "keywords": [],
#             "error": "Keyword extraction returned empty. Provide a clearer query or try again.",
#         }

#     # Reuse the search+download logic
#     result = search_and_download_pdfs(
#         keywords=kws,
#         field=field,
#         max_keywords=max_keywords,
#         max_days=max_days,
#         max_results=max_results,
#         save_dir_override=save_dir_override,
#     )
#     result["keywords"] = kws
#     return result

@mcp.tool()
def extract_keywords_only(natural_language_query: str) -> dict:
    """Fast keyword extraction only."""
    try:
        kws = get_keywords_from_llm(natural_language_query) or []
        return {"keywords": kws, "status": "success"}
    except Exception as e:
        return {"error": str(e), "keywords": []}

@mcp.tool()
def search_arxiv_titles_only(
    keywords: list[str],
    max_results: int = 5,
    max_days: int = 7
) -> dict:
    """Quick search that only returns titles and URLs, no downloads."""
    try:
        # Format query with AND for precision
        if not keywords:
            return {"error": "No keywords provided"}
            
        formatted_kws = []
        for kw in keywords[:3]:  # Limit to 3 keywords
            if ' ' in kw:
                formatted_kws.append(f'"{kw}"')
            else:
                formatted_kws.append(kw)
        
        QUERY = f"all:(" + " AND ".join(formatted_kws) + ")"
        
        client = arxiv.Client(page_size=10, delay_seconds=1)
        search = arxiv.Search(
            query=QUERY,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for r in client.results(search):
            papers.append({
                "arxiv_id": r.entry_id.split("/")[-1],
                "title": r.title,
                "authors": [a.name for a in r.authors[:3]],  # Limit authors
                "published": r.published.isoformat() if r.published else None,
                "pdf_url": r.pdf_url or r.entry_id.replace("abs", "pdf"),
                "abstract": r.summary[:200] + "..." if len(r.summary) > 200 else r.summary
            })
            
        return {"papers": papers, "query": QUERY, "count": len(papers)}
        
    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}

@mcp.tool()
def download_specific_papers(paper_urls: list[str], pdf_directory: str = None) -> dict:
    """Download specific papers by PDF URL."""
    try:
        pdf_dir = pdf_directory or config['source'][0]['pdf_path']
        os.makedirs(pdf_dir, exist_ok=True)
        
        downloaded = []
        for url in paper_urls[:5]:  # Limit to 5 downloads
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                pdf_filename = url.split("/")[-1] + '.pdf'
                pdf_path = Path(pdf_dir) / pdf_filename

                with open(pdf_path, 'wb') as f:
                    f.write(response.content)

                downloaded.append(
                    {
                        "url": url,
                        "filename": pdf_filename,
                        "path": str(pdf_path),
                        "status": "success",
                    }
                )

            except Exception as e:
                downloaded.append({"url": url, "status": "failed", "error": str(e)})

        return {
            "downloaded": downloaded,
            "success_count": sum(1 for d in downloaded if d["status"] == "success"),
        }

    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}


@mcp.tool()
def ingest_local_pdfs(
    pdf_directory: str = None,
    *,
    max_pdfs: int = 50,
    max_pages: int = 10,
    per_page_chars: Optional[int] = None,
    max_chars: Optional[int] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    embedding_model: Optional[str] = None,
    embedding_prefix: Optional[str] = None,
) -> dict:
    """Parse PDFs from ``pdf_directory`` and store them in the SQLite cache."""

    try:
        pdf_dir = pdf_directory or config['source'][0]['pdf_path']
        pdf_path = Path(pdf_dir)
        if not pdf_path.exists():
            return {"error": f"PDF directory {pdf_dir} does not exist", "database_path": str(DEFAULT_DB_PATH)}

        parsed_docs = collect_documents_from_directory(
            pdf_path,
            max_pdfs=max_pdfs,
            max_pages=max_pages,
            per_page_chars=per_page_chars,
            max_chars=max_chars,
        )

        if not parsed_docs:
            return {
                "stored": 0,
                "documents": [],
                "database_path": str(DEFAULT_DB_PATH),
                "message": f"No PDFs parsed from {pdf_dir}",
            }

        stored_model, stored_prefix = _store_documents_with_defaults(
            parsed_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_model=embedding_model,
            embedding_prefix=embedding_prefix,
        )

        return {
            "stored": len(parsed_docs),
            "documents": [doc.to_dict() for doc in parsed_docs],
            "database_path": str(DEFAULT_DB_PATH),
            "embedding_model": stored_model,
            "embedding_prefix": stored_prefix,
            "chunk_size": chunk_size or _as_int(_get_config_value("rag", "chunk_size", 1000), 1000),
            "chunk_overlap": chunk_overlap or _as_int(_get_config_value("rag", "overlap", 100), 100),
        }

    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}


@mcp.tool()
def refresh_arxiv_feed(
    natural_language_query: str,
    *,
    max_results: int = 3,
    pdf_directory: str = None,
    notify_email: Optional[str] = None,
    max_pages: int = 5,
) -> dict:
    """Search arXiv, download PDFs, store them, and optionally email a summary."""

    try:
        keywords = get_keywords_from_llm(natural_language_query) or []
        if not keywords:
            return {"error": "Keyword extraction failed", "keywords": [], "database_path": str(DEFAULT_DB_PATH)}

        search_result = search_arxiv_titles_only(keywords, max_results=max_results)
        if "error" in search_result:
            return {
                "error": search_result["error"],
                "keywords": keywords,
                "database_path": str(DEFAULT_DB_PATH),
            }

        pdf_urls = [paper["pdf_url"] for paper in search_result.get("papers", []) if paper.get("pdf_url")]
        download_result = download_specific_papers(pdf_urls, pdf_directory=pdf_directory)
        successful = [item for item in download_result.get("downloaded", []) if item.get("status") == "success" and item.get("path")]

        parsed_docs = []
        for item in successful:
            doc = parse_pdf_file(Path(item["path"]), max_pages=max_pages)
            if doc:
                parsed_docs.append(doc)

        stored_model, stored_prefix = _store_documents_with_defaults(parsed_docs)

        email_status: Optional[dict[str, object]] = None
        if notify_email:
            summary_lines = [
                f"- {paper['title']} ({paper['arxiv_id']})" for paper in search_result.get("papers", [])[:max_results]
            ]
            stored_count = len(parsed_docs)
            subject = f"Curaitor feed refresh: {stored_count} stored"
            body_lines = [
                f"Query: {natural_language_query}",
                f"Keywords: {', '.join(keywords[:5])}",
                f"Downloaded: {download_result.get('success_count', 0)} / {len(pdf_urls)}",
                f"Stored in DB: {stored_count}",
                "",
                "Top results:",
            ] + summary_lines

            try:
                send_result = gmail_send(to=notify_email, subject=subject, body="\n".join(body_lines))
            except AuthRequired as exc:
                email_status = {
                    "sent": False,
                    "requires_auth": True,
                    "auth_url": exc.auth_url,
                    "error": getattr(exc, "original_message", str(exc)),
                }
            except Exception as exc:
                email_status = {"sent": False, "error": str(exc)}
            else:
                email_status = {
                    "sent": True,
                    "message_id": send_result.get("id"),
                    "thread_id": send_result.get("threadId"),
                }

        return {
            "keywords": keywords,
            "search": search_result,
            "downloads": download_result,
            "stored": [doc.to_dict() for doc in parsed_docs],
            "stored_count": len(parsed_docs),
            "embedding_model": stored_model,
            "embedding_prefix": stored_prefix,
            "database_path": str(DEFAULT_DB_PATH),
            "email_status": email_status,
        }

    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}


@mcp.tool()
def chat_with_repository(
    question: str,
    *,
    top_doc_limit: int = 3,
    per_doc_limit: int = 2,
    embedding_model: Optional[str] = None,
    query_prefix: Optional[str] = None,
) -> dict:
    """Answer ``question`` using documents stored in the SQLite cache."""

    try:
        stored_docs = {doc["arxiv_id"]: doc for doc in list_stored_documents()}
        if not stored_docs:
            return {
                "error": "No documents are stored in the repository yet.",
                "database_path": str(DEFAULT_DB_PATH),
            }

        embedding_model_name = embedding_model or _get_config_value("rag", "embedding_model", None)
        chunks = fetch_document_chunks(embedding_model=embedding_model_name)
        if not chunks and embedding_model_name is None:
            chunks = fetch_document_chunks()

        if embedding_model_name is None:
            for chunk in chunks:
                if chunk.embedding_model:
                    embedding_model_name = chunk.embedding_model
                    break

        prefix = query_prefix
        if prefix is None:
            prefix = _get_config_value("rag", "query_prefix", None)
        query_text = f"{prefix}{question}" if prefix else question

        chunk_hits: list[tuple[DocumentChunk, float]] = []
        embedding_mode = False

        if embedding_model_name:
            vectors = []
            references: list[DocumentChunk] = []
            for chunk in chunks:
                if chunk.embedding is None:
                    continue
                references.append(chunk)
                vectors.append(np.asarray(chunk.embedding, dtype="float32"))

            if vectors:
                embedding_mode = True
                dim = vectors[0].shape[0]
                index = faiss.IndexFlatIP(dim)
                index.add(np.vstack(vectors).astype("float32"))

                query_vector = embed_texts([query_text], model_name=embedding_model_name)[0]
                query_vector_np = np.asarray(query_vector, dtype="float32").reshape(1, dim)
                search_k = min(len(references), max(top_doc_limit * per_doc_limit * 4, top_doc_limit))
                distances, indices = index.search(query_vector_np, search_k)

                for idx, score in zip(indices[0], distances[0]):
                    if idx < 0 or idx >= len(references):
                        continue
                    chunk_hits.append((references[idx], float(score)))

        relevant_papers: list[dict[str, object]] = []
        context_parts: list[str] = []

        if chunk_hits:
            doc_hits: dict[str, list[tuple[DocumentChunk, float]]] = defaultdict(list)
            for chunk, score in chunk_hits:
                doc_hits[chunk.arxiv_id].append((chunk, score))

            ranked_docs = sorted(
                doc_hits.items(),
                key=lambda item: max(score for _, score in item[1]),
                reverse=True,
            )

            for doc_id, entries in ranked_docs[:top_doc_limit]:
                entries.sort(key=lambda item: item[1], reverse=True)
                metadata = stored_docs.get(doc_id, {})
                chunk_details = []
                best_score = 0.0

                for chunk, score in entries[:per_doc_limit]:
                    best_score = max(best_score, score)
                    preview = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
                    chunk_details.append(
                        {
                            "chunk_index": chunk.chunk_index,
                            "score": score,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "text_preview": preview,
                        }
                    )

                    snippet = chunk.text if len(chunk.text) <= 1200 else chunk.text[:1200] + "..."
                    label = metadata.get("filename") or doc_id
                    context_parts.append(
                        f"[PDF: {label}] (chunk {chunk.chunk_index})\n{snippet}"
                    )

                relevant_papers.append(
                    {
                        "arxiv_id": doc_id,
                        "filename": metadata.get("filename"),
                        "path": metadata.get("path"),
                        "relevance_score": best_score,
                        "chunk_count": metadata.get("chunk_count"),
                        "chunks": chunk_details,
                        "pages_indexed": metadata.get("pages_processed"),
                        "total_pages": metadata.get("total_pages"),
                    }
                )

        if not context_parts:
            keyword_terms = [term.lower() for term in question.split() if len(term) > 3]
            fallback_scores: list[tuple[str, int, str]] = []
            for arxiv_id, text in iter_document_texts():
                lowered = text.lower()
                score = sum(lowered.count(term) for term in keyword_terms) if keyword_terms else 0
                if score:
                    snippet = text[:800] + "..." if len(text) > 800 else text
                    fallback_scores.append((arxiv_id, score, snippet))

            fallback_scores.sort(key=lambda item: item[1], reverse=True)

            for doc_id, score, snippet in fallback_scores[:top_doc_limit]:
                metadata = stored_docs.get(doc_id, {})
                label = metadata.get("filename") or doc_id
                relevant_papers.append(
                    {
                        "arxiv_id": doc_id,
                        "filename": metadata.get("filename"),
                        "path": metadata.get("path"),
                        "relevance_score": score,
                        "chunk_count": metadata.get("chunk_count"),
                    }
                )
                context_parts.append(f"[PDF: {label}]\n{snippet}")

        if not context_parts:
            return {
                "answer": "No relevant information found in the stored documents.",
                "relevant_papers": [],
                "database_path": str(DEFAULT_DB_PATH),
                "embedding_model": embedding_model_name,
                "mode": "none",
            }

        context = "\n\n---\n\n".join(context_parts)
        try:
            answer = answer_query_with_context(question, context)
        except Exception as exc:
            answer = f"Could not generate answer: {exc}"

        return {
            "answer": answer,
            "relevant_papers": relevant_papers,
            "context_chunks": len(context_parts),
            "database_path": str(DEFAULT_DB_PATH),
            "embedding_model": embedding_model_name,
            "mode": "embedding" if embedding_mode and chunk_hits else "keyword",
        }

    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}

# Add these helper functions after the existing functions

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    """Chunk text into token-based segments."""
    enc = tiktoken.get_encoding("cl100k_base")
    if not text.strip():
        return []
    toks = enc.encode(text)
    chunks_local = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(toks), step):
        window = toks[i:i+chunk_size]
        if not window: 
            break
        chunks_local.append(enc.decode(window))
        if i + chunk_size >= len(toks): 
            break
    return chunks_local or [text]

def _recency_score(iso_datetime: str, half_life_days=60, tz_str="Europe/London"):
    """Calculate recency score for papers."""
    if not iso_datetime:
        return 0.0
    try:
        dt = datetime.fromisoformat(iso_datetime)
    except Exception:
        return 0.0
    
    from dateutil import tz
    now = datetime.now(tz.gettz(tz_str))
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.gettz(tz_str))
    days = (now - dt).total_seconds() / 86400.0
    if days < 0:
        days = 0
    return 0.5 ** (days / half_life_days)

def format_context(hits, max_ctx_tokens=1800, model_enc="cl100k_base"):
    """Format retrieval hits into context string."""
    enc = tiktoken.get_encoding(model_enc)
    parts, used = [], 0
    for h in hits:
        src = f"[Source: arXiv:{h['metadata']['arxiv_id']}]"
        block = f"{src}\n{h['text'].strip()}\n"
        tok = enc.encode(block)
        if used + len(tok) > max_ctx_tokens:
            break
        parts.append(block)
        used += len(tok)
    return "\n---\n".join(parts)

def answer_query_with_context(question: str, context: str, model: str = None) -> str:
    """Ask LLM to answer using retrieved context."""
    sys_instructions = (
        "You are a concise research assistant. Answer using ONLY the provided arXiv context. "
        "Cite arXiv IDs in brackets like [arXiv:XXXX.XXXXX]. If the context is insufficient, say so briefly."
    )
    user_msg = f"Question:\n{question}\n\nContext:\n{context}\n\nAnswer in 5-8 sentences."

    max_retries, backoff_base = 3, 1.6
    def _sleep(attempt: int, retry_after: str | None):
        wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff_base ** attempt
        time_module.sleep(min(wait, 3))

    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model or raw_model,
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After"))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model or LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After"))
                    continue
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "google":
                g_model = model or LLM_MODEL
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                prompt = f"{sys_instructions}\n\n{user_msg}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                }
                resp = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                if resp.status_code == 429 and attempt < max_retries:
                    _sleep(attempt, resp.headers.get("Retry-After"))
                    continue
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

        except Exception as e:
            if attempt < max_retries:
                _sleep(attempt, None)
                continue
            return f"[ERROR] QA request failed: {e}"
    
    return "[ERROR] All attempts failed"

def summarize_docs(docs, model: str = None) -> str:
    """Summarize documents using LLM."""
    sys_instructions = (
        "You are an expert academic editor. Summarize the key findings from these research papers "
        "in 2-3 sentences. Focus on the main contributions and results."
    )
    combined_texts = "\n\n---\n\n".join(
        f"[arXiv:{d['metadata'].get('arxiv_id','')}]\n{d['text'][:500]}..." for d in docs
    )
    user_msg = f"Document Excerpts:\n{combined_texts}\n\nSummary:"

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            if provider == "openrouter":
                url = "https://openrouter.ai/api/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "curaitor-agent",
                }
                payload = {
                    "model": model or raw_model,
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "openai":
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model or LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": sys_instructions},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0,
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()

            elif provider == "google":
                g_model = model or LLM_MODEL
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{g_model}:generateContent"
                params = {"key": api_key}
                headers = {"Content-Type": "application/json"}
                prompt = f"{sys_instructions}\n\n{user_msg}"
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0},
                }
                resp = requests.post(url, params=params, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                parts = resp.json()["candidates"][0]["content"]["parts"]
                return " ".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()

        except Exception as e:
            if attempt < max_retries:
                time_module.sleep(1)
                continue
            return f"Could not generate summary: {e}"

# Replace the full_rag_pipeline function with this updated version:
# @mcp.tool()
# def full_rag_pipeline(
#     natural_language_query: str,
#     max_keywords: int = 3,
#     chunk_size: int = 512,
#     overlap: int = 50,
#     final_docs: int = 5,
#     pdf_directory: str = None
# ) -> dict:
#     """
#     Complete RAG pipeline using ALREADY DOWNLOADED PDFs: extract keywords, read PDFs, chunk, embed, retrieve, and answer.
#     Assumes PDFs are already in the specified directory.
#     """
#     try:
#         # Step 1: Extract keywords
#         print(f"[PROGRESS] Extracting keywords...")
#         kws = get_keywords_from_llm(natural_language_query) or []
#         if not kws:
#             return {"error": "Keyword extraction failed", "keywords": []}
        
#         print(f"[PROGRESS] Keywords: {kws[:max_keywords]}")
        
#         # Step 2: Read existing PDFs from directory
#         pdf_dir = pdf_directory or config['source'][0]['pdf_path']
#         pdf_path = Path(pdf_dir)
        
#         if not pdf_path.exists():
#             return {"error": f"PDF directory {pdf_dir} does not exist", "keywords": kws}
        
#         pdf_files = list(pdf_path.glob("*.pdf"))
#         if not pdf_files:
#             return {"error": f"No PDF files found in {pdf_dir}", "keywords": kws}
        
#         print(f"[PROGRESS] Found {len(pdf_files)} PDF files")
        
#         # Step 3: Process existing PDFs into documents
#         docs = []
#         for pdf_file in pdf_files[:10]:  # Limit to avoid timeout
#             try:
#                 # Extract arXiv ID from filename (assuming format like "2401.12345v1.pdf")
#                 arxiv_id = pdf_file.stem.split('.')[0] + '.' + pdf_file.stem.split('.')[1]
                
#                 # Read PDF content
#                 with open(pdf_file, 'rb') as f:
#                     reader = PdfReader(f)
#                     body_text = ""
#                     for page in reader.pages[:15]:  # Limit pages to avoid timeout
#                         page_text = page.extract_text()
#                         if page_text:
#                             body_text += page_text + "\n"
                
#                 # Create title from filename if we can't get it elsewhere
#                 title = pdf_file.stem.replace('_', ' ').replace('-', ' ')
                
#                 full_text = f"search_document: {title}\n\n{body_text}"
                
#                 docs.append({
#                     "doc_id": arxiv_id,
#                     "title": title,
#                     "text": full_text,
#                     "metadata": {
#                         "arxiv_id": arxiv_id,
#                         "pdf_path": str(pdf_file),
#                         "title": title,
#                         "authors": [],  # Would need to parse from PDF if needed
#                         "published": None,  # Would need to get from filename or content
#                     }
#                 })
                
#                 print(f"[PROGRESS] Processed {arxiv_id}")
                
#             except Exception as e:
#                 print(f"[WARN] Failed to process {pdf_file}: {e}")
#                 continue
        
#         if not docs:
#             return {"error": "No PDFs could be processed"}
        
#         print(f"[PROGRESS] Processed {len(docs)} documents")
        
#         # Step 4: Chunking
#         chunks = []
#         for d in docs:
#             parts = chunk_text(d["text"], chunk_size, overlap)
#             for j, p in enumerate(parts):
#                 chunks.append({
#                     "text": p,
#                     "doc_id": d["doc_id"],
#                     "metadata": d["metadata"],
#                     "chunk_id": f"{d['doc_id']}::chunk{j:04d}"
#                 })
        
#         print(f"[PROGRESS] Created {len(chunks)} chunks")
        
#         # Step 5: Embeddings and retrieval
#         try:
#             from sentence_transformers import SentenceTransformer, CrossEncoder
            
#             EMB_MODEL = "BAAI/bge-small-en-v1.5"  # Smaller model for speed
#             emb = SentenceTransformer(EMB_MODEL, trust_remote_code=True)
            
#             chunk_texts = [c["text"] for c in chunks]
#             X = emb.encode(chunk_texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
#             X = np.asarray(X, dtype="float32")
            
#             # Build FAISS index
#             dim = X.shape[1]
#             index = faiss.IndexFlatIP(dim)
#             index.add(X)
            
#             # Retrieve
#             query_embedding = emb.encode([f"search_query: {natural_language_query}"], normalize_embeddings=True)
#             D, I = index.search(np.asarray(query_embedding, dtype="float32"), min(20, len(chunks)))
            
#             candidates = [chunks[i] for i in I[0] if 0 <= i < len(chunks)]
            
#             # Rerank
#             reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#             pairs = [(natural_language_query, c["text"]) for c in candidates]
#             scores = reranker.predict(pairs)
            
#             # Aggregate by document and add scores
#             best_by_doc = {}
#             for cand, score in zip(candidates, scores):
#                 doc_id = cand["doc_id"]
#                 if (doc_id not in best_by_doc) or (score > best_by_doc[doc_id][1]):
#                     best_by_doc[doc_id] = (cand, score)
            
#             # Sort by combined score (no recency since we don't have dates)
#             scored_docs = []
#             raw_scores = [s for _, s in best_by_doc.values()]
#             if raw_scores:
#                 s_min, s_max = min(raw_scores), max(raw_scores)
                
#                 for doc_id, (chunk, rel_score) in best_by_doc.items():
#                     norm_score = (rel_score - s_min) / (s_max - s_min) if s_max > s_min else 1.0
#                     scored_docs.append((doc_id, chunk, rel_score, norm_score))
                
#                 scored_docs.sort(key=lambda x: x[3], reverse=True)
            
#             # Get top hits
#             hits = []
#             for _, chunk, rel_score, norm_score in scored_docs[:final_docs]:
#                 hit = dict(chunk)
#                 hit["rel_score"] = float(rel_score)
#                 hit["combined_score"] = float(norm_score)
#                 hits.append(hit)
            
#             print(f"[PROGRESS] Retrieved {len(hits)} relevant chunks")
            
#         except ImportError:
#             print("[WARN] sentence_transformers not available, using simple text matching")
#             # Fallback: simple keyword matching in chunks
#             keyword_lower = [kw.lower() for kw in kws[:max_keywords]]
#             scored_chunks = []
            
#             for chunk in chunks:
#                 text_lower = chunk["text"].lower()
#                 score = sum(1 for kw in keyword_lower if kw in text_lower)
#                 if score > 0:
#                     scored_chunks.append((chunk, score))
            
#             scored_chunks.sort(key=lambda x: x[1], reverse=True)
#             hits = []
#             for chunk, score in scored_chunks[:final_docs]:
#                 hit = dict(chunk)
#                 hit["rel_score"] = float(score)
#                 hit["combined_score"] = float(score)
#                 hits.append(hit)
        
#         # Step 6: Generate answer
#         context = format_context(hits)
#         final_answer = answer_query_with_context(natural_language_query, context)
        
#         # Create response with paper summaries
#         paper_summaries = []
#         seen_docs = set()
#         for hit in hits:
#             doc_id = hit["doc_id"]
#             if doc_id not in seen_docs:
#                 seen_docs.add(doc_id)
#                 meta = hit["metadata"]
#                 summary = summarize_docs([hit])
#                 paper_summaries.append({
#                     "arxiv_id": doc_id,
#                     "title": meta.get("title", ""),
#                     "pdf_path": meta.get("pdf_path", ""),
#                     "summary": summary,
#                     "relevance_score": hit.get("combined_score", 0)
#                 })
        
#         return {
#             "keywords": kws[:max_keywords],
#             "num_papers_processed": len(docs),
#             "paper_summaries": paper_summaries,
#             "final_answer": final_answer,
#             "status": "success"
#         }
        
#     except Exception as e:
#         return {
#             "error": f"Pipeline failed: {str(e)}",
#             "keywords": kws if 'kws' in locals() else [],
#             "status": "failed"
#         }

# @mcp.tool()
# def download_papers_only(
#     natural_language_query: str,
#     max_days: int = 7,
#     max_keywords: int = 3,
#     max_results: int = 5
# ) -> dict:
#     """
#     Lightweight function to ONLY download papers based on query. No RAG processing.
#     Use this first, then call full_rag_pipeline to process downloaded PDFs.
#     """
#     try:
#         # Step 1: Extract keywords
#         print(f"[PROGRESS] Extracting keywords...")
#         kws = get_keywords_from_llm(natural_language_query) or []
#         if not kws:
#             return {"error": "Keyword extraction failed", "keywords": []}
        
#         # Step 2: Format query and search arXiv
#         formatted_kws = []
#         for kw in kws[:max_keywords]:
#             if ' ' in kw:
#                 formatted_kws.append(f'"{kw}"')
#             else:
#                 formatted_kws.append(kw)
        
#         QUERY = f"all:(" + " AND ".join(formatted_kws) + ")"
#         print(f"[PROGRESS] Searching arXiv: {QUERY}")
        
#         # Step 3: Search and download (lightweight)
#         client = arxiv.Client(page_size=10, delay_seconds=1)
#         search = arxiv.Search(
#             query=QUERY,
#             max_results=max_results,
#             sort_by=arxiv.SortCriterion.SubmittedDate,
#             sort_order=arxiv.SortOrder.Descending
#         )
        
#         # Step 4: Download PDFs quickly
#         pdf_dir = config['source'][0]['pdf_path']
#         os.makedirs(pdf_dir, exist_ok=True)
        
#         downloaded = []
#         for r in client.results(search):
#             arxiv_id = r.entry_id.split("/")[-1]
#             pdf_url = r.pdf_url or r.entry_id.replace("abs", "pdf")
            
#             try:
#                 response = requests.get(pdf_url, timeout=30)
#                 response.raise_for_status()
                
#                 pdf_filename = pdf_url.split("/")[-1] + '.pdf'
#                 pdf_path = Path(pdf_dir) / pdf_filename
                
#                 with open(pdf_path, 'wb') as f:
#                     f.write(response.content)
                
#                 downloaded.append({
#                     "arxiv_id": arxiv_id,
#                     "title": r.title,
#                     "pdf_filename": pdf_filename,
#                     "pdf_path": str(pdf_path),
#                     "status": "success"
#                 })
                
#                 print(f"[PROGRESS] Downloaded: {pdf_filename}")
                
#             except Exception as e:
#                 downloaded.append({
#                     "arxiv_id": arxiv_id,
#                     "title": r.title,
#                     "error": str(e),
#                     "status": "failed"
#                 })
#                 print(f"[WARN] Download failed for {arxiv_id}: {e}")
        
#         return {
#             "keywords": kws[:max_keywords],
#             "query": QUERY,
#             "downloaded": downloaded,
#             "success_count": sum(1 for d in downloaded if d["status"] == "success"),
#             "total_attempted": len(downloaded),
#             "pdf_directory": pdf_dir,
#             "status": "success"
#         }
        
#     except Exception as e:
#         return {
#             "error": f"Download failed: {str(e)}",
#             "keywords": kws if 'kws' in locals() else [],
#             "status": "failed"
#         }

@mcp.tool()
def list_downloaded_pdfs(pdf_directory: str = None) -> dict:
    """
    List all PDF files in the download directory.
    """
    try:
        pdf_dir = pdf_directory or config['source'][0]['pdf_path']
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            return {"error": f"Directory {pdf_dir} does not exist", "pdfs": []}
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        stored_docs = {rec["path"]: rec for rec in list_stored_documents()}

        pdfs = []
        for pdf_file in pdf_files:
            try:
                # Try to extract arXiv ID from filename
                arxiv_id = pdf_file.stem.split('.')[0] + '.' + pdf_file.stem.split('.')[1]
                file_size = pdf_file.stat().st_size

                record = stored_docs.get(str(pdf_file))
                pdfs.append({
                    "filename": pdf_file.name,
                    "arxiv_id": arxiv_id,
                    "file_size": file_size,
                    "path": str(pdf_file),
                    "stored_in_database": bool(record),
                    "text_length": record.get("text_length") if record else None,
                    "pages_indexed": record.get("pages_processed") if record else None,
                    "total_pages": record.get("total_pages") if record else None,
                    "stored_at": record.get("stored_at") if record else None,
                    "chunk_count": record.get("chunk_count") if record else None,
                })
            except Exception:
                record = stored_docs.get(str(pdf_file))
                pdfs.append({
                    "filename": pdf_file.name,
                    "arxiv_id": "unknown",
                    "file_size": pdf_file.stat().st_size,
                    "path": str(pdf_file),
                    "stored_in_database": bool(record),
                    "text_length": record.get("text_length") if record else None,
                    "pages_indexed": record.get("pages_processed") if record else None,
                    "total_pages": record.get("total_pages") if record else None,
                    "stored_at": record.get("stored_at") if record else None,
                    "chunk_count": record.get("chunk_count") if record else None,
                })

        return {
            "pdf_directory": pdf_dir,
            "total_pdfs": len(pdfs),
            "pdfs": pdfs,
            "database_path": str(DEFAULT_DB_PATH),
        }
        
    except Exception as e:
        return {"error": str(e), "pdfs": [], "database_path": str(DEFAULT_DB_PATH)}

# @mcp.tool()
# def search_and_analyze_papers(
#     natural_language_query: str,
#     max_days: int = 7,
#     max_results: int = 5
# ) -> dict:
#     """
#     Simplified version: search papers, download, and provide analysis without embeddings.
#     """
#     try:
#         # Extract keywords
#         kws = get_keywords_from_llm(natural_language_query) or []
#         if not kws:
#             return {"error": "No keywords extracted"}
        
#         # Format search query
#         formatted_kws = [f'"{kw}"' if ' ' in kw else kw for kw in kws[:3]]
#         QUERY = f"all:(" + " AND ".join(formatted_kws) + ")"
        
#         # Search arXiv
#         client = arxiv.Client(page_size=10, delay_seconds=1)
#         search = arxiv.Search(
#             query=QUERY,
#             max_results=max_results,
#             sort_by=arxiv.SortCriterion.SubmittedDate,
#             sort_order=arxiv.SortOrder.Descending
#         )
        
#         papers = []
#         for r in client.results(search):
#             # Simple analysis without full PDF processing
#             summary = summarize_docs([{
#                 "text": f"{r.title}\n\n{r.summary}",
#                 "metadata": {"arxiv_id": r.entry_id.split("/")[-1]}
#             }])
            
#             papers.append({
#                 "arxiv_id": r.entry_id.split("/")[-1],
#                 "title": r.title,
#                 "authors": [a.name for a in r.authors[:3]],
#                 "published": r.published.isoformat() if r.published else None,
#                 "pdf_url": r.pdf_url or r.entry_id.replace("abs", "pdf"),
#                 "abstract": r.summary[:300] + "..." if len(r.summary) > 300 else r.summary,
#                 "summary": summary,
#                 "categories": list(r.categories)
#             })
        
#         # Generate overall answer
#         combined_context = "\n\n".join([
#             f"[arXiv:{p['arxiv_id']}] {p['title']}\n{p['abstract']}" 
#             for p in papers
#         ])
        
#         answer = answer_query_with_context(natural_language_query, combined_context)
        
#         return {
#             "keywords": kws[:3],
#             "query": QUERY,
#             "papers": papers,
#             "answer": answer,
#             "count": len(papers)
#         }
        
#     except Exception as e:
#         return {"error": str(e)}

# Replace the full_rag_pipeline function with this much lighter version:

@mcp.tool()
def light_rag_pipeline(
    natural_language_query: str,
    max_keywords: int = 2,
    max_pdfs: int = 3,
    max_pages_per_pdf: int = 5,
    pdf_directory: str = None
) -> dict:
    """
    LIGHTWEIGHT RAG pipeline using already downloaded PDFs. 
    Optimized for speed - processes fewer PDFs with limited pages.
    """
    try:
        # Step 1: Extract keywords (fast)
        print(f"[PROGRESS] Extracting keywords...")
        kws = get_keywords_from_llm(natural_language_query) or []
        if not kws:
            return {"error": "Keyword extraction failed", "keywords": []}
        
        print(f"[PROGRESS] Keywords: {kws[:max_keywords]}")
        
        # Step 2: Find PDFs (fast)
        pdf_dir = pdf_directory or config['source'][0]['pdf_path']
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            return {"error": f"PDF directory {pdf_dir} does not exist", "keywords": kws}
        
        pdf_files = list(pdf_path.glob("*.pdf"))[:max_pdfs]  # Limit PDFs
        if not pdf_files:
            return {"error": f"No PDF files found in {pdf_dir}", "keywords": kws}
        
        print(f"[PROGRESS] Processing {len(pdf_files)} PDFs")
        
        # Step 3: Quick text extraction (limited pages)
        parsed_docs = collect_documents_from_directory(
            pdf_path,
            max_pdfs=max_pdfs,
            max_pages=max_pages_per_pdf,
            per_page_chars=1000,
            max_chars=3000,
        )

        for doc in parsed_docs:
            print(f"[PROGRESS] Processed {doc.arxiv_id}")

        if not parsed_docs:
            return {"error": "No PDFs could be processed", "keywords": kws}

        # Persist parsed documents for later reuse and embedding-based search.
        embedding_model_used = _get_config_value("rag", "embedding_model", None)
        embedding_prefix_used = _get_config_value("rag", "embedding_prefix", None)
        if embedding_prefix_used is None and embedding_model_used:
            embedding_prefix_used = "search_document: "

        try:
            stored_model, stored_prefix = _store_documents_with_defaults(parsed_docs)
            if stored_model:
                embedding_model_used = stored_model
                embedding_prefix_used = stored_prefix
        except Exception as db_err:
            print(f"[WARN] Failed to store parsed PDFs: {db_err}")

        docs = [
            {
                "arxiv_id": doc.arxiv_id,
                "filename": doc.filename,
                "text": doc.text,
                "path": doc.path,
                "pages_processed": doc.pages_processed,
                "total_pages": doc.total_pages,
            }
            for doc in parsed_docs
        ]

        print(f"[PROGRESS] Got text from {len(docs)} documents")

        stored_docs_records = list_stored_documents()
        stored_docs = {rec["arxiv_id"]: rec for rec in stored_docs_records}

        query_prefix = _get_config_value("rag", "query_prefix", "search_query: ")
        top_doc_limit = max(1, _as_int(_get_config_value("rag", "top_k", 5), 5))
        per_doc_limit = max(1, _as_int(_get_config_value("rag", "per_doc_chunks", 2), 2))

        chunk_hits: list[tuple[str, list[tuple[DocumentChunk, float]]]] = []

        if embedding_model_used:
            chunks = fetch_document_chunks(embedding_model=embedding_model_used)
            chunks = [chunk for chunk in chunks if chunk.embedding is not None]

            if chunks:
                matrix = np.vstack([
                    np.asarray(chunk.embedding, dtype="float32") for chunk in chunks
                ])
                dim = matrix.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(matrix)

                query_text = f"{query_prefix}{natural_language_query}" if query_prefix else natural_language_query
                query_vector = embed_texts([query_text], model_name=embedding_model_used)[0]
                query_vector_np = np.asarray(query_vector, dtype="float32").reshape(1, dim)

                search_k = min(len(chunks), max(top_doc_limit * per_doc_limit * 4, top_doc_limit))
                distances, indices = index.search(query_vector_np, search_k)

                doc_hits: dict[str, list[tuple[DocumentChunk, float]]] = defaultdict(list)
                for idx, score in zip(indices[0], distances[0]):
                    if idx < 0 or idx >= len(chunks):
                        continue
                    doc_hits[chunks[idx].arxiv_id].append((chunks[idx], float(score)))

                if doc_hits:
                    ranked_docs = sorted(
                        doc_hits.items(),
                        key=lambda item: max(score for _, score in item[1]),
                        reverse=True,
                    )
                    for doc_id, entries in ranked_docs[:top_doc_limit]:
                        entries.sort(key=lambda item: item[1], reverse=True)
                        chunk_hits.append((doc_id, entries[:per_doc_limit]))

        if chunk_hits:
            context_parts: list[str] = []
            relevant_papers: list[dict[str, object]] = []

            for doc_id, entries in chunk_hits:
                metadata = stored_docs.get(doc_id, {})
                filename = metadata.get("filename") if metadata else None
                path = metadata.get("path") if metadata else None
                pages_processed = metadata.get("pages_processed") if metadata else None
                total_pages = metadata.get("total_pages") if metadata else None
                chunk_count = metadata.get("chunk_count") if metadata else None

                chunk_details = []
                best_score = 0.0
                label = filename or doc_id

                for chunk, score in entries:
                    best_score = max(best_score, score)
                    preview = chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
                    chunk_details.append(
                        {
                            "chunk_index": chunk.chunk_index,
                            "score": score,
                            "start_char": chunk.start_char,
                            "end_char": chunk.end_char,
                            "text_preview": preview,
                        }
                    )

                    context_snippet = chunk.text if len(chunk.text) <= 1200 else chunk.text[:1200] + "..."
                    context_parts.append(
                        f"[PDF: {label}] (chunk {chunk.chunk_index})\n{context_snippet}"
                    )

                relevant_papers.append(
                    {
                        "filename": filename,
                        "arxiv_id": doc_id,
                        "path": path,
                        "relevance_score": best_score,
                        "pages_indexed": pages_processed,
                        "total_pages": total_pages,
                        "chunk_count": chunk_count,
                        "chunks": chunk_details,
                    }
                )

            context = "\n\n---\n\n".join(context_parts)

            try:
                answer = answer_query_with_context(natural_language_query, context)
            except Exception as e:
                answer = f"Could not generate answer: {e}"

            return {
                "keywords": kws[:max_keywords],
                "papers_processed": len(docs),
                "relevant_papers": relevant_papers,
                "answer": answer,
                "retrieval_mode": "embedding",
                "embedding_model": embedding_model_used,
                "embedding_prefix": embedding_prefix_used,
                "query_prefix": query_prefix,
                "context_chunks": len(context_parts),
                "stored_documents": len(parsed_docs),
                "database_path": str(DEFAULT_DB_PATH),
                "status": "success"
            }

        # Fallback: keyword-based scoring when embeddings are unavailable
        keyword_lower = [kw.lower() for kw in kws[:max_keywords]]
        scored_docs = []

        for doc in docs:
            text_lower = doc["text"].lower()
            score = sum(text_lower.count(kw) for kw in keyword_lower)
            if score > 0:
                scored_docs.append((doc, score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if not scored_docs:
            return {
                "keywords": kws[:max_keywords],
                "papers_processed": len(docs),
                "relevant_papers": [],
                "answer": "No relevant content found in the processed PDFs for the given query.",
                "retrieval_mode": "keyword",
                "embedding_model": embedding_model_used,
                "stored_documents": len(parsed_docs),
                "database_path": str(DEFAULT_DB_PATH),
                "status": "success"
            }

        top_docs = scored_docs[:3]
        context_parts = []

        for doc, score in top_docs:
            text = doc["text"]
            snippet = text[:800] + "..." if len(text) > 800 else text
            context_parts.append(f"[PDF: {doc['filename']}]\n{snippet}")

        context = "\n\n---\n\n".join(context_parts)

        try:
            answer = answer_query_with_context(natural_language_query, context)
        except Exception as e:
            answer = f"Could not generate answer: {e}"

        relevant_papers = []
        for doc, score in top_docs:
            relevant_papers.append(
                {
                    "filename": doc["filename"],
                    "arxiv_id": doc["arxiv_id"],
                    "path": doc["path"],
                    "relevance_score": score,
                    "text_preview": doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                }
            )

        return {
            "keywords": kws[:max_keywords],
            "papers_processed": len(docs),
            "relevant_papers": relevant_papers,
            "answer": answer,
            "retrieval_mode": "keyword",
            "embedding_model": embedding_model_used,
            "stored_documents": len(parsed_docs),
            "database_path": str(DEFAULT_DB_PATH),
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": f"Pipeline failed: {str(e)}",
            "keywords": kws if 'kws' in locals() else [],
            "status": "failed"
        }

@mcp.tool()
def quick_pdf_search(
    query: str,
    pdf_directory: str = None,
    max_pdfs: int = 5
) -> dict:
    """
    Ultra-fast PDF search - just filenames and basic text matching.
    No PDF parsing, just filename analysis.
    """
    try:
        pdf_dir = pdf_directory or config['source'][0]['pdf_path']
        pdf_path = Path(pdf_dir)
        
        if not pdf_path.exists():
            return {"error": f"Directory {pdf_dir} does not exist"}
        
        pdf_files = list(pdf_path.glob("*.pdf"))[:max_pdfs]
        stored_docs = {rec["path"]: rec for rec in list_stored_documents()}
        
        # Extract keywords for matching
        kws = get_keywords_from_llm(query) or []
        keyword_lower = [kw.lower() for kw in kws[:3]]
        
        results = []
        for pdf_file in pdf_files:
            # Score based on filename matching
            filename_lower = pdf_file.name.lower()
            score = sum(1 for kw in keyword_lower if kw in filename_lower)

            record = stored_docs.get(str(pdf_file))
            results.append({
                "filename": pdf_file.name,
                "path": str(pdf_file),
                "size": pdf_file.stat().st_size,
                "keyword_matches": score,
                "matched_keywords": [kw for kw in keyword_lower if kw in filename_lower],
                "stored_in_database": bool(record),
                "text_length": record.get("text_length") if record else None,
                "pages_indexed": record.get("pages_processed") if record else None,
                "chunk_count": record.get("chunk_count") if record else None,
            })
        
        # Sort by relevance
        results.sort(key=lambda x: x["keyword_matches"], reverse=True)
        
        return {
            "query": query,
            "keywords": kws[:3],
            "total_pdfs": len(pdf_files),
            "results": results[:10],  # Top 10 matches
            "database_path": str(DEFAULT_DB_PATH),
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}

@mcp.tool()
def extract_pdf_text_only(
    pdf_filename: str,
    max_pages: int = 3,
    pdf_directory: str = None
) -> dict:
    """
    Extract text from a specific PDF file quickly.
    """
    try:
        pdf_dir = pdf_directory or config['source'][0]['pdf_path']
        pdf_path = Path(pdf_dir) / pdf_filename
        
        if not pdf_path.exists():
            return {"error": f"File {pdf_filename} not found in {pdf_dir}"}
        
        parsed = parse_pdf_file(
            pdf_path,
            max_pages=max_pages,
            per_page_chars=1500,
            max_chars=(max_pages * 1500) if max_pages else None,
            include_page_headers=True,
        )

        if parsed is None:
            return {
                "error": f"Could not extract text from {pdf_filename}",
                "database_path": str(DEFAULT_DB_PATH),
            }

        stored = False
        try:
            _store_documents_with_defaults([parsed])
            stored = True
        except Exception as db_err:
            print(f"[WARN] Failed to store {pdf_filename} in database: {db_err}")

        return {
            "filename": parsed.filename,
            "arxiv_id": parsed.arxiv_id,
            "pages_processed": parsed.pages_processed,
            "total_pages": parsed.total_pages,
            "text": parsed.text,
            "text_length": len(parsed.text),
            "stored_in_database": stored,
            "database_path": str(DEFAULT_DB_PATH),
            "status": "success"
        }

    except Exception as e:
        return {"error": str(e), "database_path": str(DEFAULT_DB_PATH)}

@mcp.tool()
def simple_qa_from_text(
    question: str,
    text: str,
    max_text_length: int = 2000
) -> dict:
    """
    Simple Q&A using provided text without any retrieval.
    """
    try:
        # Truncate text if too long
        if len(text) > max_text_length:
            text = text[:max_text_length] + "..."
        
        answer = answer_query_with_context(question, text)
        
        return {
            "question": question,
            "text_length": len(text),
            "answer": answer,
            "status": "success"
        }
        
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def send_email_tool(params: dict) -> TextContent:
    """
    Send an email using Gmail. Supports plain text or HTML.

    Accepted call shapes (both valid):

      1) Top-level fields (preferred):
         {
           "to": "user@example.com",
           "subject": "Hello",
           "body": "Hi there",
           "html": false,
           "cc": "cc@example.com",          # optional
           "bcc": "hidden@example.com",     # optional
           "reply_to": "me@example.com",    # optional
           "from_alias": "Agent Bot"        # optional
         }

      2) Legacy wrapper:
         { "params": { ...same fields... } }

    Required: "to" (str), "subject" (str), "body" (str).
    Optional: "html" (bool), "cc", "bcc", "reply_to", "from_alias" (str).

    Success:
      { "success": true, "message_id": "...", "thread_id": "...", "to": "...", "subject": "...", "preview": "..." }

    Validation error:
      { "success": false, "error": { "code": "validation_error", ... } }

    Auth needed :
      { "success": false, "error": { "code": "auth_required", "message": "...", "auth_url": "https://accounts.google.com/..." } }
    """
    # ---- Normalize shape ----
    payload = params.get("params") if isinstance(params, dict) and "params" in params else params
    if not isinstance(payload, dict):
        example = {"to": "user@example.com", "subject": "Hello", "body": "Hi", "html": False}
        err = {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": "Top-level object must be a dict of fields OR contain a dict under key 'params'.",
                "expected_shape": "Provide fields at top-level OR under a top-level 'params' object.",
                "example": example,
            },
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    # ---- Required fields ----
    required = ["to", "subject", "body"]
    missing = [k for k in required if k not in payload]
    if missing:
        example = {"to": "user@example.com", "subject": "Hello", "body": "Hi", "html": False}
        err = {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": f"Missing required fields: {', '.join(missing)}",
                "expected_shape": "Provide fields at top-level OR under a top-level 'params' object.",
                "example": example,
            },
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    # ---- Type checks ----
    type_errors = []
    if not isinstance(payload.get("to"), str): type_errors.append("to must be a string")
    if not isinstance(payload.get("subject"), str): type_errors.append("subject must be a string")
    if not isinstance(payload.get("body"), str): type_errors.append("body must be a string")
    if "html" in payload and not isinstance(payload["html"], bool): type_errors.append("html must be a boolean")
    for opt in ["cc", "bcc", "reply_to", "from_alias"]:
        if opt in payload and not isinstance(payload[opt], str):
            type_errors.append(f"{opt} must be a string")
    if type_errors:
        err = {
            "success": False,
            "error": {
                "code": "validation_error",
                "message": "; ".join(type_errors),
                "example": {"to":"user@example.com","subject":"Hello","body":"Hi","html":False},
            },
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    # ---- Call gmail_send ----
    allowed_keys = {"to", "subject", "body", "html", "cc", "bcc", "reply_to", "from_alias"}
    call_kwargs = {k: payload[k] for k in allowed_keys if k in payload}

    try:
        result = gmail_send(**call_kwargs)
        out = {
            "success": True,
            "message_id": result.get("id"),
            "thread_id": result.get("threadId"),
            "to": payload.get("to"),
            "subject": payload.get("subject"),
            "preview": (payload.get("body") or "")[:160],
        }
        return TextContent(type="text", text=json.dumps(out, indent=2))

    except AuthRequired as e:
        # New: Return a direct OAuth URL so the user can authenticate explicitly.
        err = {
            "success": False,
            "error": {
                "code": "auth_required",
                "message": (
                    "Authentication is required to send email. "
                    "Open the URL to grant access, then retry."
                ),
                "auth_url": e.auth_url,
                "details": e.original_message,  # e.g., "no method available for opening 'https:...'"
            },
            "input": {k: call_kwargs.get(k) for k in ("to", "subject", "html", "cc", "bcc", "reply_to", "from_alias")},
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))

    except Exception as e:
        # Generic failure
        err = {
            "success": False,
            "error": {
                "code": "send_failed",
                "message": str(e),
                "hint": "Check Gmail auth (credentials/token), scopes, and network. Ensure SCOPES includes gmail.send.",
            },
            "input": {k: call_kwargs.get(k) for k in ("to", "subject", "html", "cc", "bcc", "reply_to", "from_alias")},
        }
        return TextContent(type="text", text=json.dumps(err, indent=2))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@mcp.tool()
async def add_daily_job(hour: int, minute: int, job_id: str = "daily_my_job", topic: str = "machine learning"):
    """
    Start/ensure the scheduler is running and add a daily job at HH:MM.
    Input topic for the job to search papers on.
    Non-blocking: returns immediately.
    """

    return schedule_daily_job(hour, minute, job_id=job_id, topic=topic)

@mcp.tool()
async def delete_job(job_id: str):
    """Remove a scheduled job by its ID."""
    return remove_job(job_id)

@mcp.tool()
async def jobs():
    """List scheduled jobs."""
    return {"jobs": list_jobs()}

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
    # transport_type = os.getenv('MCP_TRANSPORT', 'sse')
    # mcp.run(transport=transport_type)