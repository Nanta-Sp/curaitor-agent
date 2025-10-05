# scheduler_service.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.events import (
    EVENT_JOB_EXECUTED,
    EVENT_JOB_ERROR,
    EVENT_JOB_MISSED,
    JobEvent,
)

# ----------------------------
# Logging (console + file)
# ----------------------------
_logging_configured = False

def _configure_logging(level: int = logging.INFO) -> None:
    global _logging_configured
    if _logging_configured:
        return

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler("data/tracker/scheduler.log", mode="a")  # append mode
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _logging_configured = True


# ----------------------------
# Singleton scheduler (module-level)
# ----------------------------
_scheduler: Optional[AsyncIOScheduler] = None

def _job_listener(event: JobEvent) -> None:
    log = logging.getLogger("scheduler.listener")
    if event.code == EVENT_JOB_EXECUTED:
        log.info("Job %s executed", event.job_id)
    elif event.code == EVENT_JOB_ERROR:
        log.exception("Job %s errored", event.job_id)
    elif event.code == EVENT_JOB_MISSED:
        log.warning("Job %s MISSED (scheduler/process may have been paused)", event.job_id)

def get_scheduler() -> AsyncIOScheduler:
    """
    Lazily create and start a single AsyncIOScheduler instance for this process.
    Uses a persistent SQLite job store so jobs are visible across processes / restarts.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    _configure_logging()

    jobstores = {
        "default": SQLAlchemyJobStore(url="sqlite:///data/tracker/mcp_jobs.sqlite"),
    }
    executors = {
        "default": ThreadPoolExecutor(10),
        "processpool": ProcessPoolExecutor(2),
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 300,  # seconds
    }

    _scheduler = AsyncIOScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=ZoneInfo("Europe/London"),
    )

    _scheduler.add_listener(
        _job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
    )

    _scheduler.start()
    logging.getLogger("scheduler").info("Scheduler started (tz=Europe/London, persistent=SQLite)")
    return _scheduler


# ----------------------------
# Example job (async)
# ----------------------------
async def my_job_async():
    logging.getLogger("scheduler.job").info(
        "Async job executed at %s", datetime.now().isoformat(timespec="seconds")
    )
    with open("data/tracker/job_run_times.txt", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\n")
    
    return print(f"[{datetime.now()}] Async job executed at the scheduled time!")

# async def send_whatsapp_message(message: str, to_number: str) -> None:
#     return

def my_job():
    """
    Wrapper to call the async job from a sync context.
    """

    asyncio.run(my_job_async())
    return "Job executed."

# ----------------------------
# Public API used by MCP tools
# ----------------------------
def schedule_daily_job(
    hour: int,
    minute: int,
    job_id: str = "daily_my_job",
    replace_existing: bool = True,
    job_func: Optional[Callable[..., object]] = None,
    *,
    args: Optional[Sequence[object]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    scheduler = get_scheduler()
    trigger = CronTrigger(hour=hour, minute=minute)  # uses scheduler's timezone
    scheduler.add_job(
        job_func or my_job,
        trigger=trigger,
        id=job_id,
        replace_existing=replace_existing,
        args=list(args) if args else None,
        kwargs=dict(kwargs) if kwargs else None,
    )
    logging.getLogger("scheduler").info(
        "Scheduled job '%s' daily at %02d:%02d", job_id, hour, minute
    )
    return {
        "ok": True,
        "message": f"Scheduled '{job_id}' daily at {hour:02d}:{minute:02d}.",
        "job_id": job_id,
        "hour": hour,
        "minute": minute,
    }


def refresh_arxiv_feed_job(
    query: str,
    *,
    max_results: int = 3,
    pdf_directory: Optional[str] = None,
    notify_email: Optional[str] = None,
    max_pages: int = 5,
) -> Dict[str, Any]:
    """Job wrapper around ``curaitor_mcp_server.refresh_arxiv_feed``."""

    import curaitor_mcp_server

    result = curaitor_mcp_server.refresh_arxiv_feed(
        natural_language_query=query,
        max_results=max_results,
        pdf_directory=pdf_directory,
        notify_email=notify_email,
        max_pages=max_pages,
    )

    logging.getLogger("scheduler.job").info(
        "Refresh arXiv feed completed: stored=%s downloads=%s",
        result.get("stored_count"),
        result.get("downloads", {}).get("success_count") if isinstance(result.get("downloads"), dict) else None,
    )
    return result


def schedule_daily_feed_refresh(
    hour: int,
    minute: int,
    *,
    query: str,
    max_results: int = 3,
    pdf_directory: Optional[str] = None,
    notify_email: Optional[str] = None,
    max_pages: int = 5,
    job_id: str = "daily_refresh_arxiv",
    replace_existing: bool = True,
) -> Dict[str, Any]:
    """Schedule the automated arXiv refresh job."""

    return schedule_daily_job(
        hour,
        minute,
        job_id=job_id,
        replace_existing=replace_existing,
        job_func=refresh_arxiv_feed_job,
        kwargs={
            "query": query,
            "max_results": max_results,
            "pdf_directory": pdf_directory,
            "notify_email": notify_email,
            "max_pages": max_pages,
        },
    )

def remove_job(job_id: str) -> Dict[str, Any]:
    scheduler = get_scheduler()
    scheduler.remove_job(job_id)
    logging.getLogger("scheduler").info("Removed job '%s'", job_id)
    return {"ok": True, "removed": job_id}

def list_jobs() -> List[Dict[str, Any]]:
    scheduler = get_scheduler()
    jobs_info: List[Dict[str, Any]] = []
    for j in scheduler.get_jobs():
        nrt = j.next_run_time.isoformat() if j.next_run_time else None
        jobs_info.append(
            {
                "id": j.id,
                "name": j.name,
                "trigger": str(j.trigger),
                "next_run_time": nrt,
                "coalesce": j.coalesce,
                "max_instances": j.max_instances,
                "misfire_grace_time": j.misfire_grace_time,
            }
        )
    return jobs_info

def shutdown_scheduler() -> str:
    global _scheduler
    if _scheduler and _scheduler.running:
        logging.getLogger("scheduler").info("Shutting down scheduler")
        _scheduler.shutdown(wait=False)
    _scheduler = None
    return "Scheduler shut down."
