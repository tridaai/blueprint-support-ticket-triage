"""HTTP API: tickets in, decision records out, agent decisions back.

A thin transport over :class:`~support_triage.service.TriageService` —
every route delegates, none re-implements. Notable behaviour:

- structured JSON logs, one line per request, correlated by a
  request id (``X-Request-ID`` header or a generated one);
- optional shared-key auth: set ``API_KEY`` and every route except the
  health/readiness probes and the console requires ``X-API-Key``;
- ``POST /tickets/analyze`` runs a ticket to the agent gate and
  returns the full decision record; a dead provider call is a 502
  with the translated error, never a half-record;
- ``POST /tickets/{id}/decision`` records approve/reject (and an
  optional assignee) — the ONLY way a run leaves ``awaiting_agent``;
- the dashboards layer: ``GET /api/stats/overview|performance|
  calibration`` aggregate the stored runs (live calibration scores
  recorded probabilities against agent outcomes), ``GET /api/policy``
  renders the composition config as data, ``GET /queue`` filters and
  searches server-side, and ``POST /api/runs/{id}/outcome`` records
  the agent's ground truth for a handled ticket;
- ``/readiness`` fails (503) when PostgreSQL is configured but
  unreachable, and says honestly when the in-memory test doubles are
  in use (records lost on restart — fine for a look, not for work).
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .config import env_str, load_dotenv
from .errors import ProviderError
from .samples import load_sample_tickets
from .schemas import TicketInput
from .service import TriageService
from .wiring import build_service

SERVICE_SLUG = "support-ticket-triage"

# ---------------------------------------------------------------------------
# structured logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger = logging.getLogger("support_triage")
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


log = _configure_logging()

# ---------------------------------------------------------------------------
# app + service (built lazily so imports/tests stay cheap)
# ---------------------------------------------------------------------------

_service: TriageService | None = None


def get_service() -> TriageService:
    global _service
    if _service is None:
        _service = build_service()
    return _service


def set_service(service: TriageService | None) -> None:
    """Test hook: inject a fully-double service (or reset with None)."""
    global _service
    _service = service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_dotenv()
    database = "PostgreSQL" if env_str("DATABASE_URL") else "in-memory (TEST DOUBLE)"
    bucket = env_str("S3_BUCKET")
    storage = f"S3 bucket {bucket!r}" if bucket else "in-memory (TEST DOUBLE)"
    log.info(
        "support-triage %s starting · store: %s · objects: %s · decision backend: %s",
        __version__,
        database,
        storage,
        env_str("DECISION_BACKEND", "offline"),
    )
    yield
    log.info("support-triage shutting down")


app = FastAPI(
    title="Trida AI Blueprint — Support Ticket Triage Agent",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
    started = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    response.headers["X-Request-ID"] = request_id
    log.info(
        "%s %s -> %s in %.1f ms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        extra={"request_id": request_id},
    )
    return response


# ---------------------------------------------------------------------------
# auth + errors
# ---------------------------------------------------------------------------


def require_api_key(request: Request) -> None:
    """Gate: open when API_KEY is unset (local demo), enforced when set."""
    expected = env_str("API_KEY")
    if not expected:
        return
    if request.headers.get("X-API-Key") != expected:
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


_PUBLIC_EXACT = {"/", "/health", "/readiness", "/favicon.svg", "/favicon.ico"}


@app.middleware("http")
async def api_key_gate(request: Request, call_next):
    path = request.url.path
    if path in _PUBLIC_EXACT or path.startswith("/static/"):
        return await call_next(request)
    try:
        require_api_key(request)
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.exception_handler(ProviderError)
async def provider_error_handler(_request: Request, exc: ProviderError):
    return JSONResponse(status_code=502, content={"detail": str(exc)})


# ---------------------------------------------------------------------------
# static console
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(str(resources.files("support_triage").joinpath("static")))
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def console():
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/favicon.svg")
def favicon_svg():
    return FileResponse(str(_STATIC_DIR / "brand" / "favicon.svg"), media_type="image/svg+xml")


@app.get("/favicon.ico")
def favicon_ico():
    return FileResponse(str(_STATIC_DIR / "brand" / "favicon.ico"), media_type="image/x-icon")


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok", "service": SERVICE_SLUG, "version": __version__}


@app.get("/readiness")
def readiness():
    database_url = env_str("DATABASE_URL")
    if not database_url:
        return {
            "status": "ready",
            "store": "in-memory test doubles (records are lost on restart)",
            "decision_backend": env_str("DECISION_BACKEND", "offline"),
        }
    store = get_service().store
    ping = getattr(store, "ping", None)
    if ping is not None and not ping():
        return JSONResponse(
            status_code=503,
            content={"status": "not ready", "store": "PostgreSQL unreachable"},
        )
    return {
        "status": "ready",
        "store": "PostgreSQL",
        "decision_backend": env_str("DECISION_BACKEND", "offline"),
    }


# ---------------------------------------------------------------------------
# tickets
# ---------------------------------------------------------------------------


@app.get("/samples")
def samples():
    return {"tickets": [t.model_dump() for t in load_sample_tickets()]}


@app.post("/tickets/analyze", status_code=201)
def analyze(ticket: TicketInput, request: Request):
    require_api_key(request)
    run = get_service().run_ticket(ticket)
    return run.result.model_dump()


@app.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    record = get_service().get_record(ticket_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
    return record.model_dump()


class DecisionRequest(BaseModel):
    decision: str  # approved | rejected
    by: str = ""
    reason: str = ""
    assignee: str = ""


@app.post("/tickets/{ticket_id}/decision")
def decide_ticket(ticket_id: str, body: DecisionRequest, request: Request):
    require_api_key(request)
    if body.decision not in ("approved", "rejected"):
        raise HTTPException(
            status_code=422, detail="decision must be 'approved' or 'rejected'"
        )
    try:
        record = get_service().record_decision(
            ticket_id,
            body.decision,
            by=body.by,
            reason=body.reason,
            assignee=body.assignee,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return record.model_dump()


class AssignRequest(BaseModel):
    assignee: str
    by: str = ""


@app.post("/tickets/{ticket_id}/assign")
def assign_ticket(ticket_id: str, body: AssignRequest, request: Request):
    require_api_key(request)
    if not body.assignee.strip():
        raise HTTPException(status_code=422, detail="assignee must not be empty")
    try:
        record = get_service().assign(ticket_id, body.assignee.strip(), by=body.by)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
    return record.model_dump()


_FLAG_PARAMS = (
    "contains_pii",
    "is_abusive",
    "mentions_legal",
    "is_outage_report",
    "needs_engineering",
)


@app.get("/queue")
def queue(
    queue: str = "",
    status: str = "",
    priority_band: str = "",
    flag: str = "",
    backend: str = "",
    q: str = "",
):
    if flag and flag not in _FLAG_PARAMS:
        raise HTTPException(
            status_code=422,
            detail=f"flag must be one of {list(_FLAG_PARAMS)}",
        )
    filters = {
        "queue": queue,
        "status": status,
        "priority_band": priority_band,
        "flag": flag,
        "backend": backend,
        "q": q,
    }
    rows = get_service().queue(filters)
    return {"tickets": rows, "count": len(rows)}


@app.get("/metrics")
def metrics():
    return get_service().metrics()


# ---------------------------------------------------------------------------
# stats, policy, outcomes (the dashboards' API)
# ---------------------------------------------------------------------------


@app.get("/api/stats/overview")
def stats_overview():
    return get_service().stats_overview()


@app.get("/api/stats/performance")
def stats_performance():
    return get_service().stats_performance()


@app.get("/api/stats/calibration")
def stats_calibration():
    """Eval-pack calibration next to live calibration, in one payload.

    The eval section is whatever the last eval run wrote (``null``
    with a note when no run has happened yet — the dashboard renders
    the honest empty state instead of the endpoint 404ing); the live
    section scores recorded probabilities against agent outcomes and
    says so plainly when no outcomes exist yet.
    """
    try:
        eval_payload = _results_payload("calibration_results.json", "calibration")
    except HTTPException:
        eval_payload = None
    return {
        "eval": eval_payload,
        "eval_note": None if eval_payload else "no calibration eval results yet — run the eval pack (make evals)",
        "live": get_service().live_calibration(),
    }


@app.get("/api/policy")
def policy():
    from .composition import policy_snapshot

    return policy_snapshot()


class OutcomeRequest(BaseModel):
    final_category: str = ""
    final_queue: str = ""
    resolved: bool = False
    by: str = ""
    # Agent-facing truth keys (was_pii, was_outage, …) or question keys;
    # normalised + validated in livecal before anything is stored.
    verify_truth: dict[str, bool] = {}
    score_truth: dict[str, int] = {}


@app.post("/api/runs/{ticket_id}/outcome")
def record_outcome(ticket_id: str, body: OutcomeRequest, request: Request):
    require_api_key(request)
    from .livecal import (
        normalise_score_truth,
        normalise_verify_truth,
        validate_final_category,
    )
    from .schemas import AgentOutcome

    try:
        outcome = AgentOutcome(
            final_category=validate_final_category(body.final_category),
            final_queue=body.final_queue.strip(),
            resolved=bool(body.resolved),
            recorded_by=body.by,
            verify_truth=normalise_verify_truth(body.verify_truth),
            score_truth=normalise_score_truth(body.score_truth),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        record = get_service().record_outcome(ticket_id, outcome)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
    return record.model_dump()


# ---------------------------------------------------------------------------
# eval + calibration results (written by the eval runners)
# ---------------------------------------------------------------------------


def _results_payload(filename: str, flag: str) -> dict:
    packaged = Path(str(resources.files("support_triage").joinpath("data").joinpath(filename)))
    repo_local = Path(__file__).resolve().parents[2] / "evals" / filename
    for candidate in (packaged, repo_local):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
    raise HTTPException(
        status_code=404,
        detail=f"no {filename} yet — run the eval pack first (see evals/; flag: {flag})",
    )


@app.get("/evals/results")
def eval_results():
    return _results_payload("eval_results.json", "evals")


@app.get("/calibration/results")
def calibration_results():
    return _results_payload("calibration_results.json", "calibration")
