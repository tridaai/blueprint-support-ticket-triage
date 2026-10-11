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
- the round-3 operations surface: ``POST /api/batches`` triages a
  list of tickets (per-ticket failures reported, never contagious),
  ``POST /api/compare`` runs one ticket through several decision
  backends side by side without queueing it, ``GET
  /api/runs/{id}/similar`` finds past tickets like this one (with
  their outcomes), and ``GET /api/events`` streams run lifecycle
  events (SSE) from the service's in-process bus;
- the round-4 watch + suggestion layer: ``GET /api/stats/drift``
  compares the recent live window's per-question Brier against the
  eval baseline (drift flags only past a sample floor), and ``GET
  /api/runs/{id}/assignment`` suggests an assignee from the roster
  with its reasons printed — advisory only, it assigns nothing;
- ``/readiness`` fails (503) when PostgreSQL is configured but
  unreachable, and says honestly when the in-memory test doubles are
  in use (records lost on restart — fine for a look, not for work).
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue as queue_module
import sys
import time
import uuid
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
# batch intake + lifecycle event stream
# ---------------------------------------------------------------------------


@app.post("/api/batches", status_code=201)
async def create_batch(request: Request):
    """Triage a list of tickets: ``{"tickets": [...]}`` or a bare list.

    Every ticket runs the full pipeline to its own gate. The response
    carries one summary per ticket — queue, band, status — or that
    ticket's error; a bad ticket never fails the batch. Batches past
    the service cap are rejected whole (422), never silently cut.
    """
    require_api_key(request)
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="request body must be JSON")
    items = payload.get("tickets") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise HTTPException(
            status_code=422, detail="body must be a ticket list or {\"tickets\": [...]}"
        )
    try:
        return get_service().run_batch(items)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/api/events")
async def events_stream(request: Request):
    """Server-sent events for the run lifecycle, from the service bus.

    Event types: ``run_completed`` (a run reached the agent gate),
    ``guardrail_blocked`` (its guardrails failed a check),
    ``outcome_recorded``, ``batch_completed``. The console subscribes
    and refreshes; anything that misses an event (a dropped stream)
    still has the plain polling endpoints — the stream is a
    convenience, never the only path to the truth.
    """
    bus = get_service().bus
    subscriber = bus.subscribe()

    async def generate():
        # NB: no request.is_disconnected() polling — under the
        # middleware stack that call competes for the receive channel
        # and can stall the stream. Client disconnects surface as a
        # cancellation here instead, and the finally still unsubscribes.
        try:
            yield ": connected\n\n"
            last_ping = time.monotonic()
            while True:
                try:
                    event = subscriber.get_nowait()
                except queue_module.Empty:
                    await asyncio.sleep(0.05)
                    if time.monotonic() - last_ping >= 15:
                        yield ": ping\n\n"
                        last_ping = time.monotonic()
                    continue
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                last_ping = time.monotonic()
        finally:
            bus.unsubscribe(subscriber)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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


@app.get("/api/stats/drift")
def stats_drift():
    """Calibration drift watch: the recent window vs the eval baseline.

    Per question, the Brier over the last DRIFT_WINDOW labelled
    outcomes next to the all-time live Brier and the eval pack's
    baseline Brier; a question flags only with enough window samples
    and a degradation past the threshold in force (both reported in
    the payload's ``window`` block). Without an eval run there is no
    baseline, and the payload says so instead of flagging.
    """
    try:
        eval_payload = _results_payload("calibration_results.json", "calibration")
        eval_per_question = eval_payload.get("per_question")
    except HTTPException:
        eval_per_question = None
    return get_service().stats_drift(eval_per_question)


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


# ---------------------------------------------------------------------------
# backend comparison + similar tickets
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    ticket: TicketInput
    backends: list[str] | None = None


@app.post("/api/compare", status_code=201)
def create_comparison(body: CompareRequest, request: Request):
    """One ticket, several decision backends, side by side.

    An experiment, not a triage: the ticket is NOT queued and no run
    record is created (the duplicate index is untouched too). The
    comparison is archived as its own record and read back at
    ``GET /api/compare/{comparison_id}``. A backend that is not
    configured answers with an error column, reported — the other
    columns still stand.
    """
    require_api_key(request)
    from .compare import CANDIDATE_BACKENDS

    unknown = [b for b in (body.backends or []) if b not in CANDIDATE_BACKENDS]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"unknown backends {unknown}; candidates are {list(CANDIDATE_BACKENDS)}",
        )
    return get_service().compare(body.ticket, body.backends)


@app.get("/api/compare/{comparison_id}")
def get_comparison(comparison_id: str):
    payload = get_service().get_comparison(comparison_id)
    if payload is None:
        raise HTTPException(status_code=404, detail=f"unknown comparison {comparison_id!r}")
    return payload


@app.get("/api/runs/{ticket_id}/similar")
def similar_tickets(ticket_id: str, limit: int = 5):
    """Past tickets nearest to this one (embedding index), each with
    its recorded outcome when an agent has recorded one."""
    if limit < 1 or limit > 20:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 20")
    try:
        return get_service().similar_tickets(ticket_id, limit=limit)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")


@app.get("/api/runs/{ticket_id}/assignment")
def assignment_suggestion(ticket_id: str):
    """A suggested assignee for a ticket at the gate — advisory only.

    Ranks the roster agents covering the ticket's queue by track
    record in the ticket's category (recorded outcomes), current
    open load, and skill overlap, with the reasons printed. This
    endpoint reads; it has never assigned anything. Assigning is
    the human's ``POST /tickets/{id}/assign``.
    """
    try:
        return get_service().assignment_suggestion(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
