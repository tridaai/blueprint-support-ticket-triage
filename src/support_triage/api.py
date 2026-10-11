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
- the round-5 policy + audit layer: ``POST /api/policy/replay``
  replays a hypothetical policy (threshold, weights, outage boost)
  over every stored run's recorded answers — no backend call,
  nothing written — and ``GET /api/runs/{id}/export`` renders one
  run as a portable audit document (JSON, or plain text with
  ``?format=text``);
- the round-6 reopen + thread layer: ``POST /api/runs/{id}/reopen``
  turns a customer's follow-up into a NEW run linked to the old one
  (prior record offered as decision context, no template draft, a
  one-band priority floor, a reopen limit that forces human
  review), and ``GET /api/threads`` / ``GET /api/threads/{id}`` /
  ``GET /api/runs/{id}/thread`` serve the conversation threads the
  runs were grouped into at intake (threads.py);
- the operator-correction layer (corrections.py): ``POST
  /api/runs/{id}/thread/attach`` / ``.../thread/detach`` / ``POST
  /api/threads/merge`` overrule the computed grouping (actor +
  reason required, chains move whole, never across customers), and
  ``POST /api/runs/{id}/duplicate`` / ``.../duplicate/unmark``
  confirm or lift a duplicate judgement — a confirmed duplicate
  leaves the calibration samples and the drift window, accounted
  in both payloads, while still counting in volume stats;
- the agent's workbench (workbench.py): ``GET
  /api/runs/{id}/workbench`` (and its ``/knowledge``,
  ``/playbook``, ``/note`` parts) joins similar tickets to their
  outcomes, serves the run's category playbook, and composes the
  internal working-note draft; ``POST .../workbench/playbook/check``
  / ``.../uncheck`` / ``.../not-applicable`` / ``.../applicable``
  record the agent's playbook progress, and ``POST
  .../workbench/note`` saves their edited note — overlays on the
  record, all of them; the triage itself is never touched;
- the QA review layer (reviews.py): ``POST /api/reviews/samples``
  draws a seeded, stratified sample over the decided runs (queue ×
  priority band; confirmed duplicates never draw; previously
  reviewed runs excluded unless the request says otherwise), ``GET
  /api/reviews/samples`` / ``/{id}`` list and read samples with
  their verdicts, ``POST /api/reviews/samples/{id}/verdicts``
  records a reviewer's per-family grading (the run's record is
  never edited), and ``GET /api/reviews/report`` serves the
  agreement report — per-family rates, per-backend split, and the
  disagreement list, with rates withheld below the verdict floor;
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
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
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


class ReplayRequest(BaseModel):
    # Every field optional — an omitted field keeps the live value,
    # so an empty body replays the current policy (and moves nothing).
    category_min_confidence: float | None = None
    weights: dict[str, float] | None = None
    outage_boost: bool | None = None


@app.post("/api/policy/replay")
def policy_replay(body: ReplayRequest, request: Request):
    """Replay a hypothetical policy over every stored run.

    The recorded question answers are re-composed in code under the
    hypothetical (confidence threshold, priority weights, outage
    boost on/off) and the runs whose queue, band or guardrail
    outcome would change are reported, before and after, with
    aggregate counts. Decisions are inputs — no backend is called —
    and nothing is persisted: the records are byte-identical after
    a replay. This is the server-side grown-up of the console's
    browser what-if, which only sketches the open queue.
    """
    require_api_key(request)
    try:
        return get_service().policy_replay(
            category_min_confidence=body.category_min_confidence,
            weights=body.weights,
            outage_boost=body.outage_boost,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


class OutcomeRequest(BaseModel):
    final_category: str = ""
    final_queue: str = ""
    resolved: bool = False
    by: str = ""
    # How it ended, in the agent's words — the workbench shows it
    # to the next agent facing a similar ticket. Optional, unjudged.
    resolution_summary: str = ""
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
            resolution_summary=body.resolution_summary.strip(),
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


@app.get("/api/runs/{ticket_id}/export")
def export_run(ticket_id: str, format: str = "json"):
    """One run as a portable audit document (exports.py).

    JSON by default; ``?format=text`` renders the same assembly as
    plain text. Intake with the injection screen's verdict, all
    eleven answers with distributions, the composition trace
    (weights, term-by-term math, routing cell, overrides), the
    guardrail verdicts, the SLA clock, the drift watch's current
    reading on this run's questions, and the recorded outcome when
    one exists. Composed in code from the stored record — no model
    text anywhere in it.
    """
    if format not in ("json", "text"):
        raise HTTPException(status_code=422, detail="format must be 'json' or 'text'")
    try:
        eval_payload = _results_payload("calibration_results.json", "calibration")
        eval_per_question = eval_payload.get("per_question")
    except HTTPException:
        eval_per_question = None
    try:
        document = get_service().export_run(ticket_id, eval_per_question)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
    if format == "text":
        from .exports import render_text

        return PlainTextResponse(render_text(document))
    return document


class ReopenRequest(BaseModel):
    message: str  # the customer's follow-up message — the new run's body
    actor: str = ""  # the agent logging the reopen (kept on the new record)
    subject: str | None = None  # defaults to the prior run's subject


@app.post("/api/runs/{ticket_id}/reopen", status_code=201)
def reopen_run(ticket_id: str, body: ReopenRequest, request: Request):
    """Reopen a handled ticket with the customer's follow-up message.

    Creates a NEW run linked to this one — the prior record is
    never mutated — and runs it through the same pipeline with the
    reopen consequences applied in code (graph.py / composition.py
    / guardrails.py): the prior run's answers and recorded outcome
    are offered to the decision call as fenced history context, the
    template first-response draft is suppressed, the priority band
    is floored one step up, and a chain that reaches REOPEN_LIMIT is
    forced to human review. The response is the new run's record.
    """
    require_api_key(request)
    if not body.message.strip():
        raise HTTPException(
            status_code=422, detail="a reopen needs the customer's follow-up message"
        )
    if not body.actor.strip():
        raise HTTPException(
            status_code=422, detail="actor is required — who is logging this reopen"
        )
    try:
        run = get_service().reopen_run(
            ticket_id,
            message=body.message,
            actor=body.actor.strip(),
            subject=body.subject,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return run.result.model_dump()


@app.get("/api/threads")
def list_threads():
    """Conversation threads over the stored runs (threads.py):
    one entry per thread — runs in order, open/reopened state, and
    the latest recorded outcome in the thread."""
    return get_service().list_threads()


@app.get("/api/threads/{thread_id}")
def get_thread(thread_id: str):
    """One thread's chain: every run in order, each with its outcome."""
    try:
        return get_service().get_thread(thread_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown thread {thread_id!r}")


@app.get("/api/runs/{ticket_id}/thread")
def run_thread(ticket_id: str):
    """The thread the run belongs to — the same chain as
    ``GET /api/threads/{thread_id}``, addressed by any of its runs."""
    try:
        return get_service().run_thread(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")


@app.get("/api/customers/dossier")
def customer_dossier(key: str = ""):
    """One customer's dossier (dossier.py), addressed by customer key.

    The key is the thread machinery's: the normalised sender email,
    or ``name:<normalised name>`` when intake carried no email. The
    dossier is every stored ticket under that key — counts, outcome
    history under the small-sample honesty, threads under the
    EFFECTIVE grouping (operator corrections applied), and the two
    history flags with their ticket-id evidence. A pure projection:
    computing it mutates nothing. An unknown key is a 404 — never
    an empty dossier dressed as knowledge.
    """
    try:
        return get_service().customer_dossier(key)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=exc.args[0] if exc.args else str(exc)
        )


@app.get("/api/runs/{ticket_id}/dossier")
def run_dossier(ticket_id: str):
    """The dossier of the customer behind one stored run — the same
    projection as ``GET /api/customers/dossier``, addressed by any
    of the customer's runs."""
    try:
        return get_service().dossier_for_run(ticket_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=exc.args[0] if exc.args else str(exc)
        )


class ThreadAttachRequest(BaseModel):
    thread_id: str  # the existing thread the run's chain should join
    actor: str = ""  # who is correcting the grouping
    reason: str = ""  # why the machine's grouping was wrong


class ThreadDetachRequest(BaseModel):
    actor: str = ""
    reason: str = ""


class ThreadMergeRequest(BaseModel):
    source_thread_id: str  # folded away
    target_thread_id: str  # survives
    actor: str = ""
    reason: str = ""


class DuplicateMarkRequest(BaseModel):
    duplicate_of: str  # the surviving run this one duplicates
    actor: str = ""
    reason: str = ""


class DuplicateUnmarkRequest(BaseModel):
    actor: str = ""
    reason: str = ""


def _correction_http(exc: Exception):
    """Map a correction refusal onto its status: KeyError → 404,
    CorrectionError → its own (409 conflict / 422 invalid)."""
    from .corrections import CorrectionError

    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=exc.args[0] if exc.args else str(exc))
    if isinstance(exc, CorrectionError):
        raise HTTPException(status_code=exc.http_status, detail=str(exc))
    raise exc


def _require_correction_fields(actor: str, reason: str) -> tuple[str, str]:
    if not actor.strip():
        raise HTTPException(
            status_code=422, detail="actor is required — a correction names who made it"
        )
    if not reason.strip():
        raise HTTPException(
            status_code=422,
            detail="reason is required — a correction states why the machine was wrong",
        )
    return actor.strip(), reason.strip()


@app.post("/api/runs/{ticket_id}/thread/attach")
def attach_run_to_thread(ticket_id: str, body: ThreadAttachRequest, request: Request):
    """Move a run's whole reopen chain into another thread.

    An overlay on the computed grouping (corrections.py): the
    record's computed thread id is preserved, the correction is
    appended with its actor and reason, and thread views resolve
    through it. The target must be an existing thread of the same
    customer. The response is the run's corrected thread view.
    """
    require_api_key(request)
    actor, reason = _require_correction_fields(body.actor, body.reason)
    try:
        return get_service().attach_run_to_thread(
            ticket_id, body.thread_id.strip(), actor=actor, reason=reason
        )
    except (KeyError, ValueError) as exc:
        _correction_http(exc)


@app.post("/api/runs/{ticket_id}/thread/detach")
def detach_run(ticket_id: str, body: ThreadDetachRequest, request: Request):
    """Move a run's whole reopen chain out into its own thread
    (``thr-op-<chain root>``). Refused (409) when the chain is
    already alone in its thread. The response is the new thread's
    view."""
    require_api_key(request)
    actor, reason = _require_correction_fields(body.actor, body.reason)
    try:
        return get_service().detach_run(ticket_id, actor=actor, reason=reason)
    except (KeyError, ValueError) as exc:
        _correction_http(exc)


@app.post("/api/threads/merge")
def merge_threads(body: ThreadMergeRequest, request: Request):
    """Fold one thread into another (the target survives): every
    run in the source thread is corrected into the target, with
    the lineage on each record. Both threads must exist and belong
    to one customer. The response is the surviving thread's view."""
    require_api_key(request)
    actor, reason = _require_correction_fields(body.actor, body.reason)
    try:
        return get_service().merge_threads(
            body.source_thread_id.strip(),
            body.target_thread_id.strip(),
            actor=actor,
            reason=reason,
        )
    except (KeyError, ValueError) as exc:
        _correction_http(exc)


@app.post("/api/runs/{ticket_id}/duplicate")
def mark_duplicate(ticket_id: str, body: DuplicateMarkRequest, request: Request):
    """Confirm this run as a duplicate of another run (same customer).

    The human's word over the intake similarity guess: the mark
    lands on this run's record only, and while it stands the run is
    excluded from calibration sampling and the drift window
    (accounted in both payloads) — it still counts in volume stats.
    A run with a live mark cannot be marked again (409, naming the
    existing mark). The response is the updated record.
    """
    require_api_key(request)
    actor, reason = _require_correction_fields(body.actor, body.reason)
    try:
        record = get_service().mark_duplicate(
            ticket_id, body.duplicate_of.strip(), actor=actor, reason=reason
        )
    except (KeyError, ValueError) as exc:
        _correction_http(exc)
    return record.model_dump()


@app.post("/api/runs/{ticket_id}/duplicate/unmark")
def unmark_duplicate(ticket_id: str, body: DuplicateUnmarkRequest, request: Request):
    """Lift a confirmed-duplicate mark. The mark entry stays on the
    record with its un-mark half filled in — the audit trail keeps
    both halves — and the run samples normally again. The response
    is the updated record."""
    require_api_key(request)
    actor, reason = _require_correction_fields(body.actor, body.reason)
    try:
        record = get_service().unmark_duplicate(ticket_id, actor=actor, reason=reason)
    except (KeyError, ValueError) as exc:
        _correction_http(exc)
    return record.model_dump()


# ---------------------------------------------------------------------------
# the agent's workbench (knowledge, playbook, working note)
# ---------------------------------------------------------------------------


class PlaybookStepRequest(BaseModel):
    step_id: str  # the playbook step being checked / unchecked
    actor: str = ""  # who is working the ticket


class PlaybookNaRequest(BaseModel):
    actor: str = ""
    reason: str = ""  # why the playbook does not fit (mark only)


class PlaybookLiftRequest(BaseModel):
    actor: str = ""


class WorkingNoteRequest(BaseModel):
    text: str  # the agent's edited note
    actor: str = ""


def _workbench_http(exc: Exception):
    """Map a workbench refusal onto its status: KeyError → 404,
    WorkbenchError → its own (409 conflict / 422 invalid)."""
    from .workbench import WorkbenchError

    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=exc.args[0] if exc.args else str(exc))
    if isinstance(exc, WorkbenchError):
        raise HTTPException(status_code=exc.http_status, detail=str(exc))
    raise exc


def _require_actor_field(actor: str) -> str:
    if not actor.strip():
        raise HTTPException(
            status_code=422, detail="actor is required — a workbench action names who made it"
        )
    return actor.strip()


@app.get("/api/runs/{ticket_id}/workbench")
def workbench_view(ticket_id: str):
    """The whole workbench for one run: resolution knowledge, the
    category playbook with its progress, and the working-note draft
    (plus the saved note, when one exists)."""
    try:
        return get_service().workbench(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")


@app.get("/api/runs/{ticket_id}/workbench/knowledge")
def workbench_knowledge(ticket_id: str):
    """How similar tickets actually ended: the ones with outcomes,
    their resolutions as recorded, and the aggregate over them —
    withheld, with the reason, when the sample is too small."""
    try:
        return get_service().workbench_knowledge(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")


@app.get("/api/runs/{ticket_id}/workbench/playbook")
def workbench_playbook(ticket_id: str):
    """The run's category playbook with the agent's progress on it."""
    try:
        return get_service().workbench_playbook(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")


@app.get("/api/runs/{ticket_id}/workbench/note")
def workbench_note(ticket_id: str):
    """The internal working-note draft, composed by code from the
    record — and the saved note, when the agent has saved one. The
    draft is not stored by reading it; saving is the POST below."""
    try:
        return get_service().working_note_draft(ticket_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ticket {ticket_id!r}")


@app.post("/api/runs/{ticket_id}/workbench/playbook/check")
def check_playbook_step(ticket_id: str, body: PlaybookStepRequest, request: Request):
    """Check one playbook step off on the record (overlay entry).
    The response is the updated record; the triage is untouched."""
    require_api_key(request)
    actor = _require_actor_field(body.actor)
    try:
        record = get_service().check_playbook_step(
            ticket_id, body.step_id.strip(), actor=actor, checked=True
        )
    except (KeyError, ValueError) as exc:
        _workbench_http(exc)
    return record.model_dump()


@app.post("/api/runs/{ticket_id}/workbench/playbook/uncheck")
def uncheck_playbook_step(ticket_id: str, body: PlaybookStepRequest, request: Request):
    """Undo a check-off (the trail keeps both entries)."""
    require_api_key(request)
    actor = _require_actor_field(body.actor)
    try:
        record = get_service().check_playbook_step(
            ticket_id, body.step_id.strip(), actor=actor, checked=False
        )
    except (KeyError, ValueError) as exc:
        _workbench_http(exc)
    return record.model_dump()


@app.post("/api/runs/{ticket_id}/workbench/playbook/not-applicable")
def mark_playbook_na(ticket_id: str, body: PlaybookNaRequest, request: Request):
    """Mark the run's playbook not-applicable, with the reason.
    Step checks are refused while the mark stands."""
    require_api_key(request)
    actor = _require_actor_field(body.actor)
    if not body.reason.strip():
        raise HTTPException(
            status_code=422,
            detail="reason is required — a not-applicable mark states why the playbook does not fit",
        )
    try:
        record = get_service().mark_playbook_na(
            ticket_id, actor=actor, reason=body.reason.strip()
        )
    except (KeyError, ValueError) as exc:
        _workbench_http(exc)
    return record.model_dump()


@app.post("/api/runs/{ticket_id}/workbench/playbook/applicable")
def lift_playbook_na(ticket_id: str, body: PlaybookLiftRequest, request: Request):
    """Lift a not-applicable mark (the mark entry stays, audited)."""
    require_api_key(request)
    actor = _require_actor_field(body.actor)
    try:
        record = get_service().lift_playbook_na(ticket_id, actor=actor)
    except (KeyError, ValueError) as exc:
        _workbench_http(exc)
    return record.model_dump()


@app.post("/api/runs/{ticket_id}/workbench/note")
def save_working_note(ticket_id: str, body: WorkingNoteRequest, request: Request):
    """Save the agent's edited internal working note on the record.
    Internal only — this surface has no send path, by design."""
    require_api_key(request)
    actor = _require_actor_field(body.actor)
    try:
        record = get_service().save_working_note(ticket_id, text=body.text, actor=actor)
    except (KeyError, ValueError) as exc:
        _workbench_http(exc)
    return record.model_dump()


# ---------------------------------------------------------------------------
# QA review (samples, verdicts, agreement report)
# ---------------------------------------------------------------------------


class ReviewSampleRequest(BaseModel):
    size: int  # runs to draw
    seed: int = 0  # the draw's seed — recorded on the sample
    created_by: str = ""  # who drew the sample
    exclude_previously_reviewed: bool = True


class ReviewVerdictRequest(BaseModel):
    ticket_id: str  # a run the sample drew
    reviewer: str = ""  # who is grading it
    families: dict[str, str] = {}  # family -> agree | disagree (all four)
    overall: str = ""  # agree | disagree
    note: str = ""


def _review_http(exc: Exception):
    """Map a review refusal onto its status: KeyError → 404,
    ReviewError → its own (409 conflict / 422 invalid)."""
    from .reviews import ReviewError

    if isinstance(exc, KeyError):
        raise HTTPException(status_code=404, detail=exc.args[0] if exc.args else str(exc))
    if isinstance(exc, ReviewError):
        raise HTTPException(status_code=exc.http_status, detail=str(exc))
    raise exc


@app.post("/api/reviews/samples", status_code=201)
def create_review_sample(body: ReviewSampleRequest, request: Request):
    """Draw a QA review sample over the decided runs (reviews.py).

    Stratified by queue and priority band, seeded and reproducible;
    confirmed duplicates never draw, and runs already reviewed in
    an earlier sample sit out unless ``exclude_previously_reviewed``
    is false. The sample is stored (seed, strata, run ids,
    exclusions by name) so the round is auditable and re-runnable.
    The response is the sample record.
    """
    require_api_key(request)
    if not body.created_by.strip():
        raise HTTPException(
            status_code=422, detail="created_by is required — a sample names who drew it"
        )
    try:
        sample = get_service().create_review_sample(
            size=body.size,
            seed=body.seed,
            created_by=body.created_by.strip(),
            exclude_previously_reviewed=body.exclude_previously_reviewed,
        )
    except ValueError as exc:
        _review_http(exc)
    return sample.model_dump()


@app.get("/api/reviews/samples")
def list_review_samples():
    """Every review sample drawn, oldest first."""
    return {
        "samples": [s.model_dump() for s in get_service().list_review_samples()]
    }


@app.get("/api/reviews/samples/{sample_id}")
def get_review_sample(sample_id: str):
    """One sample with the verdicts recorded against it so far."""
    try:
        return get_service().get_review_sample(sample_id)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown review sample {sample_id!r}"
        )


@app.post("/api/reviews/samples/{sample_id}/verdicts", status_code=201)
def record_review_verdict(sample_id: str, body: ReviewVerdictRequest, request: Request):
    """Record one reviewer's grading of one sampled run.

    Per-family agree / disagree (all four families), an overall
    verdict, and a note. The verdict lands on the review record;
    the run's triage record is never edited by being graded. One
    verdict per (sample, run, reviewer) — a repeat is a 409.
    """
    require_api_key(request)
    if not body.reviewer.strip():
        raise HTTPException(
            status_code=422, detail="reviewer is required — a verdict names who graded it"
        )
    try:
        verdict = get_service().record_review_verdict(
            sample_id,
            ticket_id=body.ticket_id.strip(),
            reviewer=body.reviewer.strip(),
            families=body.families,
            overall=body.overall,
            note=body.note,
        )
    except (KeyError, ValueError) as exc:
        _review_http(exc)
    return verdict.model_dump()


@app.get("/api/reviews/report")
def review_report(samples: str = ""):
    """The agreement report: per-family rates, per-backend split,
    overall agreement, and the disagreement list — over one sample
    (``?samples=QAS-0001``), a comma-separated set, or all samples.

    A read model: rates are withheld below the verdict floor (the
    counts always show), and nothing in the report feeds a weight,
    a threshold, or a prompt.
    """
    sample_ids = [part.strip() for part in samples.split(",") if part.strip()]
    try:
        return get_service().review_report(sample_ids or None)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=exc.args[0] if exc.args else str(exc)
        )


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
