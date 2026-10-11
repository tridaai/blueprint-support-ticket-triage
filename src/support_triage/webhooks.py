"""Gate webhooks: tell the outside world a ticket is waiting.

Optional and off by default. When ``TRIAGE_WEBHOOK_URL`` is set,
the service POSTs a compact JSON event when a run reaches the agent
gate (``run_completed``) and when its guardrails blocked
(``guardrail_blocked``)::

    {"event": "run_completed", "ticket_id": "…", "queue": "…",
     "priority_band": "…", "status": "awaiting_agent",
     "guardrails": {"passed": true, "requires_human_review": false,
                    "failed_checks": []},
     "at": "2026-…Z"}

``TRIAGE_WEBHOOK_SECRET``, when set, rides along as the
``X-Triage-Webhook-Secret`` header so the receiver can check the
event really came from this deployment.

Delivery is fire-and-forget on a daemon thread with a short
timeout: a dead webhook receiver can slow nothing and fail nothing —
failures are logged and forgotten, by design. The run that triggered
the event is already saved before the webhook is attempted.
"""

from __future__ import annotations

import json
import logging
import threading
import urllib.request

from .config import env_str

log = logging.getLogger("support_triage.webhooks")

TIMEOUT_SECONDS = 3.0
SECRET_HEADER = "X-Triage-Webhook-Secret"


def _post(url: str, headers: dict, body: dict, timeout: float) -> None:
    """One HTTP POST. Kept tiny and separate: tests stub this."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        response.read()


def _deliver(url: str, headers: dict, body: dict) -> None:
    try:
        _post(url, headers, body, TIMEOUT_SECONDS)
    except Exception as exc:  # a webhook must never surface in a run
        log.warning("webhook delivery to %s failed: %s", url, exc)


def notify(event_type: str, payload: dict) -> threading.Thread | None:
    """Fire one webhook for a lifecycle event, if configured.

    Returns the delivery thread (tests join it), or ``None`` when no
    webhook URL is configured — the common case, and a no-op.
    """
    url = env_str("TRIAGE_WEBHOOK_URL")
    if not url:
        return None
    headers: dict[str, str] = {}
    secret = env_str("TRIAGE_WEBHOOK_SECRET")
    if secret:
        headers[SECRET_HEADER] = secret
    body = {"event": event_type, **payload}
    thread = threading.Thread(
        target=_deliver, args=(url, headers, body), daemon=True, name="triage-webhook"
    )
    thread.start()
    return thread
