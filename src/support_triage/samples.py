"""Bundled synthetic sample tickets.

Every ticket in this repository is invented — names, companies,
amounts, card numbers (a well-known public test number), all of it.
They exist so the demo, the console, and a first ``curl`` show the
pipeline working across the interesting shapes: a double charge, an
outage report, a calm how-to, an injection attempt, a legal-flavoured
complaint with a card number in it, a cancellation.
"""

from __future__ import annotations

import json
from importlib import resources

from .schemas import TicketInput


def load_sample_tickets() -> list[TicketInput]:
    text = (
        resources.files("support_triage")
        .joinpath("data/sample_tickets.json")
        .read_text(encoding="utf-8")
    )
    return [TicketInput.model_validate(item) for item in json.loads(text)]
