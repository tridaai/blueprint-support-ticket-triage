"""Command line: list the sample queue, triage a ticket, record a decision.

    support-triage samples                       # the bundled synthetic tickets
    support-triage run SUP-1002                  # triage one sample to the gate
    support-triage run --file ticket.json        # triage a ticket file
    support-triage decide SUP-1002 --decision approved --by agent-1
    support-triage queue                         # what's waiting on agents
    support-triage show SUP-1002                 # the full decision record

State persists where the service persists it: PostgreSQL when
DATABASE_URL is set, per-process memory otherwise (each CLI call is
its own process — the queue/decide commands are meaningful against
the API server or a database, and say so when they find nothing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_dotenv, silence_langchain_deprecation_warnings

silence_langchain_deprecation_warnings()

from .events import PrintingSink  # noqa: E402
from .samples import load_sample_tickets  # noqa: E402
from .schemas import TicketInput  # noqa: E402
from .wiring import build_service  # noqa: E402


def _find_sample(ticket_id: str) -> TicketInput | None:
    for ticket in load_sample_tickets():
        if ticket.ticket_id == ticket_id:
            return ticket
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="support-triage")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("samples", help="list the bundled synthetic tickets")

    run_parser = sub.add_parser("run", help="triage a ticket to the agent gate")
    run_parser.add_argument("ticket_id", nargs="?", help="a bundled sample id (e.g. SUP-1002)")
    run_parser.add_argument("--file", help="path to a ticket JSON file")

    decide_parser = sub.add_parser("decide", help="record an agent decision")
    decide_parser.add_argument("ticket_id")
    decide_parser.add_argument("--decision", choices=["approved", "rejected"], required=True)
    decide_parser.add_argument("--by", default="")
    decide_parser.add_argument("--reason", default="")
    decide_parser.add_argument("--assignee", default="")

    sub.add_parser("queue", help="tickets waiting on an agent")

    show_parser = sub.add_parser("show", help="the full decision record for a ticket")
    show_parser.add_argument("ticket_id")

    args = parser.parse_args(argv)
    load_dotenv()
    service = build_service()

    if args.command == "samples":
        for ticket in load_sample_tickets():
            print(f"{ticket.ticket_id}  {ticket.subject}")
        return 0

    if args.command == "run":
        if args.file:
            ticket = TicketInput.model_validate(
                json.loads(Path(args.file).read_text(encoding="utf-8"))
            )
        elif args.ticket_id:
            ticket = _find_sample(args.ticket_id)
            if ticket is None:
                print(f"error: no bundled sample {args.ticket_id!r} (try: support-triage samples)")
                return 1
        else:
            print("error: give a sample ticket id or --file")
            return 1
        try:
            run = service.run_ticket(ticket, sink=PrintingSink())
        except Exception as exc:
            print(f"error: {exc}")
            return 1
        result = run.result
        print(
            f"\n{result.ticket_id}: {result.category} → {result.composition.queue} · "
            f"priority {result.composition.priority_score} ({result.composition.priority_band}) · "
            f"status {result.status}"
        )
        return 0

    if args.command == "decide":
        try:
            record = service.record_decision(
                args.ticket_id,
                args.decision,
                by=args.by,
                reason=args.reason,
                assignee=args.assignee,
            )
        except (KeyError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        print(f"{record.ticket_id}: decision {args.decision} recorded → status {record.status}")
        return 0

    if args.command == "queue":
        rows = service.queue()
        if not rows:
            print("queue is empty (in-memory state is per-process — run tickets via the API server or set DATABASE_URL)")
            return 0
        for row in rows:
            print(
                f"{row['ticket_id']}  [{row['priority_band']:>8} {row['priority_score']:>5}]  "
                f"{row['queue']:<24} {row['subject'][:60]}"
            )
        return 0

    if args.command == "show":
        record = service.get_record(args.ticket_id)
        if record is None:
            print(f"error: unknown ticket {args.ticket_id!r}")
            return 1
        print(record.model_dump_json(indent=2))
        return 0

    return 1  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
