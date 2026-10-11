"""Incident clustering: the outage is twenty tickets wearing twenty names.

Everything else in this system looks at one ticket, or at one
customer's tickets:

- threads (threads.py) group ONE customer's runs into a
  conversation — different customers never share a thread, by
  construction;
- the outage override (composition.py) fires per ticket, when the
  decision answers say *this* ticket reports an outage;
- similar tickets (service.similar_tickets) name one ticket's
  neighbours, one ticket at a time.

None of them can say the sentence a support lead needs at 09:40:
"these fourteen open tickets are one event." When a product
breaks, the queue fills with the same failure reported by
different customers in different words, and each report is triaged
alone. This module is the clustering layer over them.

**Proposals are computed; human acts are stored.** A cluster
proposal is a pure function of the stored run records and the
stored human acts (incidents, dismissals) — deterministic, so it
is recomputed on every read rather than persisted, and it can
never rot against a queue that has moved since it was written.
What IS stored is everything a human did about a proposal:
confirming one creates an :class:`~support_triage.schemas.Incident`
record; dismissing one stores a
:class:`~support_triage.schemas.ClusterDismissal`. The idiom is
the house's: machine judgements are recomputed (threads.py,
dossier.py), human acts are appended to a record and audited
(corrections.py, reviews.py).

**The proposal rule** (the constants below, printed on every
proposal so the rule travels with its output):

- **same placement** — one ``product_area`` AND one ``category``,
  the run record's own decided fields. A cluster never spans
  placements: tickets the triage itself put in different buckets
  are different problems until a human says otherwise.
- **one window** — members arrive within
  :data:`INCIDENT_WINDOW_HOURS` of the group's anchor (its
  earliest report). The window opens at the anchor; earlier
  reports are never pulled into a later window.
- **enough reports** — at least :data:`INCIDENT_MIN_MEMBERS`
  members. Three similar tickets is a coincidence a lead can see
  in the queue unaided; four independent reports inside hours is
  the shape of an event.
- **shared words** — every member's text shares token containment
  at or over :data:`INCIDENT_OVERLAP_THRESHOLD` with the
  ANCHOR's text, using the thread machinery's own measure
  (``threads.tokens_for`` / ``threads.overlap``: content tokens
  of subject+body, shared over the smaller set). Containment
  against the anchor — not pairwise chaining — so every member
  is tied to the first report by name, and the group cannot
  drift topic link by link. The threshold sits below the thread
  threshold (0.35) on purpose: threads match one customer's
  follow-ups, which repeat their own phrasing; an incident is
  many customers describing one failure in their own words, and
  the joint signal (same placement + window + count) carries
  what the looser wording gives up.

**Who may be proposed.** Every stored run is evidence, whatever
its status — an outbreak's first reports may already be decided
before the pattern is visible — with three exclusions, all
accounted by name on the payload: reopen runs (a follow-up is
the same customer continuing, not a new report; it rides its
chain), confirmed duplicates (corrections.py — the same customer
asking again, already linked to its survivor), and members of an
OPEN incident (already claimed by a human's confirmation).
Runs with no decided placement, or no parseable arrival time,
cannot be placed in a window and are excluded and counted too.

**Membership is an overlay, and it moves nothing.** Membership
lives on the incident record; a member run's record — answers,
composition, draft, decision, outcome — is byte-identical
whether or not it belongs to an incident (the suite pins it
across confirm, add, decline and close). No priority is changed,
no runs are merged, no member is closed when the incident
closes: closure is a named act about the incident, and each
member's own outcome stays its own flow's business. What
membership adds is purely agent-facing: a badge on the queue
row, a context line on the run's record view, a block in the
audit export, and the incident page itself — members read live
from their runs, so the page can never disagree with the queue.

**Never re-nag.** A dismissed proposal's member set is
suppressed on every later scan, exactly. A set that gains a
member is a different set — growth is new evidence — and may
propose again, naming the dismissal it outgrew. The same
discipline holds inside an incident: a ticket a lead declined
as an addition is never offered for that incident again.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from . import corrections, threads
from .schemas import ClusterDismissal, Incident, TicketResult
from .sla import parse_timestamp

# The proposal rule. Authored here, in one place, and printed on
# every proposal — a lead reading a proposal sees the rule that
# produced it, not a number they have to trust.
INCIDENT_WINDOW_HOURS = 6
INCIDENT_MIN_MEMBERS = 4
INCIDENT_OVERLAP_THRESHOLD = 0.30


class IncidentError(ValueError):
    """An incident action that cannot stand.

    ``http_status`` is the API's mapping: 409 for a state conflict
    (the proposal no longer stands, the incident is closed, the
    ticket is already a member), 422 for an action invalid on its
    face (missing actor or reason, a ticket that is not a
    candidate for the incident it is being added to).
    """

    http_status = 422


class IncidentConflict(IncidentError):
    http_status = 409


def proposal_constants() -> dict:
    """The authored rule, as data — proposals and the console
    print it beside their output."""
    return {
        "window_hours": INCIDENT_WINDOW_HOURS,
        "min_members": INCIDENT_MIN_MEMBERS,
        "overlap_threshold": INCIDENT_OVERLAP_THRESHOLD,
    }


def proposal_id_for(member_ticket_ids: list[str]) -> str:
    """A proposal's deterministic id: a fingerprint of its member
    set. The same set always computes the same id, so a lead can
    confirm or dismiss a proposal by id and the service can find
    the live proposal it names — or refuse, honestly, when the
    queue has moved and the set no longer computes."""
    digest = hashlib.sha256(
        "\n".join(sorted(member_ticket_ids)).encode("utf-8")
    ).hexdigest()
    return f"clu-{digest[:12]}"


def suggest_title(product_area: str, category: str, count: int, earliest) -> str:
    """The incident's suggested title, composed in code from the
    shared fields — no model invents it. A lead may rename at
    confirmation; the suggestion is what an unnamed confirmation
    carries."""
    area = product_area.replace("_", " ")
    kind = category.replace("_", " ")
    stamp = earliest.strftime("%H:%M") if earliest is not None else "?"
    return f"Possible incident: {kind} in {area} ({count} reports since {stamp} UTC)"


def _eligible(
    records: list[TicketResult],
    open_member_ids: set[str],
) -> tuple[list[TicketResult], dict]:
    """The runs a scan may propose over, and the exclusions by name.

    A run is evidence unless it is a follow-up (reopen), a
    confirmed duplicate, already a member of an open incident, or
    unplaceable (no decided area/category, no parseable arrival).
    """
    eligible: list[TicketResult] = []
    excluded = {
        "reopen_runs": 0,
        "confirmed_duplicates": 0,
        "open_incident_members": 0,
        "unplaced": 0,
    }
    for record in records:
        if record.reopen is not None:
            excluded["reopen_runs"] += 1
        elif corrections.is_confirmed_duplicate(record):
            excluded["confirmed_duplicates"] += 1
        elif record.ticket_id in open_member_ids:
            excluded["open_incident_members"] += 1
        elif (
            not record.product_area
            or not record.category
            or parse_timestamp(record.received_at) is None
        ):
            excluded["unplaced"] += 1
        else:
            eligible.append(record)
    return eligible, excluded


def propose_clusters(
    records: list[TicketResult],
    texts: dict[str, str],
    customers: dict[str, str],
    *,
    incidents: list[Incident],
    dismissals: list[ClusterDismissal],
) -> dict:
    """Compute the live cluster proposals over the stored runs.

    Pure and deterministic: the same records and human acts
    compute the same proposals, byte for byte. Grouping is
    greedy by arrival — within one (product_area, category)
    bucket, the earliest unclaimed run anchors a group; every
    unclaimed run inside its window whose text contains enough
    of the anchor's vocabulary joins it. A group at or over the
    member floor becomes a proposal and claims its members; a
    group that matches a dismissed member set exactly is
    suppressed (and claims its members for this scan, so the
    same set cannot re-form around a neighbour); anything smaller
    is left for a later scan, when more reports may have arrived.
    """
    open_member_ids = {
        member.ticket_id
        for incident in incidents
        if incident.status == "open"
        for member in incident.members
    }
    dismissed_sets = {dismissal.member_set for dismissal in dismissals}
    eligible, excluded = _eligible(records, open_member_ids)
    token_sets = {
        record.ticket_id: threads.tokens_for(
            record.subject, _body_of(texts.get(record.ticket_id, ""), record.subject)
        )
        for record in eligible
    }
    arrivals = {
        record.ticket_id: parse_timestamp(record.received_at) for record in eligible
    }

    buckets: dict[tuple[str, str], list[TicketResult]] = {}
    for record in eligible:
        buckets.setdefault((record.product_area, record.category), []).append(record)

    proposals: list[dict] = []
    suppressed = 0
    claimed: set[str] = set()
    for (area, category), members in sorted(buckets.items()):
        ordered = sorted(
            members, key=lambda r: (arrivals[r.ticket_id], r.ticket_id)
        )
        tried_anchors: set[str] = set()
        for anchor in ordered:
            if anchor.ticket_id in claimed or anchor.ticket_id in tried_anchors:
                continue
            tried_anchors.add(anchor.ticket_id)
            window_end = arrivals[anchor.ticket_id] + timedelta(
                hours=INCIDENT_WINDOW_HOURS
            )
            group = [anchor]
            overlaps: dict[str, float] = {anchor.ticket_id: 1.0}
            for other in ordered:
                if other.ticket_id == anchor.ticket_id:
                    continue
                if other.ticket_id in claimed:
                    continue
                arrived = arrivals[other.ticket_id]
                if arrived < arrivals[anchor.ticket_id] or arrived > window_end:
                    continue
                score = threads.overlap(
                    token_sets[other.ticket_id], token_sets[anchor.ticket_id]
                )
                if score < INCIDENT_OVERLAP_THRESHOLD:
                    continue
                group.append(other)
                overlaps[other.ticket_id] = round(score, 4)
            if len(group) < INCIDENT_MIN_MEMBERS:
                continue
            member_ids = [record.ticket_id for record in group]
            if frozenset(member_ids) in dismissed_sets:
                # The never-re-nag watermark: this exact grouping
                # was refused by a named lead. Claim the members
                # for this scan so the set cannot re-form around a
                # different anchor and nag under a new fingerprint.
                suppressed += 1
                claimed.update(member_ids)
                continue
            claimed.update(member_ids)
            proposals.append(
                _proposal_payload(
                    group, overlaps, customers, area, category, dismissals
                )
            )
    proposals.sort(
        key=lambda p: (p["signals"]["earliest_at"], p["proposal_id"])
    )
    return {
        "constants": proposal_constants(),
        "proposals": proposals,
        "suppressed_dismissed": suppressed,
        "excluded": excluded,
        "scanned_runs": len(records),
    }


def _body_of(text: str, subject: str) -> str:
    """The body half of a stored ticket text.

    ``texts`` carries the service's per-ticket text (subject line
    + body, as the duplicate check embeds it). threads.tokens_for
    tokenises subject and body together, so the subject is
    stripped back off only when it leads the text — what matters
    is that the SAME text feeds the tokeniser here as everywhere
    else; passing the whole text as the body half is equivalent
    (the subject's tokens are included once either way when the
    text starts with the subject)."""
    if text.startswith(subject):
        return text[len(subject):]
    return text


def _proposal_payload(
    group: list[TicketResult],
    overlaps: dict[str, float],
    customers: dict[str, str],
    area: str,
    category: str,
    dismissals: list[ClusterDismissal],
) -> dict:
    member_ids = [record.ticket_id for record in group]
    arrivals = [parse_timestamp(record.received_at) for record in group]
    earliest, latest = min(arrivals), max(arrivals)
    member_overlaps = [overlaps[record.ticket_id] for record in group[1:]]
    outgrown = None
    member_set = frozenset(member_ids)
    for dismissal in dismissals:
        if dismissal.member_set and dismissal.member_set < member_set:
            outgrown = {
                "dismissed_member_count": len(dismissal.member_set),
                "dismissed_by": dismissal.dismissed_by,
                "reason": dismissal.reason,
                "dismissed_at": dismissal.dismissed_at,
            }
            break
    return {
        "proposal_id": proposal_id_for(member_ids),
        "product_area": area,
        "category": category,
        "suggested_title": suggest_title(area, category, len(group), earliest),
        "anchor_ticket_id": group[0].ticket_id,
        "members": [
            {
                "ticket_id": record.ticket_id,
                "received_at": record.received_at,
                "subject": record.subject,
                "sender_name": record.sender_name,
                "status": record.status,
                "overlap": overlaps[record.ticket_id],
            }
            for record in group
        ],
        "signals": {
            **proposal_constants(),
            "member_count": len(group),
            "distinct_customers": len(
                {customers.get(tid, "") for tid in member_ids if customers.get(tid)}
            ),
            "earliest_at": group[0].received_at,
            "latest_at": group[-1].received_at,
            "span_minutes": int((latest - earliest).total_seconds() // 60),
            "min_overlap": min(member_overlaps) if member_overlaps else 1.0,
            "mean_overlap": round(
                sum(member_overlaps) / len(member_overlaps), 4
            )
            if member_overlaps
            else 1.0,
        },
        "outgrows_dismissal": outgrown,
    }


def addition_candidates(
    incident: Incident,
    records: list[TicketResult],
    texts: dict[str, str],
    *,
    incidents: list[Incident],
) -> list[dict]:
    """The runs a lead may add to one OPEN incident, computed live.

    A candidate matches the incident on its own terms: same
    placement, vocabulary tied to the incident's ANCHOR at the
    proposal threshold, arrived no earlier than the anchor and no
    later than the window after the incident's latest member
    arrival (the event is still producing reports). Members,
    declined tickets, and the proposer's own exclusions (reopens,
    confirmed duplicates, other open incidents' members) never
    appear — a declined ticket is never offered twice.
    """
    if incident.status != "open":
        return []
    by_id = {record.ticket_id: record for record in records}
    anchor = by_id.get(incident.anchor_ticket_id)
    if anchor is None:
        return []
    anchor_at = parse_timestamp(anchor.received_at)
    member_arrivals = [
        parse_timestamp(by_id[tid].received_at)
        for tid in incident.member_ids()
        if tid in by_id and parse_timestamp(by_id[tid].received_at) is not None
    ]
    if anchor_at is None or not member_arrivals:
        return []
    cutoff = max(member_arrivals) + timedelta(hours=INCIDENT_WINDOW_HOURS)
    anchor_tokens = threads.tokens_for(
        anchor.subject, _body_of(texts.get(anchor.ticket_id, ""), anchor.subject)
    )
    declined_ids = {entry.ticket_id for entry in incident.declined}
    open_member_ids = {
        member.ticket_id
        for other in incidents
        if other.status == "open"
        for member in other.members
    }
    eligible, _ = _eligible(records, open_member_ids)
    candidates: list[dict] = []
    for record in eligible:
        if record.product_area != incident.product_area:
            continue
        if record.category != incident.category:
            continue
        if record.ticket_id in declined_ids:
            continue
        arrived = parse_timestamp(record.received_at)
        if arrived is None or arrived < anchor_at or arrived > cutoff:
            continue
        score = threads.overlap(
            threads.tokens_for(
                record.subject,
                _body_of(texts.get(record.ticket_id, ""), record.subject),
            ),
            anchor_tokens,
        )
        if score < INCIDENT_OVERLAP_THRESHOLD:
            continue
        candidates.append(
            {
                "ticket_id": record.ticket_id,
                "received_at": record.received_at,
                "subject": record.subject,
                "sender_name": record.sender_name,
                "status": record.status,
                "priority_band": record.composition.priority_band,
                "queue": record.composition.queue,
                "overlap": round(score, 4),
            }
        )
    candidates.sort(key=lambda c: (c["received_at"], c["ticket_id"]))
    return candidates


def candidate_block_reason(
    incident: Incident,
    record: TicketResult,
    records: list[TicketResult],
    texts: dict[str, str],
    incidents: list[Incident],
) -> str | None:
    """Why one run is NOT an addition candidate for an incident —
    or None when it is. The add/decline actions validate against
    the same computation the candidate list shows, and refuse
    with the reason named rather than a bare no."""
    if record.ticket_id in incident.member_ids():
        return f"run {record.ticket_id} is already a member of {incident.incident_id}"
    if any(
        entry.ticket_id == record.ticket_id for entry in incident.declined
    ):
        return (
            f"run {record.ticket_id} was already declined for "
            f"{incident.incident_id} — a declined addition is not re-offered"
        )
    candidates = addition_candidates(
        incident, records, texts, incidents=incidents
    )
    if any(c["ticket_id"] == record.ticket_id for c in candidates):
        return None
    if record.product_area != incident.product_area or record.category != incident.category:
        return (
            f"run {record.ticket_id} is placed {record.product_area}/"
            f"{record.category}; {incident.incident_id} covers "
            f"{incident.product_area}/{incident.category}"
        )
    if record.reopen is not None:
        return f"run {record.ticket_id} is a reopen — a follow-up rides its chain, it is not a new report"
    if corrections.is_confirmed_duplicate(record):
        return f"run {record.ticket_id} is a confirmed duplicate — it rides its survivor's membership"
    for other in incidents:
        if other.status == "open" and other.incident_id != incident.incident_id:
            if record.ticket_id in other.member_ids():
                return f"run {record.ticket_id} already belongs to open incident {other.incident_id}"
    return (
        f"run {record.ticket_id} does not match {incident.incident_id}'s "
        "anchor on words or arrival window — it is not a candidate"
    )


def context_line(incident: Incident) -> str:
    """The agent-facing context line for a member run: which
    incident, how large, what state — the whole of what membership
    says on the run's own surfaces."""
    count = len(incident.members)
    return (
        f"incident: {incident.incident_id} · {incident.title} · "
        f"{count} member{'s' if count != 1 else ''} · {incident.status}"
    )


def incident_view(incident: Incident, records_by_id: dict, now=None) -> dict:
    """One incident as its page serves it: the stored record, its
    members read LIVE from their runs (band, queue, status, SLA
    state — the page can never disagree with the queue), and the
    count of addition candidates waiting on a lead."""
    from .sla import sla_clock

    members = []
    for member in incident.members:
        record = records_by_id.get(member.ticket_id)
        if record is None:
            members.append({**member.model_dump(), "missing": True})
            continue
        clock = sla_clock(record, now)
        members.append(
            {
                **member.model_dump(),
                "subject": record.subject,
                "sender_name": record.sender_name,
                "status": record.status,
                "category": record.category,
                "product_area": record.product_area,
                "priority_band": record.composition.priority_band,
                "priority_score": record.composition.priority_score,
                "queue": record.composition.queue,
                "received_at": record.received_at,
                "sla_state": clock["state"] if clock else None,
            }
        )
    return {
        "incident": incident.model_dump(),
        "context_line": context_line(incident),
        "members": members,
        "member_count": len(incident.members),
        "open_members": sum(1 for m in members if m.get("status") == "awaiting_agent"),
    }


def memberships_for(
    ticket_id: str, incidents: list[Incident]
) -> list[dict]:
    """Every incident a run belongs to (open and closed), as the
    run's own views carry it: the badge facts and the context
    line, nothing recomputed about the run itself."""
    found = []
    for incident in incidents:
        for member in incident.members:
            if member.ticket_id == ticket_id:
                found.append(
                    {
                        "incident_id": incident.incident_id,
                        "title": incident.title,
                        "status": incident.status,
                        "member_count": len(incident.members),
                        "origin": member.origin,
                        "added_by": member.added_by,
                        "added_at": member.added_at,
                        "context_line": context_line(incident),
                    }
                )
    found.sort(key=lambda entry: entry["incident_id"])
    return found
