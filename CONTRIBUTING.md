# Contributing

This is a reference prototype. Contributions that make the core clearer,
better tested, or easier to take and adapt are welcome.

## Dev setup

Primary workflow is [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
```

No uv? `python3 -m venv .venv && source .venv/bin/activate`, then
`make install-pip` — it installs the frozen, hash-verified set from
`constraints.txt` first, then the package itself with `--no-deps`
(pip refuses an editable install under a hashed constraints file, so
the two steps never mix).

## Before opening a PR

Run all three and make sure they pass:

```bash
make test        # pytest suite (hermetic — in-memory doubles)
make evals       # golden routing gate + calibration gate
make demo        # one ticket, traced end to end, offline
```

## Ground rules

- **Decisions, not drafts.** The decision model returns typed,
  calibrated answers behind the `DecisionEngine` port; prose is
  produced only by the deterministic template path, and nothing is
  ever sent to a customer by this system.
- **Code owns the consequences.** Priority scores, routing, and the
  guardrails are deterministic code. A change that moves a consequence
  from code into a prompt needs a very good reason and a test.
- **Synthetic data only.** Tickets, customers, and metrics in this
  repo are invented. Never commit real support data.
