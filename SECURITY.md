# Security

## Prototype — not hardened for exposure

This blueprint is a **reference prototype**. Optional API-key auth exists
(`API_KEY`), but it is a shared-secret gate for local demos, not an
identity system.

- Run it locally or inside a network you control.
- **Do not expose it to the public internet** as-is. A production
  deployment needs authenticated access, per-client data isolation,
  rate limiting, and secrets management beyond environment variables.

## Data

All tickets bundled with this repository are synthetic. Ticket text is
treated as untrusted input end to end: it is screened for prompt
injection before any model sees it, and flagged passages are fenced
out of model state. If you adapt this blueprint to real support data,
you own the privacy review — tickets routinely contain personal data,
and this prototype's PII handling is a routing flag, not a compliance
program.

## Reporting

Found something that looks like a real vulnerability in the prototype
itself? Open an issue with a minimal reproduction; do not include real
customer data in the report.
