# AGENTS.md


<!-- AGENTS-CODEX-PREFLIGHT-v1 -->
## For Codex / non-Claude agents — task-scoped context by default

If you are Codex (or any agent that does not auto-load `CLAUDE.md`):
- In lightweight mode, wait for the task, then read only the relevant repo instruction sections,
  contracts, skills, and commands. Do not run a workspace audit solely because the session began.
- `codex --full-context` restores the root audit, complete local instruction read, checklist, and
  PHASE 0.5 restatement.
- **PHASE 0.25 context proof is task-scoped (protocol v1.1).** Run it *before
  content/entity-producing work* — any task in `governance/`, editing a Core doc, or
  producing/persisting customer/product/team/vendor-facing output (docs, quotes, decks,
  emails, reports, release notes, named entities). Cycle:
  `python3 ~/repos/governance/enforcement/generate_challenge.py`, emit a `CONTEXT_PROOF`
  (format: `~/repos/governance/enforcement/test_fixtures/complete_proof.yaml`), then
  `python3 ~/repos/governance/enforcement/verify_agent_context.py --proof <file>` — proceed
  only on exit 0. **Deferred for pure code/test/build/debug work**; the commit-time spelling
  scanner (`gate_25_entity_names.py`) is the always-on net, so deferring the upfront read
  never lets a wrong spelling land. Authoritative:
  `governance/protocols/MANDATORY_CONTEXT_VERIFICATION_PROTOCOL.md`.
- Treat applicable contracts and policies from `~/repos/governance/INDEX.md` as binding once the
  task enters their scope.
<!-- /AGENTS-CODEX-PREFLIGHT-v1 -->

<!-- GOVERNANCE-PREFLIGHT-v1 -->
## Governance Pre-Flight (summary — binding rules live in governance/)

Interactive Codex sessions use lightweight startup by default: no workspace audit or PHASE 0.5
restatement before the task is known. `codex --full-context` restores both. Other launchers retain
their configured startup behavior.

After task scope is known:
- Load only the relevant repo instructions and governance contracts.
- Use the **canonical document template** for any document — do not invent a format.

**AGENTS NEVER SEND (absolute order).** No email, calendar invitation, meeting update or
cancellation, or message of any kind — to a customer or to **anyone else**. **A calendar invite
with an attendee IS a message**, as is a time change, a cancellation, and any tool call with a
`notify`/`notificationLevel`/`sendUpdates` parameter. *"Set up a call with X"* authorizes
preparing the call, **not** contacting X. Produce drafts; **Chelsea Collado handles customer
communications**, Chad transmits or delegates the rest.

This is a summary; the binding rules and full checklists live in governance (source of truth):
- `~/repos/governance/policies/AGENT_INTERACTION_POLICY.md` — startup sequence + PHASE 0.5,
  and §"External Communication and Representation — Agents Do Not Transmit"
- `~/repos/governance/standards/DOCUMENT_TEMPLATE_REGISTRY.md` — which template to use
- `~/repos/governance/INDEX.md` — master registry of all contracts, policies, gates
<!-- /GOVERNANCE-PREFLIGHT-v1 -->

## Governance Prerequisite (Non-Negotiable)

**Before any work in this repository, read and comply with:** [`~/repos/governance/INDEX.md`](../governance/INDEX.md)

All cross-repo contracts, policies, and enforcement gates in `~/repos/governance/` are binding. Repo-specific rules below may extend but never override governance contracts.

## Required Reading

This file is intentionally minimal. **You MUST also read `CLAUDE.md` in this repository** — it contains mandatory rules, contracts, and procedures that AGENTS.md does not repeat.

If both files exist, follow both. CLAUDE.md has the detailed guidance; this file ensures Codex agents discover it.

## Skills (shared index)

CHAD Suite skills live in `~/repos/claude-config/skills/<name>/SKILL.md`. A
machine-readable index of every skill (name, description, path) is at:

- `~/repos/claude-config/skills/SKILLS_INDEX.json` — load to discover available skills
- `~/repos/claude-config/skills/SKILLS_INDEX.md` — human-readable table

To use a skill, read its `SKILL.md` and follow it. Regenerate after changing
skills: `python3 ~/repos/claude-config/scripts/gen_skills_index.py`.

## Agent Rules

- Complete PHASE 0 instruction audit before any code changes
- Read `~/repos/governance/policies/AGENT_INTERACTION_POLICY.md` for full agent protocol
- 3 failed attempts at same fix → STOP and escalate
- 5 failed attempts → FORBIDDEN from further fixes
- Never modify governance documents without Chad's explicit approval
- Always include `Co-Authored-By:` line in commits identifying the agent/model
