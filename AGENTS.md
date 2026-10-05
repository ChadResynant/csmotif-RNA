# AGENTS.md


<!-- AGENTS-CODEX-PREFLIGHT-v1 -->
## For Codex / non-Claude agents — task-scoped context by default

If you are Codex (or any agent that does not auto-load `CLAUDE.md`):
- In lightweight mode, wait for the task, then read only the relevant repo instruction sections,
  contracts, skills, commands, and lessons before changing files. Startup alone does not require
  a workspace audit, complete root-context read, synchronization, checklist, or PHASE 0.5
  restatement. Mechanical gates and task-scoped context proof remain active.
- `codex --full-context` restores the root audit, complete local instruction read, checklist, and
  PHASE 0.5 restatement under the Agent Interaction Policy. Scale the execution steps and binding
  constraints to the task; never force a fixed number of steps or policies.
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

All agents — Claude, Codex, Grok, Gemini, Hermes — before starting a task:
- Follow the task-scoped startup and pre-flight requirements in the linked Agent Interaction
  Policy. Scale the plan and cited policies to the task; do not force a fixed number of steps or
  policies. Lightweight Codex work does not imply a full-context audit.
- Use the **canonical document template** for any document — do not invent a format.
- Before reporting completion, run
  `~/repos/repos-config/scripts/branch_worktree_lifecycle_gate.py --root ~/repos`; the binding
  cleanup-debt invariant is `~/repos/governance/AGENTS.md#universal-branchworktree-lifecycle-invariant`.

**AGENTS NEVER SEND (absolute order).** No email, calendar invitation, meeting update or
cancellation, or message of any kind — to a customer or to **anyone else**. **A calendar invite
with an attendee IS a message**, as is a time change, a cancellation, and any tool call with a
`notify`/`notificationLevel`/`sendUpdates` parameter. *"Set up a call with X"* authorizes
preparing the call, **not** contacting X. Produce drafts; **Chelsea Collado handles customer
communications**, Chad transmits or delegates the rest.

This is a summary; the binding rules and full checklists live in governance (source of truth):
- `~/repos/governance/policies/AGENT_INTERACTION_POLICY.md` — startup sequence + PHASE 0.5,
  and §"External Communication and Representation — Agents Do Not Transmit"
- `~/repos/claude-config/skills/prepare-cad-drawing/SKILL.md` and
  `~/repos/governance/contracts/CAD_EXPORT_FORMATS_CONTRACT.md` — engineering drawing and print
  exports, including required project/batch/date/revision/source-SHA/input-digest traceability.
- `~/repos/governance/standards/DOCUMENT_TEMPLATE_REGISTRY.md` — which template to use
- `~/repos/governance/INDEX.md` — master registry of all contracts, policies, gates
<!-- /GOVERNANCE-PREFLIGHT-v1 -->

<!-- EXECUTIVE-BREVITY-PROJECTION:BEGIN -->
## Executive Brevity Standard (generated; do not edit)

Canonical source: `governance/policies/AGENT_INTERACTION_POLICY.md#executive-brevity-standard-ebs-1`

Canonical SHA-256: `ac12a3f7444ce04b5de1c0a6bf78987eedf9e1c744c66deb12c80ac7691fe077`

Apply the exact canonical policy carried in this generated projection:

```json
{
  "schema": "resynant.executive_brevity.projection@1",
  "source": "governance/policies/AGENT_INTERACTION_POLICY.md#executive-brevity-standard-ebs-1",
  "canonical_sha256": "ac12a3f7444ce04b5de1c0a6bf78987eedf9e1c744c66deb12c80ac7691fe077",
  "policy": {
    "schema": "resynant.executive_brevity@1",
    "applies_to": [
      "claude",
      "codex",
      "grok",
      "antigravity",
      "hermes",
      "resy",
      "local-model",
      "botresynant",
      "orchestrator",
      "subagent",
      "future-agent-installation"
    ],
    "default": {
      "max_interactive_words": 300,
      "lead_with": [
        "decision",
        "verdict",
        "current_status"
      ],
      "report_only": [
        "blockers",
        "material_risks",
        "decisions_required",
        "next_action"
      ],
      "required_no_blockers_phrase": "No blockers. Proceed.",
      "default_sections": [
        "STATUS",
        "BLOCKERS",
        "DECISION NEEDED",
        "NEXT ACTION"
      ],
      "tables": "essential_only",
      "detail_storage": "durable_artifact",
      "internal_analysis_visibility": "does_not_expand_user_report"
    },
    "do_not_restate": [
      "accepted_architecture",
      "history",
      "accepted_evidence",
      "accepted_rationale",
      "prior_findings"
    ],
    "review_cycle_allowed_only_for": [
      "concrete_contradiction",
      "failed_invariant",
      "security_defect",
      "explicit_human_request"
    ],
    "detail_override_phrases": [
      "full report",
      "detailed analysis",
      "deep review"
    ],
    "detail_override_equivalents_allowed": true,
    "detail_override_scope": "current_task_only",
    "violation_class": "reporting_quality_defect",
    "work_validity_effect": "none"
  }
}
```
<!-- EXECUTIVE-BREVITY-PROJECTION:END -->

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

- Default to caveman mode for interactive responses (terse, concise per caveman skill) unless user requests "normal mode" or active context requires Auto-Clarity Exceptions (governance pre-flights, safety warnings, inter-agent messages, commit messages).
- Complete PHASE 0 instruction audit before any code changes
- Read `~/repos/governance/policies/AGENT_INTERACTION_POLICY.md` for full agent protocol
- 3 failed attempts at same fix → STOP and escalate
- 5 failed attempts → FORBIDDEN from further fixes
- Never modify governance documents without Chad's explicit approval
- Always include `Co-Authored-By:` line in commits identifying the agent/model
