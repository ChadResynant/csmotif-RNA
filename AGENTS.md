# AGENTS.md


<!-- GOVERNANCE-PREFLIGHT-v1 -->
## Governance Pre-Flight (summary — binding rules live in governance/)

All agents — Claude, Codex, Grok, Gemini, Hermes — before starting a task:
- Complete the startup audit and the **PHASE 0.5 pre-flight restatement**: restate to the
  user a 3–5 step plan plus the three most relevant governance policies, before doing the work.
- Use the **canonical document template** for any document — do not invent a format.

This is a summary; the binding rules and full checklists live in governance (source of truth):
- `~/repos/governance/policies/AGENT_INTERACTION_POLICY.md` — startup sequence + PHASE 0.5
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
