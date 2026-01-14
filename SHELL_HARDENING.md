# Shell Hardening: Local Reference

**Status:** This repo uses hard-mode shell execution gates (zero-tolerance enforcement).

## Before You Push

```bash
./tools/ci/shell_hygiene_gate_hardmode.sh
```

Exit 0? ✅ Push. | Exit 1? ❌ See patterns below.

## Fix Patterns

```python
# WRONG: os.system(...) | RIGHT: from shell_exec import run_argv; run_argv([...])
# WRONG: shell=True | RIGHT: run_bash("...") for shell features
# WRONG: os.popen(...) | RIGHT: use shell_exec module
```

## Full Docs

- Quick: `../sparky/docs/SHELL_HARDENING_QUICK_REFERENCE.md`
- Complete: `../sparky/docs/SHELL_HARDENING_PHASE_COMPLETION_2026-01-14.md`

## API (Frozen)

```python
from shell_exec import run_argv, run_bash, sh_quote
```

**You cannot introduce new shell hazards. CI prevents it.**
