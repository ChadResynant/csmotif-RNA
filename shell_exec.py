"""
shell_exec.py - Public API for External Command Execution

STABILITY: Public API (frozen as of sparky-shell-hardened-2026-01-14)

This module is the ONLY entry point for launching external processes in DoNMR/Sparky.
All subprocess calls must go through:
  - run_argv([cmd, arg1, arg2], ...) for simple commands
  - run_bash("cmd ... | filter", ...) for shell features (pipes, redirects)
  - tcsh_path() to check for NMRPipe environment

DESIGN PRINCIPLES:
  • Never use subprocess shell parameter (except run_bash which is explicit)
  • All paths quoted with sh_quote() when interpolated
  • Platform-aware (Darwin, Linux, Windows)
  • Deterministic error handling with CmdResult
  • Timeouts to prevent hangs

IMPORTS (for all code):
  from shell_exec import run_argv, run_bash, sh_quote, ExecKind

Do NOT:
  • Call subprocess module directly
  • Use legacy os-level process functions
  • Bypass sh_quote() when building shell strings

See: tools/ci/shell_hygiene_gate_hardmode.sh (enforces this at CI time)
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

PathLike = Union[str, Path]

@dataclass(frozen=True)
class CmdResult:
    argv: Sequence[str]
    returncode: int
    stdout: str
    stderr: str

class ShellExecError(RuntimeError):
    pass

def _tail(s: str, n_lines: int = 200) -> str:
    """Truncate to last N lines for readable error messages."""
    lines = s.splitlines()
    if len(lines) <= n_lines:
        return s
    return f"[...truncated {len(lines) - n_lines} lines...]\n" + "\n".join(lines[-n_lines:])

def _merge_env(env: Optional[Mapping[str, str]], clean: bool = False) -> dict:
    """Merge env, optionally starting from minimal clean environment."""
    if clean:
        # Minimal env for reproducibility
        merged = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "LANG": "C",
            "LC_ALL": "C",
        }
        # macOS temp dir matters
        if "TMPDIR" in os.environ:
            merged["TMPDIR"] = os.environ["TMPDIR"]
    else:
        merged = os.environ.copy()
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    return merged

def which(prog: str) -> Optional[str]:
    return shutil.which(prog)

def which_or_raise(prog: str, *, hint: str = "") -> str:
    p = which(prog)
    if not p:
        msg = f"Required executable not found on PATH: {prog}"
        if hint:
            msg += f"\n{hint}"
        raise ShellExecError(msg)
    return p

def run_argv(
    argv: Sequence[str],
    *,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    clean_env: bool = False,
    check: bool = True,
    timeout: Optional[float] = None,
    stdin_text: Optional[str] = None,
    _strict: bool = False,
) -> CmdResult:
    """Run command as argv list (preferred over shell).

    Warning: Detects suspicious single-item lists that might be shell strings.
    If you're passing ["rm -f *.tmp"], use run_bash() instead with proper quoting.
    """
    # Library-level tripwire: catch ["command with metacharacters"] patterns
    argv_list = list(argv)
    if len(argv_list) == 1 and isinstance(argv_list[0], str):
        item = argv_list[0]
        # Shell metacharacters that indicate "shell string in list"
        SHELL_METACHARACTERS = " \t*?[]{}|&;><$()`\"'\n"
        if any(ch in item for ch in SHELL_METACHARACTERS):
            msg = (
                f"WARNING: argv list contains single shell-like string: {item[:50]}\n"
                f"  This suggests a missing .split() or array unpacking.\n"
                f"  If intentional, use run_bash() instead.\n"
                f"  To suppress this warning, use run_argv(..., _strict=False) [default]"
            )
            import sys
            print(msg, file=sys.stderr)
            if _strict:
                raise ShellExecError(f"Strict mode: {msg}")

    cp = subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd else None,
        env=_merge_env(env, clean=clean_env),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        input=stdin_text,
    )
    if check and cp.returncode != 0:
        raise ShellExecError(
            f"Command failed (exit {cp.returncode}): {list(argv)}\n"
            f"--- stdout (tail) ---\n{_tail(cp.stdout)}\n"
            f"--- stderr (tail) ---\n{_tail(cp.stderr)}\n"
        )
    return CmdResult(argv=list(argv), returncode=cp.returncode, stdout=cp.stdout, stderr=cp.stderr)

def run_bash(
    cmd: str,
    *,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    check: bool = True,
    timeout: Optional[float] = None,
    bash_path: Optional[str] = None,
) -> CmdResult:
    """
    Use ONLY for internal/trusted command strings that require shell features
    (pipes, redirects, &&, globbing). Prefer run_argv.
    """
    bash = bash_path or which("bash") or "/bin/bash"
    return run_argv([bash, "-lc", cmd], cwd=cwd, env=env, check=check, timeout=timeout)

def tcsh_path() -> Optional[str]:
    return which("tcsh") or which("csh")

def run_tcsh_script(
    script: PathLike,
    *,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    clean_env: bool = False,  # Default: preserve env (NMRPipe needs NMRBASE etc.)
    check: bool = True,
    timeout: Optional[float] = None,
) -> CmdResult:
    """Run NMRPipe .com script under tcsh explicitly (not via shebang)."""
    tcsh = tcsh_path()
    if not tcsh:
        raise ShellExecError(
            "tcsh/csh is required to run NMRPipe .com scripts.\n"
            "macOS: brew install tcsh\n"
            "Ubuntu: sudo apt install tcsh\n"
            "RHEL/Fedora: sudo dnf install tcsh\n"
        )

    sp = Path(script)
    if not sp.exists():
        raise ShellExecError(f"Script not found: {sp}")

    # Ensure exec bit
    try:
        sp.chmod(sp.stat().st_mode | 0o111)
    except Exception:
        pass

    # -f = do not source user rc files (prevents environment leakage)
    return run_argv(
        [tcsh, "-f", str(sp)],
        cwd=cwd,
        env=env,
        clean_env=clean_env,
        check=check,
        timeout=timeout
    )

def sh_quote(s: str) -> str:
    """Use when you must embed values into run_bash(cmd)."""
    return shlex.quote(s)

class ExecKind(Enum):
    """Kind of execution to perform."""
    ARGV = "argv"           # Direct argv execution (preferred)
    BASH = "bash"           # Shell features: pipes, redirects, globs
    TCSH_SCRIPT = "tcsh_script"  # NMRPipe .com scripts

def run(
    cmd: Sequence[str] | str,
    *,
    kind: ExecKind = ExecKind.ARGV,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    clean_env: bool = False,
    check: bool = True,
    timeout: Optional[float] = None,
) -> CmdResult:
    """
    Unified shell execution interface - single entry point for all external calls.

    This is the preferred function to use. It centralizes logging, tripwires,
    and error handling in one place.

    Parameters
    ----------
    cmd : Sequence[str] or str
        Command to execute:
        - For kind=ARGV: list of strings (argv), e.g., ["echo", "hello"]
        - For kind=BASH: shell command string, e.g., "ls *.txt | wc -l"
        - For kind=TCSH_SCRIPT: path to .com script, e.g., "/path/to/script.com"
    kind : ExecKind, default=ARGV
        Execution type:
        - ARGV: Direct process execution (no shell, preferred)
        - BASH: For trusted shell features (pipes, redirects)
        - TCSH_SCRIPT: For NMRPipe .com scripts with tcsh -f
    cwd : PathLike, optional
        Working directory
    env : Mapping, optional
        Environment variables to set/override
    clean_env : bool, default=False
        If True, start from minimal clean environment (PATH, HOME, USER, LANG, LC_ALL, TMPDIR)
    check : bool, default=True
        If True, raise ShellExecError on non-zero exit
    timeout : float, optional
        Timeout in seconds

    Returns
    -------
    CmdResult
        Result with argv, returncode, stdout, stderr

    Examples
    --------
    >>> # Preferred: direct argv
    >>> result = run(["ls", "-la"], kind=ExecKind.ARGV)

    >>> # Shell features: use BASH
    >>> result = run("cat *.txt | wc -l", kind=ExecKind.BASH)

    >>> # NMRPipe: use TCSH_SCRIPT
    >>> result = run("/path/to/script.com", kind=ExecKind.TCSH_SCRIPT, cwd="/output")
    """
    if kind == ExecKind.ARGV:
        if isinstance(cmd, str):
            raise ValueError("kind=ARGV requires list, got str. Use BASH if you need shell features.")
        return run_argv(cmd, cwd=cwd, env=env, clean_env=clean_env, check=check, timeout=timeout)

    elif kind == ExecKind.BASH:
        if isinstance(cmd, (list, tuple)):
            raise ValueError("kind=BASH requires str, got list. Use ARGV for direct execution.")
        return run_bash(cmd, cwd=cwd, env=env, check=check, timeout=timeout)

    elif kind == ExecKind.TCSH_SCRIPT:
        if isinstance(cmd, (list, tuple)):
            raise ValueError("kind=TCSH_SCRIPT requires str (path), got list.")
        return run_tcsh_script(cmd, cwd=cwd, env=env, clean_env=clean_env, check=check, timeout=timeout)

    else:
        raise ValueError(f"Unknown ExecKind: {kind}")
