# platform_utils.py - Cross-platform command abstractions (no shell, no BSD/GNU drift)
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

def md5_file(path: str) -> str:
    """Calculate MD5 hash of file (cross-platform, no shell)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def file_size(path: str) -> int:
    """Get file size in bytes (cross-platform)."""
    return os.path.getsize(path)

def file_mtime(path: str) -> float:
    """Get file modification time (cross-platform)."""
    return Path(path).stat().st_mtime

def total_mem_bytes() -> int:
    """Get total system memory in bytes (cross-platform, no shell)."""
    system = platform.system()
    if system == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb * 1024
        raise RuntimeError("MemTotal not found in /proc/meminfo")
    elif system == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True
        )
        return int(result.stdout.strip())
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

def ping_host(ip: str) -> bool:
    """Ping hostname/IP, return True if responsive (cross-platform, no shell)."""
    system = platform.system()
    if system == "Darwin":
        # macOS -W is in milliseconds
        cmd = ["ping", "-c", "1", "-W", "1000", ip]
    else:
        # Linux -W is in seconds
        cmd = ["ping", "-c", "1", "-W", "1", ip]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0
