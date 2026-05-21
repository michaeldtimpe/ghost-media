#!/usr/bin/env python3
"""
Small shared utilities for the bake-off harness: atomic JSON writes (the repo's
universal tmp->rename pattern), the current git commit SHA, and a couple of
formatters. Kept tiny and dependency-light so every bench module can import it.
"""

import json
import subprocess
from pathlib import Path


def atomic_write_json(path, obj):
    """Write JSON to `path` via a temp file + rename (never a torn file).

    Uses ensure_ascii=False (repo convention — many source filenames are
    Japanese) and creates parent dirs as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.rename(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def git_commit_sha(short=False):
    """Return the current HEAD SHA, or 'unknown' outside a git tree."""
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"] if short \
            else ["git", "rev-parse", "HEAD"]
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def fmt_time(seconds):
    """H:MM:SS / M:SS, matching the existing drivers' style."""
    if seconds is None or seconds < 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bar(pct, width=25):
    pct = max(0.0, min(1.0, pct))
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)
