"""Which code produced a result: the git commit if available, else the version."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from tradingagents_quant import __version__

_REPO = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def code_version() -> str:
    """``<package version>+<short sha>`` with ``-dirty`` for uncommitted
    changes, or the package version alone outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(_REPO), "rev-parse", "--short=12", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(_REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return __version__
    return f"{__version__}+{sha}{'-dirty' if dirty else ''}" if sha else __version__
