"""Root-level compatibility shim for scripts/ci_summary.py."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci_summary import *  # noqa: F401,F403
from scripts import ci_summary as _impl

main = _impl.main

if __name__ == "__main__":
    raise SystemExit(_impl.main())