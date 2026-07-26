"""Repository entry point for the Yambda-50M SASRec experiment."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sids_rank_vs_generate.experiments.sasrec import main


if __name__ == "__main__":
    raise SystemExit(main())
