#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""
=============================================================================
02 — Build window dataset (notebook-style script)
=============================================================================

Walks through: read labels → build dataset_windows.csv → sanity checks.

Run from project root::

    python notebooks/02_build_dataset.py

Requires: pandas; uses ``src/build_dataset.py`` on ``sys.path``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PROCESSED = ROOT / "data" / "processed" / "dataset_windows.csv"
LABELS = ROOT / "data" / "labels" / "contact_frames.csv"


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# =============================================================================
# SECTION: Read labels
# =============================================================================
def run_read_labels() -> None:
    section("Read labels (contact_frames.csv)")
    import pandas as pd

    if not LABELS.is_file():
        print(f"Missing {LABELS}")
        return
    df = pd.read_csv(LABELS)
    print(df.to_string(index=False))


# =============================================================================
# SECTION: Generate dataset metadata (call build_dataset)
# =============================================================================
def run_generate_dataset_cli() -> None:
    section("Generate dataset_windows.csv via src/build_dataset.py")
    build_py = SRC / "build_dataset.py"
    if not build_py.is_file():
        print(f"Missing {build_py}")
        return
    cmd = [sys.executable, str(build_py), "--print-loo"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=False)


# =============================================================================
# SECTION: Check positive/negative balance
# =============================================================================
def run_label_balance() -> None:
    section("Positive vs negative windows (label column)")

    import pandas as pd

    if not PROCESSED.is_file():
        print(f"Run build_dataset first; expected {PROCESSED}")
        return
    df = pd.read_csv(PROCESSED)
    vc = df["label"].value_counts().sort_index()
    print(vc)
    pos = (df["label"] == 1).mean()
    print(f"\nFraction positive (contact-near windows): {pos:.4f}")
    print("Expect heavy class imbalance — report precision/recall/F1, not accuracy alone.")


# =============================================================================
# SECTION: Inspect example windows
# =============================================================================
def run_example_windows(n: int = 5) -> None:
    section("Example rows from dataset_windows.csv")

    import pandas as pd

    if not PROCESSED.is_file():
        return
    df = pd.read_csv(PROCESSED)
    print(df.head(n).to_string(index=False))

    print("\nExample **positive** windows:")
    pos = df[df["label"] == 1].head(n)
    print(pos.to_string(index=False) if len(pos) else "(none)")


def main() -> None:
    run_read_labels()
    run_generate_dataset_cli()
    run_label_balance()
    run_example_windows()
    print("\nDone.")


if __name__ == "__main__":
    main()
