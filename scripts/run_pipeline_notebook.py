#!/usr/bin/env python3
"""Run code cells from the pipeline notebook and regenerate report artifacts.

Usage:
  MODEL_MAX_ROWS=150000 MODEL_RANDOM_SEARCH_ITERS=8 MODEL_GRID_TOP_K=2 MODEL_CV_SPLITS=3 MODEL_ENABLE_SVM=1 MODEL_N_JOBS=8 MPLCONFIGDIR=/tmp ./.venv/bin/python scripts/run_pipeline_notebook.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def display(obj):
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            print(obj.head(12).to_string())
        else:
            print(obj)
    except Exception:
        print(obj)


def main() -> int:
    """Run all code cells from the pipeline notebook in order.

    The function loads `notebooks/end_to_end_pipeline.ipynb`, executes each
    code cell in a shared Python context (similar to notebook \"Run All\"), and
    returns 0 on success or 1 on first failure.
    """
    project_root = Path(__file__).resolve().parents[1]
    nb_path = project_root / "notebooks" / "end_to_end_pipeline.ipynb"
    nb = json.loads(nb_path.read_text())

    ctx = {"__name__": "__main__", "display": display}
    code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]

    print(f"Executing {len(code_cells)} code cells from {nb_path}...")

    for i, cell in enumerate(code_cells, 1):
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        print(f"\\n--- Running code cell {i}/{len(code_cells)} ---")
        try:
            exec(compile(src, f"pipeline_cell_{i}", "exec"), ctx)
        except Exception as exc:
            print(f"Cell {i} failed: {exc}")
            traceback.print_exc()
            return 1

    print("\\npipeline notebook code execution completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
