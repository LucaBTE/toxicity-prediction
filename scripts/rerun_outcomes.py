#!/usr/bin/env python3
"""Delete generated outcomes and rerun modeling notebooks from scratch."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_ROOT = PROJECT_ROOT / "notebooks"

ML_NOTEBOOKS = [
    NOTEBOOKS_ROOT / "ml-models" / "model_ridge.ipynb",
    NOTEBOOKS_ROOT / "ml-models" / "model_random_forest.ipynb",
    NOTEBOOKS_ROOT / "ml-models" / "model_svm.ipynb",
    NOTEBOOKS_ROOT / "ml-models" / "model_xgboost.ipynb",
    NOTEBOOKS_ROOT / "ml-models" / "model_weighted_ensemble.ipynb",
]

DL_NOTEBOOKS = [
    NOTEBOOKS_ROOT / "dl-models" / "model_deeppurpose_gnn.ipynb",
    NOTEBOOKS_ROOT / "dl-models" / "model_deeppurpose_cnn.ipynb",
]

INTERPRETABILITY_NOTEBOOKS = [
    NOTEBOOKS_ROOT / "ml-models" / "interpratability" / "model_interpretability.ipynb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove notebooks/**/outcome directories and rerun the selected "
            "modeling notebooks in a clean state."
        )
    )
    parser.add_argument(
        "--scope",
        choices=("ml", "all"),
        default="all",
        help="Notebook set to rerun. Default: all.",
    )
    parser.add_argument(
        "--skip-delete",
        action="store_true",
        help="Rerun notebooks without deleting existing outcome folders first.",
    )
    parser.add_argument(
        "--skip-interpretability",
        action="store_true",
        help="Do not rerun the interpretability notebook.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted/executed without changing anything.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Do not ask for confirmation before deleting outcome folders.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=-1,
        help="Per-cell execution timeout in seconds. Use -1 for no timeout. Default: -1.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def find_outcome_dirs() -> list[Path]:
    outcome_dirs = sorted(
        path
        for path in NOTEBOOKS_ROOT.rglob("outcome")
        if path.is_dir() and path.name == "outcome"
    )
    for path in outcome_dirs:
        path.relative_to(NOTEBOOKS_ROOT)
    return outcome_dirs


def selected_notebooks(args: argparse.Namespace) -> list[Path]:
    notebooks = list(ML_NOTEBOOKS)
    if args.scope == "all":
        notebooks.extend(DL_NOTEBOOKS)
    if not args.skip_interpretability:
        notebooks.extend(INTERPRETABILITY_NOTEBOOKS)
    return notebooks


def confirm_delete(outcome_dirs: list[Path], assume_yes: bool) -> None:
    if assume_yes or not outcome_dirs:
        return

    print("\nOutcome folders to delete:")
    for path in outcome_dirs:
        print(f"  - {relative(path)}")
    answer = input("\nDelete these folders and continue? [y/N] ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted.")


def delete_outcomes(outcome_dirs: list[Path], dry_run: bool) -> None:
    if not outcome_dirs:
        print("No outcome folders found.")
        return

    for path in outcome_dirs:
        print(f"Deleting {relative(path)}")
        if not dry_run:
            shutil.rmtree(path)


def run_notebook(notebook: Path, timeout: int, dry_run: bool) -> None:
    if not notebook.exists():
        raise FileNotFoundError(f"Missing notebook: {relative(notebook)}")

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        notebook.name,
    ]
    print(f"\nRunning {relative(notebook)}")
    print(f"  cwd: {relative(notebook.parent)}")
    print(f"  cmd: {' '.join(command)}")
    if dry_run:
        return

    start = time.monotonic()
    subprocess.run(command, cwd=notebook.parent, check=True)
    elapsed = time.monotonic() - start
    print(f"Finished {relative(notebook)} in {elapsed / 60:.1f} min")


def main() -> int:
    args = parse_args()
    notebooks = selected_notebooks(args)

    if not args.skip_delete:
        outcome_dirs = find_outcome_dirs()
        confirm_delete(outcome_dirs, args.yes or args.dry_run)
        delete_outcomes(outcome_dirs, args.dry_run)

    print("\nNotebook run order:")
    for notebook in notebooks:
        print(f"  - {relative(notebook)}")

    for notebook in notebooks:
        run_notebook(notebook, args.timeout, args.dry_run)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
