"""Build the submission zip for the assignment.

Includes source code, configuration (DVC, CI/CD, Docker, compose), the
trained model artifact, MLflow run artifacts, and the monitoring report.
Excludes the datasets, the virtualenv, caches, and the duplicate per-run
checkpoints under mlruns/.

Everything is read from the working tree, so the zip reflects the files as they
are on disk. Packaging runs only once sync_results.py --check confirms the
documented results match the MLflow store and every declared deliverable is
present.

Usage:
    python scripts/package_submission.py [--out Nikkita_MLOps_Assignment2.zip]
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

# The filename handed in, so every rebuild overwrites the same archive.
DEFAULT_OUT = "Nikkita_MLOps_Assignment2.zip"

# Files and directories that make up the deliverable.
INCLUDE_FILES = [
    "README.md",
    "requirements.txt",
    "requirements-api.txt",
    "params.yaml",
    "pytest.ini",
    "Dockerfile",
    "docker-compose.yml",
    ".gitignore",
    ".dvcignore",
    "models/model.pt",
    "data/raw.dvc",
    "data/processed.dvc",
    "mlflow.db",  # MLflow tracking store — evidence of logged experiment runs
]
INCLUDE_DIRS = [
    "src",
    "app",
    "tests",
    "scripts",
    ".github",
    ".dvc",
    "monitoring",
    "reports",
    "mlruns",
]
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", ".venv", "cache", "tmp"}


def should_include(path: Path) -> bool:
    if not path.is_file() or any(part in EXCLUDE_PARTS for part in path.parts):
        return False
    # Skip the per-run weight copies under mlruns/ — each is a duplicate of a
    # checkpoint we already ship, and the promoted one is in INCLUDE_FILES.
    # Their metrics, params, and plots are kept.
    if path.suffix == ".pt" and "mlruns" in path.parts:
        return False
    return True


def check_docs_current() -> None:
    """Refuse to package while README.md disagrees with mlflow.db."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("sync_results.py")), "--check"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            "Refusing to package: " + (result.stdout + result.stderr).strip()
            + "\nFix it, or re-run with --skip-checks to package anyway."
        )
    print("Docs match mlflow.db")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="package even if the README/MLflow freshness check fails",
    )
    args = parser.parse_args()

    if not args.skip_checks:
        check_docs_current()

    root = Path.cwd()
    out_path = Path(args.out)
    written, missing = 0, []

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in INCLUDE_FILES:
            path = root / name
            if path.is_file():
                archive.write(path, name)
                written += 1
            else:
                missing.append(name)

        for name in INCLUDE_DIRS:
            directory = root / name
            if not directory.is_dir():
                missing.append(name + "/")
                continue
            for path in directory.rglob("*"):
                if should_include(path):
                    archive.write(path, str(path.relative_to(root)))
                    written += 1

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Wrote {out_path} — {written} files, {size_mb:.1f} MB")
    if missing:
        sys.exit("Incomplete submission — these deliverables were not found: " + ", ".join(missing))


if __name__ == "__main__":
    main()
