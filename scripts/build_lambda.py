"""Build a Lambda deployment zip that works when built on Windows.

Three of the dependencies here ship compiled extensions: `asyncpg`,
`pydantic-core` and the wheels `sqlalchemy` builds. Installing them normally on
Windows produces Windows binaries, which import fine locally and fail inside
Lambda with an unhelpful "no module named _asyncpg". So the install is pinned to
the Lambda runtime's platform and refused if a source-only distribution would
have to be compiled, which is the failure worth having: a loud error here beats
a broken function.

`tiktoken` is excluded on purpose. It is another compiled extension, and it
fetches its encoding table over the network the first time it is imported,
which is a bad thing to do on a cold start. `embed/provider.py` already falls
back to a character bound when it is missing, and the API caps questions at
2,000 characters, so nothing about the deployed behaviour changes.

`uvicorn`, `boto3` and the development tools are excluded too. boto3 is
provided by the Lambda runtime, and shipping a second copy is 40 MB of nothing.

    python scripts/build_lambda.py
    python scripts/build_lambda.py --python-version 3.12
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from precedent.config import REPO_ROOT

log = logging.getLogger("precedent.build")

BUILD_DIR = REPO_ROOT / "build" / "lambda"
ZIP_PATH = REPO_ROOT / "build" / "precedent-lambda.zip"

# Everything the request path actually touches. Pinned loosely; the versions
# come from pyproject.
RUNTIME_REQUIREMENTS = [
    "fastapi>=0.111",
    "mangum>=0.17",
    "sqlalchemy[asyncio]>=2.0.30",
    "sqlalchemy-cockroachdb>=2.0.4",
    "asyncpg>=0.29",
    "httpx>=0.27",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "python-dotenv>=1.0",
    # GitHub App authentication. cryptography is the only compiled wheel here,
    # so it is the one to look at first if the package outgrows the 50MB direct
    # upload limit.
    "pyjwt>=2.8",
    "cryptography>=42",
    # Reads config/maintainers.yaml. It looks like a development-only
    # dependency and is not: the webhook path asks who speaks for the project on
    # every delivery, and that answer comes out of a YAML file.
    "pyyaml>=6.0",
]

# Data the application reads at runtime rather than imports. Without this the
# maintainer list is absent in deployment and `load_maintainers` returns an
# empty set, which is not an error and produces no failure: authorship checks
# quietly fall back to authorAssociation alone, which is the exact defect
# scripts/derive_maintainers.py exists to correct.
DATA_FILES = [Path("config") / "maintainers.yaml"]

# Lambda's own runtime provides these, and a second copy is dead weight.
PROVIDED_BY_RUNTIME = ("boto3", "botocore")

# Compiled files and test suites that no request will ever import, plus the
# installer's own scaffolding. `bin/` in particular is Windows console scripts,
# which are both useless in Lambda and misleading to find there.
PRUNE_PATTERNS = (
    "**/__pycache__",
    "**/*.pyc",
    "**/*.pyi",
    "**/tests",
    "**/test",
    "**/*.dist-info/RECORD",
    "bin",
    "include",
    ".lock",
)


def run(cmd: list[str]) -> None:
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def install_dependencies(target: Path, python_version: str, architecture: str) -> None:
    platform_tag = {
        "x86_64": "manylinux2014_x86_64",
        "arm64": "manylinux2014_aarch64",
    }[architecture]

    run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            str(target),
            "--python-platform",
            platform_tag,
            "--python-version",
            python_version,
            # Refuse to build from source. A source build on Windows would
            # silently produce Windows binaries for a Linux runtime.
            "--only-binary",
            ":all:",
            *RUNTIME_REQUIREMENTS,
        ]
    )


def copy_application(target: Path) -> None:
    """Copy the package itself, the static page it serves, and the data it reads."""
    source = REPO_ROOT / "src" / "precedent"
    destination = target / "precedent"
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    for relative in DATA_FILES:
        origin = REPO_ROOT / relative
        if not origin.is_file():
            raise FileNotFoundError(
                f"{relative} is missing. It is read at runtime, and shipping without it "
                "degrades silently rather than failing."
            )
        copy = target / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, copy)
        log.info("packaged %s", relative.as_posix())


def prune(target: Path) -> int:
    removed = 0
    for name in PROVIDED_BY_RUNTIME:
        for path in list(target.glob(f"{name}")) + list(target.glob(f"{name}-*.dist-info")):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
                removed += 1

    for pattern in PRUNE_PATTERNS:
        for path in list(target.glob(pattern)):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink()
            removed += 1
    return removed


def write_zip(target: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(target.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(target).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Lambda deployment package.")
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--architecture", default="x86_64", choices=["x86_64", "arm64"])
    parser.add_argument("--keep", action="store_true", help="Leave the staging directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    install_dependencies(BUILD_DIR, args.python_version, args.architecture)
    copy_application(BUILD_DIR)
    removed = prune(BUILD_DIR)
    log.info("pruned %d paths", removed)

    write_zip(BUILD_DIR, ZIP_PATH)

    unpacked = sum(p.stat().st_size for p in BUILD_DIR.rglob("*") if p.is_file())
    packed = ZIP_PATH.stat().st_size
    log.info(
        "%s\n  %.1f MB zipped, %.1f MB unpacked",
        ZIP_PATH,
        packed / 1024 / 1024,
        unpacked / 1024 / 1024,
    )

    # Lambda's own limits, checked here rather than discovered on upload.
    if packed > 50 * 1024**2:
        log.error("over the 50 MB direct upload limit; upload via S3 instead")
    if unpacked > 250 * 1024**2:
        log.error("over the 250 MB unpacked limit; the function will not deploy")
        return 1

    if not args.keep:
        shutil.rmtree(BUILD_DIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
