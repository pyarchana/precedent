"""Mirror the staged GitHub responses to S3.

The ingest writes raw pages to local disk first, deliberately, so it never had
to wait on an AWS account existing. This is the other half: the durable copy,
from which the whole corpus can be rebuilt without touching the GitHub API
again. That matters because re-ingesting 38,001 pull requests is hours of
wall clock, while re-running the transform from S3 is minutes.

Idempotent. Objects already present with a matching size are skipped, so
running it repeatedly costs almost nothing and interrupting it is safe.

    python scripts/sync_raw_to_s3.py --dry-run
    python scripts/sync_raw_to_s3.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from precedent.config import get_settings

log = logging.getLogger("precedent.s3sync")


def existing_objects(client, bucket: str, prefix: str) -> dict[str, int]:
    """Key to size for everything already under the prefix, in one pass.

    Listing once beats calling head_object per file: 3,800 round trips to
    Mumbai would dominate the runtime of a 58 MB upload.
    """
    found: dict[str, int] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found[obj["Key"]] = obj["Size"]
    return found


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Mirror staged raw pages to S3.")
    parser.add_argument("--bucket", default=settings.s3_raw_bucket or None)
    parser.add_argument("--raw-dir", type=Path, default=settings.raw_dir)
    parser.add_argument("--prefix", default="raw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bucket:
        parser.error("no bucket: pass --bucket or set S3_RAW_BUCKET in .env")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s | %(message)s",
        stream=sys.stdout,
    )

    files = sorted(p for p in args.raw_dir.rglob("*") if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    log.info(
        "%d local files, %.1f MB, target s3://%s/%s",
        len(files),
        total_bytes / 1024 / 1024,
        args.bucket,
        args.prefix,
    )

    client = boto3.client(
        "s3",
        region_name=settings.aws_region,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )

    try:
        already = existing_objects(client, args.bucket, args.prefix)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("NoSuchBucket", "AccessDenied", "InvalidAccessKeyId"):
            log.error(
                "cannot read s3://%s: %s. Check the bucket name, the region, and that "
                "the IAM user has S3 access.",
                args.bucket,
                code,
            )
            return 1
        raise

    log.info("%d objects already in the bucket", len(already))

    uploaded = skipped = 0
    uploaded_bytes = 0
    started = time.monotonic()

    for path in files:
        key = f"{args.prefix}/{path.relative_to(args.raw_dir).as_posix()}"
        size = path.stat().st_size

        if already.get(key) == size:
            skipped += 1
            continue

        if args.dry_run:
            uploaded += 1
            uploaded_bytes += size
            continue

        client.upload_file(
            str(path),
            args.bucket,
            key,
            ExtraArgs={"ContentType": "application/gzip"}
            if path.suffix == ".gz"
            else {"ContentType": "application/json"},
        )
        uploaded += 1
        uploaded_bytes += size

        if uploaded % 200 == 0:
            elapsed = time.monotonic() - started
            log.info(
                "%d uploaded, %.1f MB, %.0f files/s",
                uploaded,
                uploaded_bytes / 1024 / 1024,
                uploaded / elapsed if elapsed else 0,
            )

    verb = "would upload" if args.dry_run else "uploaded"
    log.info(
        "%s %d files (%.1f MB), skipped %d already present",
        verb,
        uploaded,
        uploaded_bytes / 1024 / 1024,
        skipped,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
