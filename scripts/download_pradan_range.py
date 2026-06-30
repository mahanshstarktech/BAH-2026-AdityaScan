#!/usr/bin/env python3
"""
Download PRADAN files into the AdityScan training cache by date range.

Credentials are read from environment variables:
  PRADAN_USER
  PRADAN_PASS

Or from a local gitignored env file:
  .env.training.local

Examples:
  python scripts/download_pradan_range.py \
    --start 2024-06-15 --end 2024-06-30 \
    --instruments helios,mag

  python scripts/download_pradan_range.py \
    --plan config/training_windows_golden_balanced.json \
    --instruments helios,mag,solexs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


LOG_FILE = ROOT / "training.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode="a"),
    ],
)
logger = logging.getLogger("download_pradan_range")


CACHE_DIR = ROOT / "data" / "pradan_cache"
PRADAN_BASE = "https://pradan1.issdc.gov.in"
ORBIT_DIR = "N00_0000"

INSTRUMENTS = {
    "solexs": {"browse_id": "solexs", "folder": "solexs"},
    "helios": {"browse_id": "hel1os", "folder": "helios"},
    "hel1os": {"browse_id": "hel1os", "folder": "helios"},
    "mag": {"browse_id": "mag", "folder": "mag"},
    "swis": {"browse_id": "swis", "folder": "swis"},
    "suit": {"browse_id": "suit", "folder": "suit"},
}


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without echoing secrets."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (key not in os.environ or os.environ[key].startswith("your_")):
            os.environ[key] = value


def _validate_credentials() -> None:
    user = os.environ.get("PRADAN_USER", "").strip()
    password = os.environ.get("PRADAN_PASS", "").strip()
    if (
        not user
        or not password
        or user == "your_pradan_username"
        or password == "your_pradan_password"
    ):
        raise SystemExit(
            "PRADAN_USER/PRADAN_PASS are empty or still set to placeholders. "
            "Edit .env.training.local with your actual PRADAN login."
        )


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _days(start: str, end: str) -> list[datetime]:
    current = _parse_date(start)
    last = _parse_date(end)
    if current > last:
        raise ValueError(f"start date {start} is after end date {end}")
    out = []
    while current <= last:
        out.append(current)
        current += timedelta(days=1)
    return out


def _windows_from_args(args) -> list[dict]:
    if args.plan:
        with Path(args.plan).open() as f:
            plan = json.load(f)
        return plan.get("windows", [])

    if not args.start or not args.end:
        raise SystemExit("Provide either --plan or both --start and --end")
    month = args.start[:7]
    return [{
        "month": month,
        "start_date": args.start,
        "end_date": args.end,
        "role": "manual_range",
    }]


def _dest_for(day: datetime, instrument: str, filename: str) -> Path:
    folder = INSTRUMENTS[instrument]["folder"]
    month_dir = CACHE_DIR / day.strftime("%Y-%m") / folder
    month_dir.mkdir(parents=True, exist_ok=True)
    return month_dir / filename


async def _download_solexs_day(session: PRADANSession, day: datetime, dry_run: bool) -> bool:
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    date_str = day.strftime("%Y%m%d")

    for version in ("v1.0", "v1.1", "v1.2", "v2.0", "v3.0"):
        filename = f"AL1_SLX_L1_{date_str}_{version}.zip"
        url = (
            f"{PRADAN_BASE}/al1/protected/downloadData/solexs/level1/"
            f"{yyyy}/{mm}/{ORBIT_DIR}/{filename}?solexs"
        )
        dest = _dest_for(day, "solexs", filename)
        if dry_run:
            logger.info("[dry-run] SoLEXS %s -> %s", url, dest)
            return True
        ok = await session.download_file(filename, dest, url)
        if ok:
            return True
    logger.warning("SoLEXS not found for %s", date_str)
    return False


async def _browse_day(session: PRADANSession, day: datetime, instrument: str) -> list[dict]:
    yyyy = day.strftime("%Y")
    mm = day.strftime("%m")
    dd = day.strftime("%d")
    date_str = day.strftime("%Y%m%d")
    browse_id = INSTRUMENTS[instrument]["browse_id"]
    day_url = (
        f"{PRADAN_BASE}/al1/protected/browse.xhtml"
        f"?id={browse_id}&date={yyyy}-{mm}-{dd}"
    )

    resp = await session._client.get(day_url)
    links = re.findall(
        rf'href="(/al1/protected/downloadData/{browse_id}[^"]+)"',
        resp.text,
        re.IGNORECASE,
    )

    files = []
    path_date = f"/{yyyy}/{mm}/{dd}/"
    for link in links:
        filename = link.split("/")[-1].split("?")[0]
        if date_str not in filename and f"{yyyy}-{mm}-{dd}" not in filename and path_date not in link:
            continue
        files.append({
            "filename": filename,
            "url": f"{PRADAN_BASE}{link}",
        })
    return files


async def _download_browsed_day(
    session: PRADANSession,
    day: datetime,
    instrument: str,
    dry_run: bool,
) -> int:
    files = await _browse_day(session, day, instrument)
    if not files:
        logger.warning("%s: no %s files found", day.strftime("%Y-%m-%d"), instrument)
        return 0

    count = 0
    for file_info in files:
        filename = file_info["filename"]
        dest = _dest_for(day, instrument, filename)
        if dry_run:
            logger.info("[dry-run] %s %s -> %s", instrument, file_info["url"], dest)
            count += 1
            continue
        ok = await session.download_file(filename, dest, file_info["url"])
        if ok:
            count += 1
    return count


async def main_async(args) -> None:
    requested = [item.strip().lower() for item in args.instruments.split(",") if item.strip()]
    unknown = [item for item in requested if item not in INSTRUMENTS]
    if unknown:
        raise SystemExit(f"Unknown instruments: {', '.join(unknown)}")

    windows = _windows_from_args(args)
    total = 0
    async with PRADANSession() as session:
        for window in windows:
            start = window["start_date"]
            end = window["end_date"]
            logger.info("Downloading PRADAN range %s to %s (%s)", start, end, window.get("role", ""))
            for day in _days(start, end):
                for instrument in requested:
                    if instrument == "solexs":
                        total += int(await _download_solexs_day(session, day, args.dry_run))
                    else:
                        total += await _download_browsed_day(session, day, instrument, args.dry_run)
                if args.sleep_seconds > 0:
                    await asyncio.sleep(args.sleep_seconds)

    logger.info("PRADAN range download complete: %d files handled", total)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PRADAN files into data/pradan_cache/YYYY-MM/<sensor>/"
    )
    parser.add_argument("--start", default="", help="YYYY-MM-DD inclusive start date")
    parser.add_argument("--end", default="", help="YYYY-MM-DD inclusive end date")
    parser.add_argument("--plan", default="", help="Training manifest JSON with windows")
    parser.add_argument(
        "--instruments",
        default="helios,mag",
        help="Comma-separated: solexs,helios,mag,swis,suit",
    )
    parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Delay between days")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs/destinations without downloading")
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env.training.local"),
        help="Optional local KEY=VALUE credential file (default: .env.training.local)",
    )
    args = parser.parse_args()

    _load_env_file(Path(args.env_file))
    _validate_credentials()

    global PRADANSession
    from pipeline.ingestion.pradan_downloader import PRADANSession

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
