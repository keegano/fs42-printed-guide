#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guide_core import (
    Event,
    _extract_year_hint,
    clean_text,
    compute_range_bounds,
    display_title,
    fetch_omdb_cover_art,
    fetch_tvdb_cover_art,
    is_movie_event,
    normalize_title_text,
    split_into_blocks,
)
from print_guide import load_catalog_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-promo editable manifest files from a catalog dump.")
    p.add_argument("--catalog", type=Path, required=True, help="Catalog dump JSON path.")
    p.add_argument("--out-dir", type=Path, default=Path("content/promos"), help="Promo manifest output directory.")
    p.add_argument("--range-mode", choices=["single", "day", "week", "month"], default="day")
    p.add_argument("--date", required=True, help="Anchor date YYYY-MM-DD")
    p.add_argument("--start", default="00:00", help="Start HH:MM (single mode)")
    p.add_argument("--hours", type=float, default=24.0, help="Hours for single mode")
    p.add_argument("--page-block-hours", type=float, default=6.0)
    p.add_argument("--max", type=int, default=16, help="Maximum promos to emit.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing promo JSON files.")
    p.add_argument("--poster-source", choices=["auto", "omdb", "tvdb", "none"], default="auto")
    p.add_argument("--tvdb-api-key", default="")
    p.add_argument("--tvdb-pin", default="")
    p.add_argument("--omdb-api-key", default="")
    return p.parse_args()


def _slug(text: str) -> str:
    raw = clean_text(text).lower()
    out = []
    prev_dash = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    s = "".join(out).strip("-")
    return s[:64] if s else "promo"


def _events_in_range(
    schedules: dict[str, list[Event]],
    start_dt: datetime,
    end_dt: datetime,
) -> list[tuple[str, str, Event]]:
    found: list[tuple[str, str, Event]] = []
    seen: set[str] = set()
    for ch, evs in schedules.items():
        for e in evs:
            if e.end <= start_dt or e.start >= end_dt:
                continue
            shown = clean_text(display_title(e.title, e.filename))
            if not shown:
                continue
            key = shown.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append((shown, clean_text(ch), e))
    return found


def _save_poster_for_event(
    out_dir: Path,
    shown: str,
    ev: Event,
    poster_source: str,
    tvdb_api_key: str,
    tvdb_pin: str,
    omdb_api_key: str,
) -> str:
    slug = _slug(shown)
    for existing in sorted(out_dir.glob(f"{slug}_poster.*")):
        if existing.is_file():
            print(f"[status] poster exists for {shown}: {existing.name}")
            return existing.name

    tmp: Path | None = None
    if poster_source in ("auto", "omdb") and omdb_api_key and is_movie_event(ev.title, ev.filename):
        print(f"[status] fetching OMDb poster for {shown}")
        tmp = fetch_omdb_cover_art(shown, _extract_year_hint(ev.filename), omdb_api_key)
    if tmp is None and poster_source in ("auto", "tvdb") and tvdb_api_key:
        print(f"[status] fetching TVDB poster for {shown}")
        tmp = fetch_tvdb_cover_art([normalize_title_text(shown)], tvdb_api_key, tvdb_pin)
    if tmp is None:
        return ""

    ext = tmp.suffix.lower() or ".jpg"
    dest = out_dir / f"{slug}_poster{ext}"
    shutil.copy2(tmp, dest)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"[status] saved poster {dest.name}")
    return dest.name


def main() -> None:
    args = parse_args()
    print(
        "[status] generating promo templates "
        f"catalog={args.catalog} out_dir={args.out_dir} mode={args.range_mode}"
    )
    print(
        "[status] options "
        f"date={args.date} start={args.start} hours={args.hours} "
        f"page_block_hours={args.page_block_hours} max={args.max} "
        f"poster_source={args.poster_source} overwrite={args.overwrite}"
    )
    _channels, _year, schedules = load_catalog_file(args.catalog)
    print(f"[status] loaded catalog channels={len(schedules)}")
    anchor = datetime.strptime(args.date, "%Y-%m-%d").date()
    start_t = datetime.strptime(args.start, "%H:%M").time()
    range_start, range_end = compute_range_bounds(args.range_mode, anchor, start_t, args.hours)
    print(
        "[status] resolved range "
        f"{range_start.strftime('%Y-%m-%d %H:%M')} -> {range_end.strftime('%Y-%m-%d %H:%M')}"
    )
    blocks = split_into_blocks(range_start, range_end, args.page_block_hours)
    print(f"[status] computed blocks={len(blocks)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    emitted = 0
    seen_ids: set[str] = set()
    for idx, (b0, b1) in enumerate(blocks, start=1):
        print(f"[status] scanning block {idx}/{len(blocks)} {b0.strftime('%m/%d %H:%M')}->{b1.strftime('%m/%d %H:%M')}")
        block_events = _events_in_range(schedules, b0, b1)
        print(f"[status] block candidates={len(block_events)}")
        for shown, ch, ev in block_events:
            sid = _slug(shown)
            if sid in seen_ids:
                print(f"[status] skip duplicate show slug={sid} show='{shown}'")
                continue
            seen_ids.add(sid)
            print(f"[status] building promo for show='{shown}' channel='{ch}' slug={sid}")
            poster_name = ""
            if args.poster_source != "none":
                poster_name = _save_poster_for_event(
                    out_dir=args.out_dir,
                    shown=shown,
                    ev=ev,
                    poster_source=args.poster_source,
                    tvdb_api_key=args.tvdb_api_key,
                    tvdb_pin=args.tvdb_pin,
                    omdb_api_key=args.omdb_api_key,
                )
            payload = {
                "id": f"promo-{sid}",
                "enabled": True,
                "range_modes": [args.range_mode],
                "title": "",
                "message_template": "Catch {show} on {weekday} at {time}!",
                "image": poster_name,
                "match_titles": [shown.lower()],
                "match_channels": [ch.lower()] if ch else [],
            }
            out_file = args.out_dir / f"promo_{sid}.json"
            if out_file.exists() and not args.overwrite:
                print(f"[status] keeping existing promo file: {out_file.name}")
                continue
            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[status] wrote promo template: {out_file.name}")
            emitted += 1
            if emitted >= args.max:
                print(f"Wrote {emitted} promo template files to {args.out_dir}")
                print("You can add additional unrelated promos (e.g., local ads) as extra JSON files in this folder.")
                return

    print(f"Wrote {emitted} promo template files to {args.out_dir}")
    print("You can add additional unrelated promos (e.g., local ads) as extra JSON files in this folder.")


if __name__ == "__main__":
    main()
