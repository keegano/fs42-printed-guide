#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guide_core import compute_range_bounds, split_into_blocks, clean_text, display_title
from print_guide import load_catalog_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate editable promo manifest templates from a catalog dump.")
    p.add_argument("--catalog", type=Path, required=True, help="Catalog dump JSON path.")
    p.add_argument("--out-dir", type=Path, default=Path("content/promos"), help="Promo manifest output directory.")
    p.add_argument("--range-mode", choices=["single", "day", "week", "month"], default="day")
    p.add_argument("--date", required=True, help="Anchor date YYYY-MM-DD")
    p.add_argument("--start", default="00:00", help="Start HH:MM (single mode)")
    p.add_argument("--hours", type=float, default=24.0, help="Hours for single mode")
    p.add_argument("--page-block-hours", type=float, default=6.0)
    p.add_argument("--max", type=int, default=16, help="Maximum templates to emit.")
    return p.parse_args()


def _events_in_range(schedules: dict[str, list], start_dt: datetime, end_dt: datetime) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
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
            found.append((shown, clean_text(ch)))
    return found


def main() -> None:
    args = parse_args()
    channels, _year, schedules = load_catalog_file(args.catalog)
    anchor = datetime.strptime(args.date, "%Y-%m-%d").date()
    start_t = datetime.strptime(args.start, "%H:%M").time()
    range_start, range_end = compute_range_bounds(args.range_mode, anchor, start_t, args.hours)
    blocks = split_into_blocks(range_start, range_end, args.page_block_hours)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    templates: list[dict] = []
    for b0, b1 in blocks:
        for shown, ch in _events_in_range(schedules, b0, b1):
            templates.append(
                {
                    "id": f"promo-{shown.lower().replace(' ', '-')[:40]}",
                    "enabled": True,
                    "range_modes": [args.range_mode],
                    "title": "",
                    "message": f"Catch {shown} at TBD!",
                    "image": "",
                    "match_titles": [shown.lower()],
                    "match_channels": [ch.lower()] if ch else [],
                }
            )
            if len(templates) >= args.max:
                break
        if len(templates) >= args.max:
            break

    out_file = args.out_dir / "generated_promos.json"
    out_file.write_text(json.dumps(templates, indent=2), encoding="utf-8")
    print(f"Wrote {out_file} with {len(templates)} template promo entries")
    print("Edit message/image values, then run print_guide.py with --content-dir")


if __name__ == "__main__":
    main()
