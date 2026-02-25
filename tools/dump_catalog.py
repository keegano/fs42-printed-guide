#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guide_core import discover_channels_from_extents, infer_year_from_extents, parse_channel_schedule
from print_guide import dump_catalog_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scan FieldStation42 schedules and dump a reusable catalog JSON.")
    p.add_argument("--fs42-dir", type=Path, required=True, help="Path to FieldStation42 repository")
    p.add_argument("--out", type=Path, required=True, help="Output catalog JSON path")
    p.add_argument("--year", type=int, default=0, help="Schedule year override (0 = infer from extents)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[status] scanning channel extents in {args.fs42_dir}")
    channels = discover_channels_from_extents(args.fs42_dir)
    if not channels:
        raise SystemExit("No channels with valid extents found")
    print(f"[status] discovered {len(channels)} channels")

    year = int(args.year)
    if year <= 0:
        print("[status] inferring schedule year from extents")
        inferred = None
        for ch in channels:
            inferred = infer_year_from_extents(args.fs42_dir, ch)
            if inferred:
                break
        if not inferred:
            raise SystemExit("Could not infer schedule year; pass --year")
        year = inferred
    print(f"[status] using schedule year {year}")

    schedules = {}
    for idx, ch in enumerate(channels, start=1):
        print(f"[status] scanning schedule {idx}/{len(channels)}: {ch}")
        try:
            schedules[ch] = parse_channel_schedule(args.fs42_dir, ch, year)
            print(f"[status] loaded {len(schedules[ch])} events for {ch}")
        except Exception as exc:
            schedules[ch] = []
            print(f"[status] failed to load {ch}: {exc}")

    print(f"[status] writing catalog dump to {args.out}")
    dump_catalog_file(args.out, args.fs42_dir, channels, year, schedules)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
