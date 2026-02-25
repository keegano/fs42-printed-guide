#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from guide_config import parse_effective_args
from guide_core import (
    Event,
    clean_text,
    compute_range_bounds,
    discover_channels_from_extents,
    display_title,
    fit_show_title,
    infer_year_from_extents,
    load_channel_numbers_from_confs,
    make_compilation_pdf,
    make_pdf,
    normalize_title_text,
    parse_channel_schedule,
    parse_date,
    parse_hhmm,
    split_into_blocks,
)


def _parse_number_map(numbers: object) -> Dict[str, str]:
    if isinstance(numbers, dict):
        return {clean_text(str(k)): clean_text(str(v)) for k, v in numbers.items()}

    text = clean_text(str(numbers or ""))
    out: Dict[str, str] = {}
    if text:
        for pair in text.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                out[clean_text(k)] = clean_text(v)
    return out


def resolve_channel_numbers(confs_dir, numbers: object) -> Dict[str, str]:
    out = load_channel_numbers_from_confs(confs_dir)
    out.update(_parse_number_map(numbers))
    return out


def _status(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[status] {clean_text(message)}")


def dump_catalog_file(path: Path, fs42_dir: Path, channels: List[str], year: int, schedules: Dict[str, List[Event]]) -> None:
    payload = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "fs42_dir": str(fs42_dir),
        "year": int(year),
        "channels": [clean_text(ch) for ch in channels],
        "schedules": {
            clean_text(ch): [
                {
                    "start": e.start.isoformat(timespec="minutes"),
                    "end": e.end.isoformat(timespec="minutes"),
                    "title": clean_text(e.title),
                    "filename": clean_text(e.filename),
                }
                for e in schedules.get(ch, [])
            ]
            for ch in channels
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_catalog_file(path: Path) -> tuple[List[str], int, Dict[str, List[Event]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Catalog must be a JSON object")

    channels_raw = raw.get("channels", [])
    if not isinstance(channels_raw, list):
        raise ValueError("Catalog 'channels' must be a list")
    channels = [clean_text(str(ch)) for ch in channels_raw if clean_text(str(ch))]
    if not channels:
        raise ValueError("Catalog has no channels")

    year = int(raw.get("year", 0))
    if year <= 0:
        raise ValueError("Catalog 'year' must be a positive integer")

    schedules_raw = raw.get("schedules", {})
    if not isinstance(schedules_raw, dict):
        raise ValueError("Catalog 'schedules' must be an object")

    schedules: Dict[str, List[Event]] = {}
    for ch in channels:
        rows = schedules_raw.get(ch, [])
        if not isinstance(rows, list):
            rows = []
        events: List[Event] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                start = datetime.fromisoformat(str(row.get("start", "")))
                end = datetime.fromisoformat(str(row.get("end", "")))
            except Exception:
                continue
            if end <= start:
                continue
            events.append(
                Event(
                    start=start,
                    end=end,
                    title=clean_text(str(row.get("title", ""))),
                    filename=clean_text(str(row.get("filename", ""))),
                )
            )
        events.sort(key=lambda e: e.start)
        schedules[ch] = events

    return channels, year, schedules


def main(argv: List[str] | None = None) -> None:
    args = parse_effective_args(argv)

    number_map = resolve_channel_numbers(args.confs_dir, args.numbers)
    if args.load_catalog:
        _status(args.status_messages, f"Loading schedule catalog from {args.load_catalog}")
        channels, catalog_year, schedules = load_catalog_file(args.load_catalog)
        year = args.year or catalog_year
        _status(args.status_messages, f"Loaded catalog: {len(channels)} channels, year {catalog_year}")
    else:
        _status(args.status_messages, f"Scanning channel extents in {args.fs42_dir}")
        channels = discover_channels_from_extents(args.fs42_dir)
        if not channels:
            raise RuntimeError("No channels with valid schedules found")
        _status(args.status_messages, f"Discovered {len(channels)} channels")

        year = args.year
        if not year:
            _status(args.status_messages, "Inferring schedule year from extents")
            inferred = None
            for ch in channels:
                inferred = infer_year_from_extents(args.fs42_dir, ch)
                if inferred:
                    break
            year = inferred or args.date.year
        _status(args.status_messages, f"Using schedule year {year}")

        schedules = {}
        for idx, ch in enumerate(channels, start=1):
            _status(args.status_messages, f"Scanning schedule {idx}/{len(channels)}: {ch}")
            try:
                schedules[ch] = parse_channel_schedule(args.fs42_dir, ch, year)
                _status(args.status_messages, f"Loaded {len(schedules[ch])} events for {ch}")
            except Exception:
                schedules[ch] = []
                _status(args.status_messages, f"Failed to load schedule for {ch}; using empty schedule")

        if args.dump_catalog:
            _status(args.status_messages, f"Writing catalog to {args.dump_catalog}")
            dump_catalog_file(args.dump_catalog, args.fs42_dir, channels, year, schedules)
            _status(args.status_messages, "Catalog write complete")

    start_dt, end_dt = compute_range_bounds(args.range_mode, args.date, args.start, args.hours)
    _status(
        args.status_messages,
        f"Preparing {args.range_mode} render window {start_dt.strftime('%Y-%m-%d %H:%M')} to {end_dt.strftime('%Y-%m-%d %H:%M')}",
    )

    title = clean_text(args.title.strip()) or clean_text(
        f"TIME TRAVEL CABLE GUIDE - {args.date.strftime('%a %b %d, %Y')} ({start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')})"
    )

    if args.range_mode == "single":
        _status(args.status_messages, "Rendering single-range PDF")
        make_pdf(
            out_path=args.out,
            grid_title=title,
            channels=channels,
            channel_numbers=number_map,
            schedules=schedules,
            start_dt=start_dt,
            end_dt=end_dt,
            step_minutes=args.step,
            double_sided_fold=args.double_sided_fold,
            fold_safe_gap=args.fold_safe_gap,
        )
    else:
        blocks = split_into_blocks(start_dt, end_dt, args.page_block_hours)
        _status(args.status_messages, f"Rendering compilation PDF with {len(blocks)} block(s)")
        make_compilation_pdf(
            out_path=args.out,
            channels=channels,
            channel_numbers=number_map,
            schedules=schedules,
            range_mode=args.range_mode,
            range_start=start_dt,
            range_end=end_dt,
            page_block_hours=args.page_block_hours,
            step_minutes=args.step,
            double_sided_fold=args.double_sided_fold,
            ads_dir=args.ads_dir,
            ad_insert_every=args.ad_insert_every,
            bottom_ads_dir=args.bottom_ads_dir,
            cover_enabled=args.cover_page,
            cover_title=args.cover_title,
            cover_subtitle=args.cover_subtitle,
            cover_period_label=args.cover_period_label,
            cover_bg_color=args.cover_bg_color,
            cover_border_size=args.cover_border_size,
            cover_text_color=args.cover_text_color,
            cover_text_outline_color=args.cover_text_outline_color,
            cover_text_outline_width=args.cover_text_outline_width,
            cover_title_font=args.cover_title_font,
            cover_title_size=args.cover_title_size,
            cover_subtitle_font=args.cover_subtitle_font,
            cover_subtitle_size=args.cover_subtitle_size,
            cover_date_font=args.cover_date_font,
            cover_date_size=args.cover_date_size,
            cover_airing_font=args.cover_airing_font,
            cover_airing_size=args.cover_airing_size,
            cover_airing_label_enabled=args.cover_airing_label_enabled,
            cover_airing_label_single_format=args.cover_airing_label_single_format,
            cover_airing_label_day_format=args.cover_airing_label_day_format,
            cover_airing_label_week_format=args.cover_airing_label_week_format,
            cover_airing_label_month_format=args.cover_airing_label_month_format,
            cover_art_source=args.cover_art_source,
            cover_art_dir=args.cover_art_dir,
            tvdb_api_key=args.tvdb_api_key,
            tvdb_pin=args.tvdb_pin,
            status_messages=args.status_messages,
            fold_safe_gap=args.fold_safe_gap,
        )

    _status(args.status_messages, f"Finished writing {args.out}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
