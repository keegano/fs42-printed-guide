#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
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


def main(argv: List[str] | None = None) -> None:
    args = parse_effective_args(argv)

    channels = discover_channels_from_extents(args.fs42_dir)
    if not channels:
        raise RuntimeError("No channels with valid schedules found")

    number_map = _parse_number_map(args.numbers)

    year = args.year
    if not year:
        inferred = None
        for ch in channels:
            inferred = infer_year_from_extents(args.fs42_dir, ch)
            if inferred:
                break
        year = inferred or args.date.year

    start_dt, end_dt = compute_range_bounds(args.range_mode, args.date, args.start, args.hours)

    schedules: Dict[str, List[Event]] = {}
    for ch in channels:
        try:
            schedules[ch] = parse_channel_schedule(args.fs42_dir, ch, year)
        except Exception:
            schedules[ch] = []

    title = clean_text(args.title.strip()) or clean_text(
        f"TIME TRAVEL CABLE GUIDE - {args.date.strftime('%a %b %d, %Y')} ({start_dt.strftime('%H:%M')}-{end_dt.strftime('%H:%M')})"
    )

    if args.range_mode == "single":
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
        )
    else:
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
            cover_art_source=args.cover_art_source,
            cover_art_dir=args.cover_art_dir,
            tvdb_api_key=args.tvdb_api_key,
            tvdb_pin=args.tvdb_pin,
        )

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
