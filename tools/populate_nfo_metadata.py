#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guide_core import (
    _fetch_omdb_movie_meta,
    _fetch_tvdb_overview_by_title,
    _fetch_tvdb_token,
    clean_text,
    display_title,
    is_movie_event,
    _extract_year_hint,
)
from print_guide import load_catalog_file
from guide_config import load_env_values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Populate Plex-compatible .nfo metadata files under FieldStation42/catalog.")
    p.add_argument("--catalog", type=Path, required=True, help="Catalog dump JSON")
    p.add_argument("--fs42-dir", type=Path, required=True, help="FieldStation42 root")
    p.add_argument("--tvdb-api-key", default="")
    p.add_argument("--tvdb-pin", default="")
    p.add_argument("--omdb-api-key", default="")
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    p.add_argument("--limit", type=int, default=0, help="Max files to write (0=all)")
    return p.parse_args()


def _find_media_path(root: Path, channel: str, filename: str) -> Path | None:
    stem = Path(clean_text(filename)).stem
    if not stem:
        return None
    channel_dir = root / "catalog" / clean_text(channel)
    if not channel_dir.exists():
        return None
    exact = list(channel_dir.rglob(f"{stem}.*"))
    print(f"[status] media lookup channel={channel} stem={stem} candidates={len(exact)}")
    for p in exact:
        if p.suffix.lower() != ".nfo":
            print(f"[status] media matched: {p}")
            return p
    print(f"[status] media lookup had no non-nfo file for {channel}/{stem}")
    return None


def _write_nfo(path: Path, title: str, plot: str, rating: str, rated: str, dry_run: bool) -> bool:
    nfo_path = path.with_suffix(".nfo")
    xml = (
        "<movie>\n"
        f"  <title>{escape(clean_text(title))}</title>\n"
        f"  <plot>{escape(clean_text(plot))}</plot>\n"
        f"  <rating>{escape(clean_text(rating))}</rating>\n"
        f"  <mpaa>{escape(clean_text(rated))}</mpaa>\n"
        "</movie>\n"
    )
    if dry_run:
        print(f"[dry-run] would write {nfo_path}")
        return True
    nfo_path.write_text(xml, encoding="utf-8")
    print(f"wrote {nfo_path}")
    return True


def main() -> None:
    args = parse_args()
    print(f"[status] starting NFO population catalog={args.catalog} fs42_dir={args.fs42_dir}")
    print(f"[status] options dry_run={args.dry_run} limit={args.limit}")
    env = load_env_values()
    if not args.tvdb_api_key:
        args.tvdb_api_key = env.get("TVDB_API_KEY", "")
    if not args.tvdb_pin:
        args.tvdb_pin = env.get("TVDB_PIN", "")
    if not args.omdb_api_key:
        args.omdb_api_key = env.get("OMDB_API_KEY", "")

    print(
        "[status] metadata sources: "
        f"tvdb={'on' if args.tvdb_api_key else 'off'} "
        f"omdb={'on' if args.omdb_api_key else 'off'}"
    )
    if not args.tvdb_api_key and not args.omdb_api_key:
        print("[status] no API keys provided/found; no NFO content can be fetched.")

    channels, _year, schedules = load_catalog_file(args.catalog)
    print(f"[status] loaded catalog channels={len(channels)}")
    token = ""
    if args.tvdb_api_key:
        try:
            token = _fetch_tvdb_token(args.tvdb_api_key, args.tvdb_pin)
            print("[status] TVDB token fetch succeeded")
        except Exception:
            token = ""
            print("[status] TVDB token fetch failed")

    written = 0
    scanned = 0
    missing_media = 0
    missing_metadata = 0
    for ch in channels:
        print(f"[status] scanning channel {ch} events={len(schedules.get(ch, []))}")
        channel_written = 0
        channel_missing_media = 0
        channel_missing_meta = 0
        for ev in schedules.get(ch, []):
            scanned += 1
            shown = clean_text(display_title(ev.title, ev.filename))
            if not shown:
                print("[status] skip event with empty display title")
                continue
            print(
                "[status] event "
                f"{ch} {ev.start.strftime('%m/%d %H:%M')}-{ev.end.strftime('%H:%M')} "
                f"title='{shown}' file='{clean_text(ev.filename)}'"
            )
            media_path = _find_media_path(args.fs42_dir, ch, ev.filename)
            if not media_path:
                missing_media += 1
                channel_missing_media += 1
                print(f"[status] skip no media match for '{shown}'")
                continue

            plot = ""
            rating = ""
            rated = ""
            title = shown

            if is_movie_event(ev.title, ev.filename) and args.omdb_api_key:
                print(f"[status] trying OMDb metadata for movie '{shown}'")
                try:
                    meta = _fetch_omdb_movie_meta(shown, _extract_year_hint(ev.filename), args.omdb_api_key)
                except Exception:
                    meta = None
                    print(f"[status] OMDb lookup failed for '{shown}'")
                if meta:
                    title = meta.title or shown
                    plot = meta.plot
                    rating = meta.imdb_rating
                    rated = meta.rated
                    print(f"[status] OMDb metadata found for '{shown}'")
            elif token:
                print(f"[status] trying TVDB overview for '{shown}'")
                try:
                    plot = _fetch_tvdb_overview_by_title(shown, token)
                except Exception:
                    plot = ""
                    print(f"[status] TVDB lookup failed for '{shown}'")

            if not plot and not rating and not rated:
                missing_metadata += 1
                channel_missing_meta += 1
                print(f"[status] skip no metadata for '{shown}'")
                continue

            if _write_nfo(media_path, title, plot, rating, rated, args.dry_run):
                written += 1
                channel_written += 1
                if args.limit and written >= args.limit:
                    print(f"limit reached ({args.limit})")
                    return
        print(
            f"[status] channel summary {ch}: "
            f"written={channel_written} missing_media={channel_missing_media} missing_metadata={channel_missing_meta}"
        )

    print(
        f"done; wrote {written} nfo file(s) "
        f"(scanned={scanned}, missing_media={missing_media}, missing_metadata={missing_metadata})"
    )


if __name__ == "__main__":
    main()
