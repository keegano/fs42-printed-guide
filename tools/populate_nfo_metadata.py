#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

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
    for p in exact:
        if p.suffix.lower() != ".nfo":
            return p
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
    channels, _year, schedules = load_catalog_file(args.catalog)
    token = ""
    if args.tvdb_api_key:
        try:
            token = _fetch_tvdb_token(args.tvdb_api_key, args.tvdb_pin)
        except Exception:
            token = ""

    written = 0
    for ch in channels:
        for ev in schedules.get(ch, []):
            shown = clean_text(display_title(ev.title, ev.filename))
            if not shown:
                continue
            media_path = _find_media_path(args.fs42_dir, ch, ev.filename)
            if not media_path:
                continue

            plot = ""
            rating = ""
            rated = ""
            title = shown

            if is_movie_event(ev.title, ev.filename) and args.omdb_api_key:
                try:
                    meta = _fetch_omdb_movie_meta(shown, _extract_year_hint(ev.filename), args.omdb_api_key)
                except Exception:
                    meta = None
                if meta:
                    title = meta.title or shown
                    plot = meta.plot
                    rating = meta.imdb_rating
                    rated = meta.rated
            elif token:
                try:
                    plot = _fetch_tvdb_overview_by_title(shown, token)
                except Exception:
                    plot = ""

            if not plot and not rating and not rated:
                continue

            if _write_nfo(media_path, title, plot, rating, rated, args.dry_run):
                written += 1
                if args.limit and written >= args.limit:
                    print(f"limit reached ({args.limit})")
                    return

    print(f"done; wrote {written} nfo file(s)")


if __name__ == "__main__":
    main()
