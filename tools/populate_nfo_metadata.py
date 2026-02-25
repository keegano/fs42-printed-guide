#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from guide_config import load_env_values
from guide_core import (
    _extract_year_hint,
    _fetch_omdb_movie_meta,
    _fetch_tvdb_overview_by_title,
    _fetch_tvdb_token,
    clean_text,
    display_title,
    is_movie_event,
)
from print_guide import load_catalog_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Populate local .nfo metadata store under content/nfo (separate from FieldStation42 catalog).")
    p.add_argument("--catalog", type=Path, required=True, help="Catalog dump JSON")
    p.add_argument("--fs42-dir", type=Path, default=None, help="Deprecated/ignored: retained for compatibility with older commands.")
    p.add_argument("--nfo-out-dir", type=Path, default=Path("content/nfo"), help="Output .nfo root directory (default: content/nfo)")
    p.add_argument("--tvdb-api-key", default="")
    p.add_argument("--tvdb-pin", default="")
    p.add_argument("--omdb-api-key", default="")
    p.add_argument("--episode-nfo", action="store_true", help="Write per-episode .nfo files instead of one top-level tvshow.nfo per series.")
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    p.add_argument("--limit", type=int, default=0, help="Max files to write (0=all)")
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
    return s[:80] if s else "unknown"


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", clean_text(text).lower())


def _write_nfo(path: Path, title: str, plot: str, rating: str, rated: str, root_tag: str, dry_run: bool) -> bool:
    xml = (
        f"<{root_tag}>\n"
        f"  <title>{escape(clean_text(title))}</title>\n"
        f"  <plot>{escape(clean_text(plot))}</plot>\n"
        f"  <rating>{escape(clean_text(rating))}</rating>\n"
        f"  <mpaa>{escape(clean_text(rated))}</mpaa>\n"
        f"</{root_tag}>\n"
    )
    if dry_run:
        print(f"[dry-run] would write {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    print(f"wrote {path}")
    return True


def _series_nfo_path(out_root: Path, channel: str, shown: str) -> Path:
    return out_root / "shows" / _slug(channel) / _slug(shown) / "tvshow.nfo"


def _episode_nfo_path(out_root: Path, channel: str, shown: str, filename: str) -> Path:
    stem = Path(clean_text(filename)).stem or shown
    return out_root / "episodes" / _slug(channel) / _slug(shown) / f"{_slug(stem)}.nfo"


def _movie_nfo_path(out_root: Path, shown: str, year_hint: str) -> Path:
    suffix = f"-{year_hint}" if clean_text(year_hint) else ""
    return out_root / "movies" / f"{_slug(shown)}{suffix}.nfo"


def main() -> None:
    args = parse_args()
    print(f"[status] starting local NFO population catalog={args.catalog} out={args.nfo_out_dir}")
    if args.fs42_dir:
        print(f"[status] note: --fs42-dir is ignored in local content mode ({args.fs42_dir})")
    print(f"[status] options dry_run={args.dry_run} limit={args.limit} episode_nfo={args.episode_nfo}")

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
        print("[status] no API keys provided/found; no metadata can be fetched.")

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
    missing_metadata = 0
    skipped_duplicate_show = 0
    skipped_duplicate_movie = 0
    processed_show_keys: set[tuple[str, str]] = set()
    processed_movie_keys: set[str] = set()

    for ch in channels:
        events = schedules.get(ch, [])
        print(f"[status] scanning channel {ch} events={len(events)}")
        channel_written = 0
        for ev in events:
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

            is_movie = is_movie_event(ev.title, ev.filename)
            plot = ""
            rating = ""
            rated = ""
            title = shown

            if is_movie:
                movie_key = _norm(shown)
                if movie_key in processed_movie_keys:
                    skipped_duplicate_movie += 1
                    print(f"[status] skip duplicate movie metadata for '{shown}'")
                    continue
                if args.omdb_api_key:
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
                else:
                    print(f"[status] skip OMDb lookup for movie '{shown}' (no key)")
            else:
                show_key = (clean_text(ch).lower(), _norm(shown))
                if show_key in processed_show_keys and not args.episode_nfo:
                    skipped_duplicate_show += 1
                    print(f"[status] skip duplicate show metadata for '{shown}' on channel={ch}")
                    continue
                if token:
                    print(f"[status] trying TVDB overview for '{shown}'")
                    try:
                        plot = _fetch_tvdb_overview_by_title(shown, token)
                    except Exception:
                        plot = ""
                        print(f"[status] TVDB lookup failed for '{shown}'")
                else:
                    print(f"[status] skip TVDB lookup for show '{shown}' (no key/token)")

            if not plot and not rating and not rated:
                missing_metadata += 1
                print(f"[status] skip no metadata for '{shown}'")
                continue

            if is_movie:
                nfo_path = _movie_nfo_path(args.nfo_out_dir, shown, _extract_year_hint(ev.filename))
                print(f"[status] writing movie nfo for '{shown}' -> {nfo_path}")
                wrote_ok = _write_nfo(nfo_path, title, plot, rating, rated, root_tag="movie", dry_run=args.dry_run)
            elif args.episode_nfo:
                nfo_path = _episode_nfo_path(args.nfo_out_dir, ch, shown, ev.filename)
                print(f"[status] writing episode nfo for '{shown}' -> {nfo_path}")
                wrote_ok = _write_nfo(nfo_path, title, plot, rating, rated, root_tag="episodedetails", dry_run=args.dry_run)
            else:
                nfo_path = _series_nfo_path(args.nfo_out_dir, ch, shown)
                print(f"[status] writing top-level show nfo for '{shown}' -> {nfo_path}")
                wrote_ok = _write_nfo(nfo_path, title, plot, rating, rated, root_tag="tvshow", dry_run=args.dry_run)

            if wrote_ok:
                written += 1
                channel_written += 1
                if is_movie:
                    processed_movie_keys.add(_norm(shown))
                elif not args.episode_nfo:
                    processed_show_keys.add((clean_text(ch).lower(), _norm(shown)))
                if args.limit and written >= args.limit:
                    print(f"limit reached ({args.limit})")
                    return
        print(f"[status] channel summary {ch}: written={channel_written}")

    print(
        f"done; wrote {written} nfo file(s) "
        f"(scanned={scanned}, missing_metadata={missing_metadata}, "
        f"skipped_duplicate_show={skipped_duplicate_show}, skipped_duplicate_movie={skipped_duplicate_movie})"
    )


if __name__ == "__main__":
    main()
