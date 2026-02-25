#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import re
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
    load_channel_content_dirs_from_confs,
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
    p.add_argument("--confs-dir", type=Path, default=None, help="Path to FieldStation42 confs directory (defaults to <fs42-dir>/confs).")
    p.add_argument("--dry-run", action="store_true", help="Do not write files")
    p.add_argument("--limit", type=int, default=0, help="Max files to write (0=all)")
    return p.parse_args()


def _find_media_path(root: Path, channel: str, filename: str) -> Path | None:
    # Deprecated exact matcher; kept for initial fast path.
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


def _norm_stem(text: str) -> str:
    t = clean_text(text).lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", "", t)
    return t


def _build_channel_media_index(root: Path, channel: str, conf_content_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    catalog_dir = root / "catalog"
    channel_dir = conf_content_dirs.get(clean_text(channel), catalog_dir / clean_text(channel))
    if conf_content_dirs.get(clean_text(channel)):
        print(f"[status] using conf content_dir for channel={channel}: {channel_dir}")
    if not channel_dir.exists() and catalog_dir.exists():
        want = _norm_stem(channel)
        candidates = [p for p in catalog_dir.iterdir() if p.is_dir()]
        matched = next((p for p in candidates if _norm_stem(p.name) == want), None)
        if matched:
            print(
                f"[status] channel directory normalized match: schedule_channel='{channel}' "
                f"-> folder='{matched.name}'"
            )
            channel_dir = matched
        else:
            for p in candidates:
                if want and want in _norm_stem(p.name):
                    print(
                        f"[status] channel directory fuzzy match: schedule_channel='{channel}' "
                        f"-> folder='{p.name}'"
                    )
                    channel_dir = p
                    break
    index: dict[str, list[Path]] = {}
    if not channel_dir.exists():
        print(f"[status] channel directory missing: {channel_dir}")
        return index
    files = [p for p in channel_dir.rglob("*") if p.is_file() and p.suffix.lower() != ".nfo"]
    print(f"[status] indexing media files for channel={channel} count={len(files)}")
    for p in files:
        k = _norm_stem(p.stem)
        if not k:
            continue
        index.setdefault(k, []).append(p)
    print(f"[status] built media index keys for channel={channel} keys={len(index)}")
    return index


def _find_media_path_indexed(
    root: Path,
    channel: str,
    filename: str,
    shown_title: str,
    channel_index: dict[str, list[Path]],
) -> Path | None:
    raw_stem = Path(clean_text(filename)).stem
    if not raw_stem:
        print(f"[status] media lookup skip: empty filename stem for title='{shown_title}'")
        return None

    # 1) normalized stem match
    target = _norm_stem(raw_stem)
    if target and target in channel_index and channel_index[target]:
        p = channel_index[target][0]
        print(
            "[status] media normalized-stem match "
            f"channel={channel} title='{shown_title}' stem='{raw_stem}' -> {p}"
        )
        return p

    # 2) fallback by normalized display title for schedules with heavily transformed filenames
    title_key = _norm_stem(shown_title)
    if title_key and title_key in channel_index and channel_index[title_key]:
        p = channel_index[title_key][0]
        print(
            "[status] media title-key match "
            f"channel={channel} title='{shown_title}' key='{title_key}' -> {p}"
        )
        return p

    # 3) token-subset fuzzy fallback
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", clean_text(raw_stem).lower()) if len(tok) >= 3]
    if tokens:
        for k, paths in channel_index.items():
            if all(tok in k for tok in tokens[:4]) and paths:
                p = paths[0]
                print(
                    "[status] media fuzzy-token match "
                    f"channel={channel} title='{shown_title}' tokens={tokens[:4]} -> {p}"
                )
                return p

    print(
        "[status] media miss "
        f"channel={channel} title='{shown_title}' stem='{raw_stem}' norm='{target}' title_norm='{title_key}'"
    )
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
    if args.confs_dir is None:
        args.confs_dir = args.fs42_dir / "confs"
    print(f"[status] starting NFO population catalog={args.catalog} fs42_dir={args.fs42_dir}")
    print(f"[status] options dry_run={args.dry_run} limit={args.limit} confs_dir={args.confs_dir}")
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
    conf_content_dirs = load_channel_content_dirs_from_confs(args.confs_dir, base_dir=args.fs42_dir)
    print(f"[status] loaded conf content_dir mappings={len(conf_content_dirs)}")
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
    media_index_cache: dict[str, dict[str, list[Path]]] = {}
    for ch in channels:
        print(f"[status] scanning channel {ch} events={len(schedules.get(ch, []))}")
        if ch not in media_index_cache:
            media_index_cache[ch] = _build_channel_media_index(args.fs42_dir, ch, conf_content_dirs)
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
            media_path = _find_media_path_indexed(
                args.fs42_dir,
                ch,
                ev.filename,
                shown,
                media_index_cache[ch],
            )
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
