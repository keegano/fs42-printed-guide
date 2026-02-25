#!/usr/bin/env python3
"""
Generate a newspaper-style black & white TV guide PDF from FieldStation42 CLI output.

Example:
  python3 fs42_print_guide.py \
    --fs42-dir /home/keegan/FieldStation42 \
    --date 2026-02-16 \
    --start 00:00 --hours 6 \
    --channels "NBC,PBS,MTV,Sitcom,Discovery,TLC,Animal Planet,Judge,TV Land,Nickelodeon,Cartoon Network,Disney,ABC Family,Boomerang,Gameshow,Adult Swim,Sci-Fi,Joss,Wizard,TCM,Musical Classics" \
    --numbers "NBC=3,PBS=4,MTV=7,Sitcom=8,Discovery=13,TLC=15,Animal Planet=16,Judge=17,TV Land=18,Nickelodeon=20,Cartoon Network=22,Disney=23,ABC Family=26,Gameshow=27,Boomerang=29,Adult Swim=24,Sci-Fi=30,Joss=32,TCM=34,Musical Classics=36,Wizard=38" \
    --out guide.pdf
"""

from __future__ import annotations

import argparse
import re
import os
import random
import calendar
import tempfile
import subprocess
import urllib.request
import urllib.error
import urllib.parse
import json
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ReportLab
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader


LINE_RE = re.compile(
    r"^(?P<mm>\d{2})/(?P<dd>\d{2}) (?P<start>\d{2}:\d{2}) - (?P<end>\d{2}:\d{2}) - (?P<title>.*?)(?: - (?P<filename>.*))?$"
)

EXTENT_RE = re.compile(
    r"^(?P<chan>.+?) schedule extents:\s*(?P<start>None|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?)\s*to\s*(?P<end>None|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?)\s*$"
)


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)
FULL_STAR = "★"
HALF_TEXT = "½"


def _pick_unicode_bold_font() -> str:
    """
    Prefer a Unicode-capable bold font for inline metadata glyphs.
    Falls back to Helvetica-Bold if no TTF is available.
    """
    font_name = "DejaVuSans-Bold"
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    if font_name in pdfmetrics.getRegisteredFontNames():
        return font_name
    if font_path.exists():
        try:
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))
            return font_name
        except Exception:
            pass
    return "Helvetica-Bold"


UNICODE_BOLD_FONT = _pick_unicode_bold_font()


def pretty_from_filename(filename: str) -> str:
    """Make a human-ish title from a file name (no extension)."""
    base = os.path.basename(filename)
    stem = os.path.splitext(base)[0]
    # common scene release separators
    stem = stem.replace(".", " ").replace("_", " ")
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def ascii_only(s: str) -> str:
    return (s or "").encode("ascii", "ignore").decode("ascii")


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", ascii_only(s)).strip()


def normalize_title_text(s: str) -> str:
    """
    Normalize display titles:
    - ASCII only
    - remove bracketed chunks like "[Foo]"
    - collapse whitespace
    """
    t = clean_text(s)
    t = re.sub(r"\S*\[[^\]]*\]\S*", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def strip_year_and_after(s: str) -> str:
    """Truncate at first 19xx/20xx (including the year)."""
    m = YEAR_RE.search(s)
    if not m:
        return s.strip()
    return s[: m.start()].rstrip()

def display_title(event_title: str, filename: str) -> str:
    """
    Decide what to show in the cell.

    - Episodes: use event_title (channel-friendly)
    - Movies: derive from filename stem, then strip at first year
    """
    event_title = normalize_title_text(event_title)
    filename = clean_text(filename)
    if EP_RE.search(filename):
        return event_title

    # If it looks like a movie rip (has a year anywhere), prefer filename-derived
    pretty = pretty_from_filename(filename)
    cleaned = strip_year_and_after(pretty)
    cleaned = normalize_title_text(cleaned)

    # Fallback: if filename didn't yield anything useful
    if cleaned:
        return cleaned
    return normalize_title_text(strip_year_and_after(event_title) or event_title)


def is_movie_event(event_title: str, filename: str) -> bool:
    filename = clean_text(filename)
    event_title = clean_text(event_title)
    if EP_RE.search(filename):
        return False
    if YEAR_RE.search(filename):
        return True
    # Conservative fallback: likely a movie block if filename is absent and title has a year.
    return bool(YEAR_RE.search(event_title))


def _extract_year_hint(text: str) -> str:
    m = YEAR_RE.search(clean_text(text))
    return m.group(1) if m else ""

@dataclass(frozen=True)
class Event:
    start: datetime
    end: datetime
    title: str
    filename: str


@dataclass(frozen=True)
class OnTonightEntry:
    title: str
    description: str


@dataclass(frozen=True)
class MovieMeta:
    title: str
    year: str
    plot: str
    imdb_rating: str
    rated: str


def _ensure_api_cache_struct(cache: Optional[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object]
    if cache is None:
        out = {}
    else:
        out = cache
    for key in ("tvdb_token", "tvdb_overview", "tvdb_cover_image", "omdb_movie"):
        if not isinstance(out.get(key), dict):
            out[key] = {}
    return out


def _load_api_cache(path: Optional[Path]) -> Dict[str, object]:
    if not path or not path.exists() or not path.is_file():
        return _ensure_api_cache_struct({})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return _ensure_api_cache_struct(data)
    except Exception:
        pass
    return _ensure_api_cache_struct({})


def _save_api_cache(path: Optional[Path], cache: Dict[str, object]) -> None:
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass


def run_fs42(fs42_dir: Path, args: List[str]) -> str:
    """
    Runs: python station_42.py <args...> inside fs42_dir and returns stdout+stderr merged.
    """
    cmd = ["python", "station_42.py"] + args
    p = subprocess.run(
        cmd,
        cwd=str(fs42_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return p.stdout

def discover_channels_from_extents(fs42_dir: Path) -> List[str]:
    """
    Uses `station_42.py -e` to discover channels that have valid schedule extents.
    Ignores any channel where start or end is None.
    Preserves CLI output order.
    """
    out = run_fs42(fs42_dir, ["-e"])
    channels = []

    for line in out.splitlines():
        line = clean_text(line)
        m = EXTENT_RE.match(line)
        if not m:
            continue

        chan = clean_text(m.group("chan"))
        start_s = m.group("start")
        end_s = m.group("end")

        if start_s == "None" or end_s == "None":
            continue

        channels.append(chan)

    return channels



def infer_year_from_extents(fs42_dir: Path, channel_name: str) -> Optional[int]:
    """
    Uses `station_42.py -e` output to infer the schedule year (e.g., 2026).
    Returns None if not found.
    """
    out = run_fs42(fs42_dir, ["-e"])
    for line in out.splitlines():
        m = EXTENT_RE.match(clean_text(line))
        if not m:
            continue
        if clean_text(m.group("chan")) != clean_text(channel_name):
            continue
        start_s = m.group("start")
        if start_s == "None":
            return None
        # "2026-02-16 00:00:00"
        y = int(start_s.split("-", 1)[0])
        return y
    return None


def parse_channel_schedule(fs42_dir: Path, channel: str, year: int) -> List[Event]:
    out = run_fs42(fs42_dir, ["-u", channel])
    events: List[Event] = []
    for line in out.splitlines():
        line = clean_text(line)
        m = LINE_RE.match(line)
        if not m:
            continue

        mm = int(m.group("mm"))
        dd = int(m.group("dd"))
        st = datetime.strptime(m.group("start"), "%H:%M").time()
        en = datetime.strptime(m.group("end"), "%H:%M").time()
        title = clean_text(m.group("title"))
        filename = clean_text(m.group("filename") or "")

        start_dt = datetime(year, mm, dd, st.hour, st.minute)
        end_dt = datetime(year, mm, dd, en.hour, en.minute)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        events.append(Event(start=start_dt, end=end_dt, title=title, filename=filename))

    events.sort(key=lambda e: e.start)
    return events

def hard_truncate_to_width(text: str, font_name: str, font_size: float, max_width_pts: float) -> str:
    """
    Chop characters off the end until the string fits max_width_pts.
    No ellipsis.
    """
    t = clean_text(text)
    if not t:
        return t

    # Fast path
    if pdfmetrics.stringWidth(t, font_name, font_size) <= max_width_pts:
        return t

    # Binary-ish chop (but simple loop is fine at small sizes)
    lo, hi = 0, len(t)
    # Find the maximum prefix that fits
    while lo < hi:
        mid = (lo + hi) // 2
        cand = t[:mid]
        if pdfmetrics.stringWidth(cand, font_name, font_size) <= max_width_pts:
            lo = mid + 1
        else:
            hi = mid
    # lo is 1 past the best fit
    best = max(0, lo - 1)
    return t[:best].rstrip()


def fit_show_title(text: str, font_name: str, font_size: float, max_width_pts: float) -> str:
    """
    Fit a title using ordered rules:
    1) Remove leading "The "
    2) Replace " And " with ", "
    3) Truncate
    4) If final word was truncated to <=3 chars but was originally >=4, drop that word
    """
    t = normalize_title_text(text)
    if not t:
        return t
    if pdfmetrics.stringWidth(t, font_name, font_size) <= max_width_pts:
        return t

    def _segment_reduce(s: str, word: str, internal_pat: str) -> str:
        parts = s.split(" - ")
        reduced = []
        for seg in parts:
            x = seg.strip()
            x = re.sub(rf"(?i)^{word}\s+", "", x).strip()
            x = re.sub(internal_pat, r"\1, ", x)
            x = re.sub(r"\s+", " ", x).strip().strip(",")
            reduced.append(x)
        return " - ".join(reduced).strip()

    # Stage 1: "the" (case-insensitive), front-removal first, then internal comma conversion.
    t = _segment_reduce(t, "the", r"(?i)(\b[^\W_][\w']*)\s+the\s+")
    t = re.sub(r"\s+", " ", t).strip()
    if pdfmetrics.stringWidth(t, font_name, font_size) <= max_width_pts:
        return t

    # Stage 2: "and" (case-insensitive), front-removal first, then internal comma conversion.
    t = _segment_reduce(t, "and", r"(?i)(\b[^\W_][\w']*)\s+and\s+(?:the\s+)?")
    t = re.sub(r"\s+", " ", t).strip()
    if pdfmetrics.stringWidth(t, font_name, font_size) <= max_width_pts:
        return t

    truncated = hard_truncate_to_width(t, font_name, font_size, max_width_pts)
    orig_words = t.split()
    trunc_words = truncated.split()
    if trunc_words and len(trunc_words) <= len(orig_words):
        idx = len(trunc_words) - 1
        orig_last = orig_words[idx]
        trunc_last = trunc_words[-1]
        if len(trunc_last) <= 3 and len(orig_last) >= 4 and trunc_last != orig_last:
            trunc_words = trunc_words[:-1]
            truncated = " ".join(trunc_words).rstrip(", ").strip()

    return truncated


def fit_title_with_badge(
    title: str,
    badge: str,
    font_name: str,
    font_size: float,
    max_width_pts: float,
) -> str:
    t = clean_text(title)
    b = re.sub(r"\s+", " ", str(badge or "")).strip()
    if not b:
        return fit_show_title(t, font_name, font_size, max_width_pts)

    badge_txt = f"[{b}]"
    badge_w = pdfmetrics.stringWidth(badge_txt, font_name, font_size)
    space_w = pdfmetrics.stringWidth(" ", font_name, font_size)

    if badge_w > max_width_pts:
        # Badge alone does not fit; fallback to title behavior.
        return fit_show_title(t, font_name, font_size, max_width_pts)

    title_max = max_width_pts - badge_w - space_w
    if title_max <= 4:
        return badge_txt

    title_fit = fit_show_title(t, font_name, font_size, title_max)
    if title_fit:
        return f"{title_fit} {badge_txt}"
    return badge_txt


def clip_events(events: List[Event], start_dt: datetime, end_dt: datetime) -> List[Event]:
    clipped = []
    for e in events:
        if e.end <= start_dt or e.start >= end_dt:
            continue
        clipped.append(e)
    return clipped


def load_channel_numbers_from_confs(confs_dir: Optional[Path]) -> Dict[str, str]:
    """
    Load channel numbers from conf JSON files.
    Expected shape per file:
      {"station_conf": {"network_name": "...", "channel_number": 3, ...}}
    """
    out: Dict[str, str] = {}
    if not confs_dir or not confs_dir.exists() or not confs_dir.is_dir():
        return out

    for p in sorted(confs_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sc = data.get("station_conf", {}) if isinstance(data, dict) else {}
            name = clean_text(str(sc.get("network_name", "")))
            number = sc.get("channel_number", None)
            if name and number is not None:
                out[name] = clean_text(str(number))
        except Exception:
            continue
    return out


def list_image_files(folder: Optional[Path]) -> List[Path]:
    if not folder:
        return []
    if not folder.exists() or not folder.is_dir():
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def choose_random(items: List[Path]) -> Optional[Path]:
    if not items:
        return None
    return random.choice(items)


def pick_cover_period_label(range_mode: str, start_dt: datetime) -> str:
    if range_mode == "month":
        return start_dt.strftime("%B %Y")
    if range_mode == "week":
        end = start_dt + timedelta(days=6)
        return f"{start_dt.strftime('%b %d')} - {end.strftime('%b %d, %Y')}"
    if range_mode == "day":
        return start_dt.strftime("%B %d, %Y")
    return start_dt.strftime("%B %Y")


def _clock_no_ampm(dt: datetime) -> str:
    return f"{dt.hour % 12 or 12}:{dt.minute:02d}"


def _build_airing_label(
    range_mode: str,
    title: str,
    when_dt: datetime,
    single_fmt: str,
    day_fmt: str,
    week_fmt: str,
    month_fmt: str,
) -> str:
    template = {
        "single": single_fmt,
        "day": day_fmt,
        "week": week_fmt,
        "month": month_fmt,
    }.get(range_mode, day_fmt)
    values = {
        "title": clean_text(title),
        "show": clean_text(title),
        "weekday": when_dt.strftime("%A"),
        "dow": when_dt.strftime("%a"),
        "time": _clock_no_ampm(when_dt),
        "md": f"{when_dt.month}/{when_dt.day}",
        "date": when_dt.strftime("%m/%d"),
    }
    try:
        return clean_text(template.format(**values))
    except Exception:
        return clean_text(template)


def pick_cover_airing_event(
    schedules: Dict[str, List[Event]],
    range_start: datetime,
    range_end: datetime,
) -> Optional[Event]:
    events = [e for evs in schedules.values() for e in evs if clean_text(e.title)]
    if not events:
        return None
    in_range = [e for e in events if range_start <= e.start < range_end]
    if in_range:
        return random.choice(in_range)
    events.sort(key=lambda e: e.start)
    return events[0]


def compute_range_bounds(range_mode: str, anchor_date: date, start_time: time, hours: float) -> Tuple[datetime, datetime]:
    if range_mode == "single":
        s = datetime.combine(anchor_date, start_time)
        return s, s + timedelta(hours=hours)
    if range_mode == "day":
        s = datetime.combine(anchor_date, time(0, 0))
        return s, s + timedelta(days=1)
    if range_mode == "week":
        s = datetime.combine(anchor_date, time(0, 0))
        return s, s + timedelta(days=7)
    # month
    s = datetime.combine(anchor_date.replace(day=1), time(0, 0))
    _, days = calendar.monthrange(s.year, s.month)
    return s, s + timedelta(days=days)


def split_into_blocks(start_dt: datetime, end_dt: datetime, block_hours: float) -> List[Tuple[datetime, datetime]]:
    blocks: List[Tuple[datetime, datetime]] = []
    step = timedelta(hours=max(0.25, block_hours))
    cur = start_dt
    while cur < end_dt:
        nxt = min(end_dt, cur + step)
        blocks.append((cur, nxt))
        cur = nxt
    return blocks


def _http_json(url: str, method: str = "GET", payload: Optional[dict] = None, headers: Optional[dict] = None) -> dict:
    data = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=req_headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8", "ignore")
    return json.loads(raw or "{}")


def fetch_tvdb_cover_art(
    show_names: List[str],
    api_key: str,
    pin: str = "",
    api_cache: Optional[Dict[str, object]] = None,
) -> Optional[Path]:
    if not show_names or not api_key:
        return None
    try:
        cache = _ensure_api_cache_struct(api_cache)
        token = _fetch_tvdb_token(api_key, pin, api_cache=cache)
        if not token:
            return None

        show = random.choice(show_names)
        show_key = clean_text(show).lower()
        cover_cache = cache.get("tvdb_cover_image", {})
        if not isinstance(cover_cache, dict):
            cover_cache = {}
            cache["tvdb_cover_image"] = cover_cache

        img_url = clean_text(str(cover_cache.get(show_key, "")))
        if not img_url:
            q = urllib.parse.quote(show)
            search = _http_json(
                f"https://api4.thetvdb.com/v4/search?query={q}&type=series",
                headers={"Authorization": f"Bearer {token}"},
            )
            results = search.get("data") or []
            if not results:
                return None
            series_id = results[0].get("tvdb_id") or results[0].get("id")
            if not series_id:
                return None

            detail = _http_json(
                f"https://api4.thetvdb.com/v4/series/{series_id}/extended",
                headers={"Authorization": f"Bearer {token}"},
            )
            d = detail.get("data", {}) or {}
            img_url = d.get("image") or ""
            if not img_url:
                arts = d.get("artworks") or []
                for a in arts:
                    if a.get("image"):
                        img_url = a["image"]
                        break
            if img_url:
                cover_cache[show_key] = clean_text(img_url)
        if not img_url:
            return None

        suffix = Path(urllib.parse.urlparse(img_url).path).suffix or ".jpg"
        tmp = tempfile.NamedTemporaryFile(prefix="tvdb_cover_", suffix=suffix, delete=False)
        tmp_path = Path(tmp.name)
        tmp.close()
        urllib.request.urlretrieve(img_url, str(tmp_path))
        return tmp_path
    except Exception:
        return None


def _fetch_tvdb_token(api_key: str, pin: str = "", api_cache: Optional[Dict[str, object]] = None) -> str:
    if not api_key:
        return ""
    cache = _ensure_api_cache_struct(api_cache)
    token_cache = cache.get("tvdb_token", {})
    if not isinstance(token_cache, dict):
        token_cache = {}
        cache["tvdb_token"] = token_cache
    cache_key = f"{api_key}|{pin}"
    cached = clean_text(str(token_cache.get(cache_key, "")))
    if cached:
        return cached
    login_payload = {"apikey": api_key}
    if pin:
        login_payload["pin"] = pin
    auth = _http_json("https://api4.thetvdb.com/v4/login", method="POST", payload=login_payload)
    token = auth.get("data", {}).get("token") or ""
    if token:
        token_cache[cache_key] = token
    return token


def _fetch_tvdb_overview_by_title(title: str, token: str, api_cache: Optional[Dict[str, object]] = None) -> str:
    if not title or not token:
        return ""
    cache = _ensure_api_cache_struct(api_cache)
    ov_cache = cache.get("tvdb_overview", {})
    if not isinstance(ov_cache, dict):
        ov_cache = {}
        cache["tvdb_overview"] = ov_cache
    cache_key = clean_text(title).lower()
    if cache_key in ov_cache:
        return clean_text(str(ov_cache.get(cache_key, "")))
    q = urllib.parse.quote(title)
    search = _http_json(
        f"https://api4.thetvdb.com/v4/search?query={q}&type=series",
        headers={"Authorization": f"Bearer {token}"},
    )
    results = search.get("data") or []
    if not results:
        return ""
    series_id = results[0].get("tvdb_id") or results[0].get("id")
    if not series_id:
        return ""
    detail = _http_json(
        f"https://api4.thetvdb.com/v4/series/{series_id}/extended",
        headers={"Authorization": f"Bearer {token}"},
    )
    d = detail.get("data", {}) or {}
    overview = clean_text(d.get("overview") or "")
    ov_cache[cache_key] = overview
    return overview


def _fetch_omdb_movie_meta(
    title: str,
    year: str,
    api_key: str,
    api_cache: Optional[Dict[str, object]] = None,
) -> Optional[MovieMeta]:
    if not title or not api_key:
        return None
    cache = _ensure_api_cache_struct(api_cache)
    om_cache = cache.get("omdb_movie", {})
    if not isinstance(om_cache, dict):
        om_cache = {}
        cache["omdb_movie"] = om_cache
    cache_key = f"{clean_text(title).lower()}|{clean_text(year)}"
    cached = om_cache.get(cache_key)
    if isinstance(cached, dict):
        return MovieMeta(
            title=clean_text(str(cached.get("title", title))),
            year=clean_text(str(cached.get("year", year))),
            plot=clean_text(str(cached.get("plot", ""))),
            imdb_rating=clean_text(str(cached.get("imdb_rating", ""))),
            rated=clean_text(str(cached.get("rated", ""))),
        )
    params = {"t": clean_text(title), "apikey": api_key}
    if year:
        params["y"] = year
    url = f"http://www.omdbapi.com/?{urllib.parse.urlencode(params)}"
    data = _http_json(url)
    if clean_text(str(data.get("Response", ""))).lower() == "false":
        om_cache[cache_key] = {}
        return None
    meta = MovieMeta(
        title=clean_text(str(data.get("Title", title))),
        year=clean_text(str(data.get("Year", year))),
        plot=clean_text(str(data.get("Plot", ""))),
        imdb_rating=clean_text(str(data.get("imdbRating", ""))),
        rated=clean_text(str(data.get("Rated", ""))),
    )
    om_cache[cache_key] = {
        "title": meta.title,
        "year": meta.year,
        "plot": meta.plot,
        "imdb_rating": meta.imdb_rating,
        "rated": meta.rated,
    }
    return meta


def _movie_meta_badge(meta: Optional[MovieMeta]) -> str:
    if not meta:
        return ""
    bits: List[str] = []
    rated = clean_text(meta.rated)
    if rated and rated not in ("N/A", "Not Rated"):
        bits.append(rated)
    try:
        score = float(meta.imdb_rating)
        stars = max(0.0, min(5.0, score / 2.0))
        half_steps = int(round(stars * 2.0))
        full = half_steps // 2
        has_half = (half_steps % 2) == 1
        stars_text = FULL_STAR * full
        if has_half:
            stars_text = f"{stars_text} {HALF_TEXT}".strip()
        if stars_text:
            bits.append(stars_text)
    except Exception:
        pass
    return " ".join(bits).strip()


def _movie_cache_key_for_event(event: Event) -> str:
    return clean_text(display_title(event.title, event.filename))


def _get_movie_meta_for_event(
    event: Event,
    omdb_api_key: str,
    movie_cache: Dict[str, MovieMeta],
    api_cache: Optional[Dict[str, object]] = None,
) -> Optional[MovieMeta]:
    if not is_movie_event(event.title, event.filename):
        return None
    key = _movie_cache_key_for_event(event)
    if not key:
        return None
    meta = movie_cache.get(key)
    if meta:
        return meta
    if not omdb_api_key:
        return None
    try:
        meta = _fetch_omdb_movie_meta(key, _extract_year_hint(event.filename), omdb_api_key, api_cache=api_cache)
    except Exception:
        meta = None
    if meta:
        movie_cache[key] = meta
    return meta


def _block_show_events(
    schedules: Dict[str, List[Event]],
    start_dt: datetime,
    end_dt: datetime,
    max_items: int = 20,
) -> List[Tuple[str, Event]]:
    names: List[Tuple[str, Event]] = []
    seen = set()
    for evs in schedules.values():
        for e in evs:
            if e.end <= start_dt or e.start >= end_dt:
                continue
            t = display_title(e.title, e.filename)
            if not t or t in seen:
                continue
            seen.add(t)
            names.append((t, e))
    random.shuffle(names)
    return names[:max_items]


def _wrap_text_lines(text: str, font_name: str, font_size: float, max_width: float) -> List[str]:
    words = clean_text(text).split()
    if not words:
        return []
    lines: List[str] = []
    cur = words[0]
    for w in words[1:]:
        candidate = f"{cur} {w}"
        if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _split_sentences(text: str) -> List[str]:
    parts = [clean_text(p) for p in re.split(r"(?<=[.!?])\s+", clean_text(text)) if clean_text(p)]
    if parts:
        return parts
    t = clean_text(text)
    return [t] if t else []


def _draw_description_columns(c, descriptions: List[OnTonightEntry], x: float, y: float, w: float, h: float) -> None:
    if not descriptions or w <= 20 or h <= 16:
        return
    c.setFillColorRGB(0, 0, 0)
    c.setStrokeColorRGB(0, 0, 0)
    c.rect(x, y, w, h, stroke=1, fill=0)

    padding = 6.0
    body_x = x + padding
    body_y = y + padding
    body_w = max(10.0, w - (2.0 * padding))
    body_h = max(10.0, h - (2.0 * padding))

    c.setFont("Helvetica-Bold", 8)
    c.drawString(body_x, y + h - 12, "ON TONIGHT")
    body_h -= 10
    body_y += 0

    cols = 2 if body_w >= 320 else 1
    col_gap = 10.0
    col_w = max(20.0, (body_w - ((cols - 1) * col_gap)) / cols)
    line_h = 8.5
    max_lines = max(2, int(body_h / line_h))
    lines_per_col = max(2, max_lines)

    col = 0
    line_idx = 0
    text_top = y + h - 22
    for entry in descriptions:
        if line_idx >= lines_per_col:
            col += 1
            line_idx = 0
        if col >= cols:
            return

        title = clean_text(entry.title)
        if not title:
            continue
        if pdfmetrics.stringWidth(title, "Helvetica-Bold", 7) > col_w:
            # Title itself cannot fit safely; stop this column as requested.
            return

        sentences = _split_sentences(entry.description)
        if not sentences:
            # No valid sentences -> remove this show and stop column.
            return

        usable_lines: List[str] = []
        prospective_idx = line_idx + 1  # reserve title line
        for sent in sentences:
            sent_lines = _wrap_text_lines(sent, "Helvetica", 7, col_w)
            # If a sentence overflows width with long token, or would exceed remaining lines,
            # drop the whole sentence.
            if any(pdfmetrics.stringWidth(ln, "Helvetica", 7) > col_w for ln in sent_lines):
                continue
            remaining = lines_per_col - prospective_idx
            if len(sent_lines) > remaining:
                continue
            usable_lines.extend(sent_lines)
            prospective_idx += len(sent_lines)

        if not usable_lines:
            # Remove whole show and end this column.
            return

        cx = body_x + col * (col_w + col_gap)
        cy = text_top - (line_idx * line_h)
        if cy < body_y:
            return
        c.setFont("Helvetica-Bold", 7)
        c.drawString(cx, cy, title)
        line_idx += 1

        c.setFont("Helvetica", 7)
        for ln in usable_lines:
            cy = text_top - (line_idx * line_h)
            if cy < body_y:
                return
            c.drawString(cx, cy, ln)
            line_idx += 1


def _build_block_descriptions(
    schedules: Dict[str, List[Event]],
    start_dt: datetime,
    end_dt: datetime,
    tvdb_api_key: str,
    tvdb_pin: str,
    omdb_api_key: str,
    desc_cache: Dict[str, str],
    movie_cache: Dict[str, MovieMeta],
    token_holder: Dict[str, str],
    api_cache: Optional[Dict[str, object]] = None,
    max_items: int = 8,
) -> List[OnTonightEntry]:
    candidates = _block_show_events(schedules, start_dt, end_dt, max_items=max_items * 3)
    if not candidates:
        return []
    out: List[OnTonightEntry] = []
    token = token_holder.get("token", "")
    if tvdb_api_key and not token:
        try:
            token = _fetch_tvdb_token(tvdb_api_key, tvdb_pin, api_cache=api_cache)
        except Exception:
            token = ""
        token_holder["token"] = token

    for title, ev in candidates:
        desc = ""
        if is_movie_event(ev.title, ev.filename):
            key = clean_text(title)
            meta = movie_cache.get(key)
            if not meta and omdb_api_key:
                try:
                    meta = _fetch_omdb_movie_meta(title, _extract_year_hint(ev.filename), omdb_api_key, api_cache=api_cache)
                except Exception:
                    meta = None
                if meta:
                    movie_cache[key] = meta
            desc = clean_text(meta.plot if meta else "")
        else:
            desc = desc_cache.get(title, "")
            if not desc and token:
                try:
                    desc = _fetch_tvdb_overview_by_title(title, token, api_cache=api_cache)
                except Exception:
                    desc = ""
                desc_cache[title] = desc
        if desc:
            out.append(OnTonightEntry(title=title, description=desc))
        if len(out) >= max_items:
            break
    return out


def time_label(dt: datetime) -> str:
    return dt.strftime("%I:%M").lstrip("0").replace(":00", "") + ("p" if dt.hour >= 12 else "a")


class GuideTimelineFlowable(Flowable):
    def __init__(
        self,
        channels: List[str],
        channel_numbers: Dict[str, str],
        schedules: Dict[str, List[Event]],
        start_dt: datetime,
        end_dt: datetime,
        step_minutes: int,
        omdb_api_key: str = "",
        movie_cache: Optional[Dict[str, MovieMeta]] = None,
        movie_inline_meta: bool = True,
        api_cache: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.channel_numbers = channel_numbers
        self.schedules = schedules
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.step_minutes = step_minutes
        self.omdb_api_key = clean_text(omdb_api_key)
        self.movie_cache = movie_cache if movie_cache is not None else {}
        self.movie_inline_meta = movie_inline_meta
        self.api_cache = api_cache
        self.cell_font = UNICODE_BOLD_FONT
        self.first_col = (1.15 * 0.75) * inch
        self.header_h = 0.26 * inch
        self.row_h = 0.24 * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.header_h + (len(self.channels) * self.row_h)
        return self.width, self.height

    def _x_for(self, dt: datetime, timeline_w: float, total_seconds: float) -> float:
        secs = (dt - self.start_dt).total_seconds()
        secs = max(0.0, min(total_seconds, secs))
        return self.first_col + (timeline_w * (secs / total_seconds))

    def _safe_header_label_x(self, x: float, label: str, page_width: float) -> float:
        label_w = pdfmetrics.stringWidth(label, "Helvetica-Bold", 7)
        min_x = self.first_col + 1.0 + (label_w / 2.0)
        max_x = page_width - 1.0 - (label_w / 2.0)
        if max_x <= min_x:
            return x
        return min(max(x, min_x), max_x)

    def _draw_channel_label(self, c, row_bottom: float, channel_name: str, channel_num: str) -> None:
        y = row_bottom + ((self.row_h - 7) / 2.0) + 1
        c.setFont("Helvetica-Bold", 7)

        name = hard_truncate_to_width(channel_name, "Helvetica-Bold", 7, self.first_col - 4)
        if not channel_num:
            c.drawCentredString(self.first_col / 2.0, y, name)
            return

        bubble_r = 4.3
        bubble_d = bubble_r * 2.0
        gap = 2.0
        num = hard_truncate_to_width(channel_num, "Helvetica-Bold", 5.5, bubble_d - 2)
        name_max_w = max(1.0, self.first_col - bubble_d - gap - 4)
        name_fit = hard_truncate_to_width(name, "Helvetica-Bold", 7, name_max_w)
        name_w = pdfmetrics.stringWidth(name_fit, "Helvetica-Bold", 7)

        group_w = bubble_d + gap + name_w
        start_x = max(1.0, (self.first_col - group_w) / 2.0)

        cx = start_x + bubble_r
        cy = row_bottom + (self.row_h / 2.0)
        c.circle(cx, cy, bubble_r, stroke=1, fill=0)
        c.setFont("Helvetica-Bold", 5.5)
        c.drawCentredString(cx, cy - 2.0, num)

        c.setFont("Helvetica-Bold", 7)
        c.drawString(start_x + bubble_d + gap, y, name_fit)

    def draw(self) -> None:
        c = self.canv
        width = self.width
        height = self.height
        timeline_w = max(1.0, width - self.first_col)
        total_seconds = max(1.0, (self.end_dt - self.start_dt).total_seconds())

        # Border and main separators only (no per-slot vertical grid cells).
        c.setLineWidth(0.25)
        c.rect(0, 0, width, height, stroke=1, fill=0)
        c.line(0, height - self.header_h, width, height - self.header_h)
        c.line(self.first_col, 0, self.first_col, height)

        # Header
        c.setFont("Helvetica-Bold", 7)
        c.drawCentredString(self.first_col / 2.0, height - self.header_h + 6, "CH")
        ticks: List[datetime] = []
        cur = self.start_dt
        while cur <= self.end_dt:
            ticks.append(cur)
            cur += timedelta(minutes=self.step_minutes)

        for cur in ticks:
            x = self._x_for(cur, timeline_w, total_seconds)
            c.line(x, height - self.header_h, x, height - self.header_h + 3)
            label = time_label(cur)
            safe_x = self._safe_header_label_x(x, label, width)
            c.drawCentredString(safe_x, height - self.header_h + 6, label)

        # Rows and proportional event blocks
        for idx, raw_ch in enumerate(self.channels):
            ch = clean_text(raw_ch)
            row_top = height - self.header_h - (idx * self.row_h)
            row_bottom = row_top - self.row_h

            c.line(0, row_bottom, width, row_bottom)

            num = clean_text(self.channel_numbers.get(raw_ch, ""))
            self._draw_channel_label(c, row_bottom, ch, num)

            evs = clip_events(self.schedules.get(raw_ch, []), self.start_dt, self.end_dt)
            for e in evs:
                ev_start = max(e.start, self.start_dt)
                ev_end = min(e.end, self.end_dt)
                if ev_end <= ev_start:
                    continue

                x0 = self._x_for(ev_start, timeline_w, total_seconds)
                x1 = self._x_for(ev_end, timeline_w, total_seconds)
                box_w = max(0.5, x1 - x0)
                box_y = row_bottom + 0.5
                box_h = max(1.0, self.row_h - 1.0)
                c.rect(x0, box_y, box_w, box_h, stroke=1, fill=0)

                text_w = box_w - 4
                if text_w <= 0:
                    continue

                shown = display_title(e.title, e.filename)
                if self.movie_inline_meta:
                    meta = _get_movie_meta_for_event(e, self.omdb_api_key, self.movie_cache, api_cache=self.api_cache)
                    badge = _movie_meta_badge(meta)
                    if badge:
                        shown = fit_title_with_badge(shown, badge, self.cell_font, 6.5, text_w)
                    else:
                        shown = fit_show_title(shown, self.cell_font, 6.5, text_w)
                else:
                    shown = fit_show_title(shown, self.cell_font, 6.5, text_w)
                if not shown:
                    continue
                c.setFont(self.cell_font, 6.5)
                c.drawString(x0 + 2, row_bottom + ((self.row_h - 6.5) / 2.0) + 1, shown)


class FoldedGuideTimelineFlowable(Flowable):
    """
    Draw two guide halves side-by-side on one page for fold printing.
    Left panel = first half of time range, right panel = second half.
    """
    def __init__(
        self,
        channels: List[str],
        channel_numbers: Dict[str, str],
        schedules: Dict[str, List[Event]],
        start_dt: datetime,
        split_dt: datetime,
        end_dt: datetime,
        step_minutes: int,
        safe_gap: float = 0.0,
        omdb_api_key: str = "",
        movie_cache: Optional[Dict[str, MovieMeta]] = None,
        movie_inline_meta: bool = True,
        api_cache: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        cache = movie_cache if movie_cache is not None else {}
        self.left = GuideTimelineFlowable(
            channels=channels,
            channel_numbers=channel_numbers,
            schedules=schedules,
            start_dt=start_dt,
            end_dt=split_dt,
            step_minutes=step_minutes,
            omdb_api_key=omdb_api_key,
            movie_cache=cache,
            movie_inline_meta=movie_inline_meta,
            api_cache=api_cache,
        )
        self.right = GuideTimelineFlowable(
            channels=channels,
            channel_numbers=channel_numbers,
            schedules=schedules,
            start_dt=split_dt,
            end_dt=end_dt,
            step_minutes=step_minutes,
            omdb_api_key=omdb_api_key,
            movie_cache=cache,
            movie_inline_meta=movie_inline_meta,
            api_cache=api_cache,
        )
        self.gap = (0.12 + max(0.0, safe_gap)) * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        panel_w = max(1.0, (availWidth - self.gap) / 2.0)
        _, lh = self.left.wrap(panel_w, availHeight)
        _, rh = self.right.wrap(panel_w, availHeight)
        self.height = max(lh, rh)
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        panel_w = max(1.0, (self.width - self.gap) / 2.0)

        c.saveState()
        self.left.wrap(panel_w, self.height)
        self.left.drawOn(c, 0, 0)
        c.restoreState()

        c.saveState()
        self.right.wrap(panel_w, self.height)
        self.right.drawOn(c, panel_w + self.gap, 0)
        c.restoreState()


class FoldHeaderFlowable(Flowable):
    def __init__(self, left_text: str, right_text: str, font_size: float = 7.0) -> None:
        super().__init__()
        self.left_text = clean_text(left_text)
        self.right_text = clean_text(right_text)
        self.font_size = font_size
        self.height = 0.14 * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        y = (self.height - self.font_size) / 2.0 + 1
        c.setFont("Helvetica-Bold", self.font_size)
        c.drawString(0, y, self.left_text)
        c.drawRightString(self.width, y, self.right_text)


def draw_image_fit(c, image_path: Path, x: float, y: float, w: float, h: float) -> bool:
    try:
        ir = ImageReader(str(image_path))
        iw, ih = ir.getSize()
        if iw <= 0 or ih <= 0:
            return False
        scale = min(w / iw, h / ih)
        dw = iw * scale
        dh = ih * scale
        dx = x + (w - dw) / 2.0
        dy = y + (h - dh) / 2.0
        c.drawImage(ir, dx, dy, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
        return True
    except Exception:
        return False


def _parse_hex_color(hex_value: str, fallback: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Tuple[float, float, float]:
    raw = clean_text(hex_value).lstrip("#")
    if len(raw) == 3:
        raw = "".join([ch * 2 for ch in raw])
    if len(raw) != 6:
        return fallback
    try:
        return tuple(int(raw[i : i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore[return-value]
    except Exception:
        return fallback


def draw_image_cover(c, image_path: Path, x: float, y: float, w: float, h: float) -> bool:
    """
    Draw image using "cover" behavior: fills target rect and crops overflow.
    """
    try:
        ir = ImageReader(str(image_path))
        iw, ih = ir.getSize()
        if iw <= 0 or ih <= 0 or w <= 0 or h <= 0:
            return False

        scale = max(w / iw, h / ih)
        dw = iw * scale
        dh = ih * scale
        dx = x + (w - dw) / 2.0
        dy = y + (h - dh) / 2.0

        c.saveState()
        clip = c.beginPath()
        clip.rect(x, y, w, h)
        c.clipPath(clip, stroke=0, fill=0)
        c.drawImage(ir, dx, dy, width=dw, height=dh, preserveAspectRatio=True, mask="auto")
        c.restoreState()
        return True
    except Exception:
        return False


def draw_cover_panel(
    c,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    subtitle: str,
    period_label: str,
    art_path: Optional[Path] = None,
    bg_color_hex: str = "FFFFFF",
    border_size_in: float = 0.0,
    text_color_hex: str = "FFFFFF",
    text_outline_color_hex: str = "000000",
    text_outline_width: float = 1.2,
    title_font: str = "Helvetica-Bold",
    title_size: float = 28.0,
    subtitle_font: str = "Helvetica",
    subtitle_size: float = 13.0,
    date_font: str = "Helvetica-Bold",
    date_size: float = 18.0,
    date_y: float = 0.18,
    airing_label: str = "",
    airing_font: str = "Helvetica-Bold",
    airing_size: float = 14.0,
    airing_y: float = 0.11,
) -> None:
    bg = _parse_hex_color(bg_color_hex, fallback=(1.0, 1.0, 1.0))
    text_rgb = _parse_hex_color(text_color_hex, fallback=(1.0, 1.0, 1.0))
    outline_rgb = _parse_hex_color(text_outline_color_hex, fallback=(0.0, 0.0, 0.0))
    c.setFillColorRGB(*bg)
    c.rect(x, y, width, height, stroke=0, fill=1)

    border = max(0.0, border_size_in) * inch
    inner_w = max(0.0, width - (2.0 * border))
    inner_h = max(0.0, height - (2.0 * border))
    text_max_w = max(24.0, width - (2.0 * max(border, 0.12 * inch)))
    if art_path and inner_w > 0 and inner_h > 0:
        draw_image_cover(c, art_path, x + border, y + border, inner_w, inner_h)

    cx = x + (width / 2.0)
    _draw_outlined_centered_text(c, clean_text(title), cx, y + height * 0.86, title_font, title_size, text_rgb, outline_rgb, text_outline_width)
    if subtitle:
        _draw_outlined_centered_text(
            c,
            clean_text(subtitle),
            cx,
            y + height * 0.80,
            subtitle_font,
            subtitle_size,
            text_rgb,
            outline_rgb,
            text_outline_width,
        )
    if period_label:
        _draw_outlined_centered_text(
            c,
            clean_text(period_label),
            cx,
            y + height * max(0.0, min(1.0, date_y)),
            date_font,
            date_size,
            text_rgb,
            outline_rgb,
            text_outline_width,
        )
    if airing_label:
        _draw_outlined_centered_wrapped_text(
            c,
            clean_text(airing_label),
            cx,
            y + height * max(0.0, min(1.0, airing_y)),
            airing_font,
            airing_size,
            text_rgb,
            outline_rgb,
            text_outline_width,
            max_width=text_max_w,
            max_lines=2,
        )


def _draw_outlined_centered_text(
    c,
    text: str,
    center_x: float,
    y: float,
    font_name: str,
    font_size: float,
    fill_rgb: Tuple[float, float, float],
    outline_rgb: Tuple[float, float, float],
    outline_width: float,
) -> None:
    t = clean_text(text)
    if not t:
        return
    c.setFont(font_name, font_size)
    ow = max(0.0, outline_width)
    if ow > 0:
        c.setFillColorRGB(*outline_rgb)
        for dx, dy in [(-ow, 0), (ow, 0), (0, -ow), (0, ow), (-ow, -ow), (-ow, ow), (ow, -ow), (ow, ow)]:
            c.drawCentredString(center_x + dx, y + dy, t)
    c.setFillColorRGB(*fill_rgb)
    c.drawCentredString(center_x, y, t)


def _truncate_with_ellipsis(text: str, font_name: str, font_size: float, max_width: float) -> str:
    t = clean_text(text)
    if not t:
        return t
    if pdfmetrics.stringWidth(t, font_name, font_size) <= max_width:
        return t
    ell = "..."
    fit = hard_truncate_to_width(t, font_name, font_size, max_width - pdfmetrics.stringWidth(ell, font_name, font_size))
    return clean_text(f"{fit}{ell}")


def _wrap_cover_text(
    text: str,
    font_name: str,
    font_size: float,
    max_width: float,
    max_lines: int,
) -> List[str]:
    lines = _wrap_text_lines(text, font_name, font_size, max_width)
    if len(lines) <= max_lines:
        return lines
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :])
    head.append(_truncate_with_ellipsis(tail, font_name, font_size, max_width))
    return head


def _draw_outlined_centered_wrapped_text(
    c,
    text: str,
    center_x: float,
    bottom_line_y: float,
    font_name: str,
    font_size: float,
    fill_rgb: Tuple[float, float, float],
    outline_rgb: Tuple[float, float, float],
    outline_width: float,
    max_width: float,
    max_lines: int = 2,
    line_gap: float = 1.15,
) -> None:
    lines = _wrap_cover_text(text, font_name, font_size, max_width, max_lines=max_lines)
    if not lines:
        return
    for i, line in enumerate(reversed(lines)):
        y = bottom_line_y + (i * font_size * line_gap)
        _draw_outlined_centered_text(c, line, center_x, y, font_name, font_size, fill_rgb, outline_rgb, outline_width)


class FullPageImageFlowable(Flowable):
    def __init__(self, image_path: Path, frame_height: float) -> None:
        super().__init__()
        self.image_path = image_path
        self.frame_height = frame_height

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        draw_image_fit(self.canv, self.image_path, 0, 0, self.width, self.height)


class CoverPageFlowable(Flowable):
    def __init__(
        self,
        frame_height: float,
        title: str,
        subtitle: str,
        period_label: str,
        airing_label: str = "",
        art_path: Optional[Path] = None,
        bg_color: str = "FFFFFF",
        border_size: float = 0.0,
        text_color: str = "FFFFFF",
        text_outline_color: str = "000000",
        text_outline_width: float = 1.2,
        title_font: str = "Helvetica-Bold",
        title_size: float = 28.0,
        subtitle_font: str = "Helvetica",
        subtitle_size: float = 13.0,
        date_font: str = "Helvetica-Bold",
        date_size: float = 18.0,
        date_y: float = 0.18,
        airing_font: str = "Helvetica-Bold",
        airing_size: float = 14.0,
        airing_y: float = 0.11,
    ) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.title = clean_text(title)
        self.subtitle = clean_text(subtitle)
        self.period_label = clean_text(period_label)
        self.airing_label = clean_text(airing_label)
        self.art_path = art_path
        self.bg_color = clean_text(bg_color) or "FFFFFF"
        self.border_size = max(0.0, border_size)
        self.text_color = clean_text(text_color) or "FFFFFF"
        self.text_outline_color = clean_text(text_outline_color) or "000000"
        self.text_outline_width = max(0.0, text_outline_width)
        self.title_font = clean_text(title_font) or "Helvetica-Bold"
        self.title_size = max(1.0, title_size)
        self.subtitle_font = clean_text(subtitle_font) or "Helvetica"
        self.subtitle_size = max(1.0, subtitle_size)
        self.date_font = clean_text(date_font) or "Helvetica-Bold"
        self.date_size = max(1.0, date_size)
        self.date_y = max(0.0, min(1.0, date_y))
        self.airing_font = clean_text(airing_font) or "Helvetica-Bold"
        self.airing_size = max(1.0, airing_size)
        self.airing_y = max(0.0, min(1.0, airing_y))

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        draw_cover_panel(
            c=self.canv,
            x=0,
            y=0,
            width=self.width,
            height=self.height,
            title=self.title,
            subtitle=self.subtitle,
            period_label=self.period_label,
            airing_label=self.airing_label,
            art_path=self.art_path,
            bg_color_hex=self.bg_color,
            border_size_in=self.border_size,
            text_color_hex=self.text_color,
            text_outline_color_hex=self.text_outline_color,
            text_outline_width=self.text_outline_width,
            title_font=self.title_font,
            title_size=self.title_size,
            subtitle_font=self.subtitle_font,
            subtitle_size=self.subtitle_size,
            date_font=self.date_font,
            date_size=self.date_size,
            date_y=self.date_y,
            airing_font=self.airing_font,
            airing_size=self.airing_size,
            airing_y=self.airing_y,
        )


class GuideWithBottomAdFlowable(Flowable):
    def __init__(
        self,
        guide: Flowable,
        ad_path: Optional[Path],
        frame_height: float,
        descriptions: Optional[List[OnTonightEntry]] = None,
    ) -> None:
        super().__init__()
        self.guide = guide
        self.ad_path = ad_path
        self.frame_height = frame_height
        self.gap = 0.08 * inch
        self.descriptions = descriptions or []

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        gw, gh = self.guide.wrap(self.width, self.frame_height)
        guide_y = max(0.0, self.frame_height - gh)
        self.guide.drawOn(c, 0, guide_y)

        available = guide_y - self.gap
        if not self.ad_path:
            if self.descriptions and available > 24:
                _draw_description_columns(c, self.descriptions, 0, 0, self.width, available)
            return
        if available <= 12:
            return
        draw_image_fit(c, self.ad_path, 0, 0, self.width, available)


class CompilationGuidePageFlowable(Flowable):
    def __init__(
        self,
        frame_height: float,
        left_header: str,
        right_header: str,
        guide: Flowable,
        bottom_ad: Optional[Path] = None,
        bottom_descriptions: Optional[List[OnTonightEntry]] = None,
    ) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.left_header = clean_text(left_header)
        self.right_header = clean_text(right_header)
        self.guide = guide
        self.bottom_ad = bottom_ad
        self.bottom_descriptions = bottom_descriptions or []
        self.header_h = 0.14 * inch
        self.gap = 0.06 * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        c.setFont("Helvetica-Bold", 7)
        y_hdr = self.frame_height - self.header_h + 2
        c.drawString(0, y_hdr, self.left_header)
        c.drawRightString(self.width, y_hdr, self.right_header)
        c.line(0, self.frame_height - self.header_h, self.width, self.frame_height - self.header_h)

        content_h = self.frame_height - self.header_h - self.gap
        _, guide_h = self.guide.wrap(self.width, content_h)
        ad_h = 0.0
        if self.bottom_ad and content_h - guide_h > 16:
            ad_h = content_h - guide_h - self.gap

        if ad_h > 0:
            draw_image_fit(c, self.bottom_ad, 0, 0, self.width, ad_h)
        elif self.bottom_descriptions and content_h - guide_h > 24:
            desc_h = content_h - guide_h - self.gap
            _draw_description_columns(c, self.bottom_descriptions, 0, 0, self.width, desc_h)
            ad_h = desc_h

        guide_y = ad_h + (self.gap if ad_h > 0 else 0.0)
        self.guide.drawOn(c, 0, guide_y)


@dataclass(frozen=True)
class BookletPageSpec:
    kind: str  # guide|cover|blank
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    header_left: str = ""
    header_right: str = ""
    bottom_ad: Optional[Path] = None
    bottom_descriptions: Optional[List[OnTonightEntry]] = None
    cover_title: str = ""
    cover_subtitle: str = ""
    cover_period_label: str = ""
    cover_airing_label: str = ""
    cover_art_path: Optional[Path] = None
    cover_bg_color: str = "FFFFFF"
    cover_border_size: float = 0.0
    cover_text_color: str = "FFFFFF"
    cover_text_outline_color: str = "000000"
    cover_text_outline_width: float = 1.2
    cover_title_font: str = "Helvetica-Bold"
    cover_title_size: float = 28.0
    cover_subtitle_font: str = "Helvetica"
    cover_subtitle_size: float = 13.0
    cover_date_font: str = "Helvetica-Bold"
    cover_date_size: float = 18.0
    cover_date_y: float = 0.18
    cover_airing_font: str = "Helvetica-Bold"
    cover_airing_size: float = 14.0
    cover_airing_y: float = 0.11


def _compute_fold_split(start_dt: datetime, end_dt: datetime, step_minutes: int) -> datetime:
    total = end_dt - start_dt
    half = start_dt + timedelta(seconds=total.total_seconds() / 2.0)
    step_s = max(60, step_minutes * 60)
    offset_s = int((half - start_dt).total_seconds())
    snapped_s = int(round(offset_s / step_s) * step_s)
    split_dt = start_dt + timedelta(seconds=snapped_s)
    if split_dt <= start_dt:
        split_dt = start_dt + timedelta(seconds=step_s)
    if split_dt >= end_dt:
        split_dt = end_dt - timedelta(seconds=step_s)
    return split_dt


def impose_booklet_pages(logical_pages: List[BookletPageSpec]) -> List[Tuple[BookletPageSpec, BookletPageSpec]]:
    """
    Convert booklet page order into printer order for duplex fold printing.
    Input pages are reading order (1..N). Output is a list of physical sides
    where each entry is (left_half, right_half).
    """
    pages = list(logical_pages)
    while len(pages) % 4 != 0:
        pages.append(BookletPageSpec(kind="blank"))

    total = len(pages)
    out: List[Tuple[BookletPageSpec, BookletPageSpec]] = []
    for sheet in range(total // 4):
        front_left = pages[total - (2 * sheet) - 1]
        front_right = pages[(2 * sheet)]
        back_left = pages[(2 * sheet) + 1]
        back_right = pages[total - (2 * sheet) - 2]
        out.append((front_left, front_right))
        out.append((back_left, back_right))
    return out


class BookletHalfPageFlowable(Flowable):
    def __init__(
        self,
        frame_height: float,
        spec: BookletPageSpec,
        channels: List[str],
        channel_numbers: Dict[str, str],
        schedules: Dict[str, List[Event]],
        step_minutes: int,
        omdb_api_key: str = "",
        movie_cache: Optional[Dict[str, MovieMeta]] = None,
        movie_inline_meta: bool = True,
        api_cache: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.spec = spec
        self.channels = channels
        self.channel_numbers = channel_numbers
        self.schedules = schedules
        self.step_minutes = step_minutes
        self.omdb_api_key = clean_text(omdb_api_key)
        self.movie_cache = movie_cache if movie_cache is not None else {}
        self.movie_inline_meta = movie_inline_meta
        self.api_cache = api_cache
        self.header_h = 0.14 * inch
        self.gap = 0.06 * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def _draw_cover(self, c) -> None:
        draw_cover_panel(
            c=c,
            x=0,
            y=0,
            width=self.width,
            height=self.height,
            title=self.spec.cover_title,
            subtitle=self.spec.cover_subtitle,
            period_label=self.spec.cover_period_label,
            airing_label=self.spec.cover_airing_label,
            art_path=self.spec.cover_art_path,
            bg_color_hex=self.spec.cover_bg_color,
            border_size_in=self.spec.cover_border_size,
            text_color_hex=self.spec.cover_text_color,
            text_outline_color_hex=self.spec.cover_text_outline_color,
            text_outline_width=self.spec.cover_text_outline_width,
            title_font=self.spec.cover_title_font,
            title_size=self.spec.cover_title_size,
            subtitle_font=self.spec.cover_subtitle_font,
            subtitle_size=self.spec.cover_subtitle_size,
            date_font=self.spec.cover_date_font,
            date_size=self.spec.cover_date_size,
            date_y=self.spec.cover_date_y,
            airing_font=self.spec.cover_airing_font,
            airing_size=self.spec.cover_airing_size,
            airing_y=self.spec.cover_airing_y,
        )

    def _draw_guide(self, c) -> None:
        c.setFont("Helvetica-Bold", 7)
        y_hdr = self.frame_height - self.header_h + 2
        c.drawString(0, y_hdr, self.spec.header_left)
        c.drawRightString(self.width, y_hdr, self.spec.header_right)
        c.line(0, self.frame_height - self.header_h, self.width, self.frame_height - self.header_h)

        content_h = self.frame_height - self.header_h - self.gap
        guide = GuideTimelineFlowable(
            channels=self.channels,
            channel_numbers=self.channel_numbers,
            schedules=self.schedules,
            start_dt=self.spec.start_dt,  # type: ignore[arg-type]
            end_dt=self.spec.end_dt,  # type: ignore[arg-type]
            step_minutes=self.step_minutes,
            omdb_api_key=self.omdb_api_key,
            movie_cache=self.movie_cache,
            movie_inline_meta=self.movie_inline_meta,
            api_cache=self.api_cache,
        )
        content: Flowable = guide
        if self.spec.bottom_ad or self.spec.bottom_descriptions:
            content = GuideWithBottomAdFlowable(
                guide=guide,
                ad_path=self.spec.bottom_ad,
                frame_height=content_h,
                descriptions=self.spec.bottom_descriptions,
            )

        self.guide = content
        _, guide_h = self.guide.wrap(self.width, content_h)
        guide_y = max(0.0, content_h - guide_h)
        self.guide.drawOn(c, 0, guide_y)

    def draw(self) -> None:
        c = self.canv
        c.setFillColorRGB(1, 1, 1)
        c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
        c.setFillColorRGB(0, 0, 0)

        if self.spec.kind == "cover":
            self._draw_cover(c)
            return
        if self.spec.kind == "guide":
            self._draw_guide(c)
            return


class BookletSpreadFlowable(Flowable):
    def __init__(self, frame_height: float, left: Flowable, right: Flowable, safe_gap: float = 0.0) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.left = left
        self.right = right
        self.gap = max(0.0, safe_gap) * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        panel_w = max(1.0, (self.width - self.gap) / 2.0)

        self.left.wrap(panel_w, self.height)
        self.left.drawOn(c, 0, 0)
        self.right.wrap(panel_w, self.height)
        self.right.drawOn(c, panel_w + self.gap, 0)


def make_pdf(
    out_path: Path,
    grid_title: str,
    channels: List[str],
    channel_numbers: Dict[str, str],
    schedules: Dict[str, List[Event]],
    start_dt: datetime,
    end_dt: datetime,
    step_minutes: int,
    double_sided_fold: bool = False,
    fold_safe_gap: float = 0.0,
    omdb_api_key: str = "",
    movie_inline_meta: bool = True,
    api_cache: Optional[Dict[str, object]] = None,
) -> None:
    page_size = landscape(letter)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=page_size,
        leftMargin=0.35 * inch,
        rightMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
        title=grid_title,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=16,
        spaceAfter=6,
    )
    story = []
    movie_cache: Dict[str, MovieMeta] = {}
    runtime_api_cache = _ensure_api_cache_struct(api_cache)

    if double_sided_fold:
        split_dt = _compute_fold_split(start_dt, end_dt, step_minutes)

        story.append(
            FoldHeaderFlowable(
                left_text="CABLE GUIDE",
                right_text=start_dt.strftime("%a %b %d, %Y"),
                font_size=7.0,
            )
        )
        story.append(Spacer(1, 0.04 * inch))
        story.append(
            FoldedGuideTimelineFlowable(
                channels=channels,
                channel_numbers=channel_numbers,
                schedules=schedules,
                start_dt=start_dt,
                split_dt=split_dt,
                end_dt=end_dt,
                step_minutes=step_minutes,
                safe_gap=fold_safe_gap,
                omdb_api_key=omdb_api_key,
                movie_cache=movie_cache,
                movie_inline_meta=movie_inline_meta,
                api_cache=runtime_api_cache,
            )
        )
    else:
        story.append(Paragraph(grid_title, title_style))
        story.append(Spacer(1, 0.08 * inch))
        story.append(
            GuideTimelineFlowable(
                channels=channels,
                channel_numbers=channel_numbers,
                schedules=schedules,
                start_dt=start_dt,
                end_dt=end_dt,
                step_minutes=step_minutes,
                omdb_api_key=omdb_api_key,
                movie_cache=movie_cache,
                movie_inline_meta=movie_inline_meta,
                api_cache=runtime_api_cache,
            )
        )
    doc.build(story)


def make_compilation_pdf(
    out_path: Path,
    channels: List[str],
    channel_numbers: Dict[str, str],
    schedules: Dict[str, List[Event]],
    range_mode: str,
    range_start: datetime,
    range_end: datetime,
    page_block_hours: float,
    step_minutes: int,
    double_sided_fold: bool = False,
    ads_dir: Optional[Path] = None,
    ad_insert_every: int = 0,
    bottom_ads_dir: Optional[Path] = None,
    cover_enabled: bool = False,
    cover_title: str = "Time Travel Cable Guide",
    cover_subtitle: str = "",
    cover_period_label: str = "",
    cover_bg_color: str = "FFFFFF",
    cover_border_size: float = 0.0,
    cover_text_color: str = "FFFFFF",
    cover_text_outline_color: str = "000000",
    cover_text_outline_width: float = 1.2,
    cover_title_font: str = "Helvetica-Bold",
    cover_title_size: float = 28.0,
    cover_subtitle_font: str = "Helvetica",
    cover_subtitle_size: float = 13.0,
    cover_date_font: str = "Helvetica-Bold",
    cover_date_size: float = 18.0,
    cover_date_y: float = 0.18,
    cover_airing_font: str = "Helvetica-Bold",
    cover_airing_size: float = 14.0,
    cover_airing_y: float = 0.11,
    cover_airing_label_enabled: bool = True,
    cover_airing_label_single_format: str = "{time}",
    cover_airing_label_day_format: str = "Catch {show} at {time}!",
    cover_airing_label_week_format: str = "Catch {show} on {weekday} at {time}!",
    cover_airing_label_month_format: str = "Catch {show} on {md} at {time}!",
    cover_art_source: str = "none",
    cover_art_dir: Optional[Path] = None,
    tvdb_api_key: str = "",
    tvdb_pin: str = "",
    omdb_api_key: str = "",
    movie_inline_meta: bool = True,
    api_cache_enabled: bool = True,
    api_cache_file: Optional[Path] = Path(".cache/printed_guide_api_cache.json"),
    status_messages: bool = False,
    fold_safe_gap: float = 0.0,
) -> None:
    def _status(message: str) -> None:
        if status_messages:
            print(f"[status] {clean_text(message)}")

    page_size = landscape(letter)
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=page_size,
        leftMargin=0.35 * inch,
        rightMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
        title=clean_text(cover_title) or "Time Travel Cable Guide",
    )

    frame_h = max(1.0, doc.height - 12.0)
    story: List[Flowable] = []
    tmp_files: List[Path] = []
    runtime_api_cache = _load_api_cache(api_cache_file) if api_cache_enabled else _ensure_api_cache_struct({})

    full_ads = list_image_files(ads_dir)
    bottom_ads = list_image_files(bottom_ads_dir)
    cover_folder_art = list_image_files(cover_art_dir)

    cover_art: Optional[Path] = None
    cover_airing_label = ""
    cover_event: Optional[Event] = None
    if cover_enabled:
        _status(f"Selecting cover art (source={cover_art_source})")
        source = cover_art_source.lower().strip()
        if source in ("folder", "auto"):
            cover_art = choose_random(cover_folder_art)
            if cover_art:
                _status(f"Selected cover from folder: {cover_art.name}")
        if not cover_art and source in ("tvdb", "auto"):
            cover_event = pick_cover_airing_event(schedules, range_start, range_end)
            picked_title = normalize_title_text(cover_event.title) if cover_event else ""
            candidates = [picked_title] if picked_title else sorted(
                {normalize_title_text(e.title) for evs in schedules.values() for e in evs if e.title}
            )
            _status(f"Trying TVDB cover lookup from {len(candidates)} title candidate(s)")
            fetched = fetch_tvdb_cover_art(candidates, api_key=tvdb_api_key, pin=tvdb_pin, api_cache=runtime_api_cache)
            if fetched:
                cover_art = fetched
                tmp_files.append(fetched)
                _status(f"Selected cover from TVDB download: {fetched.name}")
            if cover_airing_label_enabled and cover_event:
                cover_airing_label = _build_airing_label(
                    range_mode=range_mode,
                    title=cover_event.title,
                    when_dt=cover_event.start,
                    single_fmt=cover_airing_label_single_format,
                    day_fmt=cover_airing_label_day_format,
                    week_fmt=cover_airing_label_week_format,
                    month_fmt=cover_airing_label_month_format,
                )
                if cover_airing_label:
                    _status(f"TVDB airing label: {cover_airing_label}")

        if not cover_art:
            _status("No cover art selected; using text-only cover")

        if not double_sided_fold:
            period = clean_text(cover_period_label) or pick_cover_period_label(range_mode, range_start)
            story.append(
                CoverPageFlowable(
                    frame_height=frame_h,
                    title=cover_title,
                    subtitle=cover_subtitle,
                    period_label=period,
                    airing_label=cover_airing_label,
                    art_path=cover_art,
                    bg_color=cover_bg_color,
                    border_size=cover_border_size,
                    text_color=cover_text_color,
                    text_outline_color=cover_text_outline_color,
                    text_outline_width=cover_text_outline_width,
                    title_font=cover_title_font,
                    title_size=cover_title_size,
                    subtitle_font=cover_subtitle_font,
                    subtitle_size=cover_subtitle_size,
                    date_font=cover_date_font,
                    date_size=cover_date_size,
                    date_y=cover_date_y,
                    airing_font=cover_airing_font,
                    airing_size=cover_airing_size,
                    airing_y=cover_airing_y,
                )
            )

    blocks = split_into_blocks(range_start, range_end, page_block_hours)
    desc_cache: Dict[str, str] = {}
    movie_cache: Dict[str, MovieMeta] = {}
    token_holder: Dict[str, str] = {}
    _status(f"Computed {len(blocks)} schedule block(s) for compilation")
    if double_sided_fold:
        logical_pages: List[BookletPageSpec] = []
        if cover_enabled:
            period = clean_text(cover_period_label) or pick_cover_period_label(range_mode, range_start)
            logical_pages.append(
                BookletPageSpec(
                    kind="cover",
                    cover_title=cover_title,
                    cover_subtitle=cover_subtitle,
                    cover_period_label=period,
                    cover_airing_label=cover_airing_label,
                    cover_art_path=cover_art,
                    cover_bg_color=cover_bg_color,
                    cover_border_size=cover_border_size,
                    cover_text_color=cover_text_color,
                    cover_text_outline_color=cover_text_outline_color,
                    cover_text_outline_width=cover_text_outline_width,
                    cover_title_font=cover_title_font,
                    cover_title_size=cover_title_size,
                    cover_subtitle_font=cover_subtitle_font,
                    cover_subtitle_size=cover_subtitle_size,
                    cover_date_font=cover_date_font,
                    cover_date_size=cover_date_size,
                    cover_date_y=cover_date_y,
                    cover_airing_font=cover_airing_font,
                    cover_airing_size=cover_airing_size,
                    cover_airing_y=cover_airing_y,
                )
            )

        for b0, b1 in blocks:
            split_dt = _compute_fold_split(b0, b1, step_minutes)
            bottom_ad_left = choose_random(bottom_ads) if bottom_ads else None
            bottom_ad_right = choose_random(bottom_ads) if bottom_ads else None
            bottom_desc_left: List[OnTonightEntry] = []
            bottom_desc_right: List[OnTonightEntry] = []
            if not bottom_ad_left:
                _status(
                    f"Generating On Tonight fallback for {b0.strftime('%m/%d %H:%M')} - {split_dt.strftime('%H:%M')} (left half)"
                )
                bottom_desc_left = _build_block_descriptions(
                    schedules=schedules,
                    start_dt=b0,
                    end_dt=split_dt,
                    tvdb_api_key=tvdb_api_key,
                    tvdb_pin=tvdb_pin,
                    omdb_api_key=omdb_api_key,
                    desc_cache=desc_cache,
                    movie_cache=movie_cache,
                    token_holder=token_holder,
                    api_cache=runtime_api_cache,
                )
                _status(f"On Tonight entries (left): {len(bottom_desc_left)}")
            if not bottom_ad_right:
                _status(
                    f"Generating On Tonight fallback for {split_dt.strftime('%m/%d %H:%M')} - {b1.strftime('%H:%M')} (right half)"
                )
                bottom_desc_right = _build_block_descriptions(
                    schedules=schedules,
                    start_dt=split_dt,
                    end_dt=b1,
                    tvdb_api_key=tvdb_api_key,
                    tvdb_pin=tvdb_pin,
                    omdb_api_key=omdb_api_key,
                    desc_cache=desc_cache,
                    movie_cache=movie_cache,
                    token_holder=token_holder,
                    api_cache=runtime_api_cache,
                )
                _status(f"On Tonight entries (right): {len(bottom_desc_right)}")
            logical_pages.append(
                BookletPageSpec(
                    kind="guide",
                    start_dt=b0,
                    end_dt=split_dt,
                    header_left="CABLE GUIDE",
                    header_right=f"{b0.strftime('%a %b %d, %Y %H:%M')} - {split_dt.strftime('%H:%M')}",
                    bottom_ad=bottom_ad_left,
                    bottom_descriptions=bottom_desc_left,
                )
            )
            logical_pages.append(
                BookletPageSpec(
                    kind="guide",
                    start_dt=split_dt,
                    end_dt=b1,
                    header_left="CABLE GUIDE",
                    header_right=f"{split_dt.strftime('%a %b %d, %Y %H:%M')} - {b1.strftime('%H:%M')}",
                    bottom_ad=bottom_ad_right,
                    bottom_descriptions=bottom_desc_right,
                )
            )

        if cover_enabled:
            logical_pages.append(BookletPageSpec(kind="blank"))
            _status("Added blank back-cover half-page")

        imposed = impose_booklet_pages(logical_pages)
        _status(f"Booklet imposition produced {len(imposed)} physical side(s)")
        for i, (left_spec, right_spec) in enumerate(imposed):
            if i > 0:
                story.append(PageBreak())
            story.append(
                BookletSpreadFlowable(
                    frame_height=frame_h,
                    left=BookletHalfPageFlowable(
                        frame_height=frame_h,
                        spec=left_spec,
                        channels=channels,
                        channel_numbers=channel_numbers,
                        schedules=schedules,
                        step_minutes=step_minutes,
                        omdb_api_key=omdb_api_key,
                        movie_cache=movie_cache,
                        movie_inline_meta=movie_inline_meta,
                        api_cache=runtime_api_cache,
                    ),
                    right=BookletHalfPageFlowable(
                        frame_height=frame_h,
                        spec=right_spec,
                        channels=channels,
                        channel_numbers=channel_numbers,
                        schedules=schedules,
                        step_minutes=step_minutes,
                        omdb_api_key=omdb_api_key,
                        movie_cache=movie_cache,
                        movie_inline_meta=movie_inline_meta,
                        api_cache=runtime_api_cache,
                    ),
                    safe_gap=fold_safe_gap,
                )
            )

        doc.build(story)
        _status("PDF build complete (booklet mode)")
        if api_cache_enabled:
            _save_api_cache(api_cache_file, runtime_api_cache)
        for p in tmp_files:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        return

    for i, (b0, b1) in enumerate(blocks):
        if story:
            story.append(PageBreak())

        if double_sided_fold:
            split_dt = _compute_fold_split(b0, b1, step_minutes)

            guide_flow: Flowable = FoldedGuideTimelineFlowable(
                channels=channels,
                channel_numbers=channel_numbers,
                schedules=schedules,
                start_dt=b0,
                split_dt=split_dt,
                end_dt=b1,
                step_minutes=step_minutes,
                safe_gap=fold_safe_gap,
                omdb_api_key=omdb_api_key,
                movie_cache=movie_cache,
                movie_inline_meta=movie_inline_meta,
                api_cache=runtime_api_cache,
            )
        else:
            guide_flow = GuideTimelineFlowable(
                channels=channels,
                channel_numbers=channel_numbers,
                schedules=schedules,
                start_dt=b0,
                end_dt=b1,
                step_minutes=step_minutes,
                omdb_api_key=omdb_api_key,
                movie_cache=movie_cache,
                movie_inline_meta=movie_inline_meta,
                api_cache=runtime_api_cache,
            )

        bottom_ad = choose_random(bottom_ads) if bottom_ads else None
        bottom_desc: List[OnTonightEntry] = []
        if not bottom_ad:
            _status(f"Generating On Tonight fallback for {b0.strftime('%m/%d %H:%M')} - {b1.strftime('%H:%M')}")
            bottom_desc = _build_block_descriptions(
                schedules=schedules,
                start_dt=b0,
                end_dt=b1,
                tvdb_api_key=tvdb_api_key,
                tvdb_pin=tvdb_pin,
                omdb_api_key=omdb_api_key,
                desc_cache=desc_cache,
                movie_cache=movie_cache,
                token_holder=token_holder,
                api_cache=runtime_api_cache,
            )
            _status(f"On Tonight entries: {len(bottom_desc)}")
        header_left = "CABLE GUIDE"
        header_right = f"{b0.strftime('%a %b %d, %Y %H:%M')} - {b1.strftime('%H:%M')}"
        story.append(
            CompilationGuidePageFlowable(
                frame_height=frame_h,
                left_header=header_left,
                right_header=header_right,
                guide=guide_flow,
                bottom_ad=bottom_ad,
                bottom_descriptions=bottom_desc,
            )
        )

        if ad_insert_every > 0 and full_ads and (i + 1) % ad_insert_every == 0 and (i + 1) < len(blocks):
            story.append(PageBreak())
            story.append(FullPageImageFlowable(choose_random(full_ads), frame_h))

    doc.build(story)
    _status("PDF build complete")
    if api_cache_enabled:
        _save_api_cache(api_cache_file, runtime_api_cache)

    for p in tmp_files:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def parse_hhmm(s: str) -> time:
    return datetime.strptime(s, "%H:%M").time()
