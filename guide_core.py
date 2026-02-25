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
from reportlab.lib.utils import ImageReader


LINE_RE = re.compile(
    r"^(?P<mm>\d{2})/(?P<dd>\d{2}) (?P<start>\d{2}:\d{2}) - (?P<end>\d{2}:\d{2}) - (?P<title>.*?)(?: - (?P<filename>.*))?$"
)

EXTENT_RE = re.compile(
    r"^(?P<chan>.+?) schedule extents:\s*(?P<start>None|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?)\s*to\s*(?P<end>None|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(\.\d+)?)\s*$"
)


YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
EP_RE = re.compile(r"\bS\d{2}E\d{2}\b", re.IGNORECASE)


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

@dataclass(frozen=True)
class Event:
    start: datetime
    end: datetime
    title: str
    filename: str


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
) -> Optional[Path]:
    if not show_names or not api_key:
        return None
    try:
        login_payload = {"apikey": api_key}
        if pin:
            login_payload["pin"] = pin
        auth = _http_json("https://api4.thetvdb.com/v4/login", method="POST", payload=login_payload)
        token = auth.get("data", {}).get("token") or ""
        if not token:
            return None

        show = random.choice(show_names)
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
    ) -> None:
        super().__init__()
        self.channels = channels
        self.channel_numbers = channel_numbers
        self.schedules = schedules
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.step_minutes = step_minutes
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
        cur = self.start_dt
        while cur <= self.end_dt:
            x = self._x_for(cur, timeline_w, total_seconds)
            c.line(x, height - self.header_h, x, height - self.header_h + 3)
            c.drawCentredString(x, height - self.header_h + 6, time_label(cur))
            cur += timedelta(minutes=self.step_minutes)

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
                shown = fit_show_title(shown, "Helvetica-Bold", 6.5, text_w)
                if not shown:
                    continue
                c.setFont("Helvetica-Bold", 6.5)
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
    ) -> None:
        super().__init__()
        self.left = GuideTimelineFlowable(
            channels=channels,
            channel_numbers=channel_numbers,
            schedules=schedules,
            start_dt=start_dt,
            end_dt=split_dt,
            step_minutes=step_minutes,
        )
        self.right = GuideTimelineFlowable(
            channels=channels,
            channel_numbers=channel_numbers,
            schedules=schedules,
            start_dt=split_dt,
            end_dt=end_dt,
            step_minutes=step_minutes,
        )
        self.gap = 0.12 * inch

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
        art_path: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.title = clean_text(title)
        self.subtitle = clean_text(subtitle)
        self.period_label = clean_text(period_label)
        self.art_path = art_path

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        if self.art_path:
            draw_image_fit(c, self.art_path, 0, 0, self.width, self.height)
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, self.height * 0.60, self.width, self.height * 0.40, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)
        else:
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)

        c.setFont("Helvetica-Bold", 28)
        c.drawCentredString(self.width / 2.0, self.height * 0.86, self.title)
        if self.subtitle:
            c.setFont("Helvetica", 13)
            c.drawCentredString(self.width / 2.0, self.height * 0.80, self.subtitle)
        if self.period_label:
            c.setFont("Helvetica-Bold", 18)
            c.drawCentredString(self.width / 2.0, self.height * 0.72, self.period_label)


class GuideWithBottomAdFlowable(Flowable):
    def __init__(self, guide: Flowable, ad_path: Optional[Path], frame_height: float) -> None:
        super().__init__()
        self.guide = guide
        self.ad_path = ad_path
        self.frame_height = frame_height
        self.gap = 0.08 * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        gw, gh = self.guide.wrap(self.width, self.frame_height)
        guide_y = max(0.0, self.frame_height - gh)
        self.guide.drawOn(c, 0, guide_y)

        if not self.ad_path:
            return
        available = guide_y - self.gap
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
    ) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.left_header = clean_text(left_header)
        self.right_header = clean_text(right_header)
        self.guide = guide
        self.bottom_ad = bottom_ad
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
    cover_title: str = ""
    cover_subtitle: str = ""
    cover_period_label: str = ""
    cover_art_path: Optional[Path] = None


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
    ) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.spec = spec
        self.channels = channels
        self.channel_numbers = channel_numbers
        self.schedules = schedules
        self.step_minutes = step_minutes
        self.header_h = 0.14 * inch
        self.gap = 0.06 * inch

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def _draw_cover(self, c) -> None:
        if self.spec.cover_art_path:
            draw_image_fit(c, self.spec.cover_art_path, 0, 0, self.width, self.height)
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, self.height * 0.55, self.width, self.height * 0.45, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)
        else:
            c.setFillColorRGB(1, 1, 1)
            c.rect(0, 0, self.width, self.height, stroke=0, fill=1)
            c.setFillColorRGB(0, 0, 0)

        c.setFont("Helvetica-Bold", 16)
        c.drawCentredString(self.width / 2.0, self.height * 0.86, clean_text(self.spec.cover_title))
        if self.spec.cover_subtitle:
            c.setFont("Helvetica", 9)
            c.drawCentredString(self.width / 2.0, self.height * 0.79, clean_text(self.spec.cover_subtitle))
        if self.spec.cover_period_label:
            c.setFont("Helvetica-Bold", 12)
            c.drawCentredString(self.width / 2.0, self.height * 0.70, clean_text(self.spec.cover_period_label))

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
        )
        content: Flowable = guide
        if self.spec.bottom_ad:
            content = GuideWithBottomAdFlowable(guide=guide, ad_path=self.spec.bottom_ad, frame_height=content_h)

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
    def __init__(self, frame_height: float, left: Flowable, right: Flowable) -> None:
        super().__init__()
        self.frame_height = frame_height
        self.left = left
        self.right = right

    def wrap(self, availWidth: float, availHeight: float) -> Tuple[float, float]:
        self.width = availWidth
        self.height = self.frame_height
        return self.width, self.height

    def draw(self) -> None:
        c = self.canv
        panel_w = self.width / 2.0

        self.left.wrap(panel_w, self.height)
        self.left.drawOn(c, 0, 0)
        self.right.wrap(panel_w, self.height)
        self.right.drawOn(c, panel_w, 0)


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
    cover_art_source: str = "none",
    cover_art_dir: Optional[Path] = None,
    tvdb_api_key: str = "",
    tvdb_pin: str = "",
) -> None:
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

    full_ads = list_image_files(ads_dir)
    bottom_ads = list_image_files(bottom_ads_dir)
    cover_folder_art = list_image_files(cover_art_dir)

    cover_art: Optional[Path] = None
    if cover_enabled:
        source = cover_art_source.lower().strip()
        if source in ("folder", "auto"):
            cover_art = choose_random(cover_folder_art)
        if not cover_art and source in ("tvdb", "auto"):
            candidates = sorted({normalize_title_text(e.title) for evs in schedules.values() for e in evs if e.title})
            fetched = fetch_tvdb_cover_art(candidates, api_key=tvdb_api_key, pin=tvdb_pin)
            if fetched:
                cover_art = fetched
                tmp_files.append(fetched)

        if not double_sided_fold:
            period = clean_text(cover_period_label) or pick_cover_period_label(range_mode, range_start)
            story.append(
                CoverPageFlowable(
                    frame_height=frame_h,
                    title=cover_title,
                    subtitle=cover_subtitle,
                    period_label=period,
                    art_path=cover_art,
                )
            )

    blocks = split_into_blocks(range_start, range_end, page_block_hours)
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
                    cover_art_path=cover_art,
                )
            )

        for b0, b1 in blocks:
            split_dt = _compute_fold_split(b0, b1, step_minutes)
            bottom_ad_left = choose_random(bottom_ads) if bottom_ads else None
            bottom_ad_right = choose_random(bottom_ads) if bottom_ads else None
            logical_pages.append(
                BookletPageSpec(
                    kind="guide",
                    start_dt=b0,
                    end_dt=split_dt,
                    header_left="CABLE GUIDE",
                    header_right=f"{b0.strftime('%a %b %d, %Y %H:%M')} - {split_dt.strftime('%H:%M')}",
                    bottom_ad=bottom_ad_left,
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
                )
            )

        if cover_enabled:
            logical_pages.append(BookletPageSpec(kind="blank"))

        imposed = impose_booklet_pages(logical_pages)
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
                    ),
                    right=BookletHalfPageFlowable(
                        frame_height=frame_h,
                        spec=right_spec,
                        channels=channels,
                        channel_numbers=channel_numbers,
                        schedules=schedules,
                        step_minutes=step_minutes,
                    ),
                )
            )

        doc.build(story)
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
            )
        else:
            guide_flow = GuideTimelineFlowable(
                channels=channels,
                channel_numbers=channel_numbers,
                schedules=schedules,
                start_dt=b0,
                end_dt=b1,
                step_minutes=step_minutes,
            )

        bottom_ad = choose_random(bottom_ads) if bottom_ads else None
        header_left = "CABLE GUIDE"
        header_right = f"{b0.strftime('%a %b %d, %Y %H:%M')} - {b1.strftime('%H:%M')}"
        story.append(
            CompilationGuidePageFlowable(
                frame_height=frame_h,
                left_header=header_left,
                right_header=header_right,
                guide=guide_flow,
                bottom_ad=bottom_ad,
            )
        )

        if ad_insert_every > 0 and full_ads and (i + 1) % ad_insert_every == 0 and (i + 1) < len(blocks):
            story.append(PageBreak())
            story.append(FullPageImageFlowable(choose_random(full_ads), frame_h))

    doc.build(story)

    for p in tmp_files:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def parse_hhmm(s: str) -> time:
    return datetime.strptime(s, "%H:%M").time()
