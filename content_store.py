from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalize_key(value: object) -> str:
    return clean_text(value).lower()


@dataclass(frozen=True)
class CoverSpec:
    id: str
    title: str
    subtitle: str
    period_label: str
    airing_label: str
    image: Optional[Path]
    enabled: bool
    range_modes: List[str]


@dataclass(frozen=True)
class PromoSpec:
    id: str
    title: str
    message: str
    image: Optional[Path]
    enabled: bool
    match_titles: List[str]
    match_channels: List[str]
    range_modes: List[str]


@dataclass(frozen=True)
class NfoMeta:
    title: str
    plot: str
    rated: str
    imdb_rating: str


@dataclass
class NfoIndex:
    by_filename_stem: Dict[str, NfoMeta]
    by_title: Dict[str, NfoMeta]

    def lookup(self, title: str, filename: str) -> Optional[NfoMeta]:
        stem = normalize_key(Path(clean_text(filename)).stem)
        if stem and stem in self.by_filename_stem:
            return self.by_filename_stem[stem]
        key = normalize_key(title)
        if key and key in self.by_title:
            return self.by_title[key]
        return None


def _read_json_files(folder: Path) -> List[dict]:
    if not folder.exists() or not folder.is_dir():
        return []
    out: List[dict] = []
    for p in sorted(folder.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            out.append({"_path": p, **data})
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    out.append({"_path": p, **item})
    return out


def load_cover_specs(content_dir: Path) -> List[CoverSpec]:
    rows = _read_json_files(content_dir / "covers")
    specs: List[CoverSpec] = []
    for row in rows:
        src = row.get("_path")
        image = clean_text(row.get("image"))
        image_path = (src.parent / image).resolve() if image and isinstance(src, Path) else None
        specs.append(
            CoverSpec(
                id=clean_text(row.get("id")) or (src.stem if isinstance(src, Path) else "cover"),
                title=clean_text(row.get("title")),
                subtitle=clean_text(row.get("subtitle")),
                period_label=clean_text(row.get("period_label")),
                airing_label=clean_text(row.get("airing_label")),
                image=image_path if image_path and image_path.exists() else None,
                enabled=bool(row.get("enabled", True)),
                range_modes=[normalize_key(v) for v in row.get("range_modes", []) if clean_text(v)],
            )
        )
    return [s for s in specs if s.enabled]


def load_promo_specs(content_dir: Path) -> List[PromoSpec]:
    rows = _read_json_files(content_dir / "promos")
    specs: List[PromoSpec] = []
    for row in rows:
        src = row.get("_path")
        image = clean_text(row.get("image"))
        image_path = (src.parent / image).resolve() if image and isinstance(src, Path) else None
        specs.append(
            PromoSpec(
                id=clean_text(row.get("id")) or (src.stem if isinstance(src, Path) else "promo"),
                title=clean_text(row.get("title")),
                message=clean_text(row.get("message")),
                image=image_path if image_path and image_path.exists() else None,
                enabled=bool(row.get("enabled", True)),
                match_titles=[normalize_key(v) for v in row.get("match_titles", []) if clean_text(v)],
                match_channels=[normalize_key(v) for v in row.get("match_channels", []) if clean_text(v)],
                range_modes=[normalize_key(v) for v in row.get("range_modes", []) if clean_text(v)],
            )
        )
    return [s for s in specs if s.enabled]


def pick_cover_spec(specs: List[CoverSpec], range_mode: str) -> Optional[CoverSpec]:
    mode = normalize_key(range_mode)
    candidates = [s for s in specs if not s.range_modes or mode in s.range_modes]
    if not candidates:
        return None
    return random.choice(candidates)


def pick_promo_spec(
    specs: List[PromoSpec],
    range_mode: str,
    title_hints: Iterable[str],
    channel_hints: Iterable[str],
) -> Optional[PromoSpec]:
    mode = normalize_key(range_mode)
    title_set = {normalize_key(t) for t in title_hints if clean_text(t)}
    ch_set = {normalize_key(c) for c in channel_hints if clean_text(c)}
    candidates = [s for s in specs if not s.range_modes or mode in s.range_modes]
    if not candidates:
        return None

    scored: List[tuple[int, PromoSpec]] = []
    for spec in candidates:
        score = 0
        if spec.match_titles and title_set.intersection(spec.match_titles):
            score += 4
        if spec.match_channels and ch_set.intersection(spec.match_channels):
            score += 2
        if not spec.match_titles and not spec.match_channels:
            score += 1
        scored.append((score, spec))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score = scored[0][0]
    top = [spec for score, spec in scored if score == best_score]
    return random.choice(top)


def load_nfo_index(fs42_dir: Path) -> NfoIndex:
    root = fs42_dir / "catalog"
    by_stem: Dict[str, NfoMeta] = {}
    by_title: Dict[str, NfoMeta] = {}
    if not root.exists() or not root.is_dir():
        return NfoIndex(by_filename_stem=by_stem, by_title=by_title)

    for p in root.rglob("*.nfo"):
        try:
            xml = ET.fromstring(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        title = clean_text(xml.findtext("title"))
        plot = clean_text(xml.findtext("plot"))
        rated = clean_text(xml.findtext("mpaa")) or clean_text(xml.findtext("rated"))
        imdb_rating = clean_text(xml.findtext("rating"))
        if not title and not plot and not rated and not imdb_rating:
            continue
        meta = NfoMeta(title=title, plot=plot, rated=rated, imdb_rating=imdb_rating)
        stem = normalize_key(p.stem)
        if stem and stem not in by_stem:
            by_stem[stem] = meta
        tkey = normalize_key(title)
        if tkey and tkey not in by_title:
            by_title[tkey] = meta

    return NfoIndex(by_filename_stem=by_stem, by_title=by_title)
