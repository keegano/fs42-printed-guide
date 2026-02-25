from __future__ import annotations

import argparse
import json
import os
import tomllib
from pathlib import Path
from typing import Any, Dict, Optional

from guide_core import parse_date, parse_hhmm

DEFAULT_FS42_DIR = Path(__file__).resolve().parent.parent / "FieldStation42"

DEFAULTS: Dict[str, Any] = {
    "start": parse_hhmm("18:00"),
    "hours": 8.0,
    "range_mode": "single",
    "page_block_hours": 6.0,
    "fold_safe_gap": 0.1,
    "step": 30,
    "numbers": "",
    "confs_dir": Path("confs"),
    "year": 0,
    "fs42_dir": DEFAULT_FS42_DIR,
    "out": Path("tv_guide.pdf"),
    "title": "",
    "double_sided_fold": False,
    "ads_dir": None,
    "ad_insert_every": 0,
    "bottom_ads_dir": None,
    "cover_page": False,
    "cover_title": "Time Travel Cable Guide",
    "cover_subtitle": "",
    "cover_period_label": "",
    "cover_bg_color": "FFFFFF",
    "cover_border_size": 0.0,
    "cover_text_color": "FFFFFF",
    "cover_text_outline_color": "000000",
    "cover_text_outline_width": 1.2,
    "cover_title_font": "Helvetica-Bold",
    "cover_title_size": 28.0,
    "cover_subtitle_font": "Helvetica",
    "cover_subtitle_size": 13.0,
    "cover_date_font": "Helvetica-Bold",
    "cover_date_size": 18.0,
    "cover_date_y": 0.18,
    "cover_airing_font": "Helvetica-Bold",
    "cover_airing_size": 14.0,
    "cover_airing_y": 0.11,
    "cover_airing_label_enabled": True,
    "cover_airing_label_single_format": "{time}",
    "cover_airing_label_day_format": "Catch {show} at {time}!",
    "cover_airing_label_week_format": "Catch {show} on {weekday} at {time}!",
    "cover_airing_label_month_format": "Catch {show} on {md} at {time}!",
    "cover_art_source": "none",
    "cover_art_dir": None,
    "tvdb_api_key": "",
    "tvdb_pin": "",
    "load_catalog": None,
    "dump_catalog": None,
    "status_messages": True,
}

REQUIRED_KEYS = ("date",)


def _parse_env_file(path: Path) -> Dict[str, str]:
    if not path.exists() or not path.is_file():
        return {}
    out: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip("\"").strip("'")
        if key:
            out[key] = val
    return out


def load_env_values() -> Dict[str, str]:
    # Support running from project root or other cwd.
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]
    merged: Dict[str, str] = {}
    for p in candidate_paths:
        merged.update(_parse_env_file(p))

    # Process env vars override file values.
    merged.update({k: v for k, v in os.environ.items() if k in ("TVDB_API_KEY", "TVDB_PIN")})
    return merged


def _normalize_config_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        key = k.replace("-", "_")
        if key == "station42_dir":
            key = "fs42_dir"
        out[key] = v
    return out


def load_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    elif path.suffix.lower() in (".toml", ".tml"):
        data = tomllib.loads(raw)
    else:
        raise ValueError(f"Unsupported config format: {path}. Use .json or .toml")
    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON/TOML object")
    return _normalize_config_keys(data)


def _coerce_config_values(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)

    path_keys = ("fs42_dir", "out", "ads_dir", "bottom_ads_dir", "cover_art_dir", "confs_dir", "load_catalog", "dump_catalog")
    for k in path_keys:
        if k in out and out[k] is not None and not isinstance(out[k], Path):
            out[k] = Path(str(out[k]))

    if "date" in out and isinstance(out["date"], str):
        out["date"] = parse_date(out["date"])
    if "start" in out and isinstance(out["start"], str):
        out["start"] = parse_hhmm(out["start"])

    int_keys = ("step", "year", "ad_insert_every")
    for k in int_keys:
        if k in out and out[k] is not None and not isinstance(out[k], int):
            out[k] = int(out[k])

    float_keys = (
        "hours",
        "page_block_hours",
        "fold_safe_gap",
        "cover_border_size",
        "cover_title_size",
        "cover_subtitle_size",
        "cover_date_size",
        "cover_text_outline_width",
        "cover_airing_size",
        "cover_date_y",
        "cover_airing_y",
    )
    for k in float_keys:
        if k in out and out[k] is not None and not isinstance(out[k], float):
            out[k] = float(out[k])

    bool_keys = ("double_sided_fold", "cover_page", "status_messages", "cover_airing_label_enabled")
    for k in bool_keys:
        if k in out and not isinstance(out[k], bool):
            if isinstance(out[k], str):
                out[k] = out[k].strip().lower() in ("1", "true", "yes", "on")
            else:
                out[k] = bool(out[k])

    return out


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    p.add_argument("--config", type=Path, help="Path to JSON/TOML config file.")
    p.add_argument("--station42-dir", dest="fs42_dir", type=Path, help="Path to FieldStation42 repo (where station_42.py lives).")
    p.add_argument("--fs42-dir", type=Path, help="Path to FieldStation42 repo (where station_42.py lives).")
    p.add_argument("--date", type=parse_date, help="Guide date: YYYY-MM-DD")
    p.add_argument("--start", type=parse_hhmm, help="Start time HH:MM")
    p.add_argument("--hours", type=float, help="How many hours to include from start time (single mode).")
    p.add_argument("--range-mode", choices=["single", "day", "week", "month"], help="single=existing behavior; day/week/month=create multi-page compilation.")
    p.add_argument("--page-block-hours", type=float, help="Hours shown per guide page in day/week/month compilation.")
    p.add_argument("--step", type=int, help="Minutes per header tick")
    p.add_argument("--numbers", type=str, help='Comma-separated mapping like "NBC=3,PBS=4"')
    p.add_argument("--confs-dir", type=Path, help="Directory containing channel conf JSON files with station_conf.network_name/channel_number.")
    p.add_argument("--year", type=int, help="Override schedule year (0 = infer from -e using first channel).")
    p.add_argument("--out", type=Path, help="Output PDF path")
    p.add_argument("--title", type=str, help="Optional title override.")
    p.add_argument("--double-sided-fold", action="store_true", help="Render guide as two side-by-side time-range halves on one page for fold printing.")
    p.add_argument("--fold-safe-gap", type=float, help="Extra center gutter in inches for folded output to tolerate off-center folds.")
    p.add_argument("--ads-dir", type=Path, help="Folder of full-page ad images to intersperse between guide pages.")
    p.add_argument("--ad-insert-every", type=int, help="Insert a full-page ad after every N guide pages (0 disables).")
    p.add_argument("--bottom-ads-dir", type=Path, help="Folder of ad images to place below the guide when there is remaining page space.")
    p.add_argument("--cover-page", action="store_true", help="Include a cover page before the guide pages.")
    p.add_argument("--cover-title", type=str, help='Cover title, e.g. "Time Travel Cable Guide".')
    p.add_argument("--cover-subtitle", type=str, help='Cover subtitle, e.g. "March 2026".')
    p.add_argument("--cover-period-label", type=str, help="Optional explicit period label on cover. If omitted, inferred from range mode/date.")
    p.add_argument("--cover-bg-color", type=str, help="Cover background/border color hex, e.g. FFFFFF or 101820.")
    p.add_argument("--cover-border-size", type=float, help="Border inset in inches around cover art.")
    p.add_argument("--cover-text-color", type=str, help="Cover text color hex.")
    p.add_argument("--cover-text-outline-color", type=str, help="Cover text outline color hex.")
    p.add_argument("--cover-text-outline-width", type=float, help="Cover text outline width in points.")
    p.add_argument("--cover-title-font", type=str, help="Cover title font name.")
    p.add_argument("--cover-title-size", type=float, help="Cover title font size (pt).")
    p.add_argument("--cover-subtitle-font", type=str, help="Cover subtitle font name.")
    p.add_argument("--cover-subtitle-size", type=float, help="Cover subtitle font size (pt).")
    p.add_argument("--cover-date-font", type=str, help="Cover period/date font name.")
    p.add_argument("--cover-date-size", type=float, help="Cover period/date font size (pt).")
    p.add_argument("--cover-date-y", type=float, help="Cover period/date vertical position as fraction of cover height (0-1).")
    p.add_argument("--cover-airing-font", type=str, help="Cover airing-label font name.")
    p.add_argument("--cover-airing-size", type=float, help="Cover airing-label font size (pt).")
    p.add_argument("--cover-airing-y", type=float, help="Cover airing-label vertical position as fraction of cover height (0-1).")
    p.add_argument("--cover-airing-label-enabled", action="store_true", help="Enable airing label text on TVDB cover selection.")
    p.add_argument("--no-cover-airing-label", dest="cover_airing_label_enabled", action="store_false", help="Disable airing label text on TVDB cover selection.")
    p.add_argument("--cover-airing-label-single-format", type=str, help="Template for single range airing label.")
    p.add_argument("--cover-airing-label-day-format", type=str, help="Template for day range airing label.")
    p.add_argument("--cover-airing-label-week-format", type=str, help="Template for week range airing label.")
    p.add_argument("--cover-airing-label-month-format", type=str, help="Template for month range airing label.")
    p.add_argument("--cover-art-source", choices=["none", "folder", "tvdb", "auto"], help="Cover art source.")
    p.add_argument("--cover-art-dir", type=Path, help="Folder for cover art images (used by cover-art-source folder/auto).")
    p.add_argument("--tvdb-api-key", type=str, help="TVDB API key for cover-art-source tvdb/auto.")
    p.add_argument("--tvdb-pin", type=str, help="TVDB PIN (if required by your TVDB account/app).")
    p.add_argument("--load-catalog", type=Path, help="Load pre-scanned channel schedules/catalog JSON instead of querying station_42.py.")
    p.add_argument("--dump-catalog", type=Path, help="Write scanned channel schedules/catalog JSON for later reuse.")
    p.add_argument("--status-messages", action="store_true", help="Print progress/status messages while scanning and rendering (default on).")
    p.add_argument("--no-status-messages", dest="status_messages", action="store_false", help="Disable progress/status messages.")
    return p


def parse_effective_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    bootstrap_ns, _ = bootstrap.parse_known_args(argv)

    cfg = _coerce_config_values(load_config_file(bootstrap_ns.config))

    parser = _build_cli_parser()
    cli_ns = parser.parse_args(argv)
    cli_values = vars(cli_ns)
    cli_values.pop("config", None)
    env_values = load_env_values()

    merged: Dict[str, Any] = dict(DEFAULTS)
    if env_values.get("TVDB_API_KEY"):
        merged["tvdb_api_key"] = env_values["TVDB_API_KEY"]
    if env_values.get("TVDB_PIN"):
        merged["tvdb_pin"] = env_values["TVDB_PIN"]
    merged.update(cfg)
    merged.update(cli_values)

    missing = [k for k in REQUIRED_KEYS if merged.get(k) is None]
    if missing:
        msg = ", ".join(missing)
        raise SystemExit(f"Missing required options (CLI or config): {msg}")

    return argparse.Namespace(**merged)
