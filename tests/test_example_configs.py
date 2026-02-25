from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cfg = pytest.importorskip("guide_config")


def test_example_single_evening_json_parses():
    path = ROOT / "examples" / "single_evening.json"
    args = cfg.parse_effective_args(["--config", str(path)])

    assert args.range_mode == "single"
    assert args.double_sided_fold is True
    assert args.fold_safe_gap == 0.15
    assert args.out == Path("out/single_evening.pdf")


def test_example_month_booklet_toml_parses():
    path = ROOT / "examples" / "month_booklet.toml"
    args = cfg.parse_effective_args(["--config", str(path)])

    assert args.range_mode == "month"
    assert args.cover_page is True
    assert args.fold_safe_gap == 0.2
    assert args.ad_insert_every == 4


def test_example_cached_iterate_toml_parses():
    path = ROOT / "examples" / "cached_iterate.toml"
    args = cfg.parse_effective_args(["--config", str(path)])

    assert args.range_mode == "week"
    assert args.load_catalog == Path("cache/march_week_catalog.json")
    assert args.dump_catalog == Path("cache/march_week_catalog.json")
    assert args.cover_art_source == "auto"
