from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure local module import works when tests run from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pg = pytest.importorskip("print_guide")
core = pytest.importorskip("guide_core")


PAYLOAD_DIR = Path(__file__).resolve().parent / "test_payloads"


def _payload_for_channel(channel: str) -> str:
    name_map = {
        "NBC": "u_NBC.txt",
        "PBS": "u_PBS.txt",
        "MTV": "u_MTV.txt",
        "USA": "u_USA.txt",
        "Joss": "u_Joss.txt",
        "Boomerang": "u_Boomerang.txt",
        "TV Land": "u_TV_Land.txt",
    }
    return (PAYLOAD_DIR / name_map[channel]).read_text(encoding="utf-8")


@pytest.fixture
def fake_fs42(monkeypatch):
    extents = (PAYLOAD_DIR / "e.txt").read_text(encoding="utf-8")

    def _fake_run_fs42(_fs42_dir: Path, args: list[str]) -> str:
        if args == ["-e"]:
            return extents
        if len(args) == 2 and args[0] == "-u":
            ch = args[1]
            return _payload_for_channel(ch)
        raise AssertionError(f"Unexpected args: {args}")

    monkeypatch.setattr(core, "run_fs42", _fake_run_fs42)


def test_discover_channels_filters_none_extents(fake_fs42):
    channels = pg.discover_channels_from_extents(Path("/unused"))

    assert "WeatherTV" not in channels
    assert "NBC" in channels
    assert "PBS" in channels
    assert channels[0] == "NBC"


def test_infer_year_from_extents(fake_fs42):
    year = pg.infer_year_from_extents(Path("/unused"), "NBC")
    assert year == 2026


def test_parse_channel_schedule_handles_missing_filename(fake_fs42):
    events = pg.parse_channel_schedule(Path("/unused"), "MTV", 2026)

    assert len(events) > 0
    assert events[0].title == "Rock"
    assert events[0].filename == ""


def test_parse_channel_schedule_episode_line(fake_fs42):
    events = pg.parse_channel_schedule(Path("/unused"), "NBC", 2026)

    first = events[0]
    assert first.start.year == 2026
    assert first.title == "Er"
    assert "S02E20" in first.filename


def test_display_title_episode_prefers_event_title():
    shown = pg.display_title(
        "Er",
        "ER - S02E20 - Fevers of Unknown Origin WEBRip-1080p.mkv",
    )
    assert shown == "Er"


def test_display_title_movie_style_uses_filename():
    shown = pg.display_title("Some Block", "The.Matrix.1999.1080p.BluRay.mkv")
    assert shown == "The Matrix"


def test_normalize_title_removes_bracketed_tokens():
    assert pg.normalize_title_text("[Foo] Movie") == "Movie"
    assert pg.normalize_title_text("Movie [1080p]") == "Movie"


def test_fit_show_title_applies_the_and_reduction():
    text = "The Chronicles of Narnia - the Lion the Witch and the Wardrobe"
    out = pg.fit_show_title(text, "Helvetica-Bold", 6.5, 120)
    assert out.startswith("Chronicles of Narnia")
    assert "- Lion, Witch" in out


def test_compute_range_bounds_week_and_month():
    s, e = pg.compute_range_bounds("week", pg.parse_date("2026-03-10"), pg.parse_hhmm("05:00"), 8.0)
    assert (e - s).days == 7

    ms, me = pg.compute_range_bounds("month", pg.parse_date("2026-02-16"), pg.parse_hhmm("00:00"), 8.0)
    assert ms.strftime("%Y-%m-%d") == "2026-02-01"
    assert me.strftime("%Y-%m-%d") == "2026-03-01"


def test_split_into_blocks_count():
    start = pg.datetime(2026, 2, 24, 0, 0)
    end = pg.datetime(2026, 2, 25, 0, 0)
    blocks = pg.split_into_blocks(start, end, 6.0)
    assert len(blocks) == 4
    assert blocks[0] == (start, pg.datetime(2026, 2, 24, 6, 0))


def test_resolve_channel_numbers_uses_confs_and_cli_override():
    confs = Path(__file__).resolve().parent / "fixtures" / "confs"
    merged = pg.resolve_channel_numbers(confs, "NBC=99,New Channel=77")

    # Loaded from fixture confs (with CLI override for NBC).
    assert merged["NBC"] == "99"
    assert merged["PBS"] == "4"
    assert merged["New Channel"] == "77"
