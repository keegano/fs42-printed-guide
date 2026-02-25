from __future__ import annotations

import json
import sys
import base64
from datetime import datetime
from pathlib import Path

import pytest

# Ensure local module import works when tests run from repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pg = pytest.importorskip("print_guide")
core = pytest.importorskip("guide_core")


PAYLOAD_DIR = Path(__file__).resolve().parent / "test_payloads"
REAL_CATALOG_DUMP = PAYLOAD_DIR / "catalog_dump.json"


def _write_tiny_png(path: Path) -> None:
    tiny_png = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xx2kAAAAASUVORK5CYII="
    )
    path.write_bytes(base64.b64decode(tiny_png))


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


def test_catalog_dump_and_load_round_trip(tmp_path: Path):
    channels = ["NBC"]
    schedules = {
        "NBC": [
            core.Event(
                start=datetime(2026, 3, 1, 9, 0),
                end=datetime(2026, 3, 1, 9, 30),
                title="Morning Show",
                filename="Morning.Show.S01E01.mkv",
            )
        ]
    }
    out = tmp_path / "catalog.json"
    pg.dump_catalog_file(out, Path("/tmp/fs42"), channels, 2026, schedules)
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["year"] == 2026
    assert raw["channels"] == ["NBC"]

    loaded_channels, loaded_year, loaded_schedules = pg.load_catalog_file(out)
    assert loaded_channels == ["NBC"]
    assert loaded_year == 2026
    assert loaded_schedules["NBC"][0].title == "Morning Show"


def test_main_uses_loaded_catalog_without_scanning(monkeypatch, tmp_path: Path):
    catalog = tmp_path / "catalog.json"
    pg.dump_catalog_file(
        catalog,
        Path("/tmp/fs42"),
        ["NBC"],
        2026,
        {
            "NBC": [
                core.Event(
                    start=datetime(2026, 3, 1, 9, 0),
                    end=datetime(2026, 3, 1, 10, 0),
                    title="Cached Show",
                    filename="",
                )
            ]
        },
    )

    called = {"discover": 0, "make_pdf": 0}

    def fail_discover(_dir):
        called["discover"] += 1
        raise AssertionError("discover should not be called when --load-catalog is used")

    def fake_make_pdf(**kwargs):
        called["make_pdf"] += 1
        assert kwargs["channels"] == ["NBC"]
        assert kwargs["schedules"]["NBC"][0].title == "Cached Show"

    monkeypatch.setattr(pg, "discover_channels_from_extents", fail_discover)
    monkeypatch.setattr(pg, "make_pdf", fake_make_pdf)

    pg.main(
        [
            "--load-catalog",
            str(catalog),
            "--date",
            "2026-03-01",
            "--range-mode",
            "single",
            "--out",
            str(tmp_path / "out.pdf"),
        ]
    )

    assert called["discover"] == 0
    assert called["make_pdf"] == 1


def test_load_real_catalog_dump():
    channels, year, schedules = pg.load_catalog_file(REAL_CATALOG_DUMP)
    assert year == 2026
    assert len(channels) >= 20
    assert "NBC" in channels
    assert len(schedules["NBC"]) > 10
    assert schedules["NBC"][0].start < schedules["NBC"][0].end


def test_real_catalog_compilation_with_ads(tmp_path: Path, monkeypatch):
    channels, _, schedules = pg.load_catalog_file(REAL_CATALOG_DUMP)
    monkeypatch.setattr(core, "choose_random", lambda items: items[0] if items else None)

    ads_dir = tmp_path / "ads"
    bottom_ads = tmp_path / "bottom_ads"
    cover_dir = tmp_path / "cover"
    ads_dir.mkdir()
    bottom_ads.mkdir()
    cover_dir.mkdir()
    _write_tiny_png(ads_dir / "ad.png")
    _write_tiny_png(bottom_ads / "bottom.png")
    _write_tiny_png(cover_dir / "cover.png")

    out = tmp_path / "real_catalog_compilation.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers={},
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 2, 24, 0, 0),
        range_end=datetime(2026, 2, 25, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        ads_dir=ads_dir,
        ad_insert_every=1,
        bottom_ads_dir=bottom_ads,
        cover_enabled=True,
        cover_title="Time Travel Cable Guide",
        cover_subtitle="Real Dump",
        cover_art_source="folder",
        cover_art_dir=cover_dir,
        cover_bg_color="102030",
        cover_border_size=0.12,
        cover_title_font="Helvetica-Bold",
        cover_title_size=24,
        cover_subtitle_font="Helvetica",
        cover_subtitle_size=11,
        cover_date_font="Helvetica-Bold",
        cover_date_size=15,
    )

    reader = pytest.importorskip("pypdf").PdfReader(str(out))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    # day 24h + 12h blocks = 2 guide pages, +1 cover, +1 interstitial ad after page 1 = 4 pages.
    assert len(reader.pages) == 4
    assert "Time Travel Cable Guide" in text
    assert "CABLE GUIDE" in text


def test_tvdb_cover_adds_airing_label_from_catalog(tmp_path: Path, monkeypatch):
    channels, _, schedules = pg.load_catalog_file(REAL_CATALOG_DUMP)
    cover = tmp_path / "cover.png"
    _write_tiny_png(cover)

    monkeypatch.setattr(core, "fetch_tvdb_cover_art", lambda *_a, **_k: cover)
    monkeypatch.setattr(core.random, "choice", lambda items: items[0])

    out = tmp_path / "tvdb_cover_label.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers={},
        schedules=schedules,
        range_mode="week",
        range_start=datetime(2026, 2, 24, 0, 0),
        range_end=datetime(2026, 3, 3, 0, 0),
        page_block_hours=24,
        step_minutes=30,
        cover_enabled=True,
        cover_title="Time Travel Cable Guide",
        cover_subtitle="Week",
        cover_art_source="tvdb",
        cover_airing_label_enabled=True,
        cover_airing_label_week_format="{title} playing {weekday} at {time}",
    )

    text = "\n".join((p.extract_text() or "") for p in pytest.importorskip("pypdf").PdfReader(str(out)).pages)
    assert "playing" in text


def test_bottom_descriptions_used_when_no_bottom_ads(tmp_path: Path, monkeypatch):
    channels, _, schedules = pg.load_catalog_file(REAL_CATALOG_DUMP)
    monkeypatch.setattr(core, "_fetch_tvdb_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(core, "_fetch_tvdb_overview_by_title", lambda title, token: f"{title} overview from TVDB")
    monkeypatch.setattr(core.random, "shuffle", lambda items: None)

    out = tmp_path / "desc_fallback.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers={},
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 2, 24, 0, 0),
        range_end=datetime(2026, 2, 25, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        cover_enabled=False,
        bottom_ads_dir=None,
        tvdb_api_key="fake-key",
    )

    text = "\n".join((p.extract_text() or "") for p in pytest.importorskip("pypdf").PdfReader(str(out)).pages)
    assert "ON TONIGHT" in text
    assert "overview from TVDB" in text


def test_block_descriptions_skip_missing_instead_of_placeholder(monkeypatch):
    _, _, schedules = pg.load_catalog_file(REAL_CATALOG_DUMP)
    monkeypatch.setattr(core, "_fetch_tvdb_token", lambda *_a, **_k: "tok")
    monkeypatch.setattr(core, "_fetch_tvdb_overview_by_title", lambda *_a, **_k: "")
    monkeypatch.setattr(core, "_fetch_omdb_movie_meta", lambda *_a, **_k: None)

    entries = core._build_block_descriptions(
        schedules=schedules,
        start_dt=datetime(2026, 2, 24, 0, 0),
        end_dt=datetime(2026, 2, 24, 6, 0),
        tvdb_api_key="k",
        tvdb_pin="",
        omdb_api_key="k",
        desc_cache={},
        movie_cache={},
        token_holder={},
        max_items=5,
    )
    assert isinstance(entries, list)
    assert all(isinstance(e, core.OnTonightEntry) for e in entries)
    assert all("See schedule listing" not in e.description for e in entries)
