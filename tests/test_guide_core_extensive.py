from __future__ import annotations

import base64
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

core = pytest.importorskip("guide_core")
PdfReader = pytest.importorskip("pypdf").PdfReader


SNAPSHOT_MARKERS = json.loads((Path(__file__).parent / "snapshots" / "pdf_expected_markers.json").read_text(encoding="utf-8"))


def _mk_event(start_h: int, start_m: int, end_h: int, end_m: int, title: str, filename: str = "") -> core.Event:
    d = date(2026, 3, 5)
    st = datetime.combine(d, time(start_h, start_m))
    en = datetime.combine(d, time(end_h, end_m))
    if en <= st:
        en += timedelta(days=1)
    return core.Event(start=st, end=en, title=title, filename=filename)


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _write_tiny_png(path: Path) -> None:
    tiny_png = (
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xx2kAAAAASUVORK5CYII="
    )
    path.write_bytes(base64.b64decode(tiny_png))


def test_text_helpers():
    assert core.pretty_from_filename("Movie.Title_2020.mkv") == "Movie Title 2020"
    assert core.ascii_only("Caf\xe9") == "Caf"
    assert core.clean_text("  A   B  ") == "A B"
    assert core.normalize_title_text("[Foo]  The  Movie") == "The Movie"
    assert core.strip_year_and_after("The Matrix 1999 Remaster") == "The Matrix"


def test_display_title_variants():
    assert core.display_title("Er", "ER - S01E01 - Pilot.mkv") == "Er"
    assert core.display_title("Ignored", "The.Matrix.1999.1080p.mkv") == "The Matrix"
    assert core.display_title("Fallback 2001 Version", "") == "Fallback"


def test_parse_helpers_and_labels():
    assert core.parse_date("2026-03-05") == date(2026, 3, 5)
    assert core.parse_hhmm("09:30") == time(9, 30)
    assert core.time_label(datetime(2026, 3, 5, 9, 0)) == "9a"
    assert core.time_label(datetime(2026, 3, 5, 13, 30)) == "1:30p"


def test_width_fit_helpers():
    t = "The Chronicles of Narnia - the Lion the Witch and the Wardrobe"
    out = core.fit_show_title(t, "Helvetica-Bold", 6.5, 120)
    assert out.startswith("Chronicles of Narnia")
    assert "Lion, Witch" in out
    assert core.hard_truncate_to_width("abc", "Helvetica", 8, 100) == "abc"


def test_clip_events_and_ranges():
    evs = [
        _mk_event(8, 0, 8, 30, "A"),
        _mk_event(8, 30, 9, 0, "B"),
        _mk_event(9, 0, 9, 30, "C"),
    ]
    clipped = core.clip_events(evs, datetime(2026, 3, 5, 8, 15), datetime(2026, 3, 5, 9, 0))
    assert [e.title for e in clipped] == ["A", "B"]

    s, e = core.compute_range_bounds("single", date(2026, 3, 5), time(9, 0), 2.5)
    assert (e - s) == timedelta(hours=2.5)
    ws, we = core.compute_range_bounds("week", date(2026, 3, 5), time(0, 0), 0)
    assert (we - ws).days == 7
    ms, me = core.compute_range_bounds("month", date(2026, 2, 20), time(0, 0), 0)
    assert ms.day == 1 and me.month == 3 and me.day == 1

    blocks = core.split_into_blocks(datetime(2026, 3, 5, 0, 0), datetime(2026, 3, 5, 6, 0), 2)
    assert len(blocks) == 3


def test_cover_period_label_modes():
    s = datetime(2026, 3, 5, 0, 0)
    assert core.pick_cover_period_label("month", s) == "March 2026"
    assert "2026" in core.pick_cover_period_label("week", s)
    assert core.pick_cover_period_label("day", s).endswith("2026")


def test_build_airing_label_formats():
    dt = datetime(2026, 2, 28, 20, 30)
    week = core._build_airing_label(
        "week",
        "Rugrats",
        dt,
        single_fmt="{time}",
        day_fmt="{time}",
        week_fmt="{title} playing {weekday} at {time}",
        month_fmt="{md} at {time}",
    )
    month = core._build_airing_label(
        "month",
        "Rugrats",
        dt,
        single_fmt="{time}",
        day_fmt="{time}",
        week_fmt="{title} playing {weekday} at {time}",
        month_fmt="{md} at {time}",
    )
    day = core._build_airing_label(
        "day",
        "Rugrats",
        datetime(2026, 2, 28, 5, 0),
        single_fmt="{time}",
        day_fmt="{time}",
        week_fmt="{title} playing {weekday} at {time}",
        month_fmt="{md} at {time}",
    )
    assert "Rugrats playing" in week
    assert "Saturday" in week
    assert month == "2/28 at 8:30"
    assert day == "5:00"


def test_wrap_cover_text_for_catch_message_width():
    label = "Catch The Really Long Adventures of Captain Super Cartoon Squad at 8:30!"
    lines = core._wrap_cover_text(label, "Helvetica-Bold", 14, max_width=220, max_lines=2)
    assert 1 <= len(lines) <= 2
    for line in lines:
        assert core.pdfmetrics.stringWidth(line, "Helvetica-Bold", 14) <= 220.5


def test_wrap_cover_text_truncates_when_needed():
    label = "Catch " + ("VeryLongShowName " * 30) + "at 9:00!"
    lines = core._wrap_cover_text(label, "Helvetica-Bold", 14, max_width=180, max_lines=2)
    assert len(lines) == 2
    assert lines[-1].endswith("...")


def test_list_image_files_and_choose_random(tmp_path: Path, monkeypatch):
    (tmp_path / "a.jpg").write_text("x")
    (tmp_path / "b.png").write_text("x")
    (tmp_path / "c.txt").write_text("x")

    files = core.list_image_files(tmp_path)
    assert [p.name for p in files] == ["a.jpg", "b.png"]
    monkeypatch.setattr(core.random, "choice", lambda items: items[-1])
    assert core.choose_random(files).name == "b.png"
    assert core.choose_random([]) is None


def test_load_channel_numbers_from_confs_fixture(tmp_path: Path):
    conf_dir = Path(__file__).parent / "fixtures" / "confs"
    loaded = core.load_channel_numbers_from_confs(conf_dir)
    assert loaded.get("NBC") == "3"
    assert loaded.get("PBS") == "4"

    # Invalid JSON files should be ignored.
    (tmp_path / "broken.json").write_text("{not valid", encoding="utf-8")
    merged = core.load_channel_numbers_from_confs(tmp_path)
    assert merged == {}


def test_run_fs42_executes_local_station_script(tmp_path: Path):
    script = tmp_path / "station_42.py"
    script.write_text(
        """
import sys
print('OK', ' '.join(sys.argv[1:]))
""".strip()
    )
    out = core.run_fs42(tmp_path, ["-e"])
    assert "OK -e" in out


def test_discover_infer_parse_with_stubbed_run(monkeypatch):
    extents = "\n".join(
        [
            "NBC schedule extents: 2026-03-05 00:00:00 to 2026-03-06 00:00:00",
            "WeatherTV schedule extents: None to None",
        ]
    )
    sched = "\n".join(
        [
            "03/05 09:00 - 09:30 - Test Show - Test.Show.S01E01.mkv",
            "03/05 09:30 - 10:00 - Movie Night - The.Matrix.1999.1080p.mkv",
            "03/05 10:00 - 10:30 - Music Block",
        ]
    )

    def fake(_dir: Path, args: list[str]) -> str:
        if args == ["-e"]:
            return extents
        if args == ["-u", "NBC"]:
            return sched
        return ""

    monkeypatch.setattr(core, "run_fs42", fake)
    channels = core.discover_channels_from_extents(Path("/unused"))
    assert channels == ["NBC"]
    assert core.infer_year_from_extents(Path("/unused"), "NBC") == 2026

    evs = core.parse_channel_schedule(Path("/unused"), "NBC", 2026)
    assert len(evs) == 3
    assert evs[-1].filename == ""


def test_http_json_get_and_post(monkeypatch):
    class _Resp:
        def __init__(self, data: bytes):
            self._data = data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return self._data

    def fake_urlopen(req, timeout=15):
        if req.method == "POST":
            return _Resp(b'{"posted": true}')
        return _Resp(b'{"ok": true}')

    monkeypatch.setattr(core.urllib.request, "urlopen", fake_urlopen)
    got = core._http_json("http://example.local/x")
    posted = core._http_json("http://example.local/y", method="POST", payload={"a": 1})
    assert got["ok"] is True
    assert posted["posted"] is True


def test_fetch_tvdb_cover_art_success(monkeypatch, tmp_path: Path):
    calls = {"n": 0}

    def fake_http(url: str, method: str = "GET", payload=None, headers=None):
        if url.endswith("/login"):
            return {"data": {"token": "abc"}}
        if "search" in url:
            return {"data": [{"id": 123}]}
        return {"data": {"image": "http://example.com/poster.jpg"}}

    def fake_retrieve(url: str, path: str):
        calls["n"] += 1
        _write_tiny_png(Path(path))
        return path, None

    monkeypatch.setattr(core, "_http_json", fake_http)
    monkeypatch.setattr(core.urllib.request, "urlretrieve", fake_retrieve)
    monkeypatch.setattr(core.random, "choice", lambda items: items[0])

    p = core.fetch_tvdb_cover_art(["Some Show"], api_key="k")
    assert p is not None and p.exists() and calls["n"] == 1
    p.unlink(missing_ok=True)


def test_fetch_tvdb_cover_art_no_token(monkeypatch):
    monkeypatch.setattr(core, "_http_json", lambda *a, **k: {"data": {}})
    assert core.fetch_tvdb_cover_art(["Some Show"], api_key="k") is None


def test_movie_cover_prefers_omdb_over_tvdb(tmp_path: Path, monkeypatch):
    channels, numbers, schedules = _sample_schedules()
    movie_ev = core.Event(
        start=datetime(2026, 3, 1, 20, 0),
        end=datetime(2026, 3, 1, 22, 0),
        title="Movie Block",
        filename="The.Matrix.1999.1080p.BluRay.mkv",
    )
    schedules["NBC"].append(movie_ev)

    cover_png = tmp_path / "cover.png"
    _write_tiny_png(cover_png)

    monkeypatch.setattr(core, "pick_cover_airing_event", lambda *_a, **_k: movie_ev)
    monkeypatch.setattr(core, "fetch_omdb_cover_art", lambda *a, **k: cover_png)
    monkeypatch.setattr(core, "fetch_tvdb_cover_art", lambda *a, **k: (_ for _ in ()).throw(AssertionError("TVDB should not be used")))

    out = tmp_path / "movie_cover_pref.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 1, 0, 0),
        range_end=datetime(2026, 3, 2, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        cover_enabled=True,
        cover_art_source="tvdb",
        omdb_api_key="ok",
        tvdb_api_key="unused",
    )
    assert out.exists()


def test_api_cache_reuses_tvdb_and_omdb_calls(monkeypatch):
    calls = {"n": 0}

    def fake_http(url: str, method: str = "GET", payload=None, headers=None):
        calls["n"] += 1
        if "thetvdb.com/v4/login" in url:
            return {"data": {"token": "tok"}}
        if "thetvdb.com/v4/search" in url:
            return {"data": [{"id": 123}]}
        if "thetvdb.com/v4/series/123/extended" in url:
            return {"data": {"overview": "Overview text", "image": "http://example.com/poster.jpg"}}
        if "omdbapi.com" in url:
            return {"Response": "True", "Title": "The Matrix", "Year": "1999", "Plot": "Plot text", "imdbRating": "7.6", "Rated": "R"}
        return {}

    monkeypatch.setattr(core, "_http_json", fake_http)
    cache = {}

    t1 = core._fetch_tvdb_token("k", api_cache=cache)
    t2 = core._fetch_tvdb_token("k", api_cache=cache)
    assert t1 == t2 == "tok"

    o1 = core._fetch_tvdb_overview_by_title("Rugrats", "tok", api_cache=cache)
    o2 = core._fetch_tvdb_overview_by_title("Rugrats", "tok", api_cache=cache)
    assert o1 == o2 == "Overview text"

    m1 = core._fetch_omdb_movie_meta("The Matrix", "1999", "ok", api_cache=cache)
    m2 = core._fetch_omdb_movie_meta("The Matrix", "1999", "ok", api_cache=cache)
    assert m1 is not None and m2 is not None
    # login + tvdb search + tvdb detail + omdb = 4 network calls total when cache works
    assert calls["n"] == 4


def test_draw_image_fit(tmp_path: Path):
    png = tmp_path / "tiny.png"
    _write_tiny_png(png)

    from reportlab.pdfgen import canvas

    out = tmp_path / "img.pdf"
    c = canvas.Canvas(str(out))
    ok = core.draw_image_fit(c, png, 0, 0, 100, 100)
    c.showPage()
    c.save()
    assert ok is True

    c2 = canvas.Canvas(str(tmp_path / "img2.pdf"))
    not_ok = core.draw_image_fit(c2, tmp_path / "missing.png", 0, 0, 100, 100)
    c2.showPage()
    c2.save()
    assert not_ok is False


def test_draw_image_cover(tmp_path: Path):
    png = tmp_path / "tiny.png"
    _write_tiny_png(png)

    from reportlab.pdfgen import canvas

    out = tmp_path / "cover_img.pdf"
    c = canvas.Canvas(str(out))
    ok = core.draw_image_cover(c, png, 0, 0, 100, 50)
    c.showPage()
    c.save()
    assert ok is True


def test_draw_description_columns_skips_show_when_no_fitting_sentence(tmp_path: Path):
    from reportlab.pdfgen import canvas

    out = tmp_path / "ontonight_trim.pdf"
    c = canvas.Canvas(str(out))
    entries = [
        core.OnTonightEntry(
            title="A Very Very Long Show Title That Does Not Fit",
            description="This sentence will not fit either.",
        )
    ]
    core._draw_description_columns(c, entries, 0, 0, 80, 60)
    c.showPage()
    c.save()

    txt = _extract_pdf_text(out)
    # Header may still draw, but the show should be removed entirely.
    assert "ON TONIGHT" in txt
    assert "Very Very Long Show" not in txt


def test_draw_description_columns_renders_bold_title_and_sentence(tmp_path: Path):
    from reportlab.pdfgen import canvas

    out = tmp_path / "ontonight_ok.pdf"
    c = canvas.Canvas(str(out))
    entries = [core.OnTonightEntry(title="Rugrats", description="Tommy leads a toy hunt. Angelica objects.")]
    core._draw_description_columns(c, entries, 0, 0, 300, 120)
    c.showPage()
    c.save()

    txt = _extract_pdf_text(out)
    assert "Rugrats:" in txt
    assert "Tommy leads a toy hunt." in txt


def test_draw_description_columns_empty_descriptions_shows_fallback(tmp_path: Path):
    from reportlab.pdfgen import canvas

    out = tmp_path / "ontonight_empty.pdf"
    c = canvas.Canvas(str(out))
    core._draw_description_columns(c, [], 0, 0, 250, 100)
    c.showPage()
    c.save()

    txt = _extract_pdf_text(out)
    assert "ON TONIGHT" in txt
    assert "No descriptions generated; check source filters." not in txt


def test_movie_meta_badge_rendered_inline(monkeypatch, tmp_path: Path):
    channels = ["MTV"]
    numbers = {"MTV": "7"}
    schedules = {
        "MTV": [
                core.Event(
                    start=datetime(2026, 3, 5, 9, 0),
                    end=datetime(2026, 3, 5, 12, 0),
                    title="Movie Block",
                    filename="The.Matrix.1999.1080p.BluRay.mkv",
                )
        ]
    }

    monkeypatch.setattr(
        core,
        "_fetch_omdb_movie_meta",
        lambda *a, **k: core.MovieMeta(
            title="The Matrix",
            year="1999",
            plot="A computer hacker learns reality is simulated.",
            imdb_rating="7.4",
            rated="R",
        ),
    )

    out = tmp_path / "movie_inline.pdf"
    core.make_pdf(
        out_path=out,
        grid_title="CABLE GUIDE",
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        start_dt=datetime(2026, 3, 5, 9, 0),
        end_dt=datetime(2026, 3, 5, 12, 0),
        step_minutes=30,
        omdb_api_key="fake",
        movie_inline_meta=True,
    )
    txt = _extract_pdf_text(out)
    assert "R" in txt
    assert "★" in txt
    assert "½" in txt
    assert "[" not in txt


def test_fit_title_with_badge_parts_drops_badge_when_space_tight():
    title, badge = core.fit_title_with_badge_parts(
        title="The Matrix Reloaded",
        badge="R ★★★ ½",
        font_name="Helvetica-Bold",
        font_size=6.5,
        max_width_pts=55.0,
    )
    assert title
    assert badge == ""


def test_block_descriptions_movie_prepends_rating_badge():
    schedules = {
        "Movies": [
            core.Event(
                start=datetime(2026, 3, 5, 20, 0),
                end=datetime(2026, 3, 5, 22, 0),
                title="Movie Night",
                filename="The.Matrix.1999.1080p.mkv",
            )
        ]
    }
    nfo = core.cs.NfoIndex(
        by_filename_stem={},
        by_title={
            "the matrix": core.cs.NfoMeta(
                title="The Matrix",
                plot="A hacker learns the truth.",
                rated="R",
                imdb_rating="7.4",
            )
        },
    )
    entries = core._build_block_descriptions(
        schedules=schedules,
        start_dt=datetime(2026, 3, 5, 19, 0),
        end_dt=datetime(2026, 3, 5, 23, 0),
        ignored_channels=None,
        ignored_titles=None,
        nfo_index=nfo,
        max_items=3,
    )
    assert entries
    assert entries[0].description.startswith("R ★★★ ½")


def _sample_schedules() -> tuple[list[str], dict[str, str], dict[str, list[core.Event]]]:
    channels = ["NBC", "PBS"]
    numbers = {"NBC": "3", "PBS": "4"}
    schedules = {
        "NBC": [
            _mk_event(9, 0, 9, 30, "Test Show", "Test.Show.S01E01.mkv"),
            _mk_event(9, 30, 10, 0, "Movie Slot", "The.Matrix.1999.1080p.mkv"),
        ],
        "PBS": [_mk_event(9, 0, 10, 0, "Nature", "")],
    }
    return channels, numbers, schedules


def test_make_pdf_single_and_fold_output(tmp_path: Path):
    channels, numbers, schedules = _sample_schedules()

    single = tmp_path / "single.pdf"
    core.make_pdf(
        out_path=single,
        grid_title="CABLE GUIDE",
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        start_dt=datetime(2026, 3, 5, 9, 0),
        end_dt=datetime(2026, 3, 5, 10, 0),
        step_minutes=30,
        double_sided_fold=False,
    )
    single_text = _extract_pdf_text(single)
    for marker in SNAPSHOT_MARKERS["single"]:
        assert marker in single_text
    assert len(PdfReader(str(single)).pages) == 1

    fold = tmp_path / "fold.pdf"
    core.make_pdf(
        out_path=fold,
        grid_title="Unused Title",
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        start_dt=datetime(2026, 3, 5, 9, 0),
        end_dt=datetime(2026, 3, 5, 11, 0),
        step_minutes=30,
        double_sided_fold=True,
    )
    fold_text = _extract_pdf_text(fold)
    for marker in SNAPSHOT_MARKERS["fold"]:
        assert marker in fold_text
    assert len(PdfReader(str(fold)).pages) == 1


def test_make_compilation_pdf_with_cover_ads_and_bottom_ads(tmp_path: Path, monkeypatch):
    channels, numbers, schedules = _sample_schedules()

    ads_dir = tmp_path / "ads"
    bottom_ads = tmp_path / "bottom_ads"
    cover_dir = tmp_path / "cover"
    ads_dir.mkdir()
    bottom_ads.mkdir()
    cover_dir.mkdir()
    _write_tiny_png(ads_dir / "ad.png")
    _write_tiny_png(bottom_ads / "bottom.png")
    _write_tiny_png(cover_dir / "cover.png")

    monkeypatch.setattr(core, "choose_random", lambda items: items[0] if items else None)

    out = tmp_path / "compilation.pdf"
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    (content_dir / "promos").mkdir(parents=True)
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 1, 0, 0),
        range_end=datetime(2026, 3, 2, 0, 0),
        page_block_hours=6,
        step_minutes=30,
        double_sided_fold=False,
        ads_dir=ads_dir,
        ad_insert_every=2,
        bottom_ads_dir=bottom_ads,
        cover_enabled=True,
        cover_title="Time Travel Cable Guide",
        cover_subtitle="March 2026",
        cover_period_label="",
        cover_art_source="folder",
        cover_art_dir=cover_dir,
        content_dir=content_dir,
        tvdb_api_key="",
        tvdb_pin="",
    )

    reader = PdfReader(str(out))
    # day(24h) with block 6h = 4 guide pages, +1 cover, +1 ad after page 2 => 6 pages
    assert len(reader.pages) == 6
    txt = _extract_pdf_text(out)
    for marker in SNAPSHOT_MARKERS["compilation"]:
        assert marker in txt


def test_impose_booklet_pages_order():
    logical = [core.BookletPageSpec(kind="blank", header_right=str(i)) for i in range(1, 9)]
    imposed = core.impose_booklet_pages(logical)

    order = [(left.header_right, right.header_right) for left, right in imposed]
    assert order == [
        ("8", "1"),
        ("2", "7"),
        ("6", "3"),
        ("4", "5"),
    ]


def test_make_compilation_pdf_folded_booklet_imposition(tmp_path: Path, monkeypatch):
    channels, numbers, schedules = _sample_schedules()
    monkeypatch.setattr(core, "choose_random", lambda items: items[0] if items else None)

    out = tmp_path / "folded_booklet.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 1, 0, 0),
        range_end=datetime(2026, 3, 2, 0, 0),
        page_block_hours=6,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=True,
        cover_title="Time Travel Cable Guide",
        cover_subtitle="March 2026",
    )

    reader = PdfReader(str(out))
    # logical pages: 1 cover + 8 guide halves + 1 back cover blank = 10, padded to 12 => 6 physical sides
    assert len(reader.pages) == 6

    first_page_text = reader.pages[0].extract_text() or ""
    second_page_text = reader.pages[1].extract_text() or ""
    assert "Time Travel Cable Guide" in first_page_text
    assert "CABLE GUIDE" not in first_page_text
    assert "CABLE GUIDE" in second_page_text


def test_make_compilation_pdf_interstitial_catch_page(tmp_path: Path, monkeypatch):
    channels, numbers, schedules = _sample_schedules()
    monkeypatch.setattr(core, "pick_cover_airing_event", lambda *a, **k: schedules["NBC"][0])
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    (content_dir / "promos").mkdir(parents=True)

    out = tmp_path / "catch_interstitial.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 1, 0, 0),
        range_end=datetime(2026, 3, 2, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        ad_insert_every=1,
        interstitial_source="catch",
        cover_enabled=False,
        content_dir=content_dir,
    )
    text = _extract_pdf_text(out)
    assert "Catch Tonight" not in text
    # In file-content mode, interstitials are only inserted when a matching promo
    # manifest exists with non-empty content.
    assert "Catch " not in text


def test_make_compilation_pdf_booklet_back_cover_catch(tmp_path: Path, monkeypatch):
    channels, numbers, schedules = _sample_schedules()
    monkeypatch.setattr(core, "pick_cover_airing_event", lambda *a, **k: schedules["NBC"][0])
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    (content_dir / "promos").mkdir(parents=True)

    out = tmp_path / "booklet_back_catch.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 1, 0, 0),
        range_end=datetime(2026, 3, 2, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=True,
        back_cover_catch_enabled=True,
        content_dir=content_dir,
    )
    text = _extract_pdf_text(out)
    assert "Catch Tonight" not in text
    assert "Catch " not in text


def test_make_compilation_pdf_booklet_back_cover_catch_without_front_cover(tmp_path: Path, monkeypatch):
    channels, numbers, schedules = _sample_schedules()
    monkeypatch.setattr(core, "pick_cover_airing_event", lambda *a, **k: schedules["NBC"][0])
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    (content_dir / "promos").mkdir(parents=True)

    out = tmp_path / "booklet_back_only_catch.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 1, 0, 0),
        range_end=datetime(2026, 3, 2, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=False,
        back_cover_catch_enabled=True,
        content_dir=content_dir,
    )
    text = _extract_pdf_text(out)
    assert "Catch " not in text


def test_make_compilation_pdf_interstitial_promo_template_fills_placeholders(tmp_path: Path):
    channels, numbers, schedules = _sample_schedules()
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    promos = content_dir / "promos"
    promos.mkdir(parents=True)
    (promos / "promo_nbc.json").write_text(
        json.dumps(
            {
                "id": "promo-nbc",
                "enabled": True,
                "range_modes": ["day"],
                "title": "",
                "message_template": "Catch {show} on {weekday} at {time}!",
                "image": "",
                "match_titles": ["test show"],
                "match_channels": ["nbc"],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "interstitial_placeholder.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 5, 0, 0),
        range_end=datetime(2026, 3, 6, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        ad_insert_every=1,
        interstitial_source="catch",
        cover_enabled=False,
        content_dir=content_dir,
    )
    text = _extract_pdf_text(out)
    assert "Catch Test Show on Thursday at 9:00!" in text


def test_make_compilation_pdf_prefers_matched_promo_over_generic(tmp_path: Path):
    channels, numbers, schedules = _sample_schedules()
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    promos = content_dir / "promos"
    promos.mkdir(parents=True)
    (promos / "promo_generic.json").write_text(
        json.dumps(
            {
                "id": "promo-generic",
                "enabled": True,
                "range_modes": [],
                "title": "Tony's Pizza",
                "message_template": "Open late every night.",
                "image": "",
                "match_titles": [],
                "match_channels": [],
            }
        ),
        encoding="utf-8",
    )
    (promos / "promo_matched.json").write_text(
        json.dumps(
            {
                "id": "promo-matched",
                "enabled": True,
                "range_modes": [],
                "title": "",
                "message_template": "Catch {show} on {weekday} at {time}!",
                "image": "",
                "match_titles": ["test show"],
                "match_channels": ["nbc"],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "prefer_matched.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 5, 0, 0),
        range_end=datetime(2026, 3, 6, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=False,
        content_dir=content_dir,
    )
    text = _extract_pdf_text(out)
    assert "Catch Test Show on Thursday at 9:00!" in text

    # Generic promos may be consumed for booklet padding, but the matched promo
    # must be used for the back cover first (outer sheet first side).
    reader = PdfReader(str(out))
    first_page = reader.pages[0].extract_text() or ""
    assert "Catch Test Show on Thursday at 9:00!" in first_page


def test_make_compilation_pdf_booklet_never_reuses_same_promo(tmp_path: Path):
    channels, numbers, schedules = _sample_schedules()
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    promos = content_dir / "promos"
    promos.mkdir(parents=True)
    (promos / "promo_one.json").write_text(
        json.dumps(
            {
                "id": "promo-one",
                "enabled": True,
                "range_modes": [],
                "title": "Promo One",
                "message_template": "First promo",
                "image": "",
                "match_titles": [],
                "match_channels": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "no_reuse.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 5, 0, 0),
        range_end=datetime(2026, 3, 6, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=False,
        content_dir=content_dir,
    )
    reader = PdfReader(str(out))
    pages_with_promo = 0
    for p in reader.pages:
        txt = p.extract_text() or ""
        if "Promo One" in txt:
            pages_with_promo += 1
    assert pages_with_promo <= 1


def test_booklet_back_cover_promo_stays_on_outer_sheet(tmp_path: Path):
    channels, numbers, schedules = _sample_schedules()
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    promos = content_dir / "promos"
    promos.mkdir(parents=True)
    (promos / "promo_back.json").write_text(
        json.dumps(
            {
                "id": "promo-back",
                "enabled": True,
                "range_modes": [],
                "title": "BACK COVER PROMO",
                "message_template": "Outer sheet promo",
                "image": "",
                "match_titles": [],
                "match_channels": [],
            }
        ),
        encoding="utf-8",
    )

    out = tmp_path / "back_cover_position.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 5, 0, 0),
        range_end=datetime(2026, 3, 6, 0, 0),
        page_block_hours=6,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=True,
        cover_title="Time Travel Cable Guide",
        content_dir=content_dir,
    )

    reader = PdfReader(str(out))
    first = reader.pages[0].extract_text() or ""
    second = reader.pages[1].extract_text() or ""
    assert "BACK COVER PROMO" in first
    assert "BACK COVER PROMO" not in second


def test_booklet_padding_uses_promos_before_blanks(tmp_path: Path, capsys):
    channels, numbers, schedules = _sample_schedules()
    content_dir = tmp_path / "content"
    (content_dir / "covers").mkdir(parents=True)
    promos = content_dir / "promos"
    promos.mkdir(parents=True)
    for idx in range(1, 5):
        (promos / f"promo_{idx}.json").write_text(
            json.dumps(
                {
                    "id": f"promo-{idx}",
                    "enabled": True,
                    "range_modes": [],
                    "title": f"PROMO {idx}",
                    "message_template": f"Message {idx}",
                    "image": "",
                    "match_titles": [],
                    "match_channels": [],
                }
            ),
            encoding="utf-8",
        )

    out = tmp_path / "pad_promos.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="day",
        range_start=datetime(2026, 3, 5, 0, 0),
        range_end=datetime(2026, 3, 6, 0, 0),
        page_block_hours=12,
        step_minutes=30,
        double_sided_fold=True,
        cover_enabled=False,
        content_dir=content_dir,
        status_messages=True,
    )
    captured = capsys.readouterr().out
    assert "Inserted booklet pad promo before back cover" in captured
    assert "Inserted booklet pad blank before back cover (no unused promos left)" not in captured


def test_ontonight_generation_retries_until_renderable(tmp_path: Path, monkeypatch, capsys):
    channels, numbers, schedules = _sample_schedules()
    attempts = {"n": 0}
    nfo_dir = tmp_path / "nfo"
    nfo_dir.mkdir(parents=True, exist_ok=True)
    (nfo_dir / "test show.nfo").write_text(
        "<tvshow><title>Test Show</title><plot>Test Show has a valid plot. Another sentence.</plot></tvshow>",
        encoding="utf-8",
    )

    def _mock_renderable(_entries, _box_width=250.0, **_kwargs):
        attempts["n"] += 1
        return attempts["n"] >= 3

    monkeypatch.setattr(core, "_has_renderable_ontonight_content", _mock_renderable)

    out = tmp_path / "retry_ontonight.pdf"
    core.make_compilation_pdf(
        out_path=out,
        channels=channels,
        channel_numbers=numbers,
        schedules=schedules,
        range_mode="single",
        range_start=datetime(2026, 3, 5, 9, 0),
        range_end=datetime(2026, 3, 5, 12, 0),
        page_block_hours=3,
        step_minutes=30,
        cover_enabled=False,
        nfo_dir=nfo_dir,
        status_messages=True,
    )
    captured = capsys.readouterr().out
    assert "WARNING: On Tonight generation attempt 1/10 had no renderable entries" in captured
    assert "WARNING: On Tonight generation attempt 2/10 had no renderable entries" in captured
    assert "On Tonight generation recovered" in captured


def test_build_block_descriptions_avoids_repeats_until_exhausted():
    schedules = {
        "NBC": [
            core.Event(
                start=datetime(2026, 3, 5, 9, 0),
                end=datetime(2026, 3, 5, 10, 0),
                title="Alpha Show",
                filename="alpha_show.mkv",
            ),
            core.Event(
                start=datetime(2026, 3, 5, 10, 0),
                end=datetime(2026, 3, 5, 11, 0),
                title="Beta Show",
                filename="beta_show.mkv",
            ),
        ]
    }
    nfo = core.cs.NfoIndex(
        by_filename_stem={},
        by_title={
            "alpha show": core.cs.NfoMeta(title="Alpha Show", plot="Alpha plot.", rated="", imdb_rating=""),
            "beta show": core.cs.NfoMeta(title="Beta Show", plot="Beta plot.", rated="", imdb_rating=""),
        },
    )
    used = {"alpha show"}

    entries = core._build_block_descriptions(
        schedules=schedules,
        start_dt=datetime(2026, 3, 5, 9, 0),
        end_dt=datetime(2026, 3, 5, 12, 0),
        ignored_channels=None,
        ignored_titles=None,
        nfo_index=nfo,
        used_titles=used,
        max_items=2,
    )
    assert [e.title.lower() for e in entries] == ["beta show"]

    # Now everything is used; repeats are allowed.
    used.update({"beta show"})
    entries2 = core._build_block_descriptions(
        schedules=schedules,
        start_dt=datetime(2026, 3, 5, 9, 0),
        end_dt=datetime(2026, 3, 5, 12, 0),
        ignored_channels=None,
        ignored_titles=None,
        nfo_index=nfo,
        used_titles=used,
        max_items=2,
    )
    assert len(entries2) >= 1
    assert entries2[0].title.lower() in {"alpha show", "beta show"}


def test_flowable_wraps():
    channels, numbers, schedules = _sample_schedules()
    guide = core.GuideTimelineFlowable(channels, numbers, schedules, datetime(2026, 3, 5, 9, 0), datetime(2026, 3, 5, 10, 0), 30)
    w, h = guide.wrap(500, 700)
    assert w == 500 and h > 0

    folded = core.FoldedGuideTimelineFlowable(
        channels,
        numbers,
        schedules,
        datetime(2026, 3, 5, 9, 0),
        datetime(2026, 3, 5, 10, 0),
        datetime(2026, 3, 5, 11, 0),
        30,
        safe_gap=0.2,
    )
    w2, h2 = folded.wrap(500, 700)
    assert w2 == 500 and h2 > 0
    assert folded.gap > 0.12 * core.inch

    hdr = core.FoldHeaderFlowable("L", "R")
    w3, h3 = hdr.wrap(500, 700)
    assert w3 == 500 and h3 > 0


def test_header_label_edges_are_inset():
    channels, numbers, schedules = _sample_schedules()
    guide = core.GuideTimelineFlowable(channels, numbers, schedules, datetime(2026, 3, 5, 9, 0), datetime(2026, 3, 5, 11, 0), 30)
    guide.wrap(500, 700)

    left_safe = guide._safe_header_label_x(guide.first_col, "9a", 500)
    right_safe = guide._safe_header_label_x(500, "11a", 500)

    assert left_safe > guide.first_col
    assert right_safe < 500
