from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
REAL_CATALOG_DUMP = ROOT / "tests" / "test_payloads" / "catalog_dump.json"


def test_generate_promo_templates_writes_separate_files(tmp_path: Path):
    out_dir = tmp_path / "promos"
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "generate_promo_templates.py"),
        "--catalog",
        str(REAL_CATALOG_DUMP),
        "--out-dir",
        str(out_dir),
        "--range-mode",
        "day",
        "--date",
        "2026-02-24",
        "--page-block-hours",
        "12",
        "--max",
        "3",
        "--poster-source",
        "none",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))

    files = sorted(out_dir.glob("promo_*.json"))
    assert len(files) == 3
    payload = json.loads(files[0].read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "{weekday}" in payload.get("message_template", "")
    assert isinstance(payload.get("match_titles", []), list)


def test_generate_promo_templates_accepts_missing_space_before_poster_source(tmp_path: Path):
    out_dir = tmp_path / "promos"
    # Intentionally mimic user typo: no space before --poster-source.
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "generate_promo_templates.py"),
        "--catalog",
        str(REAL_CATALOG_DUMP),
        "--out-dir",
        str(out_dir) + "--poster-source",
        "tvdb",
        "--range-mode",
        "day",
        "--date",
        "2026-02-24",
        "--page-block-hours",
        "12",
        "--max",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    files = sorted(out_dir.glob("promo_*.json"))
    assert len(files) == 1


def test_save_poster_and_backfill_existing_promo(tmp_path: Path, monkeypatch):
    mod = __import__("tools.generate_promo_templates", fromlist=["_save_poster_for_event", "main"])

    # fake downloaded image
    dl = tmp_path / "dl.jpg"
    dl.write_bytes(b"fakeimg")
    monkeypatch.setattr(mod, "fetch_tvdb_cover_art", lambda *a, **k: dl)

    ev = mod.Event(
        start=datetime(2026, 2, 24, 12, 0),
        end=datetime(2026, 2, 24, 13, 0),
        title="Test Show",
        filename="Test.Show.S01E01.mkv",
    )
    out_dir = tmp_path / "promos"
    out_dir.mkdir()
    poster = mod._save_poster_for_event(
        out_dir=out_dir,
        shown="Test Show",
        ev=ev,
        poster_source="tvdb",
        tvdb_api_key="k",
        tvdb_pin="",
        omdb_api_key="",
    )
    assert poster
    assert (out_dir / poster).exists()

    existing = out_dir / "promo_test-show.json"
    existing.write_text(
        json.dumps(
            {
                "id": "promo-test-show",
                "enabled": True,
                "message_template": "Catch {show}",
                "image": "",
                "match_titles": ["test show"],
                "match_channels": [],
            }
        ),
        encoding="utf-8",
    )

    # emulate the backfill branch
    payload = json.loads(existing.read_text(encoding="utf-8"))
    if not payload.get("image") and poster:
        payload["image"] = poster
        existing.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    out_payload = json.loads(existing.read_text(encoding="utf-8"))
    assert out_payload["image"] == poster
