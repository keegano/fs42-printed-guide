from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

cfg = pytest.importorskip("guide_config")


def test_parse_effective_args_from_json_config(tmp_path: Path):
    conf = {
        "fs42_dir": "/tmp/fs42",
        "date": "2026-03-01",
        "range_mode": "month",
        "page_block_hours": 4,
        "double_sided_fold": True,
        "numbers": {"NBC": "3", "PBS": "4"},
    }
    p = tmp_path / "guide.json"
    p.write_text(json.dumps(conf), encoding="utf-8")

    args = cfg.parse_effective_args(["--config", str(p)])

    assert args.fs42_dir == Path("/tmp/fs42")
    assert args.date.strftime("%Y-%m-%d") == "2026-03-01"
    assert args.range_mode == "month"
    assert args.page_block_hours == 4.0
    assert args.double_sided_fold is True
    assert isinstance(args.numbers, dict)


def test_parse_effective_args_from_toml_config(tmp_path: Path):
    p = tmp_path / "guide.toml"
    p.write_text(
        """
fs42_dir = "/tmp/fs42"
date = "2026-03-02"
start = "06:30"
out = "bundle.pdf"
cover_page = true
""".strip(),
        encoding="utf-8",
    )

    args = cfg.parse_effective_args(["--config", str(p)])

    assert args.start.strftime("%H:%M") == "06:30"
    assert args.out == Path("bundle.pdf")
    assert args.cover_page is True


def test_cli_overrides_config(tmp_path: Path):
    p = tmp_path / "guide.json"
    p.write_text(
        json.dumps(
            {
                "fs42_dir": "/tmp/fs42",
                "date": "2026-03-01",
                "range_mode": "week",
                "hours": 12,
            }
        ),
        encoding="utf-8",
    )

    args = cfg.parse_effective_args(
        [
            "--config",
            str(p),
            "--range-mode",
            "single",
            "--hours",
            "3",
            "--out",
            "custom.pdf",
        ]
    )

    assert args.range_mode == "single"
    assert args.hours == 3.0
    assert args.out == Path("custom.pdf")


def test_load_config_file_rejects_unsupported_extension(tmp_path: Path):
    p = tmp_path / "guide.yaml"
    p.write_text("fs42_dir: /tmp/fs42", encoding="utf-8")
    with pytest.raises(ValueError):
        cfg.load_config_file(p)


def test_parse_effective_args_requires_date():
    with pytest.raises(SystemExit):
        cfg.parse_effective_args(["--fs42-dir", "/tmp/fs42"])


def test_parse_effective_args_uses_default_station42_dir():
    args = cfg.parse_effective_args(["--date", "2026-03-01"])
    assert args.fs42_dir == cfg.DEFAULT_FS42_DIR


def test_station42_dir_alias_is_supported():
    args = cfg.parse_effective_args(["--station42-dir", "/tmp/s42", "--date", "2026-03-01"])
    assert args.fs42_dir == Path("/tmp/s42")


def test_catalog_options_and_status_toggle():
    args = cfg.parse_effective_args(
        [
            "--date",
            "2026-03-01",
            "--load-catalog",
            "scan.json",
            "--dump-catalog",
            "dump.json",
            "--fold-safe-gap",
            "0.2",
            "--no-status-messages",
        ]
    )
    assert args.load_catalog == Path("scan.json")
    assert args.dump_catalog == Path("dump.json")
    assert args.fold_safe_gap == 0.2
    assert args.status_messages is False


def test_env_file_supplies_tvdb_credentials(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
TVDB_API_KEY=file-key
TVDB_PIN=file-pin
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    args = cfg.parse_effective_args(["--fs42-dir", "/tmp/fs42", "--date", "2026-03-01"])
    assert args.tvdb_api_key == "file-key"
    assert args.tvdb_pin == "file-pin"


def test_cli_overrides_env_for_tvdb_credentials(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TVDB_API_KEY=file-key\nTVDB_PIN=file-pin\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    args = cfg.parse_effective_args(
        [
            "--fs42-dir",
            "/tmp/fs42",
            "--date",
            "2026-03-01",
            "--tvdb-api-key",
            "cli-key",
            "--tvdb-pin",
            "cli-pin",
        ]
    )
    assert args.tvdb_api_key == "cli-key"
    assert args.tvdb_pin == "cli-pin"
