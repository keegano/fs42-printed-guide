from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


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
