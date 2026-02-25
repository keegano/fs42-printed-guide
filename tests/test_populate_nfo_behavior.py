from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

mod = __import__("tools.populate_nfo_metadata", fromlist=["_series_nfo_path", "_movie_nfo_path", "_episode_nfo_path", "_target_nfo_path"])


def test_series_nfo_path_is_under_local_content_store():
    root = Path("/tmp/content/nfo")
    got = mod._series_nfo_path(root, "NBC", "Law & Order")
    assert str(got).endswith("/tmp/content/nfo/shows/nbc/law-order/tvshow.nfo")


def test_movie_nfo_path_is_under_local_content_store():
    root = Path("/tmp/content/nfo")
    got = mod._movie_nfo_path(root, "The Matrix", "1999")
    assert str(got).endswith("/tmp/content/nfo/movies/the-matrix-1999.nfo")


def test_episode_nfo_path_is_under_local_content_store():
    root = Path("/tmp/content/nfo")
    got = mod._episode_nfo_path(root, "NBC", "Law & Order", "Law & Order - S01E01.mkv")
    assert str(got).endswith("/tmp/content/nfo/episodes/nbc/law-order/law-order-s01e01.nfo")


def test_target_nfo_path_prefers_series_file_when_not_episode_mode():
    root = Path("/tmp/content/nfo")
    got = mod._target_nfo_path(root, "NBC", "Law & Order", "Law & Order - S01E02.mkv", is_movie=False, episode_nfo=False)
    assert str(got).endswith("/tmp/content/nfo/shows/nbc/law-order/tvshow.nfo")
