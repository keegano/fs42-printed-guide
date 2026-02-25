# Printed Guide

Generate printable TV guide PDFs from FieldStation42 schedule output.

## Overview

`print_guide.py` reads channel extents (`station_42.py -e`) and per-channel schedules (`station_42.py -u <channel>`), parses events, normalizes titles, and renders a print-focused PDF.

Current rendering supports:
- Proportional timeline layout (event blocks positioned by true start/end time)
- Optional folded layout (two side-by-side timeline halves on one page)
- Single-window guide output and multi-page compilation output
- Optional ad insertion and optional cover page with image art

## Data Flow

1. Discover channels with valid extents from `-e` output.
2. Infer schedule year from extents unless overridden.
3. Parse `-u` schedule lines into `Event(start, end, title, filename)` objects.
4. Normalize titles:
- ASCII-only sanitization
- Remove bracketed fragments
- Episode/movie display title selection
- Width fitting with case-insensitive reduction rules for `the` / `and`
5. Render to PDF.

Channel numbers are sourced from `confs/*.json` files by default using:
- `station_conf.network_name`
- `station_conf.channel_number`

Those numbers are rendered as circled badges next to channel names in the guide.

## Rendering Modes

### Single Range (`--range-mode single`)
Renders one guide window based on:
- `--date`
- `--start`
- `--hours`

Optional fold mode (`--double-sided-fold`) renders two side-by-side halves on the same page.

### Compilation (`--range-mode day|week|month`)
Builds a larger PDF by concatenating many guide pages:
- `day`: 24h from selected date
- `week`: 7 days from selected date
- `month`: full calendar month of selected date

Per-page time window is controlled by `--page-block-hours`.

## Ads and Cover Features

Compilation mode supports:
- Full-page interstitial ads from `--ads-dir`, inserted every N guide pages via `--ad-insert-every`
- Bottom ad fill from `--bottom-ads-dir` (ad image placed below guide when page has remaining vertical space)
- Optional cover page (`--cover-page`) with configurable text and optional art

Cover art sources (`--cover-art-source`):
- `none`
- `folder` (random image from `--cover-art-dir`)
- `tvdb` (random show title -> TVDB lookup/download)
- `auto` (folder first, then TVDB fallback)

TVDB options:
- `--tvdb-api-key`
- `--tvdb-pin` (optional)

## Config Files

All CLI options can also be supplied via `--config <file.json|file.toml>`.

Precedence:
1. Built-in defaults
2. Config file values
3. Explicit CLI flags

Required fields (`fs42_dir`, `date`) can come from either CLI or config.

Example `guide.toml`:

```toml
fs42_dir = "/path/to/FieldStation42"
date = "2026-03-01"
range_mode = "month"
page_block_hours = 6
double_sided_fold = true
cover_page = true
cover_title = "Time Travel Cable Guide"
cover_subtitle = "March 2026"
cover_art_source = "folder"
cover_art_dir = "./assets/cover"
```

## CLI

Core options:
- `--config FILE.json|FILE.toml`
- `--fs42-dir PATH`
- `--date YYYY-MM-DD`
- `--start HH:MM`
- `--hours FLOAT`
- `--range-mode single|day|week|month`
- `--page-block-hours FLOAT`
- `--step INT`
- `--numbers "CHAN=NUM,..."`
- `--confs-dir PATH`
- `--year INT`
- `--out FILE.pdf`
- `--double-sided-fold`

Compilation extras:
- `--ads-dir PATH`
- `--ad-insert-every INT`
- `--bottom-ads-dir PATH`
- `--cover-page`
- `--cover-title TEXT`
- `--cover-subtitle TEXT`
- `--cover-period-label TEXT`
- `--cover-art-source none|folder|tvdb|auto`
- `--cover-art-dir PATH`
- `--tvdb-api-key KEY`
- `--tvdb-pin PIN`

## Example

```bash
python3 print_guide.py \
  --fs42-dir /path/to/FieldStation42 \
  --date 2026-03-01 \
  --range-mode month \
  --page-block-hours 6 \
  --double-sided-fold \
  --cover-page \
  --cover-title "Time Travel Cable Guide" \
  --cover-subtitle "March 2026" \
  --cover-art-source folder \
  --cover-art-dir ./assets/cover \
  --ads-dir ./assets/ads/full \
  --ad-insert-every 4 \
  --bottom-ads-dir ./assets/ads/bottom \
  --out march_guide.pdf
```

## Project Layout

- `print_guide.py`: thin CLI entrypoint / orchestration
- `guide_config.py`: config-file loading + merge with CLI options
- `guide_core.py`: parsing, title normalization, rendering, compilation
- `main.py`: entrypoint wrapper
- `tests/test_payloads/`: captured FieldStation42 output fixtures
