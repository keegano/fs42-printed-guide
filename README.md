# Printed Guide

Generate printable TV guide PDFs from FieldStation42 schedule output.

## Overview

`print_guide.py` reads channel extents (`station_42.py -e`) and per-channel schedules (`station_42.py -u <channel>`), parses events, normalizes titles, and renders a print-focused PDF.
By default it looks for FieldStation42 at `../FieldStation42` (adjacent to this repo), and you can override that with config/CLI.

Current rendering supports:
- Proportional timeline layout (event blocks positioned by true start/end time)
- Optional folded layout (two side-by-side timeline halves on one page)
- Single-window guide output and multi-page compilation output
- Optional ad insertion and optional cover page with image art
- Progress status messages during station scanning and PDF assembly
- Progress status messages while generating "On Tonight" blocks
- Catalog dump/load to reuse scanned schedules without re-running FieldStation42
- Configurable folded safe-gap at center fold for print tolerance
- Cover styling controls (background color, border inset, and title/date fonts)
- Default outlined cover text for readability on arbitrary backgrounds

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

With `--double-sided-fold`, compilation mode now performs booklet imposition:
- each guide block is split into left/right time halves as separate booklet pages
- booklet pages are reordered into printer sheet order for duplex fold printing
- cover art is placed on the right half of the outer front sheet
- the back-cover half is currently blank

## Ads and Cover Features

Compilation mode supports:
- Full-page interstitial ads from `--ads-dir`, inserted every N guide pages via `--ad-insert-every`
- Bottom ad fill from `--bottom-ads-dir` (ad image placed below guide when page has remaining vertical space)
- Optional cover page (`--cover-page`) with configurable text and optional art
- Cover art can be rendered as a full-bleed image inside a configurable border
  filled by `--cover-bg-color` (useful for trim/fold safety)
- TVDB covers can include an airing label (for example: `"Rugrats playing Tuesday at 7:00"`)
  using configurable templates per range mode (`single/day/week/month`)
- If no bottom ad is available, guide pages can fill lower space with TVDB show-description columns for titles airing in that block
- "On Tonight" titles are bold; entries without a usable description are skipped
- If no usable entries remain, a fallback message is shown instead of an empty block
- Optional Catch promo interstitial pages can be generated instead of image ads

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

Required field: `date` (and `fs42_dir` only when not using `--load-catalog`).

## Catalog Cache Workflow

To avoid repeatedly scanning slow `station_42.py` calls:

1. Run once with `--dump-catalog ./cache/march_scan.json`.
2. Iterate on rendering options with `--load-catalog ./cache/march_scan.json`.

Catalog files include discovered channels, inferred year, and parsed schedule events.
See `examples/real_catalog_ads.toml` for a realistic cached-catalog + ad setup.

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
- `--station42-dir PATH` (preferred alias)
- `--fs42-dir PATH` (backward compatible alias)
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
- `--fold-safe-gap FLOAT` (inches of extra center gutter for folded pages)

Compilation extras:
- `--ads-dir PATH`
- `--ad-insert-every INT`
- `--interstitial-source ads|catch|none`
- `--bottom-ads-dir PATH`
- `--cover-page`
- `--cover-title TEXT`
- `--cover-subtitle TEXT`
- `--cover-period-label TEXT`
- `--cover-bg-color HEX`
- `--cover-border-size FLOAT`
- `--cover-text-color HEX`
- `--cover-text-outline-color HEX`
- `--cover-text-outline-width FLOAT`
- `--cover-title-font NAME`
- `--cover-title-size FLOAT`
- `--cover-subtitle-font NAME`
- `--cover-subtitle-size FLOAT`
- `--cover-date-font NAME`
- `--cover-date-size FLOAT`
- `--cover-date-y FLOAT`
- `--cover-airing-font NAME`
- `--cover-airing-size FLOAT`
- `--cover-airing-y FLOAT`
- `--cover-airing-label-enabled` / `--no-cover-airing-label`
- `--cover-airing-label-single-format TEXT`
- `--cover-airing-label-day-format TEXT`
- `--cover-airing-label-week-format TEXT`
- `--cover-airing-label-month-format TEXT`
- `--cover-art-source none|folder|tvdb|auto`
- `--cover-art-dir PATH`
- `--tvdb-api-key KEY`
- `--tvdb-pin PIN`
- `--omdb-api-key KEY`
- `--api-cache-file PATH`
- `--no-api-cache`
- `--dump-catalog PATH`
- `--load-catalog PATH`
- `--status-messages` / `--no-status-messages`
- `--ignore-list-file PATH`
- `--back-cover-catch-enabled` / `--no-back-cover-catch`

`--fold-safe-gap` applies to folded single pages and folded booklet compilation spreads.
Time tick labels on guide headers are inset at the edges so start/end labels stay
inside print-safe bounds and clear of the `CH` column and fold gutter.

Cover text defaults to white with black outline (`--cover-text-color` and
`--cover-text-outline-color`) so title/date remain printable over arbitrary art.
Date and airing-label lines are positioned near the bottom by default (`cover_date_y=0.18`,
`cover_airing_y=0.11`) and are configurable.

Movie metadata:
- When `--omdb-api-key` is provided, movie blocks can include inline badges
  like `[PG-13 ★★★ ½]` in the guide.
- For movie cover picks (`cover_art_source=tvdb/auto`), OMDb poster is preferred.
- "On Tonight" description fallback will use OMDb plot summaries for movies,
  TVDB overviews for series.
- TVDB/OMDb responses are cached to disk by default (`.cache/printed_guide_api_cache.json`)
  to reduce API usage and avoid free-tier overages.

Ignore list:
- Use `--ignore-list-file` with JSON like:
  `{"channels": ["MTV"], "titles": ["Rock"]}`
- Ignored channels/titles are excluded from cover candidate selection and "On Tonight" content.

When `--interstitial-source catch` is used with `--ad-insert-every`, inserted interstitial
pages use generated "Catch Tonight" copy instead of image ads.

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
