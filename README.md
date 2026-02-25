# Printed Guide

Generate printable TV guide PDFs from FieldStation42 schedules.

## Intended Workflow

Rendering is now file-first and offline:

1. Scan schedules from FieldStation42 (or load a catalog dump).
2. Prepare editable content files in `content/`:
- `content/covers/*.json`
- `content/promos/*.json`
3. Populate media metadata into Plex-compatible `.nfo` files under FieldStation42 `catalog/`.
4. Run `print_guide.py` to render. Guide generation does not call external APIs.

If content/metadata is missing, rendering continues and prints status messages about what was skipped.

## Folder Layout

- `print_guide.py`: CLI entrypoint
- `guide_config.py`: config/CLI merge
- `guide_core.py`: parsing and PDF rendering
- `content_store.py`: cover/promo manifest loading + NFO indexing
- `tools/generate_promo_templates.py`: generate editable promo templates from schedules
- `tools/populate_nfo_metadata.py`: optional helper to write `.nfo` metadata files
- `content/covers/`: cover manifests
- `content/promos/`: promo manifests

## Content Manifest Format

### Cover manifest (`content/covers/*.json`)

```json
{
  "id": "default-cover",
  "enabled": true,
  "range_modes": ["day", "week", "month", "single"],
  "title": "Time Travel Cable Guide",
  "subtitle": "Curated Edition",
  "period_label": "",
  "airing_label": "",
  "image": "cover_default.jpg"
}
```

### Promo manifest (`content/promos/*.json`)

```json
{
  "id": "promo-rugrats",
  "enabled": true,
  "range_modes": ["day", "week", "month"],
  "title": "",
  "message": "Catch Rugrats at 7:00 PM!",
  "image": "rugrats_promo.jpg",
  "match_titles": ["rugrats"],
  "match_channels": ["nickelodeon"]
}
```

Notes:
- `image` paths are relative to the manifest file directory.
- Matching is case-insensitive.
- Empty promo entries are ignored.

## NFO Metadata

Guide descriptions and movie ratings come from `.nfo` files.

Lookup behavior:
- First by filename stem from the scheduled filename.
- Fallback by normalized title.

Expected fields:
- `<title>`
- `<plot>`
- `<rating>`
- `<mpaa>`

## Tools

### Generate promo templates

```bash
python3 tools/generate_promo_templates.py \
  --catalog ./cache/march_week_catalog.json \
  --date 2026-03-01 \
  --range-mode week \
  --page-block-hours 4 \
  --out-dir ./content/promos
```

### Populate NFO metadata

```bash
python3 tools/populate_nfo_metadata.py \
  --catalog ./cache/march_week_catalog.json \
  --fs42-dir ../FieldStation42 \
  --tvdb-api-key "$TVDB_API_KEY" \
  --tvdb-pin "$TVDB_PIN" \
  --omdb-api-key "$OMDB_API_KEY"
```

Run with `--dry-run` first to preview writes.

## Rendering

### Example command

```bash
python3 print_guide.py \
  --station42-dir ../FieldStation42 \
  --content-dir ./content \
  --load-catalog ./cache/march_week_catalog.json \
  --date 2026-03-01 \
  --range-mode week \
  --page-block-hours 4 \
  --double-sided-fold \
  --cover-page \
  --interstitial-source catch \
  --out out/week_cached.pdf
```

## Booklet Rules

- Folded booklet mode performs page imposition for duplex printing.
- A safe center gap is configurable via `--fold-safe-gap`.
- Back cover always receives a promo page.
- Blank filler pages are avoided by adding promo filler pages when booklet padding is needed.

## Config

All CLI options can be set with `--config <json|toml>`.

See:
- `examples/single_evening.json`
- `examples/month_booklet.toml`
- `examples/cached_iterate.toml`
- `examples/real_catalog_ads.toml`
