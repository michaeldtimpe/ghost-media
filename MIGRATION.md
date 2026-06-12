# Moving ghost-media to a new host

Checklist for relocating the working setup (repo + generated data + media).
Written 2026-06 from the M5 Max setup; sizes are as of that date.

## 1. What must travel

### Repo
`git clone git@github.com:michaeldtimpe/ghost-media.git` — code, docs,
committed baselines (`bench/perceptual_*.json`), and `examples/`.

### Generated data (gitignored — copy it; regenerating is expensive)

| What | Where | Size | Regeneration cost if lost |
|------|-------|------|---------------------------|
| Raw video analyses | repo root `*.analysis.json`, `_manifest.json`, `_audit_summary.json` | ~50 MB | hours (media-analyzer over all footage) |
| Enriched corpus | `enriched/` (`*.enriched.json`, `*.clip_embeddings.json`, `*.quality.json`, `frames/`) | 1.2 GB | days — the vision bake-off + rescan effort |
| Text flags | `text_flags/` | 2 MB | hours (scan_text_fast.py) |
| Local footage staging | `raw_footage/` | 533 MB | re-copy from archive drive |
| Demucs stems | `demucs_output/` | varies | regenerable (`extract_lyrics.py`) — optional |

### The `sets` symlink target
`sets` in the repo root is a **symlink** → `/Volumes/archive/temp/media-analysis/sets`
(9.3 GB: `.deep-analysis.json`, `.lyrics.json`, rendered videos, `*_mv_build/`).
On the new host either recreate it (`ln -s <new-location> sets`) or make
`sets/` a real directory — the pipeline only cares that `./sets/` resolves.
The rendered `.mp4`s are outputs; only the `.deep-analysis.json` and
`.lyrics.json` files are pipeline inputs.

### Archive-drive media (needed for rendering + analysis, not for selection-only work)
- **Source footage** (single canonical root): `/Volumes/archive/3000/3100/visuals/library/`
  with one subdir per collection — `raw/` (the original corpus),
  `arche-loops/`, `synesthesia/`, `hungry-ghost/`, `_staging/` (pre-import
  parking; excluded from the live index).
- **DJ-set audio**: `/Volumes/archive/3000/3100/sets` (per-set dirs with masters)

## 2. Media paths (one module, env-overridable)

All media locations live in **`media_paths.py`** — every script imports from
it; nothing else hardcodes archive paths:

```python
FOOTAGE_ROOT  # default /Volumes/archive/3000/3100/visuals/library   (env GHOST_FOOTAGE_ROOT)
SETS_ROOT     # default /Volumes/archive/3000/3100/sets              (env GHOST_SETS_ROOT)
CACHE_ROOT    # default ~/ghost-media-cache                          (env GHOST_MEDIA_CACHE)
```

On a new host, either mount the archive at the same path or set the env
vars. `media_paths.find_source()` resolves via a one-time recursive filename
index (NFC-normalized — macOS/SMB filesystems return NFD names) with the old
fuzzy fallback, so the paths recorded inside `.analysis.json` files do
**not** need to be valid. Run `scripts/remap_media_paths.py --apply` to
canonicalize them anyway, and use its default verify mode (`exit 0` = every
analysis resolves to exactly one file, no duplicate filenames across
collections) as the post-move health check.

### Local cache (gigabit-ethernet hosts)

Batch analysis makes ~5 decode passes per video (scene detect, motion,
embeddings, text scan, loops); warm a collection onto local NVMe and point
the whole pipeline at it via the env override:

```bash
python3 media_paths.py --warm hungry-ghost
GHOST_FOOTAGE_ROOT=~/ghost-media-cache/library python3 analyze_visuals_library.py ...
# afterwards: python3 scripts/remap_media_paths.py --apply   (re-canonicalize stored paths)
```

`media_paths.cached_path()` additionally offers per-file copy-through with
LRU eviction at `GHOST_MEDIA_CACHE_MAX_GB` (default 100). The assembler
reads straight from the NAS by default — its access pattern (a few seeks per
clip) doesn't benefit enough to justify the copy.

## 3. Environment setup

- macOS on Apple Silicon is the tested platform (torch uses MPS). The
  assembler itself only needs **numpy + FFmpeg** and runs anywhere.
- `./setup.sh` → creates `.venv` with the full dependency set
  (torch/open_clip for CLIP embeddings + the lyric→CLIP scoring dim).
- `brew install ffmpeg`
- Optional, by pipeline stage: Ollama + `qwen2.5vl:7b` (vision enrichment),
  demucs + openai-whisper (lyrics extraction).

> **Known gap on the M5 Max (2026-06):** no `.venv` was ever created after
> the machine migration, so renders since late May ran on system python3
> without torch — scoring dim 11 (lyric→CLIP similarity) silently skipped.
> Run `setup.sh` on the new host to restore parity with pre-migration renders.

## 4. Post-move verification

1. `python3 smoke_test.py` — scoring/selection/bench invariants, no media needed.
2. `python3 scripts/capture_perceptual_baseline.py` — selection-only run over
   the copied JSON corpus (~30 s/set, no media files needed); compare against
   `bench/perceptual_baselines.json`.
3. Short render with the frame-exact verifier:
   ```bash
   python3 assemble_v2.py --set waiting-to-begin-2024 --segment 300 420 -o sets/sync_check.mp4
   ```
   Expect `Verifying frame counts... all N clips frame-exact`. Any off-plan
   clip means an ffmpeg/codec difference on the new host — see lessons.md
   "# Render side".
4. Full render of one set; compare the `[4b/5] Perceptual diversity` block
   against `bench/perceptual_results.json`.
