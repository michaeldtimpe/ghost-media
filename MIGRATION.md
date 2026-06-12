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
- **Source footage**: `/Volumes/archive/3000/3100/visuals/raw visuals footage` (57 videos)
- **DJ-set audio**: `/Volumes/archive/3000/3100/sets` (per-set dirs with masters)

## 2. Hardcoded paths to update

`assemble_v2.py` has two absolute constants — everything else is
`BASE_DIR`-relative:

```python
SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")
SETS_DIR   = Path("/Volumes/archive/3000/3100/sets")   # SET_CONFIGS audio paths derive from this
```

Point these at the new media locations (or mount the archive at the same
path). `find_source_video()` fuzzy-matches filenames inside `SOURCE_DIR`, so
the footage can live anywhere as long as the constant is right — the paths
recorded inside old `.analysis.json` files do **not** need to be valid.

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
