# Agent Guide

Context for AI agents (Claude Code, Copilot, etc.) working on this project.

## What This Project Does

Generates algorithmic music videos for DJ sets. Source video clips are analyzed for visual features (motion, color, brightness, style) and matched against audio features (energy, bass, spectral content, tempo) to produce beat-synced video edits.

## Key Files to Read First

1. **`assemble_v2.py`** — The main assembler. Start here to understand the full pipeline: scene database construction, phrase feature extraction, adaptive merging, scoring, diversity enforcement, and ffmpeg assembly.

2. **`ARCHITECTURE.md`** — System design, data flow, file formats, scoring weights, and diversity parameters.

3. **`analyze_dj_set_deep.py`** — Audio analysis that produces the `.deep-analysis.json` files consumed by the assembler.

4. **`enrich_analyses.py`** — Vision model enrichment that adds semantic labels to video scenes.

5. **`extract_lyrics.py`** — Demucs vocal separation + Whisper transcription. Produces `.lyrics.json` with timestamped lyric segments and keyword indices mapped to phrases.

6. **`bench/` + `bench/TESTPLAN.md`** — the vision-engine bake-off harness (CLI `bench_run.py`). Scores VLMs (Ollama / MLX / Claude) on the footage to pick the production enrichment engine; the winner feeds `enrich_analyses.py --sampling-plan`. Read `bench/TESTPLAN.md` first — it has the lineup, metrics, and a **status/resume** section. Bake-off is **complete**; production = `mlx-qwen7b` + `mlx-internvl` parse-fail fallback (V1 cascade), full corpus rescanned 2026-05-28.

7. **`audio_field_audit.md`** — the per-field contract for `.deep-analysis.json` (schema 2.2.0): who computes each field, who reads it, byte cost, recommendation. Authoritative reference for what stays in the JSON. See also `lessons.md` "# Audio side" for the underlying rationale.

## Common Tasks

### Adding a new DJ set

1. Add entry to `SET_CONFIGS` in `assemble_v2.py` with analysis file, audio path, output name, and optional `style_hints`
2. Run `analyze_dj_set_deep.py` on the audio file to produce the `.deep-analysis.json`
3. Run `assemble_v2.py --set <name>`

### Extracting lyrics for a set

1. Install deps: `pip install demucs openai-whisper`
2. Run `python3 extract_lyrics.py --set <name>` (or `--all`)
3. Demucs separates vocals (~5-15 min per hour of audio), Whisper transcribes
4. Output: `sets/<name>.lyrics.json` — automatically loaded by assembler on next run
5. Use `--whisper-model medium` or `large` for better accuracy at cost of speed

### Adding source videos

1. Place video files in the source directory
2. Run the video analyzer to produce `.analysis.json`
3. Run `enrich_analyses.py` to add semantic descriptions
4. Run `scan_text_fast.py` to flag English text
5. Run `generate_clip_embeddings.py` to produce `.clip_embeddings.json` sidecars
6. Run `flag_quality.py` to produce `.quality.json` sidecars (cheap; reads JSON only)
7. The assembler will automatically pick up new scenes (and their CLIP/quality sidecars) on next run

### Running the quality pass

`flag_quality.py` flags dead scenes (black, blown-out, frozen, flat, near-duplicate)
from the existing enriched timelines — no video re-decoding, no archive drive needed.

1. `python3 flag_quality.py` (or `--video <substring>`, `--skip-existing`)
2. Writes `enriched/<src>.quality.json` (per-scene `quality_score` + `flags`)
3. The assembler applies it as a soft penalty (scoring dim 12, `QUALITY_PENALTY_WEIGHT`)
4. Thresholds + per-flag penalties are tunable constants at the top of `flag_quality.py`.
   `near_dup` is a deliberately gentle nudge (a loop is usable footage); black/blown/frozen
   are strong. The assembler logs the quality of its selection for tuning.

### Choosing a vision backend

`enrich_analyses.py --backend {ollama|claude-cli|anthropic-api}` (or `$GHOST_VISION_BACKEND`),
`--model <name>`:

- **`ollama`** (default) — local qwen2.5vl, free + offline. Best for bulk overnight runs.
- **`claude-cli`** — Claude via the Claude Code CLI on your **subscription** (zero marginal cost).
  Best for quality and hard-case re-rating. Ensure `ANTHROPIC_API_KEY` is **unset** (the health
  check warns if it's set, since it would route to metered API billing); for unattended batches
  use `claude setup-token` → export `CLAUDE_CODE_OAUTH_TOKEN`.
- **`anthropic-api`** — metered Anthropic SDK (`pip install anthropic`, `ANTHROPIC_API_KEY`).
  Opt-in only, when you want API speed.

`--reenrich-flagged` re-runs only flagged scenes (failed validation/parse, or `quality_score <
--quality-threshold`) — run it on `claude-cli` to clean up the cheap local model's hard cases at
no marginal cost.

### Running the vision-engine bake-off

The `bench/` harness compares VLM engines on a representative footage subset to pick the
production enrichment engine. It fixes the prior 2.6%-coverage / generic-tag problem with an
**adaptive sampler** (one description per *distinct visual state*, not 1 frame/30s) and folds
text detection into the vision pass (`has_english_text`). Full plan + status: `bench/TESTPLAN.md`.

1. `python3 bench_run.py health` — backend availability (mlx reports "not installed" until `pip install mlx-vlm`).
2. `python3 bench_run.py plan` — build `enriched/<stem>.sampling_plan.json` + print the coverage histogram.
3. `python3 bench_run.py run --engines ollama-qwen7b,mlx-gemma3,baseline,clip-ceiling [--judge]` — run; resumable/pausable.
4. `python3 bench_run.py compare [--video SUBSTR]` — CLIP-anchored scoreboard / side-by-side description dump.
5. Pilot rescan with the winner: `python3 enrich_analyses.py --backend <winner> --sampling-plan --video <clip>`.

Engines are defined in `bench/config.py` (`ENGINES` / `DEFAULT_ENGINES`). Composite rank is
**objective metrics only** (within-video Recall@1 + coverage + text-F1 + adjacent discriminability);
the LLM-judge is audit-only. Models load from the local archive `<local-model-archive>` or HF/Ollama;
prefer the archive before downloading. `claude-cli` needs `env -u ANTHROPIC_API_KEY` (subscription).

### Searching scenes by description

`python3 query_scenes.py "golden liquid high energy" [--top N] [--min-quality F |
--quality-weight] [--diversity λ] [--extract]` — natural-language search over the existing CLIP
embeddings. `--diversity` applies MMR suppression so near-identical scenes don't dominate;
`--extract` ffmpeg-dumps top hits to `/tmp` for preview (needs the archive drive).

### Tuning clip selection

All parameters are constants at the top of `assemble_v2.py`:
- **Hard-block diversity**: `VARIETY_WINDOW`, `SCENE_VARIETY_WINDOW`, `MAX_SOURCE_USAGE_MULT`, `REUSE_PENALTY`, `TOP_CANDIDATES`
- **Perceptual MMR rerank**: `MMR_LAMBDA` (0.5), `MMR_POOL` (80), `MMR_RECENT_WINDOW` (10), `MMR_DIAGNOSTIC_PHRASES` (10)
- **Near-duplicate hard skip threshold**: `NEAR_DUP_SIM` (0.97, in `flag_quality.py`)
- **Phrase merging**: `MERGE_ENERGY_THRESHOLD`, `MERGE_ENERGY_DELTA`, `MAX_MERGED_DURATION`
- **Scene filtering**: `MIN_SCENE_DURATION`, `MAX_SCENE_DURATION`
- **Per-set creative direction**: `style_hints` dict in `SET_CONFIGS`

The MMR pool ceiling matters more than `MMR_LAMBDA` for tuning — see `lessons.md` "# Selection side". The diagnostic log at `bench/mmr_diagnostics.log` (gitignored) captures per-candidate scoring + cosine penalty for the first 10 phrases of each run; spot-check that before changing constants.

### Auditing perceptual diversity

- `scripts/audit_repeats.py` — Phase 0 forensic audit. Classifies the chronological selection against three substrates: literal `(source, scene_index)` repeats (would indicate constraint-code bug), within-source perceptual (cosine ≥ 0.95 within 5 phrases), cross-source perceptual (cosine ≥ 0.90 within 5 phrases). Emits a histogram across (window × threshold) cells. Use to confirm a "constant repeats" complaint *before* designing a fix.
- `scripts/capture_perceptual_baseline.py` — selection-only run (no ffmpeg), dumps `compute_perceptual_diversity()` metrics for one or more sets to a committed JSON file. ~30s/set vs ~12 min full render; useful for tuning. Selection-only SKIPS lyrics-CLIP scoring, so use the assembler's `[4b/5]` block from a full render for the deliverable acceptance numbers.

### Tuning style hints

Each set in `SET_CONFIGS` can have a `style_hints` dict:

```python
"style_hints": {
    "preferred_styles": ["abstract", "particle", "fractal"],  # +0.5 bonus
    "avoid_styles": ["cinematic"],                             # -1.0 penalty
    "color_preference": "cool",          # bias warm/cool/neutral
    "preferred_sources": ["FreeFormLiquid"],  # +0.3 bonus
    "avoid_sources": ["show me lyrics"],     # -2.0 penalty (effective exclude)
    "energy_response": "follow",         # or "contrast" to invert
}
```

## Data Locations

| Data | Path | Format |
|------|------|--------|
| Source videos | `/Volumes/archive/3000/3100/visuals/library/<collection>/` (canonical root; see `media_paths.py`, env-overridable) | MP4/AVI/MOV |
| DJ set audio | `/Volumes/archive/3000/3100/sets/<set-name>/` | MP3 |
| Video analyses | `*.analysis.json` (project root) | JSON |
| Enriched analyses | `enriched/*.enriched.json` | JSON |
| Text flags | `text_flags/*.text_flags.json` | JSON |
| Audio analyses | `sets/*.deep-analysis.json` | JSON |
| Lyrics | `sets/*.lyrics.json` | JSON |
| Demucs stems | `demucs_output/htdemucs/*/vocals.wav` | WAV |
| Output videos | `sets/*_music_video.mp4` | MP4 |

## Important Patterns

- **All timelines are sampled at 8 Hz** (both audio and video analysis)
- **Scenes are the atomic unit** for clip selection — not frames, not whole videos
- **Adaptive enrichment** — `enrich_analyses.py --sampling-plan` covers one description per distinct visual state (not 1 frame per 30s as in pre-bake-off pipelines)
- **Text filtering is per-second** — if any second in a scene range has text, the scene is excluded
- **The random seed is fixed** (`--seed 42` default) for reproducible builds. Change seed to get different clip selections with same scoring.
- **Phrase merging happens after feature extraction** — the merge step combines adjacent low-energy phrases into longer clip holds
- **Lyrics are optional** — if no `.lyrics.json` exists for a set, the assembler skips lyric-visual matching gracefully
- **Whisper hallucinations** — common on instrumental sections; the extractor filters these (e.g. "thank you for watching", "[music]"). If you see bad lyrics data, check the hallucination patterns in `extract_lyrics.py`
- **`.deep-analysis.json` schema 2.2.0** carries a non-blocking `beat_quality` block with IOI outliers, octave-doubling rate + max run length, and metronomic deviation. Warnings print on the analyzer console but the file always writes; assembler reads `bpm_timeline.confidence` (with `.get()` fallback for pre-2.1.0 files).
- **`assemble_v2.py` scoring constants are tuning-sensitive** — the weights in `score_scene` co-evolved with the diversity windows. Land new scoring components at <0.5 weight or as multipliers on existing terms by convention.
- **Diversity is now perceptual, not just metadata** — `select_clips` applies a near_dup hard skip (Phase A) AND an MMR re-rank over a top-80 score pool (Phase C). Metadata-level windows (`VARIETY_WINDOW`, `SCENE_VARIETY_WINDOW`) are still in place; MMR closes the gap when "different filename" still meant "same image." See `lessons.md` "# Selection side".
- **Hash-suffix duplicate filenames are silent corpus poisoning** — analyzer reruns can produce both `X.webm` and `X-abc123.webm` as separate "sources." Audit periodically with `scripts/audit_repeats.py` or grep `enriched/` for paired `-[a-f0-9]{5,6}` suffixes. Dedup offsite (don't just rm without backup).

## Dependencies

Requires: Python 3.10+, numpy, librosa, scipy, FFmpeg, Ollama (for enrichment/text scanning only), demucs + openai-whisper (for lyrics extraction only), open_clip + torch (for CLIP embeddings and scene search).

The `claude-cli` vision backend needs the Claude Code CLI (uses your subscription); the `anthropic-api` backend needs `pip install anthropic` + `ANTHROPIC_API_KEY` (metered, optional).

The assembler (`assemble_v2.py`) only needs numpy and FFmpeg — it reads pre-computed JSON files. All sidecars are optional and degrade gracefully: no `.lyrics.json` skips lyric matching, no `.clip_embeddings.json` skips CLIP similarity, no `.quality.json` leaves every scene at `quality_score = 1.0` (no penalty).

## Things to Watch Out For

- Source videos are on an external drive (`/Volumes/archive/`). The drive must be mounted for analysis, enrichment, text scanning, and assembly.
- Some source filenames contain Japanese characters and special characters. Path matching uses fuzzy substring comparison to handle Unicode normalization differences.
- The `.deep-analysis.json` files are **13-15 MB each on schema 2.2.0** (down from 37-44 MB on 2.0.0 — see Phase 4 pruning in `audio_field_audit.md`).
- Generated music videos are 2-3 GB each.
- Vision model inference is slow (~45-65 seconds per frame via Ollama; MLX is ~1.9× faster). The fast text scanner mitigates this with adaptive sampling.
- `sets/` may be a **symlink** into the archive drive on dev machines (`ln -s /Volumes/archive/temp/media-analysis/sets sets`). The assembler resolves `analysis` paths via `BASE_DIR / "sets" / ...`. Untracked, gitignored.
- Moving to a new machine? **[MIGRATION.md](MIGRATION.md)** has the full copy list (which gitignored data is expensive to regenerate), the hardcoded paths to update (`SOURCE_DIR` / `SETS_DIR` in `assemble_v2.py`), and a post-move verification checklist including the frame-exact render check.
