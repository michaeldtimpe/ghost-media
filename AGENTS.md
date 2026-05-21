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

### Searching scenes by description

`python3 query_scenes.py "golden liquid high energy" [--top N] [--min-quality F |
--quality-weight] [--diversity λ] [--extract]` — natural-language search over the existing CLIP
embeddings. `--diversity` applies MMR suppression so near-identical scenes don't dominate;
`--extract` ffmpeg-dumps top hits to `/tmp` for preview (needs the archive drive).

### Tuning clip selection

All parameters are constants at the top of `assemble_v2.py`:
- **Diversity**: `VARIETY_WINDOW`, `SCENE_VARIETY_WINDOW`, `REUSE_PENALTY`, `MAX_SOURCE_USAGE_MULT`, `TOP_CANDIDATES`
- **Phrase merging**: `MERGE_ENERGY_THRESHOLD`, `MERGE_ENERGY_DELTA`, `MAX_MERGED_DURATION`
- **Scene filtering**: `MIN_SCENE_DURATION`, `MAX_SCENE_DURATION`
- **Per-set creative direction**: `style_hints` dict in `SET_CONFIGS`

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
| Source videos | `/Volumes/archive/3000/3100/visuals/raw visuals footage/` | MP4/AVI/MOV |
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
- **Enrichment is sparse** — 1 frame per 30 seconds, propagated to nearby scenes
- **Text filtering is per-second** — if any second in a scene range has text, the scene is excluded
- **The random seed is fixed** (`--seed 42` default) for reproducible builds. Change seed to get different clip selections with same scoring.
- **Phrase merging happens after feature extraction** — the merge step combines adjacent low-energy phrases into longer clip holds
- **Lyrics are optional** — if no `.lyrics.json` exists for a set, the assembler skips lyric-visual matching gracefully
- **Whisper hallucinations** — common on instrumental sections; the extractor filters these (e.g. "thank you for watching", "[music]"). If you see bad lyrics data, check the hallucination patterns in `extract_lyrics.py`

## Dependencies

Requires: Python 3.10+, numpy, librosa, scipy, FFmpeg, Ollama (for enrichment/text scanning only), demucs + openai-whisper (for lyrics extraction only), open_clip + torch (for CLIP embeddings and scene search).

The `claude-cli` vision backend needs the Claude Code CLI (uses your subscription); the `anthropic-api` backend needs `pip install anthropic` + `ANTHROPIC_API_KEY` (metered, optional).

The assembler (`assemble_v2.py`) only needs numpy and FFmpeg — it reads pre-computed JSON files. All sidecars are optional and degrade gracefully: no `.lyrics.json` skips lyric matching, no `.clip_embeddings.json` skips CLIP similarity, no `.quality.json` leaves every scene at `quality_score = 1.0` (no penalty).

## Things to Watch Out For

- Source videos are on an external drive (`/Volumes/archive/`). The drive must be mounted for analysis, enrichment, text scanning, and assembly.
- Some source filenames contain Japanese characters and special characters. Path matching uses fuzzy substring comparison to handle Unicode normalization differences.
- The `.deep-analysis.json` files are 37-44 MB each (large timelines at 8 Hz across 1+ hour sets).
- Generated music videos are 2-3 GB each.
- Vision model inference is slow (~45-65 seconds per frame via Ollama). The fast text scanner mitigates this with adaptive sampling.
