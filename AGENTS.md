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
5. The assembler will automatically pick up new scenes on next run

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

Requires: Python 3.10+, numpy, librosa, scipy, FFmpeg, Ollama (for enrichment/text scanning only), demucs + openai-whisper (for lyrics extraction only).

The assembler (`assemble_v2.py`) only needs numpy and FFmpeg — it reads pre-computed JSON files. Lyrics are optional; if no `.lyrics.json` exists, the lyric matching term is simply skipped.

## Things to Watch Out For

- Source videos are on an external drive (`/Volumes/archive/`). The drive must be mounted for analysis, enrichment, text scanning, and assembly.
- Some source filenames contain Japanese characters and special characters. Path matching uses fuzzy substring comparison to handle Unicode normalization differences.
- The `.deep-analysis.json` files are 37-44 MB each (large timelines at 8 Hz across 1+ hour sets).
- Generated music videos are 2-3 GB each.
- Vision model inference is slow (~45-65 seconds per frame via Ollama). The fast text scanner mitigates this with adaptive sampling.
