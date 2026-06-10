# Media Analysis — Algorithmic Music Video Generator

Generates music videos for DJ sets by matching visual clips to audio features. The pipeline analyzes a library of source videos (VJ loops, music videos, drone footage, motion graphics) and a DJ set's audio, then algorithmically selects and assembles clips that sync to the music's energy, tempo, color temperature, and mood.

## Quick Start

### Prerequisites

- Python 3.10+ with `numpy`, `librosa`, `scipy`
- [FFmpeg](https://ffmpeg.org/) for video extraction/assembly
- [Ollama](https://ollama.com/) with `qwen2.5vl:7b` for vision enrichment and text detection
- [Demucs](https://github.com/facebookresearch/demucs) + [Whisper](https://github.com/openai/whisper) for lyrics extraction (`pip install demucs openai-whisper`)
- Source videos in `/Volumes/archive/3000/3100/visuals/raw visuals footage/`
- DJ set audio in `/Volumes/archive/3000/3100/sets/<set-name>/`

### Pipeline Overview

```
Source Videos          DJ Set Audio
     |                      |
     v                      v
1. analyze (scan)    2. analyze_dj_set_deep
     |                      |
     v                      v
3. enrich (vision)   3b. extract_lyrics
     |                (Demucs + Whisper)
     v                      |
4. scan_text_fast           |
     |                      |
     +----------+-----------+
                |
                v
         5. assemble_v2
                |
                v
          Music Video (.mp4)
```

### Running Each Step

```bash
# 1. Analyze source videos (scene detection, motion, color, brightness)
#    Produces .analysis.json per video — already done for 73 videos

# 2. Deep audio analysis of a DJ set
python3 analyze_dj_set_deep.py --input "/path/to/set.mp3" --tracklist tracklist.txt

# 3. Enrich with vision model descriptions (requires Ollama running)
python3 enrich_analyses.py

# 4. Scan for English text (adaptive — fast)
python3 scan_text_fast.py              # scan all unscanned videos
python3 scan_text_fast.py --status     # show scan progress
python3 scan_text_fast.py --only drone # only scan videos matching "drone"

# 4b. Extract lyrics (Demucs vocal separation + Whisper transcription)
python3 extract_lyrics.py --set blue-sky-genesis-2025   # one set
python3 extract_lyrics.py --all                          # all sets
python3 extract_lyrics.py --status                       # show status
python3 extract_lyrics.py --whisper-model medium          # higher accuracy

# 5. Assemble music video
python3 assemble_v2.py --set blue-sky-genesis-2025    # one set
python3 assemble_v2.py --all                           # all sets
python3 assemble_v2.py --set blue-sky-genesis-2025 --phrase-bars 8  # longer phrases
python3 assemble_v2.py --set blue-sky-genesis-2025 --segment 120 360  # time range
```

## Current Sets

| Set Name | Tracks | Status |
|----------|--------|--------|
| blue-sky-genesis-2025 | 42 | Rendered |
| boxing-day-2025 | ~40 | Rendered |
| cheerleader-exodus-2025 | ~40 | Rendered |
| waiting-to-begin-2024 | ~35 | Rendered |
| will-call-2025 | ~40 | Rendered |

## Key Features

- **Audio-visual matching**: Motion, brightness, color temperature, saturation, and contrast are all scored against audio features (energy, spectral centroid, bass ratio, percussive balance)
- **Adaptive phrase merging**: Low-energy sustained sections automatically get longer clip holds; high-energy drops get fast cuts
- **Diversity enforcement**: Source variety window (15 phrases), scene variety window (30), usage ceiling per source, steep reuse penalty
- **Style hints**: Per-set creative direction — preferred/avoided visual styles, color palette bias, source preferences
- **Text filtering**: English text detected via vision model is excluded from clip selection
- **Semantic enrichment**: Vision model provides visual style, mood, content tags for smarter matching
- **Lyrics matching**: Vocal lyrics extracted via Demucs + Whisper; keywords matched against clip content tags for contextual visual pairing

## Configuration

Edit `SET_CONFIGS` in `assemble_v2.py` to add sets or tune style hints. See the style hints reference comment in that file for all available options.

Key tuning parameters at the top of `assemble_v2.py`:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `VARIETY_WINDOW` | 15 | No same source video within N clips |
| `SCENE_VARIETY_WINDOW` | 30 | No same scene within N clips |
| `TOP_CANDIDATES` | 20 | Random pool size for clip selection |
| `REUSE_PENALTY` | 0.5 | Per-reuse score multiplier decay |
| `MAX_SOURCE_USAGE_MULT` | 2.0 | Usage cap = (total_clips / sources) * this |
| `MERGE_ENERGY_THRESHOLD` | 0.35 | Merge phrases below this energy |
| `MERGE_ENERGY_DELTA` | 0.12 | Merge if energy delta < this |
| `MAX_MERGED_DURATION` | 45.0 | Max merged phrase duration (seconds) |

## File Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system design.

## Other Scripts

- `render_sizzle.py` — Reactive audio visualization (spectrum, particles, waveform)
- `vision_model_contest.py` — Evaluation framework for comparing vision models
- `assemble_music_video.py` — v1 assembler (superseded by assemble_v2.py)
- `analyze_dj_set.py` — Basic audio analysis (superseded by deep version)
- `cleanup_enrichment.py` — Fix artifacts in enrichment data
