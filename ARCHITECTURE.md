# Architecture

## System Overview

The system has four independent analysis pipelines that feed into a single assembly step. Each pipeline produces JSON artifacts that are consumed downstream.

```
┌─────────────────────────────────────────────────────────────┐
│                     SOURCE VIDEOS                           │
│  /Volumes/archive/3000/3100/visuals/raw visuals footage/    │
│  57 videos: VJ loops, music videos, drone, motion graphics  │
└────────┬──────────────────────┬─────────────────────────────┘
         │                      │
    ┌────▼────┐           ┌─────▼──────┐
    │ analyze │           │   enrich   │
    │ (scene  │           │  (vision   │
    │ detect, │           │   model)   │
    │ motion, │           │            │
    │ color)  │           │ qwen2.5vl  │
    └────┬────┘           └─────┬──────┘
         │                      │
   .analysis.json         .enriched.json
         │                      │
         │    ┌────────────┐    │
         │    │ scan_text  │    │
         │    │  (English  │    │
         │    │   text     │    │
         │    │  detect)   │    │
         │    └─────┬──────┘    │
         │          │           │
         │   .text_flags.json   │
         │          │           │
┌────────▼──────────▼───────────▼──────┐
│          SCENE DATABASE              │
│  Merges: scenes + motion + color +   │
│  brightness + semantics - text       │
│  ~19,000 usable scenes              │
└────────────────┬─────────────────────┘
                 │
                 │   ┌──────────────────────────┐
                 │   │   DJ SET AUDIO            │
                 │   │   analyze_dj_set_deep.py  │
                 │   │                           │
                 │   │   7-band energy, HPSS,    │
                 │   │   onsets, chroma, BPM,    │
                 │   │   spectral, phrases       │
                 │   └─────────┬────────────────┘
                 │             │
                 │      .deep-analysis.json
                 │             │
                 │   ┌─────────▼────────────────┐
                 │   │   LYRICS EXTRACTION       │
                 │   │   extract_lyrics.py        │
                 │   │                           │
                 │   │   Demucs → vocal isolation │
                 │   │   Whisper → transcription  │
                 │   │   → keyword extraction     │
                 │   └─────────┬────────────────┘
                 │             │
                 │        .lyrics.json
                 │             │
          ┌──────▼─────────────▼──────┐
          │      ASSEMBLE v2          │
          │                           │
          │  1. Build scene database  │
          │  2. Extract phrase feats  │
          │  3. Adaptive phrase merge │
          │  3b. Load lyrics keywords │
          │  4. Score + select clips  │
          │  5. ffmpeg extract/concat │
          └───────────┬───────────────┘
                      │
               Music Video (.mp4)
```

## Data Flow & File Formats

### Video Analysis (`.analysis.json`)

Produced by the initial video scanner. Contains raw measurements at ~8 Hz:

```
{
  "file": { "name", "path", "duration_sec" },
  "scenes": [ { scene_index, start_sec, end_sec, duration_sec } ],
  "motion": { "timeline": [ { time_sec, mean_motion, max_motion } ] },
  "brightness": { "timeline": [ { time_sec, brightness, contrast } ] },
  "colors": { "timeline": [ { time_sec, dominant_colors: [{rgb, hex, percentage}] } ] },
  "metadata": { "duration_sec", "fps", "resolution", "analyzer_version" }
}
```

### Enriched Analysis (`.enriched.json`)

Extends analysis with vision model semantics. Same structure as above, plus:

```
{
  "frame_analyses": [
    {
      "scene_index": 5,
      "time_sec": 30.0,
      "analysis": {
        "visual_description": "...",
        "color_palette": { "dominant_colors", "color_relationship", "temperature" },
        "composition": { "layout", "implied_motion", "motion_description" },
        "visual_style": "abstract|geometric|fractal|liquid|cinematic|...",
        "mood": { "tone": "...", "energy": "calm|building|moderate|intense|chaotic" },
        "content_tags": ["tag1", "tag2"]
      }
    }
  ],
  "enrichment_metadata": { "model", "frames_analyzed", "inference_times" }
}
```

Sampling: 1 frame per 30 seconds, 4-30 frames per video. Semantic data is propagated to nearby scenes during database building.

### Text Flags (`.text_flags.json`)

Per-second flags for English text presence:

```
{
  "video": "name",
  "source": "/path/to/video",
  "duration_sec": 120.5,
  "flags": {
    "0": { "time_sec": 0.0, "has_english_text": true, "description": "..." },
    "60": { "time_sec": 60.0, "has_english_text": false }
  },
  "status": "complete",
  "scan_method": "adaptive|1fps",
  "text_frame_count": 42,
  "text_ratio": 0.35
}
```

Two scanners exist:
- `scan_text.py` — 1fps exhaustive scan (used for initial 18 target videos)
- `scan_text_fast.py` — adaptive scan: every 60s coarse, binary-search boundaries when text found. ~60x faster for text-free videos.

### Deep Audio Analysis (`.deep-analysis.json`)

Multi-dimensional audio feature extraction at 8 Hz:

```
{
  "tracks": [
    {
      "title", "start_sec", "end_sec",
      "bpm": { "mean", "min", "max" },
      "energy": { "mean", "peak", "std" },
      "bands": { "sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance" },
      "harmonic_percussive": { "harmonic_mean", "percussive_mean", "balance" },
      "spectral": { "centroid_mean_hz", "flatness_mean", "brightness", "texture" }
    }
  ],
  "multiband_energy": [ { "time_sec", "sub_bass", "bass", ..., "total_rms" } ],
  "hpss_timeline": [ { "time_sec", "harmonic", "percussive" } ],
  "onsets": [ { "time_sec", "strength" } ],
  "spectral_timeline": [ { "time_sec", "centroid_hz", "flux" } ],
  "bpm_timeline": [ { "time_sec", "bpm" } ],
  "phrases": {
    "four_bar": [ { "start_sec", "end_sec", "beat_count", "energy_mean", "energy_shape" } ],
    "eight_bar": [...],
    "sixteen_bar": [...]
  }
}
```

### Lyrics (`.lyrics.json`)

Extracted via Demucs (vocal separation) + Whisper (transcription):

```
{
  "set_name": "blue-sky-genesis-2025",
  "demucs_model": "htdemucs",
  "whisper_model": "base",
  "stats": { "total_segments", "total_keywords", "vocal_coverage_sec" },
  "all_keywords": ["fire", "sky", "love", ...],
  "segments": [
    {
      "start_sec": 45.2, "end_sec": 48.7,
      "text": "set the sky on fire",
      "keywords": ["set", "sky", "fire"],
      "confidence": 0.35,
      "words": [ { "word": "sky", "start": 46.1, "end": 46.4 } ]
    }
  ],
  "track_summary": {
    "Track Title": { "segment_count": 5, "top_keywords": ["sky", "fire"] }
  },
  "phrase_lyrics": {
    "four_bar": { "0": ["sky", "fire"], "3": ["love", "night"] },
    "eight_bar": { ... },
    "sixteen_bar": { ... }
  }
}
```

The `phrase_lyrics` index maps phrase indices to keyword lists, consumed directly by the assembler's scoring function.

## Assembly Pipeline Detail

### Scene Database Construction

1. Load all `.enriched.json` files from `enriched/`
2. For each video's scenes, merge in:
   - Motion data (optical flow mean/peak from timeline)
   - Brightness and contrast
   - Dominant colors → computed temperature (warm/cool) and saturation
   - Semantic data from nearest frame analysis (visual style, mood, tags)
3. Filter out scenes < 1.5s or > 120s
4. Filter out scenes overlapping with English text flags
5. Result: ~19,000 candidate scenes with full feature vectors

### Phrase Feature Extraction

1. Load `.deep-analysis.json` for the target DJ set
2. Use 4-bar phrases as base units (~7-8s at 120 BPM)
3. Per phrase, extract:
   - Energy (normalized RMS)
   - Bass ratio (sub_bass + bass vs total)
   - Audio brightness (spectral centroid)
   - Percussive ratio (from HPSS)
   - BPM, energy shape

### Adaptive Phrase Merging

Adjacent 4-bar phrases are merged when:
- Both below energy threshold (0.35)
- Energy delta < 0.12 (stable, not transitioning)
- Same track (no cross-track merges)
- Previous phrase shape is "sustain" or "decay"
- Combined duration < 45 seconds

This produces longer clip holds for breakdowns and ambient passages, while keeping fast cuts for drops and buildups.

### Scoring (per scene vs. per phrase)

Each scene is scored against each phrase on 8 dimensions:

| Dimension | Weight | Logic |
|-----------|--------|-------|
| Motion-energy | 1.0 | Visual motion ↔ audio energy |
| Brightness | 0.8 | Visual brightness ↔ spectral centroid |
| Color temperature | 0.7 | Warm colors ↔ bass-heavy audio |
| Duration fit | 0.9 | Scene length ↔ target (3-15s based on energy) |
| Saturation | 0.5 | Color saturation ↔ energy level |
| Contrast | 0.5 | Visual contrast ↔ percussive ratio |
| Semantic | 0.1-0.4 | Mood energy match (calm↔calm, intense↔intense) |
| Style hints | variable | Per-set creative direction bonus/penalty |
| Lyrics match | 0.25/match | Lyric keywords matched against clip content_tags (cap 0.8) |

### Diversity Enforcement

| Mechanism | Value | Effect |
|-----------|-------|--------|
| Source variety window | 15 | Hard block: same source video can't repeat within 15 clips |
| Scene variety window | 30 | Hard block: exact same scene can't repeat within 30 clips |
| Reuse penalty | 0.5 | Score *= 1/(1 + 0.5 * uses) — steep diminishing returns |
| Usage ceiling | 2.0x | No source exceeds (total_clips / n_sources) * 2 uses |
| Random pool | 20 | Weighted random from top 20 candidates (not top 1) |

### Video Assembly

1. Extract each selected clip segment via ffmpeg (scale to 1080p, pad black)
2. Concatenate with 8-frame crossfades
3. Overlay DJ set audio starting at the first phrase's timestamp
4. Output: MP4 @ 30fps, AAC 192kbps

## External Dependencies

| Tool | Version | Purpose |
|------|---------|---------|
| FFmpeg | any recent | Video extraction, scaling, concat, audio mux |
| Ollama | 0.3+ | Local vision model inference |
| qwen2.5vl:7b | — | Vision-language model for enrichment + text detection |
| numpy | 1.24+ | Numerical operations |
| librosa | 0.10+ | Audio feature extraction (deep analysis) |
| scipy | 1.10+ | Signal processing |
| demucs | 4.0+ | Vocal separation from DJ set audio |
| openai-whisper | latest | Speech-to-text transcription of isolated vocals |

## Directory Layout

```
ghost-media/                 # (the generated data dirs/files below are gitignored)
├── media_analyzer/          # installable toolkit package (CLI: media-analyzer)
├── pyproject.toml  setup.sh  examples/
├── *.analysis.json          # 73 raw video analyses
├── enriched/
│   └── *.enriched.json      # 53 vision-enriched analyses
├── text_flags/
│   └── *.text_flags.json    # 18 text-scanned (more pending)
├── sets/
│   ├── *.deep-analysis.json # 5 DJ set audio analyses
│   ├── *_music_video.mp4    # Generated music videos
│   └── *_mv_build/          # Intermediate build artifacts
├── _manifest.json           # Processing manifest for all source videos
├── extract_lyrics.py        # Demucs + Whisper lyrics extraction
├── demucs_output/           # Demucs separated stems (vocals.wav etc)
├── assemble_v2.py           # Main assembler (current)
├── analyze_dj_set_deep.py   # Deep audio analysis
├── enrich_analyses.py       # Vision model enrichment
├── scan_text_fast.py        # Adaptive text scanner
├── scan_text.py             # Original 1fps text scanner
├── render_sizzle.py         # Audio-reactive visualizer
├── vision_model_contest.py  # Vision model evaluation
├── cleanup_enrichment.py    # Post-enrichment fixes
├── analyze_dj_set.py        # Basic audio analysis (legacy)
└── assemble_music_video.py  # v1 assembler (legacy)
```
