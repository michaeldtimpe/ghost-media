# media-analyzer

Local media analysis toolkit that produces ML-friendly JSON. Designed as a building block for automated music video editing — analyze your audio and footage library, then use the structured output to make intelligent matching and sync decisions.

## Features

### Audio Analysis
- **Tempo & beats** — BPM estimation, precise beat timestamps
- **Song sections** — Structural segmentation with boundary timestamps
- **Energy curve** — RMS loudness over time, dynamic range
- **Spectral features** — Brightness, warmth, spectral centroid, bandwidth, zero-crossing rate
- **Key detection** — Musical key and mode (major/minor) via Krumhansl-Kessler profiles
- **Mood inference** — Heuristic valence/arousal estimation with descriptive tags

### Video Analysis
- **Metadata** — Duration, resolution, FPS, codecs, file size
- **Scene detection** — Automatic scene boundary timestamps (ffmpeg + OpenCV fallback)
- **Dominant colors** — Per-frame color extraction via k-means clustering
- **Motion intensity** — Optical flow analysis over time
- **Brightness & contrast** — Luminance curves for visual matching

## Quick Start

### Prerequisites
- Python 3.10+
- ffmpeg (`brew install ffmpeg`)

### Install

```bash
cd media-analyzer
chmod +x setup.sh
./setup.sh
source .venv/bin/activate
```

Or manually:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

### Single File Analysis

```bash
# Analyze an audio file
media-analyzer audio ~/music/track.mp3

# Analyze a video file
media-analyzer video ~/footage/clip.mp4

# Custom output path
media-analyzer audio ~/music/track.mp3 -o ~/analysis/track.json

# Skip motion analysis for faster video processing
media-analyzer video ~/footage/clip.mp4 --skip-motion
```

### Batch Analysis

```bash
# Analyze everything in a directory
media-analyzer batch ~/media-library/

# Recursive scan with 8 parallel workers
media-analyzer batch ~/media-library/ -r -w 8

# Audio files only
media-analyzer batch ~/media-library/ --audio-only -r

# Video files only, skip motion for speed
media-analyzer batch ~/media-library/ --video-only --skip-motion -r

# Custom output directory
media-analyzer batch ~/media-library/ -o ~/analysis-output/ -r
```

### Output

Each file produces a `.analysis.json` file. Batch mode also creates a `_batch_manifest.json` with a summary of all processed files.

Default output location:
- Single file: `<input>.analysis.json` (next to the source file)
- Batch mode: `<directory>/analysis_output/` (configurable with `-o`)

## JSON Schema

See `examples/` for complete example outputs.

### Audio Output Fields

| Field | Description |
|-------|-------------|
| `tempo.bpm` | Estimated beats per minute |
| `tempo.beat_times_sec` | Array of beat positions in seconds |
| `energy.rms_energy` | Energy curve sampled at 2Hz |
| `energy.dynamic_range_db` | Loudness range in dB |
| `spectral.brightness` | `dark` / `moderate` / `bright` / `very_bright` |
| `spectral.warmth` | `very_warm` / `warm` / `neutral` / `cold` |
| `key.key_label` | e.g. "A minor", with confidence score |
| `sections.sections` | Array of `{label, start_sec, end_sec, duration_sec}` |
| `mood.primary_mood` | `energetic_positive` / `calm_positive` / `intense_dark` / `somber` / `neutral` |
| `mood.valence` | 0 (negative) to 1 (positive) |
| `mood.arousal` | 0 (calm) to 1 (energetic) |

### Video Output Fields

| Field | Description |
|-------|-------------|
| `metadata.video.{width,height,fps}` | Resolution and frame rate |
| `scenes.scenes` | Array of `{scene_index, start_sec, end_sec, duration_sec}` |
| `colors.timeline[].dominant_colors` | Top N colors as RGB, hex, and percentage |
| `motion.motion_level` | `static` / `low` / `moderate` / `high` / `very_high` |
| `motion.timeline[].mean_motion` | Per-sample motion intensity |
| `brightness.average_brightness` | 0 (black) to 1 (white) |
| `brightness.brightness_level` | `very_dark` / `dark` / `moderate` / `bright` / `very_bright` |

## CLI Options

### Audio Options
| Option | Default | Description |
|--------|---------|-------------|
| `--sr` | 22050 | Analysis sample rate |
| `--compact` | false | Compact JSON output |

### Video Options
| Option | Default | Description |
|--------|---------|-------------|
| `--scene-threshold` | 30.0 | Scene sensitivity (lower = more scenes) |
| `--color-interval` | 1.0 | Color sampling interval (seconds) |
| `--motion-interval` | 0.5 | Motion sampling interval (seconds) |
| `--brightness-interval` | 0.5 | Brightness sampling interval (seconds) |
| `--n-colors` | 3 | Dominant colors per sample |
| `--skip-motion` | false | Skip optical flow (much faster) |

### Batch Options
| Option | Default | Description |
|--------|---------|-------------|
| `-r, --recursive` | false | Scan subdirectories |
| `-w, --workers` | 4 | Parallel worker processes |
| `--audio-only` | false | Only process audio files |
| `--video-only` | false | Only process video files |
| `-o, --output-dir` | `<dir>/analysis_output` | Output directory |

## Supported Formats

### Audio
mp3, wav, flac, ogg, m4a, aac, aiff, wma, opus, ape, alac

### Video
mp4, mov, mkv, avi, webm, wmv, flv, m4v, mpg, mpeg, 3gp, ogv, ts, mts

## Architecture

```
media_analyzer/
├── __init__.py
├── cli.py              # Unified CLI entry point
├── audio_analyzer.py   # Audio analysis (librosa-based)
└── video_analyzer.py   # Video analysis (OpenCV + ffmpeg)
```

The JSON output is designed to be consumed by downstream tools — a matching engine, an editing agent, or any ML pipeline. The schema is versioned (`schema_version`) so consumers can handle format evolution.

## Next Steps (Future Work)

- **Content tagging** — Add CLIP-based visual content descriptors per scene
- **Audio-video matching** — Score footage against a music track's features
- **Edit decision list (EDL)** — Auto-generate cut lists synced to beats
- **Waveform/spectrogram export** — Visual representations for UI display
