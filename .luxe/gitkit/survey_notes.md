## Survey Notes: ghost-media

### What the project IS and does

An **algorithmic music-video generator** that ingests a library of source video footage and a DJ set's audio, analyzes both independently, then auto-selects and assembles clips synced to the music's rhythm, energy, and lyrics. It's a batch-processing pipeline, not a web service.

### Architecture & main modules/layers

**Four analysis pipelines → single assembly step** (see `ARCHITECTURE.md`):

| Layer | Module(s) | Purpose |
|-------|-----------|---------|
| **Video analysis** | `analyze_footage.py`, `media_analyzer/video_analyzer.py` | Scene detection, motion (optical flow), brightness, color extraction at ~8 Hz |
| **Vision enrichment** | `enrich_analyses.py`, `vision_backends.py`, `vision_schema.py` | Pluggable vision-model backend (ollama/qwen2.5vl:7b default, claude-cli, anthropic-api) adds semantic descriptions (mood, style, tags) |
| **Text detection** | `scan_text.py`, `scan_text_fast.py` | English text overlay detection (1fps exhaustive or adaptive binary-search) |
| **Audio analysis** | `analyze_dj_set.py`, `analyze_dj_set_deep.py`, `media_analyzer/audio_analyzer.py` | 7-band energy, HPSS, onsets, chroma, BPM, spectral features, phrase segmentation at 8 Hz |
| **Lyrics extraction** | `extract_lyrics.py` | Demucs vocal separation → Whisper transcription → keyword extraction |
| **Quality culling** | `flag_quality.py` | Per-scene quality scores (black/blown/frozen/flat/near-dup) |
| **CLIP embeddings** | `generate_clip_embeddings.py`, `clip_utils.py` | 512-dim scene embeddings for semantic search and scoring |
| **Assembly** | `assemble_v2.py` (1620 LOC) | **The core**: builds scene database, scores scenes vs phrases, selects with diversity constraints, ffmpeg concat |
| **Benchmarks** | `bench/` | Vision model contest (`vision_model_contest.py`), hybrid scoring experiments, MMR diagnostics |
| **Toolkit package** | `media_analyzer/` | Installable CLI (`media-analyzer`) wrapping video/audio analyzers |

### Entrypoints & data flow

```
Source videos ──► analyze_footage.py ──► .analysis.json
                    │
                    ▼
enrich_analyses.py ──► .enriched.json (vision model semantics)
                    │
                    ├──► generate_clip_embeddings.py ──► .clip_embeddings.json
                    ├──► flag_quality.py ──► .quality.json
                    └──► scan_text_fast.py ──► .text_flags.json

DJ set audio ──► analyze_dj_set_deep.py ──► .deep-analysis.json
                    │
                    ▼
extract_lyrics.py ──► .lyrics.json

All JSON sidecars ──► assemble_v2.py ──► ffmpeg ──► music_video.mp4
```

Entry points are **standalone scripts** invoked from the command line. The `media_analyzer` package (`media_analyzer/cli.py`) provides a thin `media-analyzer` CLI wrapper for single-file audio/video analysis.

### Key domain entities

- **SceneClip** (`assemble_v2.py`): The core dataclass representing a candidate clip — source video, scene index, motion/brightness/color/semantic features, CLIP embedding, quality score.
- **Phrase** (`assemble_v2.py`): 4-bar audio phrase unit with energy, bass ratio, BPM, onset density, etc.
- **Scene database**: ~19,000 candidate scenes built from enriched analyses, filtered by duration (1.5–120s) and text-overlay overlap.
- **Set config** (`assemble_v2.py:44-75`): Dict mapping set names to analysis file, audio file, output name, and style hints.

### Cross-cutting concerns

**Authn/Authz**: None. This is a local batch-processing tool. No authentication, no authorization, no user accounts.

**Input/webhook validation**: No web endpoints. Input validation is schema-level — `vision_schema.normalize_analysis` snaps enum fields to allowed vocabularies and coerces list fields (`vision_schema.py`). The assembler uses `.get()` defaults for backward compatibility with older JSON schemas (e.g., pre-2.1.0 `bpm_timeline.confidence`).

**Secrets**: No secrets in the repo. Cloud backends (`anthropic-api`, `claude-cli`) rely on environment variables or the user's local CLI config. `bench/keys.py` exists for benchmark API keys (gitignored).

**Persistence**: File-system based. JSON sidecars in `enriched/`, `text_flags/`, `sets/`. State tracking via `enriched/state.json`. No database.

**External calls**: Heavy — ffmpeg (video extraction/concat), ffprobe (metadata), ollama (local vision inference), Demucs (vocal separation), Whisper (transcription), CLIP/torch (embeddings), optional anthropic API. All are subprocess calls or library imports.

**Hardcoded paths**: `SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")` and `SETS_DIR` in `assemble_v2.py:40-41` and `enrich_analyses.py:48`. These are machine-specific and would break on any other system.

### Highest risk / refactor surface

1. **`assemble_v2.py` (1620 LOC)** — The single largest file, containing the entire assembly pipeline: scene database construction, phrase feature extraction, scoring (12 dimensions), diversity enforcement (MMR, variety windows, fallback cascade), and ffmpeg assembly. This is the highest concentration of business logic and the most likely place for subtle bugs. The scoring table alone spans 20+ lines of complex conditional logic.

2. **Hardcoded filesystem paths** — `SOURCE_DIR` and `SETS_DIR` in `assemble_v2.py:40-41` and `enrich_analyses.py:48` are absolute paths to external storage. No configuration abstraction.

3. **No test infrastructure** — `bench/` is a model benchmarking harness, not a unit test suite. No pytest, no CI, no automated regression tests. The only "tests" are manual perceptual baselines in `bench/perceptual_baselines.json`.

4. **`flag_quality.py` threshold tuning** — `NEAR_DUP_SIM` was recently tightened from 0.985 → 0.97 (`flag_quality.py:49`). Quality thresholds are magic numbers with no validation or documentation of their empirical basis beyond comments.

5. **`vision_backends.py`** — Pluggable backend interface is clean, but the refusal-detection logic (`_looks_like_refusal`) uses substring matching on the first 200 chars, which is fragile against model output variations.

6. **`enrich_analyses.py` (904 LOC)** — Stateful pipeline with pause/resume, dry-run, and progress tracking. Complex state machine in `STATE_FILE`.
