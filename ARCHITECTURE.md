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
│  brightness + semantics - text +     │
│  CLIP emb + quality + loop flags     │
│  ~6,500 usable scenes               │
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

The enrichment schema is enforced by `vision_schema.normalize_analysis`: enum fields (`visual_style`, `mood.energy`, `color_palette.temperature`, `color_palette.color_relationship`, `composition.implied_motion`) are snapped onto their allowed vocabularies (or `"unclear"`), and list fields are coerced to flat string lists. Any corrections are recorded under a reserved `_validation` key (`{field: {raw, normalized}}`). The backend + model that produced a frame are stamped under `_provenance`. **Both `_validation` and `_provenance` are metadata — downstream consumers (the assembler) read only named fields and ignore them.** Enrichment runs through a pluggable backend (`vision_backends.py`): `ollama` (local, default), `claude-cli` (Claude on the user's subscription — zero marginal cost), or `anthropic-api` (opt-in, metered).

### CLIP Embeddings (`.clip_embeddings.json`)

One sidecar per source video (in `enriched/`), produced by `generate_clip_embeddings.py` (CLIP `ViT-B-32/openai`, shared loader in `clip_utils.py`). Holds a 512-dim L2-normalized vector per scene (encoded from the scene's midpoint frame):

```
{
  "model": "ViT-B-32/openai",
  "embedding_dim": 512,
  "scenes": [ { "scene_index", "frame_time_sec", "embedding": [512 floats] } ]
}
```

Consumed by the assembler (scoring dim 11: lyric-text ↔ visual similarity) and by `query_scenes.py` for natural-language search.

### Scene Quality (`.quality.json`)

One sidecar per source video (in `enriched/`), produced by `flag_quality.py` from the existing brightness/motion timelines (no video re-decoding) plus the CLIP embeddings for within-source near-duplicate detection:

```
{
  "model_version": "1.0",
  "thresholds": { ... },
  "scenes": [
    {
      "scene_index": 12,
      "quality_score": 0.1,      // combined, used by the assembler
      "technical_score": 0.1,    // black / blown_out / frozen
      "editorial_score": 1.0,    // low_info / near_dup
      "flags": ["black"],
      "metrics": { "brightness_p95": 0.02, "motion_max": 0.005, ... }
    }
  ]
}
```

Flags: `black` (brightness p95 < 0.05), `blown_out` (brightness p5 > 0.95), `frozen` (peak motion < 0.05), `low_info` (mean contrast < 0.04), `near_dup` (CLIP cosine ≥ 0.985 vs an earlier kept scene in the same source). The assembler turns `quality_score` into a soft, non-destructive penalty (scoring dim 12).

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

Schema **2.2.0**. Multi-dimensional audio feature extraction at 8 Hz. The
per-field contract (which fields are persisted, which are consumed, which are
strategic latent signals) is the authoritative reference for what stays in
this file: see [`audio_field_audit.md`](audio_field_audit.md).

```
{
  "schema_version": "2.2.0",
  "analyzer": "dj-set-analyzer-deep",
  "file": { "name", "path", "duration_sec", "sample_rate" },
  "global": { "bpm", "bpm_range", "beat_count", "key", "onset_count", "onset_rate_per_sec" },
  "beat_quality": {
    "ioi_median_sec",
    "ioi_outlier_count", "ioi_outlier_pct",
    "octave_doubling_pct", "octave_doubling_run_max",
    "metronomic_deviation_max_sec", "metronomic_deviation_final_sec",
    "metronomic_deviation_valid",
    "octave_corrected_windows", "octave_corrected_times_sec"
  },
  "tracks": [
    {
      "title", "start_sec", "end_sec",
      "bpm": { "mean", "min", "max", "std" },
      "energy": { "mean", "peak", "std" },
      "bands": { "sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance" },
      "harmonic_percussive": { "harmonic_mean", "percussive_mean", "balance" },
      "spectral": { "centroid_mean_hz", "flatness_mean", "brightness", "texture" },
      "chroma_dominant"
    }
  ],
  "transitions": [ { "from_track", "to_track", "boundary_sec", "energy_*", "bass_*", "bpm_*", "hp_ratio_*", "flux_*", "type" } ],
  "beats": { "times_sec": [...], "downbeats_sec": [...],
             "downbeat_estimator", "downbeat_bar_offset", "downbeat_bar_margin",
             "phrase_anchor_offset", "phrase_anchor_candidate_16", "phrase_anchor_margin" },
  "bpm_timeline": [ { "time_sec", "bpm", "confidence" } ],
  "multiband_energy": [ { "time_sec", "total_rms", "sub_bass", "bass", "presence", "brilliance" } ],
  "hpss_timeline": [ { "time_sec", "harmonic", "percussive" } ],
  "onsets": { "count", "times_sec": [...] },
  "spectral_timeline": [ { "time_sec", "centroid_hz" } ],
  "key_timeline": [ { "time_sec", "key", "mode", "label", "confidence" } ],
  "phrases": {
    "four_bar": [ { "start_sec", "end_sec", "beat_count", "energy_mean", "energy_peak", "energy_shape" } ],
    "eight_bar": [...],
    "sixteen_bar": [...]
  }
}
```

**Schema 2.1.0 vs 2.0.0:** the major timelines dropped per-item fields with
no current reader to cut JSON size ~64% (37 → 13 MB on a 70-min set):
`chroma_timeline` (full), `multiband.{low_mid, mid, high_mid}`,
`spectral.{bandwidth_hz, flatness, rolloff_hz, contrast_mean, flux}`,
`hpss_timeline.hp_ratio`, `onsets.strength_envelope`, `beats.features`. Internal
aggregators (`analyze_tracks_deep`, `analyze_transitions`) still receive the
full per-item field set during compute; pruning happens only at result-dict
assembly. Kept as latent signals for future key/transition-aware sequencing:
`multiband.presence/brilliance`, `key_timeline`, `transitions[]`.

**`beat_quality`** is non-blocking observability on `librosa.beat.beat_track`:
IOI outlier rate (±15% fractional tolerance vs median), octave-doubling
percent + max consecutive-window run, metronomic deviation (max + final
absolute deviation from a constant-tempo extrapolation, with a `_valid`
guard set false when octave-doubling exceeds 10%). The `metronomic_deviation`
metric is *not* tracker drift — it measures distance from a synthetic
metronome built on `global_bpm`, which legitimately accumulates on long DJ
sets with tempo ramping. See `lessons.md` under "# Audio side".

**`bpm_timeline.confidence`** is consumed by the assembler (see scoring table
below). Pre-2.1.0 files lack it; assembler's `.get()` defaults preserve
existing behavior on those.

**Schema 2.2.0 vs 2.1.0 (cut-timing refinement):** `phrases[].end_sec` is now
the start of the *next* phrase (full musical span; pre-2.2.0 undershot by one
beat and the assembler compensated — the workaround is now gated on
`schema_version`). `bpm_timeline` is repaired before persisting: octave-locked
windows are folded onto the rolling median and keep `confidence × 0.25`
(`beat_quality.octave_corrected_*` records the corrections). `beats` gains a
*heuristic* downbeat estimate (`downbeats_sec` + margins) persisted for
observability/audit; phrase anchoring stays opt-in (`--anchor-phrases`)
because the candidate signals disagree on this corpus (see lessons.md). The
assembler additionally snaps planned cut frames to `beats.times_sec` and
moves phrase boundaries off sung words (`adjust_cuts_for_vocals`, word-level
Whisper timings, ±2 beats max shift) before clip selection.

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
4. **Sub-scene salvage**: scenes overlapping English text flags are split
   around the flagged seconds (±`TEXT_MARGIN_SEC`=1s padding) and the clean
   sub-ranges ≥ `SALVAGE_MIN_DURATION`=1.5s are kept (`salvage_subscenes`;
   `--no-salvage` restores whole-scene discard). Sub-clips share the parent's
   `scene_index`, so variety windows / reuse penalties / near-dup treat them
   as one scene; measured features (motion, brightness, colors) are
   recomputed per sub-range; `loopable` is forced off for trimmed ranges
   (the SSIM check validated the parent's endpoints). Recovers ~43 min of
   footage and returns 3 fully-text-excluded sources to rotation on the
   current corpus. Scenes flagged end-to-end stay excluded.
5. Result: ~6,400 candidate scenes with full feature vectors (after duration
   + text filters + salvage, across the 52-source enriched corpus)

### Phrase Feature Extraction

1. Load `.deep-analysis.json` for the target DJ set
2. Use 4-bar phrases as base units (~7-8s at 120 BPM)
3. Per phrase, extract:
   - Energy (normalized RMS)
   - Bass ratio (sub_bass + bass vs total)
   - Audio brightness (spectral centroid)
   - Percussive ratio (from HPSS)
   - BPM, energy shape
   - **`onset_density_rank`** — onsets/sec within the phrase window (corrected
     to extend by one beat past the last beat-time so the denominator is the
     actual phrase span), then percentile-ranked across all phrases in the set
   - **`bpm_confidence`** — per-phrase average of `bpm_timeline.confidence`,
     normalized by the per-set 95th-percentile divisor (avoids hardcoded scales)

### Adaptive Phrase Merging

Adjacent 4-bar phrases are merged when:
- Both below energy threshold (0.35)
- Energy delta < 0.12 (stable, not transitioning)
- Same track (no cross-track merges)
- Previous phrase shape is "sustain" or "decay"
- Combined duration < 45 seconds

This produces longer clip holds for breakdowns and ambient passages, while keeping fast cuts for drops and buildups.

### Scoring (per scene vs. per phrase)

Each scene is scored against each phrase. The first six-and-a-half are weighted
match terms; the rest are additive bonuses/penalties (see `score_scene` in
`assemble_v2.py`):

| # | Dimension | Weight | Logic |
|---|-----------|--------|-------|
| 1 | Motion-energy | 1.0 | Visual motion ↔ audio energy |
| 2 | Brightness | 0.8 | Visual brightness ↔ spectral centroid |
| 3 | Color temperature | 0.7 | Warm colors ↔ bass-heavy audio |
| 4 | Duration fit | 0.9 × `(0.6 + 0.4 × bpm_confidence)` | Scene length ↔ target (3-15s based on energy), **speed-fit aware**: judged on the closest duration the clip can *present* within the render speed range `[SPEED_FIT_MIN, SPEED_FIT_MAX]` (a 5.6s scene is a perfect 8s phrase at 0.7×). Also **soft-de-emphasized** on low-confidence BPM phrases (breakdowns / intros / transitions); at zero confidence, term retains 60% weight — mild, not "disable". |
| 5 | Saturation | 0.5 | Color saturation ↔ energy level |
| 6 | Contrast | 0.5 | Visual contrast ↔ percussive ratio |
| 6b | **Onset-density ↔ motion-jitter** | 0.4 | Per-set percentile-ranked `onsets/sec` per phrase × per-corpus percentile-ranked `motion_std` per clip. Orthogonal to energy: distinguishes lots-of-small-percussion from one-sustained-pad at the same loudness. Both sides percentile-ranked so the match is scale-free (insensitive to BPM regime or motion magnitude). |
| 7 | Semantic | 0.1-0.4 | Mood energy match (calm↔calm, intense↔intense) |
| 8 | Style hints | variable | Per-set creative direction bonus/penalty |
| 9 | Lyrics match | 0.25/match | Lyric keywords matched against clip content_tags (cap 0.8) |
| 10 | Loopability | +0.15 | Prefer loopable clips for phrases longer than the scene |
| 11 | CLIP similarity | ×0.6 | Lyric-text embedding ↔ scene visual embedding (cosine) |
| 12 | Scene quality | ×3.0 | **Non-destructive** soft penalty `-(1-quality_score)*3` — a dead scene (black/blown/frozen) sinks ~3 pts but is never excluded outright |

**`bpm_confidence`** is the per-phrase average of `bpm_timeline.confidence`
divided by the set's 95th-percentile (avoids hardcoded-scale drift across
heterogeneous sets). Falls back to `1.0` when reading a pre-2.1.0 JSON without
a `confidence` field (preserves prior behavior).

**`motion_std`** is added to `SceneClip` (per-scene `np.std(motion_vals)` over
the optical-flow timeline) and percentile-ranked across the full scene database
post-build. Stored as `motion_std_rank` on each clip; the comparison against
phrase-side `onset_density_rank` is a pure rank-vs-rank distance — no scalar
scales involved.

New scoring components land at <0.5 weight or as multipliers on existing terms
by convention. The weights here co-evolved with the diversity windows in
`select_clips` and the `MAX_SOURCE_USAGE_MULT` cap; a primary-axis weight (1.0+)
on a new dimension would destabilize that balance. See `lessons.md` under
"# Audio side" for the rationale.

### Diversity Enforcement

`select_clips` (`assemble_v2.py:742+`) layers metadata-level constraints
(hard blocks on filename + scene index) with perceptual-level constraints
(near-duplicate exclusion + MMR re-ranking in CLIP space).

| Mechanism | Constant | Effect |
|-----------|-------|--------|
| Near-duplicate hard skip | `NEAR_DUP_SIM = 0.97` (in `flag_quality.py`) | Hard block: skip clips flagged `near_dup` (CLIP cosine ≥ 0.97 vs an earlier kept scene in the same source). Participates in the fallback cascade alongside `VARIETY_WINDOW` — both are perceptual-diversity constraints. |
| Source variety window | `VARIETY_WINDOW = 15` | Hard block: same source video can't repeat within 15 clips |
| Scene variety window | `SCENE_VARIETY_WINDOW = 30` | Hard block: exact same `(source, scene_index)` can't repeat within 30 clips |
| Usage ceiling | `MAX_SOURCE_USAGE_MULT = 2.0` | Cap: no source exceeds `(total_clips / n_sources) * 2` uses |
| Reuse penalty | `REUSE_PENALTY = 0.5` | Soft: `score *= 1/(1 + 0.5 * uses)` — steep diminishing returns |
| Scene reuse penalty | `SCENE_REUSE_PENALTY = 1.0` | Soft, per `(source, scene_index)`: re-showing the exact scene is far more visible than re-visiting a source, so identical frames decay faster than the source-level penalty. Window-mode offset variation additionally makes long-scene reuses show different footage. |
| MMR pool size | `MMR_POOL = 80` | After hard blocks + scoring, top-80 by raw score enter MMR re-rank. Tuning showed pool ceiling matters more than λ — a narrower pool of 30 left one validation set worse off because all top-scoring candidates were perceptually similar. |
| MMR diversity weight | `MMR_LAMBDA = 0.5` | `mmr_score = norm(raw) − λ × max_cosine(candidate, last MMR_RECENT_WINDOW selected)` |
| MMR recent window | `MMR_RECENT_WINDOW = 10` | Compare each candidate against the last 10 picked clips |
| Random pool | `TOP_CANDIDATES = 20` | Weighted random from top-20 of the MMR-reranked list (not top 1) |

**Fallback cascade** — when hard blocks empty the candidate pool, `select_clips`
relaxes constraints progressively, in three levels:

1. Drop perceptual-diversity blocks (`near_dup` hard skip + `VARIETY_WINDOW`)
   together; keep scene dedup + usage ceiling.
2. Drop scene dedup; keep usage ceiling.
3. Drop usage ceiling — last resort. Should essentially never fire on a healthy
   corpus.

**MMR re-rank** prevents perceptually-similar clips from sitting back-to-back
even when their `source_name` and `scene_index` differ. Pre-MMR, the assembler
log could report "40/40 sources used" while the viewer still perceived
constant repeats; MMR closes that gap by reading the CLIP embedding space the
scoring stack already loaded into each `SceneClip.clip_embedding`. Per-phrase
min-max score normalization is mandatory — raw scores vary in magnitude per
phrase, which would otherwise make `λ` brittle. See `lessons.md` "# Selection
side" for the empirical rationale (the pool ceiling matters more than λ).

**Diagnostic log** at `bench/mmr_diagnostics.log` captures per-candidate
`(raw_score, norm_score, max_recent_cosine, mmr_score)` for the first
`MMR_DIAGNOSTIC_PHRASES = 10` phrases of each run. Gitignored runtime artefact.

### Perceptual Diversity Observability

After `select_clips` returns, `compute_perceptual_diversity()` measures the
final ordered selection and emits a `[4b/5] Perceptual diversity` block:

- Consecutive-pair CLIP cosine — mean and median across `selection[i] ↔ selection[i+1]`
- Close-pair counts: `pairs ≥ 0.85 / 0.80 / 0.75 within 5 phrases`
- Close-pair counts: `pairs ≥ 0.90 / 0.85 within 30 phrases`

These ride with every assembler run going forward. Baselines for the validation
sets are committed at `bench/perceptual_baselines.json`; post-uplift results at
`bench/perceptual_results.json`.

### Video Assembly (frame-exact, v2.1)

Every clip is planned in **frames anchored to the audio timeline**: clip *i*
ends at the frame nearest `(phrase_i.end_sec − timeline_start)`, so per-clip
rounding can never accumulate and every cut lands on a phrase boundary.
Per-clip render mode (chosen in `select_clips`, executed by `extract_clip`):

| Mode | When | Mechanics |
|------|------|-----------|
| `window` | scene ≥ 1.35 × phrase | Cut a phrase-length window; start offset varies per reuse so a re-used scene shows different footage |
| `speedfit` | scene within 0.7–1.35 × phrase | Play the whole scene speed-adjusted (`setpts`, slow-mo or speed-up) to land exactly on the frame count |
| `loop` | loopable-flagged, or > `PINGPONG_MAX_SCENE` | Repeat scene start-to-start, trim to frames |
| `pingpong` | any other short scene (≤ 15s) | Forward+reverse cycle (seam frame de-duplicated), repeated to fill — seamless for any content |

1. Extract each clip at exactly its planned frame count (`-frames:v`, never
   wall-clock `-t` on the output side), scaled to 1080p, padded black. A failed
   extraction renders **black filler** of the planned frame count — dropping a
   clip would shift every later cut off the audio timeline.
2. Verify: ffprobe every rendered clip; assert `rendered frames == planned
   frames`, print net drift if any.
3. Concatenate (hard cuts on phrase boundaries — beat-aligned cuts, no
   crossfades) and overlay DJ set audio starting at the first phrase's timestamp.
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
| open_clip + torch | latest | CLIP embeddings (generation, assembler dim 11, scene query) |
| Claude Code CLI | 2.1+ | `claude-cli` vision backend (uses your subscription) |
| anthropic | latest | `anthropic-api` vision backend (optional, metered) |

## Directory Layout

```
ghost-media/                 # (the generated data dirs/files below are gitignored)
├── media_analyzer/          # installable toolkit package (CLI: media-analyzer)
├── pyproject.toml  setup.sh  examples/
├── *.analysis.json          # 73 raw video analyses
├── enriched/
│   ├── *.enriched.json          # 53 vision-enriched analyses
│   ├── *.clip_embeddings.json   # per-scene CLIP vectors (semantic search/matching)
│   └── *.quality.json           # per-scene quality scores + cull flags
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
├── enrich_analyses.py       # Vision enrichment (pluggable backend)
├── vision_schema.py         # Shared enrichment prompt + enum normalizer
├── vision_backends.py       # Vision backends: ollama / claude-cli / anthropic-api
├── generate_clip_embeddings.py # CLIP per-scene embedding generator
├── clip_utils.py            # Shared CLIP loader + image/text encoders
├── flag_quality.py          # Per-scene quality / cull pass → .quality.json
├── detect_loops.py          # Loop detection (SSIM first/last frame) → enriched
├── query_scenes.py          # Natural-language scene search over CLIP embeddings
├── smoke_test.py            # Fast feature-interaction checks
├── scan_text_fast.py        # Adaptive text scanner
├── scan_text.py             # Original 1fps text scanner
├── render_sizzle.py         # Audio-reactive visualizer
├── vision_model_contest.py  # Vision model evaluation
├── cleanup_enrichment.py    # Post-enrichment fixes
├── analyze_dj_set.py        # Basic audio analysis (legacy)
└── assemble_music_video.py  # v1 assembler (legacy)
```
