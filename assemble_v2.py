#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MUSIC VIDEO ASSEMBLER v2
  Scene-based clip matching using full analysis data
═══════════════════════════════════════════════════════════════════════════════

Fixes from v1:
  1. Scenes as clip units (19k+ scenes vs 513 frames)
  2. Flattened scoring — no single dimension dominates
  3. Hard variety enforcement — no source repeats within N phrases
  4. Real motion/color/brightness from video analysis (not qwen guesses)
  5. Scene duration matching — fast cuts for energy, long holds for grooves

v2.1 render layer (frame-exact sync contract):
  6. Every clip is planned in frames anchored to the audio timeline — cuts
     land on phrase boundaries (v2.0 truncated short scenes and padded
     +0.5s per clip, drifting off the audio within minutes)
  7. Scenes shorter than their phrase are slow-mo'd (speedfit), ping-pong
     looped, or plain-looped — never truncated
  8. Long scenes are windowed with varied start offsets per reuse; exact
     scene reuse carries its own penalty (SCENE_REUSE_PENALTY)

Pipeline:
  1. Load enriched files → build scene database with merged features
  2. Load deep audio analysis → extract phrase features
  3. Score scenes against phrases using motion, color, brightness, semantics
  4. Select with hard variety constraints
  5. Extract + assemble via ffmpeg
"""

import json
import math
import os
import subprocess
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
ENRICHED_DIR = BASE_DIR / "enriched"
TEXT_FLAGS_DIR = BASE_DIR / "text_flags"
# Canonical media locations + filename index live in media_paths
# (env-overridable: GHOST_FOOTAGE_ROOT / GHOST_SETS_ROOT / GHOST_MEDIA_CACHE).
import media_paths
SOURCE_DIR = media_paths.FOOTAGE_ROOT
SETS_DIR = media_paths.SETS_ROOT

# Hungry-ghost music videos where on-screen text/captions are *intrinsic*
# (not incidental titles the text-flagger trims) — comedy/lyric videos that
# would clash with abstract DJ-set visuals. Substring-matched against
# source_name; applied as avoid_sources to the existing abstract sets below.
# The footage stays globally selectable; this only steers the 5 legacy sets.
# Edit freely — text-flagging already excludes incidental on-screen text, so
# this is purely an aesthetic "keep recognizable caption-MVs out" list.
CAPTION_HEAVY_SOURCES = [
    "Jazz Emu",            # D_H_A_R_N_T_Z_ — wall-to-wall robot-narration captions
    "GO BANANAS",          # LITTLE BIG — on-screen text throughout
    "HYPNODANCER",         # LITTLE BIG — on-screen text throughout
    "Young Folks",         # Peter Bjorn And John — recurring caption gags
    "Sexy and I Know It",  # LMFAO — recurring on-screen text
]

# Set definitions: analysis file → audio file → output name
SET_CONFIGS = {
    "blue-sky-genesis-2025": {
        "analysis": "Mtimpe MIX1 02DEC MASTER MP3 V6.deep-analysis.json",
        "audio": SETS_DIR / "blue-sky-genesis-2025" / "Mtimpe MIX1 02DEC MASTER MP3  V6  .mp3",
        "output_name": "blue-sky-genesis-2025_music_video.mp4",
        "style_hints": {"avoid_sources": CAPTION_HEAVY_SOURCES},
    },
    "boxing-day-2025": {
        "analysis": "Mtimpe MIX 30OCT MASTER WAV v5.deep-analysis.json",
        "audio": SETS_DIR / "boxing-day-2025" / "Mtimpe MIX 30OCT MASTER MP3 v5.mp3",
        "output_name": "boxing-day-2025_music_video.mp4",
        "style_hints": {"avoid_sources": CAPTION_HEAVY_SOURCES},
    },
    "cheerleader-exodus-2025": {
        "analysis": "Mtimpe MIX2 10DEC MASTER WAV v3.deep-analysis.json",
        "audio": SETS_DIR / "cheerleader-exodus-2025" / "Mtimpe MIX2 10DEC MASTER MP3 v3.mp3",
        "output_name": "cheerleader-exodus-2025_music_video.mp4",
        "style_hints": {"avoid_sources": CAPTION_HEAVY_SOURCES},
    },
    "waiting-to-begin-2024": {
        "analysis": "arche august dj set rev 4.deep-analysis.json",
        "audio": SETS_DIR / "waiting-to-begin-2024" / "arche august dj set rev 4.mp3",
        "output_name": "waiting-to-begin-2024_music_video.mp4",
        "style_hints": {"avoid_sources": CAPTION_HEAVY_SOURCES},
    },
    "will-call-2025": {
        "analysis": "Mtimpe MIX 09JUN MASTER WAV v4.deep-analysis.json",
        "audio": SETS_DIR / "will-call-2025" / "Mtimpe MIX 09JUN MASTER MP3 v4.mp3",
        "output_name": "will-call-2025_music_video.mp4",
        "style_hints": {"avoid_sources": CAPTION_HEAVY_SOURCES},
    },
}

# ─── Style Hints Reference ───────────────────────────────────────────────
# Each set can have a style_hints dict with any of these keys:
#
#   "preferred_styles": ["abstract", "liquid", "fractal", ...]
#       → +0.5 score bonus for clips matching these visual_style values
#
#   "avoid_styles": ["cinematic", "photorealistic", ...]
#       → -1.0 score penalty for clips matching these
#
#   "color_preference": "warm" | "cool" | "neutral"
#       → shifts color temperature matching toward preferred palette
#
#   "preferred_sources": ["Disengaging", "FreeFormLiquid", ...]
#       → +0.3 bonus for clips from these source videos (substring match)
#
#   "avoid_sources": ["show me lyrics", ...]
#       → -2.0 penalty (effectively excludes these sources)
#
#   "energy_response": "follow" | "contrast"
#       → "follow" (default): high audio energy → high visual energy
#       → "contrast": inverts the energy mapping for dreamy/surreal feel
#
# Example:
#   "style_hints": {
#       "preferred_styles": ["abstract", "particle", "fractal"],
#       "avoid_styles": ["cinematic"],
#       "color_preference": "cool",
#       "energy_response": "follow",
#   }

MIN_SCENE_DURATION = 1.5   # ignore scenes shorter than this
MAX_SCENE_DURATION = 120   # ignore extremely long static scenes
VARIETY_WINDOW = 15        # hard block: no same source within N phrases
SCENE_VARIETY_WINDOW = 30  # hard block: no same scene within N phrases
TOP_CANDIDATES = 20        # random pool size for weighted selection
REUSE_PENALTY = 0.5        # steeper logarithmic penalty per reuse
SCENE_REUSE_PENALTY = 1.0  # per-scene analogue of REUSE_PENALTY — re-showing the
                           # exact scene is far more visible than re-visiting a source
MAX_SOURCE_USAGE_MULT = 2.0  # cap any source at (total_phrases / n_sources) * this
QUALITY_PENALTY_WEIGHT = 3.0  # soft penalty for dead scenes: -(1-quality)*this (see flag_quality.py)

# ─── Frame-Exact Rendering ────────────────────────────────────────────────
# Every clip is planned in *frames* anchored to the audio timeline: clip i
# always ends at the frame nearest (phrase_i.end_sec - timeline_start), so
# per-clip rounding can never accumulate and cuts stay on phrase boundaries.
# Scenes shorter than their phrase are stretched (slow-mo), ping-pong looped,
# or plain-looped instead of truncated — truncation was the v2.0 sync bug.
FPS = 30                   # output frame rate; all durations are planned as frame counts
SPEED_FIT_MIN = 0.7        # slowest playback (slow-mo) used to stretch a scene onto a phrase
SPEED_FIT_MAX = 1.35       # fastest playback used to compress a scene onto a phrase

# Beat-locked looping: retune loop/pingpong cycles to an integer number of
# beats so the seam lands ON the beat — visible repetition reads as rhythm,
# not glitch. Constant-rate only (no ramps); falls back to native-speed
# loop/pingpong when no integer beat count fits within the speed band.
BEATLOOP_SPEED_MIN = 0.8
BEATLOOP_SPEED_MAX = 1.25
BEATLOOP_BPM_MIN = 60.0    # distrust phrase BPM outside this band
BEATLOOP_BPM_MAX = 200.0

# Canonical lyric↔tag matching: exclude canonical terms that tag more than
# this fraction of the corpus — ubiquitous terms ('abstract' in an
# abstract-art corpus) make the lyric bonus indiscriminate.
CANON_MATCH_MAX_DF = 0.03
PINGPONG_MAX_SCENE = 15.0  # ping-pong only for scenes ≤ this — ffmpeg reverse buffers
                           # the whole clip in RAM (~3 MB/frame at 1080p)

# ─── MMR Diversity Re-ranking (Phase C) ──────────────────────────────────
# After hard-block filter and soft scoring, the top-N scored candidates pass
# through MMR re-ranking which penalises perceptual similarity to recently-
# selected clips before the weighted-random pool is built. Closes the metadata-
# vs-perceptual diversity gap: filenames being different doesn't mean the
# embeddings are different.
MMR_LAMBDA = 0.5              # diversity penalty weight; 0 = no MMR, 1 = pure dedup
                              # (tuned across 3 sets: λ=0.4 made cheerleader-exodus
                              # WORSE; λ=0.7 also regressed it. λ=0.5 is the only
                              # value where all 3 sets improve from baseline.)
MMR_RECENT_WINDOW = 10        # how many recently-selected clips to compare against
MMR_POOL = 80                 # post-score candidate pool size, pre-MMR
                              # (wider than initially planned: the score pool is
                              # often perceptually homogeneous, so MMR needs more
                              # to choose from)
MMR_DIAGNOSTIC_PHRASES = 10   # log per-candidate MMR diagnostics for the first N phrases
MMR_DIAG_PATH = BASE_DIR / "bench" / "mmr_diagnostics.log"

# ─── Adaptive Phrase Merging ──────────────────────────────────────────────
# Controls how 4-bar phrases get merged for sustained/low-energy sections
MERGE_ENERGY_THRESHOLD = 0.35   # merge adjacent phrases below this energy
MERGE_ENERGY_DELTA = 0.12       # merge if energy difference < this (stable)
MAX_MERGED_DURATION = 45.0      # never merge beyond this duration (seconds)
MIN_MERGED_DURATION = 6.0       # minimum phrase duration after merge


# ─── Scene Database ────────────────────────────────────────────────────────

@dataclass
class SceneClip:
    source_name: str
    source_path: str
    scene_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    # From video analysis (measured)
    motion_mean: float       # optical flow mean
    motion_peak: float       # optical flow peak
    motion_std: float        # optical flow std within scene — within-clip jitter
    brightness_mean: float   # 0-1
    contrast_mean: float     # 0-1
    dominant_colors: list    # list of {rgb, hex, percentage}
    color_temperature: float # computed from rgb: <0.5 = cool, >0.5 = warm
    color_saturation: float  # computed from dominant colors
    # From semantic enrichment (if available)
    visual_style: str
    mood_energy: str
    content_tags: list
    has_semantic: bool
    # Canonicalized content_tags (scripts/canonicalize_tags.py): free-form
    # VLM tags mapped onto the ~200-term canonical vocabulary so synonyms
    # ("sea"/"ocean") land on the same term for lyric matching.
    canonical_tags: list = field(default_factory=list)
    # Loop detection
    loopable: bool = False
    loop_similarity: float = 0.0
    # CLIP embedding
    clip_embedding: np.ndarray = None
    # Quality pass (flag_quality.py) — 1.0 = clean, lower = dead/duplicate
    quality_score: float = 1.0
    cull_flags: list = field(default_factory=list)
    # Enriched-file stem (sanitized) — the reliable key for CLIP/quality sidecars,
    # since source_name keeps the original (unsanitized) filename.
    enriched_stem: str = ""
    # Salvaged sub-range ordinal within the parent scene (0 = whole scene).
    # All sub-ranges share the parent's scene_index, so variety windows,
    # reuse penalties, and near-dup treat them as one scene.
    sub_index: int = 0
    # Percentile rank of motion_std across the full scene database (0-1).
    # Populated post-build by _assign_motion_std_ranks; used by onset_match.
    motion_std_rank: float = 0.5
    # Percentile rank of motion_mean across the corpus (0-1). Motion magnitude
    # spans ~600x between sources (lessons.md), so the raw /15 normalization
    # saturates; the rank side of dim 1 is scale-free.
    motion_mean_rank: float = 0.5


def compute_color_temperature(colors):
    """Estimate warm/cool from dominant RGB colors. 0=cool, 1=warm."""
    if not colors:
        return 0.5
    total_weight = 0
    warmth = 0
    for c in colors:
        rgb = c.get("rgb", [128, 128, 128])
        pct = c.get("percentage", 0.2)
        r, g, b = rgb
        # Warm = more red/yellow, cool = more blue
        w = (r * 1.2 + g * 0.5) / (b + g + r + 1)
        warmth += w * pct
        total_weight += pct
    return min(1.0, warmth / (total_weight + 1e-10))


def compute_color_saturation(colors):
    """Estimate average saturation from dominant colors."""
    if not colors:
        return 0.3
    sats = []
    for c in colors:
        rgb = c.get("rgb", [128, 128, 128])
        r, g, b = [v / 255.0 for v in rgb]
        mx, mn = max(r, g, b), min(r, g, b)
        if mx == 0:
            sats.append(0)
        else:
            sats.append((mx - mn) / mx)
    return float(np.mean(sats))


def _sanitize_for_match(name):
    """Normalize a filename for fuzzy matching by stripping special chars."""
    import re
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()


def find_source_video(file_info):
    """Find the actual source video file (filename index + fuzzy fallback)."""
    return media_paths.find_source(file_info)


def load_text_flags():
    """Load text flags and build a set of (source_name, second) pairs that have English text."""
    text_seconds = {}  # source_stem → set of seconds with English text
    if not TEXT_FLAGS_DIR.exists():
        return text_seconds

    for tf_path in TEXT_FLAGS_DIR.glob("*.text_flags.json"):
        data = json.loads(tf_path.read_text())
        if data.get("status") != "complete":
            continue
        source_stem = Path(data.get("source", "")).stem
        tf_stem = tf_path.stem.replace(".text_flags", "")
        flagged = set()
        for sec_str, info in data.get("flags", {}).items():
            if info.get("has_english_text"):
                flagged.add(int(sec_str))
        if flagged:
            # Store under both the source stem and the text_flag stem for
            # matching. UNION across files: a source can have several flag
            # files (main scan, *.runtime_bans, *.render_check) — assignment
            # here would silently drop all but the last-globbed one.
            text_seconds.setdefault(source_stem, set()).update(flagged)
            text_seconds.setdefault(tf_stem, set()).update(flagged)

    return text_seconds


# Sub-scene salvage: instead of discarding a whole scene because one stretch
# has text, subtract the flagged seconds (padded by the flag resolution) and
# keep the clean sub-ranges. Flags are per-second, so the data already exists.
TEXT_MARGIN_SEC = 1.0        # padding around flagged seconds (flag resolution)
SALVAGE_MIN_DURATION = 1.5   # keep clean sub-ranges at least this long


def scene_text_spans(clip_source_name, start_sec, end_sec, text_seconds):
    """Flagged-second intervals overlapping [start_sec, end_sec), merged.

    Each flagged second s covers [s, s+1). Returns a sorted list of
    (lo, hi) spans; empty list = scene is clean.
    """
    if not text_seconds:
        return []
    flagged = set()
    for key, flagged_secs in text_seconds.items():
        if key in clip_source_name or clip_source_name in key:
            for sec in range(int(start_sec), int(end_sec) + 1):
                if sec in flagged_secs:
                    flagged.add(sec)
    if not flagged:
        return []
    spans = []
    for sec in sorted(flagged):
        if spans and sec <= spans[-1][1]:
            spans[-1][1] = sec + 1
        else:
            spans.append([float(sec), float(sec + 1)])
    return [(lo, hi) for lo, hi in spans]


def scene_has_text(clip_source_name, start_sec, end_sec, text_seconds):
    """Check if a scene overlaps with any English text seconds."""
    return bool(scene_text_spans(clip_source_name, start_sec, end_sec, text_seconds))


def salvage_subscenes(start_sec, end_sec, flagged_spans):
    """Subtract padded flagged spans from a scene; return clean sub-ranges.

    Each returned (lo, hi) is at least SALVAGE_MIN_DURATION long and at
    least TEXT_MARGIN_SEC away from any flagged second.
    """
    clean = []
    cursor = start_sec
    for lo, hi in flagged_spans:
        lo -= TEXT_MARGIN_SEC
        hi += TEXT_MARGIN_SEC
        if lo > cursor:
            clean.append((cursor, min(lo, end_sec)))
        cursor = max(cursor, hi)
    if cursor < end_sec:
        clean.append((cursor, end_sec))
    return [(lo, hi) for lo, hi in clean if hi - lo >= SALVAGE_MIN_DURATION]


def build_scene_database(text_seconds=None, salvage=True):
    """Build scene database from all enriched files, merging all data sources.

    salvage=True splits partially text-flagged scenes into clean sub-ranges
    (see salvage_subscenes) instead of discarding them whole. --no-salvage
    restores the old discard behavior for A/B comparison.
    """
    scenes = []
    n_excluded = n_salvaged_scenes = n_subclips = 0
    recovered_sec = 0.0
    files = sorted(ENRICHED_DIR.iterdir())
    files = [f for f in files if f.name.endswith('.enriched.json')]

    for f in files:
        data = json.loads(f.read_text())
        file_info = data.get("file", {})
        source_path = find_source_video(file_info)
        source_name = file_info.get("name", f.stem)
        src_stem = f.name.replace(".enriched.json", "")  # sanitized; matches sidecar filenames

        # Get timelines
        motion_tl = data.get("motion", {}).get("timeline", [])
        brightness_tl = data.get("brightness", {}).get("timeline", [])
        color_tl = data.get("colors", {}).get("timeline", [])

        # Index timelines by time for fast lookup
        def get_range(timeline, start, end, key):
            return [e[key] for e in timeline if start <= e.get("time_sec", 0) < end]

        def get_colors_at(timeline, start, end):
            """Get averaged dominant colors in a time range."""
            relevant = [e for e in timeline if start <= e.get("time_sec", 0) < end]
            if not relevant:
                return []
            # Use the middle sample's colors
            mid = relevant[len(relevant) // 2]
            return mid.get("dominant_colors", [])

        # Get scene list
        scene_data = data.get("scenes", {})
        scene_list = scene_data.get("scenes", []) if isinstance(scene_data, dict) else scene_data

        # Build frame analysis index by scene
        fa_by_scene = {}
        for fa in data.get("frame_analyses", []):
            si = fa.get("scene_index")
            if si is not None and "analysis" in fa:
                fa_by_scene[si] = fa["analysis"]

        for s in scene_list:
            dur = s.get("duration_sec", 0)
            if dur < MIN_SCENE_DURATION or dur > MAX_SCENE_DURATION:
                continue

            start = s.get("start_sec", 0)
            end = s.get("end_sec", start)
            si = s.get("scene_index", 0)

            # Semantic (from enrichment or scene-level)
            semantic = s.get("semantic", {}) or fa_by_scene.get(si, {})
            has_semantic = bool(semantic)
            mood = semantic.get("mood", {}) if semantic else {}
            style = semantic.get("visual_style", "") if semantic else ""
            tags = semantic.get("content_tags", []) if semantic else []

            # Text handling: salvage clean sub-ranges of partially flagged
            # scenes; discard only what's actually contaminated.
            sub_ranges = [(start, end)]
            trimmed = False
            spans = scene_text_spans(source_name, start, end, text_seconds) \
                if text_seconds else []
            if spans:
                if not salvage:
                    n_excluded += 1
                    continue
                sub_ranges = salvage_subscenes(start, end, spans)
                if not sub_ranges:
                    n_excluded += 1
                    continue
                trimmed = True
                n_salvaged_scenes += 1
                n_subclips += len(sub_ranges)
                recovered_sec += sum(hi - lo for lo, hi in sub_ranges)

            for sub_idx, (sub_start, sub_end) in enumerate(sub_ranges):
                # Measured features recomputed per sub-range — accurate for
                # the footage actually used, not the parent average.
                motion_vals = get_range(motion_tl, sub_start, sub_end, "mean_motion")
                motion_peaks = get_range(motion_tl, sub_start, sub_end, "max_motion")
                bright_vals = get_range(brightness_tl, sub_start, sub_end, "brightness")
                contrast_vals = get_range(brightness_tl, sub_start, sub_end, "contrast")
                colors = get_colors_at(color_tl, sub_start, sub_end)

                clip = SceneClip(
                    source_name=source_name,
                    source_path=source_path,
                    scene_index=si,
                    start_sec=sub_start,
                    end_sec=sub_end,
                    duration_sec=sub_end - sub_start,
                    motion_mean=float(np.mean(motion_vals)) if motion_vals else 0,
                    motion_peak=float(np.max(motion_peaks)) if motion_peaks else 0,
                    motion_std=float(np.std(motion_vals)) if motion_vals else 0,
                    brightness_mean=float(np.mean(bright_vals)) if bright_vals else 0.3,
                    contrast_mean=float(np.mean(contrast_vals)) if contrast_vals else 0.3,
                    dominant_colors=colors,
                    color_temperature=compute_color_temperature(colors),
                    color_saturation=compute_color_saturation(colors),
                    visual_style=style,
                    mood_energy=mood.get("energy", ""),
                    content_tags=tags,
                    canonical_tags=(semantic.get("canonical_tags", [])
                                    if semantic else []),
                    has_semantic=has_semantic,
                    # The SSIM loop check validated the PARENT's first/last
                    # frames; a trimmed sub-range has different endpoints.
                    loopable=s.get("loopable", False) and not trimmed,
                    loop_similarity=0.0 if trimmed else s.get("loop_similarity", 0.0),
                    enriched_stem=src_stem,
                    sub_index=sub_idx if trimmed else 0,
                )
                scenes.append(clip)

    # Load CLIP embeddings from sidecar files
    clip_embeddings = {}  # (source_name, scene_index) → np.array
    for emb_file in ENRICHED_DIR.glob("*.clip_embeddings.json"):
        emb_data = json.loads(emb_file.read_text())
        src_name = emb_file.name.replace(".clip_embeddings.json", "")
        for se in emb_data.get("scenes", []):
            key = (src_name, se["scene_index"])
            clip_embeddings[key] = np.array(se["embedding"], dtype=np.float32)

    if clip_embeddings:
        matched = 0
        for clip in scenes:
            key = (clip.enriched_stem, clip.scene_index)
            if key in clip_embeddings:
                clip.clip_embedding = clip_embeddings[key]
                matched += 1
        print(f"    CLIP embeddings: {matched}/{len(scenes)} scenes")

    # Load quality scores from sidecar files (flag_quality.py)
    quality = {}  # (source_name, scene_index) → (score, flags)
    for q_file in ENRICHED_DIR.glob("*.quality.json"):
        q_data = json.loads(q_file.read_text())
        src_name = q_file.name.replace(".quality.json", "")
        for qs in q_data.get("scenes", []):
            quality[(src_name, qs["scene_index"])] = (
                qs.get("quality_score", 1.0), qs.get("flags", []))
    if quality:
        flagged = 0
        for clip in scenes:
            key = (clip.enriched_stem, clip.scene_index)
            if key in quality:
                clip.quality_score, clip.cull_flags = quality[key]
                if clip.cull_flags:
                    flagged += 1
        print(f"    Quality scores: {len(scenes) - flagged}/{len(scenes)} clean, {flagged} flagged")

    # Percentile-rank motion_std across the corpus so onset_match is
    # dataset-relative, not tied to a hardcoded scale.
    if scenes:
        stds = np.array([c.motion_std for c in scenes])
        ranks = stds.argsort().argsort().astype(np.float32) / max(len(stds) - 1, 1)
        for c, r in zip(scenes, ranks):
            c.motion_std_rank = float(r)
        means = np.array([c.motion_mean for c in scenes])
        m_ranks = means.argsort().argsort().astype(np.float32) / max(len(means) - 1, 1)
        for c, r in zip(scenes, m_ranks):
            c.motion_mean_rank = float(r)

    if n_salvaged_scenes:
        print(f"    Salvage: {n_subclips} sub-scenes from {n_salvaged_scenes} "
              f"flagged scenes (+{recovered_sec:.0f}s recovered), "
              f"{n_excluded} fully excluded")
    elif n_excluded:
        print(f"    Text filter: {n_excluded} scenes excluded"
              f"{' (salvage disabled)' if not salvage else ''}")

    return scenes


# ─── Audio Features ────────────────────────────────────────────────────────

@dataclass
class PhraseFeatures:
    start_sec: float
    end_sec: float
    duration_sec: float
    energy: float           # 0-1
    energy_peak: float
    energy_shape: str
    bass_ratio: float       # 0-1, how bass-heavy
    brightness_audio: float # spectral centroid normalized
    percussive_ratio: float # 0-1
    bpm: float
    track_title: str
    # Percentile rank of onsets/sec within the current set (0-1).
    # Independent of energy: distinguishes "lots of small percussive hits"
    # from "one sustained pad" at the same loudness.
    onset_density_rank: float = 0.5
    # Average BPM-detection confidence over the phrase, normalized to [0, 1]
    # using divisor calibrated from the actual bpm_timeline range (Phase 0).
    bpm_confidence: float = 1.0
    lyrics_keywords: list = field(default_factory=list)  # keywords from vocals
    # lyrics_keywords mapped onto the canonical tag vocabulary (run_set 3c).
    canonical_keywords: list = field(default_factory=list)
    clip_text_embedding: np.ndarray = None  # CLIP text encoding of lyrics


def _schema_tuple(data):
    """Parse schema_version into a comparable (major, minor) tuple."""
    try:
        parts = str(data.get("schema_version", "0.0.0")).split(".")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return (0, 0)


# Padding around word spans when judging whether a cut bisects a word: a cut
# this close to a word edge is perceptually on the boundary, not mid-word.
VOCAL_WORD_PAD_SEC = 0.15
# How far a phrase boundary may move to clear a word, in beats either side.
VOCAL_SHIFT_MAX_BEATS = 2
# When vocals are continuous across every candidate beat (no word-clear cut
# possible), merge the boundary away instead — don't cut mid-line — unless the
# combined hold would exceed this. 0 disables merging (boundary stays on beat).
VOCAL_MERGE_MAX_SEC = 20.0


def _merge_phrase_pair(a, b):
    """Duration-weighted merge of two adjacent PhraseFeatures."""
    dur = a.duration_sec + b.duration_sec
    wa, wb = a.duration_sec / dur, b.duration_sec / dur
    return PhraseFeatures(
        start_sec=a.start_sec, end_sec=b.end_sec, duration_sec=dur,
        energy=a.energy * wa + b.energy * wb,
        energy_peak=max(a.energy_peak, b.energy_peak),
        energy_shape="sustain",
        bass_ratio=a.bass_ratio * wa + b.bass_ratio * wb,
        brightness_audio=a.brightness_audio * wa + b.brightness_audio * wb,
        percussive_ratio=a.percussive_ratio * wa + b.percussive_ratio * wb,
        bpm=a.bpm * wa + b.bpm * wb,
        track_title=a.track_title,
        onset_density_rank=a.onset_density_rank * wa + b.onset_density_rank * wb,
        bpm_confidence=min(a.bpm_confidence, b.bpm_confidence),
    )


def adjust_cuts_for_vocals(phrase_features, lyric_segments, beat_times):
    """Keep phrase boundaries off sung words.

    For each interior boundary that falls inside a word span (±VOCAL_WORD_PAD_SEC):
    1. Try beats within ±VOCAL_SHIFT_MAX_BEATS, nearest first, taking the first
       one clear of all words. Clearance relaxes progressively (comfortable pad
       → tight pad → strict inter-word gap): a cut squeezed between two words
       still beats one that bisects a word.
    2. If every candidate is mid-word (continuous vocals), don't cut mid-line:
       merge the two phrases into one hold, unless the combined duration would
       exceed VOCAL_MERGE_MAX_SEC — then the boundary stays put (an on-beat
       mid-phrase cut is acceptable; an off-beat one is not).

    Boundaries move in matched end/start pairs so the frame-exact planner
    downstream needs no changes. Returns (new_phrase_list, moved, merged).
    """
    if not lyric_segments or len(beat_times) < 2 or len(phrase_features) < 2:
        return phrase_features, 0, 0
    word_spans = []
    for seg in lyric_segments:
        for w in seg.get("words", []):
            if w.get("end", 0) > w.get("start", 0):
                word_spans.append((w["start"], w["end"]))
    if not word_spans:
        return phrase_features, 0, 0
    word_spans.sort()
    starts = np.array([s for s, _ in word_spans])
    ends = np.array([e for _, e in word_spans])
    bt = np.asarray(beat_times, dtype=np.float64)

    def in_word(t, pad):
        # Spans overlap at most locally; check the few whose start precedes t.
        idx = np.searchsorted(starts, t + pad, side="right")
        lo = max(0, idx - 12)  # spans are ≤ ~1.5s; 12 covers the densest vocals
        return bool(np.any((starts[lo:idx] - pad < t) & (t < ends[lo:idx] + pad)))

    out = [phrase_features[0]]
    moved = merged = 0
    for nxt in phrase_features[1:]:
        cur = out[-1]
        boundary = cur.end_sec
        if not in_word(boundary, VOCAL_WORD_PAD_SEC):
            out.append(nxt)
            continue

        idx = int(np.searchsorted(bt, boundary))
        candidates = [float(bt[j])
                      for j in range(idx - VOCAL_SHIFT_MAX_BEATS,
                                     idx + VOCAL_SHIFT_MAX_BEATS + 1)
                      if 0 <= j < len(bt)]
        candidates.sort(key=lambda t: abs(t - boundary))

        def usable(t, pad):
            # Keep both phrases non-degenerate after the shift.
            return (t > cur.start_sec + 1.0 and t < nxt.end_sec - 1.0
                    and abs(t - boundary) > 1e-4 and not in_word(t, pad))

        new_t = None
        for pad in (VOCAL_WORD_PAD_SEC, VOCAL_WORD_PAD_SEC / 3, 0.0):
            new_t = next((t for t in candidates if usable(t, pad)), None)
            if new_t is not None:
                break

        if new_t is not None:
            cur.end_sec = new_t
            cur.duration_sec = new_t - cur.start_sec
            nxt.start_sec = new_t
            nxt.duration_sec = nxt.end_sec - new_t
            out.append(nxt)
            moved += 1
        elif (VOCAL_MERGE_MAX_SEC > 0
                and cur.duration_sec + nxt.duration_sec <= VOCAL_MERGE_MAX_SEC):
            out[-1] = _merge_phrase_pair(cur, nxt)
            merged += 1
        else:
            out.append(nxt)
    return out, moved, merged


def extract_phrase_features(data, phrases):
    """Extract audio features per phrase.

    Back-compat: new fields (onset_density_rank, bpm_confidence) gracefully
    default if a pre-2.1.0 .deep-analysis.json doesn't carry their sources.
    """
    multiband = data["multiband_energy"]
    hpss = data["hpss_timeline"]
    spectral = data["spectral_timeline"]
    bpm_tl = data["bpm_timeline"]
    tracks = data.get("tracks", [])

    # New in 2.1.0: onset times for per-phrase density, BPM confidence for
    # weighting duration matching. Both fall back to neutral defaults if
    # missing so old JSONs still score.
    onset_times = np.array(data.get("onsets", {}).get("times_sec", []), dtype=np.float64)
    global_bpm = float(data.get("global", {}).get("bpm", 128.0))
    # Pre-2.2.0: detect_phrases set end_sec to the last beat's timestamp,
    # undershooting the true phrase span by ~1 beat; extend when reading old
    # JSONs. 2.2.0+ end_sec is the next phrase's start — no correction.
    end_undershoots = _schema_tuple(data) < (2, 2)
    # Per-set 95th-pct of bpm confidence: divisor that normalizes typical
    # strong-tempo material to ~1.0. Falls back to 8.0 (close to Phase 0
    # observation of 8.28 on the smoke set) when the timeline lacks it.
    conf_vals = [b.get("confidence") for b in bpm_tl if "confidence" in b]
    conf_divisor = float(np.percentile(conf_vals, 95)) if conf_vals else 8.0
    conf_divisor = max(conf_divisor, 0.1)  # guard against degenerate case

    # First pass: collect raw onsets/sec per phrase so we can percentile-rank.
    raw_densities: list[float] = []
    phrase_basics: list[dict] = []

    for phrase in phrases:
        start, end = phrase["start_sec"], phrase["end_sec"]

        mb = [m for m in multiband if start <= m["time_sec"] < end]
        hp = [h for h in hpss if start <= h["time_sec"] < end]
        sp = [s for s in spectral if start <= s["time_sec"] < end]
        bp = [b for b in bpm_tl if start <= b["time_sec"] < end]

        if not mb:
            continue

        phrase_end_effective = end + 60.0 / global_bpm if end_undershoots else end
        n_onsets = (np.searchsorted(onset_times, phrase_end_effective)
                    - np.searchsorted(onset_times, start)) if len(onset_times) else 0
        phrase_dur = max(phrase_end_effective - start, 1e-6)
        raw_densities.append(float(n_onsets) / phrase_dur)
        phrase_basics.append({
            "phrase": phrase, "mb": mb, "hp": hp, "sp": sp, "bp": bp,
            "start": start, "end": end,
        })

    # Percentile-rank onset density within the set. Per-set normalization
    # handles tempo-sensitivity (174 BPM DnB vs 92 BPM hip-hop both produce
    # valid "busy" rankings) and is robust to hi-hat-roll outliers.
    if raw_densities:
        ranks_arr = np.array(raw_densities).argsort().argsort().astype(np.float32)
        density_ranks = (ranks_arr / max(len(ranks_arr) - 1, 1)).tolist()
    else:
        density_ranks = []

    results = []
    for basic, dens_rank in zip(phrase_basics, density_ranks):
        phrase = basic["phrase"]
        start, end = basic["start"], basic["end"]
        mb, hp, sp, bp = basic["mb"], basic["hp"], basic["sp"], basic["bp"]

        rms = [m["total_rms"] for m in mb]
        bass = [m["sub_bass"] + m["bass"] for m in mb]
        total = [m["total_rms"] for m in mb]
        e_mean = np.mean(rms)

        # Spectral centroid → audio "brightness" 0-1
        centroids = [s["centroid_hz"] for s in sp] if sp else [2000]
        audio_brightness = min(np.mean(centroids) / 8000, 1.0)

        # Bass ratio
        bass_ratio = np.mean(bass) / (np.mean(total) + 1e-10)
        bass_ratio = min(bass_ratio, 1.0)

        # Percussive ratio
        if hp:
            harm = np.mean([h["harmonic"] for h in hp])
            perc = np.mean([h["percussive"] for h in hp])
            perc_ratio = perc / (harm + perc + 1e-10)
        else:
            perc_ratio = 0.5

        bpm = np.mean([b["bpm"] for b in bp]) if bp else 128

        # BPM-detection confidence over this phrase. Average then normalize
        # via per-set 95th-pct divisor (Phase 0 calibration). Falls back to
        # 1.0 (max) when the timeline lacks confidence (old schema).
        conf_present = [b.get("confidence") for b in bp if "confidence" in b]
        if conf_present:
            bpm_conf = min(float(np.mean(conf_present)) / conf_divisor, 1.0)
        else:
            bpm_conf = 1.0

        track = ""
        for tr in tracks:
            if tr["start_sec"] <= start < tr["end_sec"]:
                track = tr["title"]
                break

        results.append(PhraseFeatures(
            start_sec=start, end_sec=end,
            duration_sec=end - start,
            energy=min(e_mean / 0.45, 1.0),
            energy_peak=min(max(rms) / 0.5, 1.0),
            energy_shape=phrase.get("energy_shape", "sustain"),
            bass_ratio=float(bass_ratio),
            brightness_audio=float(audio_brightness),
            percussive_ratio=float(perc_ratio),
            bpm=float(bpm),
            track_title=track,
            onset_density_rank=float(dens_rank),
            bpm_confidence=float(bpm_conf),
        ))

    return results


# ─── Scoring ───────────────────────────────────────────────────────────────
# All components normalized to 0-1, then weighted equally-ish.
# No single factor can dominate.

def score_scene(clip: SceneClip, phrase: PhraseFeatures, style_hints=None) -> float:
    """Score a scene clip against an audio phrase. All components 0-1."""
    if style_hints is None:
        style_hints = {}

    # Energy response mode: "contrast" inverts the energy target
    energy_response = style_hints.get("energy_response", "follow")
    target_energy = phrase.energy if energy_response == "follow" else (1.0 - phrase.energy)

    # 1. Motion-energy match (visual motion should match audio energy).
    #    Blend of raw /15 normalization (absolute: a static clip is static
    #    everywhere) and corpus percentile rank (scale-free: motion magnitude
    #    spans ~600x between sources, so raw saturates for whole sources).
    #    Landed soft at 50/50; ratchet after baseline comparison.
    clip_motion_norm = min(clip.motion_mean / 15.0, 1.0)
    motion_match = 0.5 * (1.0 - abs(clip_motion_norm - target_energy)) \
        + 0.5 * (1.0 - abs(clip.motion_mean_rank - target_energy))

    # 2. Brightness match (bright audio → bright visuals)
    bright_match = 1.0 - abs(clip.brightness_mean - phrase.brightness_audio)

    # 3. Color temperature match
    target_warmth = phrase.bass_ratio
    color_pref = style_hints.get("color_preference")
    if color_pref == "warm":
        target_warmth = min(target_warmth + 0.25, 1.0)
    elif color_pref == "cool":
        target_warmth = max(target_warmth - 0.25, 0.0)
    temp_match = 1.0 - abs(clip.color_temperature - target_warmth)

    # 4. Duration suitability — speed-fit aware. The render layer can play a
    #    clip anywhere in [SPEED_FIT_MIN, SPEED_FIT_MAX], so judge the closest
    #    duration the clip can *present*, not its native length. This only
    #    raises dur_match for near-fits (a 5.6s scene is a perfect 8s phrase
    #    at 0.7x); exact fits are unaffected.
    target_dur = 3.0 + (1.0 - target_energy) * 12.0
    presentable_min = clip.duration_sec / SPEED_FIT_MAX
    presentable_max = clip.duration_sec / SPEED_FIT_MIN
    effective_dur = min(max(target_dur, presentable_min), presentable_max)
    dur_ratio = effective_dur / target_dur
    if dur_ratio < 0.3:
        dur_match = dur_ratio / 0.3 * 0.5
    elif dur_ratio > 3.0:
        dur_match = 0.5 / (dur_ratio / 3.0)
    else:
        dur_match = 1.0 - abs(1.0 - dur_ratio) * 0.3

    # 5. Saturation match (high energy → saturated, low → desaturated)
    sat_match = 1.0 - abs(clip.color_saturation - target_energy * 0.7)

    # 6. Contrast match (percussive → high contrast)
    contrast_match = 1.0 - abs(clip.contrast_mean - phrase.percussive_ratio)

    # 6b. Onset-density ↔ motion-jitter match. Both sides are percentile-ranked
    # within their respective pools (phrases-per-set, scenes-per-corpus) so the
    # comparison is scale-free: busy phrases prefer high-jitter clips
    # regardless of absolute tempo or motion magnitude. Orthogonal to energy.
    onset_match = 1.0 - abs(clip.motion_std_rank - phrase.onset_density_rank)

    # 7. Semantic bonus
    semantic_bonus = 0.0
    if clip.has_semantic:
        semantic_bonus = 0.1
        if clip.mood_energy == "intense" and target_energy > 0.7:
            semantic_bonus += 0.15
        elif clip.mood_energy == "calm" and target_energy < 0.3:
            semantic_bonus += 0.15
        elif clip.mood_energy == "moderate" and 0.3 <= target_energy <= 0.7:
            semantic_bonus += 0.1

    # 8. Style hints bonus/penalty
    style_bonus = 0.0
    preferred_styles = style_hints.get("preferred_styles", [])
    avoid_styles = style_hints.get("avoid_styles", [])
    if clip.visual_style:
        if clip.visual_style in preferred_styles:
            style_bonus += 0.5
        if clip.visual_style in avoid_styles:
            style_bonus -= 1.0

    preferred_sources = style_hints.get("preferred_sources", [])
    avoid_sources = style_hints.get("avoid_sources", [])
    for ps in preferred_sources:
        if ps.lower() in clip.source_name.lower():
            style_bonus += 0.3
            break
    for avs in avoid_sources:
        if avs.lower() in clip.source_name.lower():
            style_bonus -= 2.0
            break

    # 9. Lyrics-visual matching
    #    Exact lexical matches PLUS canonical-vocabulary matches, so synonyms
    #    ("ocean" lyric ↔ "sea" tag → both canonical "water") finally count.
    lyrics_bonus = 0.0
    if phrase.lyrics_keywords and clip.has_semantic and clip.content_tags:
        lyric_set = set(kw.lower() for kw in phrase.lyrics_keywords)
        tag_set = set(tag.lower() for tag in clip.content_tags)
        matches = lyric_set & tag_set
        if phrase.canonical_keywords and clip.canonical_tags:
            matches = matches | (set(phrase.canonical_keywords)
                                 & set(clip.canonical_tags))
        if matches:
            # Scaled bonus: more matches = bigger bonus, but diminishing
            lyrics_bonus = min(len(matches) * 0.25, 0.8)

    # 10. Loopability bonus — prefer loopable clips for long phrases
    loop_bonus = 0.0
    if clip.loopable and phrase.duration_sec > clip.duration_sec:
        loop_bonus = 0.15

    # 11. CLIP semantic similarity (lyrics text → visual embedding)
    clip_sim_bonus = 0.0
    if (phrase.clip_text_embedding is not None and clip.clip_embedding is not None):
        sim = float(np.dot(phrase.clip_text_embedding, clip.clip_embedding))
        clip_sim_bonus = max(0, sim) * 0.6

    # 12. Scene quality — soft, non-destructive penalty (flag_quality.py).
    #     A dead scene (black/blown/frozen) sinks ~3 pts but stays selectable.
    quality_penalty = -(1.0 - clip.quality_score) * QUALITY_PENALTY_WEIGHT

    # Soft de-emphasis of duration matching when the per-phrase BPM detection
    # is unconfident (breakdowns, intros/outros, transitions). At zero
    # confidence dur_match retains 60% of its weight — mild, not "near-disable."
    dur_confidence_weight = 0.6 + 0.4 * phrase.bpm_confidence

    # Weighted combination
    score = (
        motion_match * 1.0 +
        bright_match * 0.8 +
        temp_match * 0.7 +
        dur_match * 0.9 * dur_confidence_weight +
        sat_match * 0.5 +
        contrast_match * 0.5 +
        onset_match * 0.4 +
        semantic_bonus +
        style_bonus +
        lyrics_bonus +
        loop_bonus +
        clip_sim_bonus +
        quality_penalty
    )

    return score


# ─── MMR Re-ranking (Phase C) ──────────────────────────────────────────────

def _l2_normalize(matrix):
    """L2-normalize rows; zero-rows stay zero."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    return matrix / norms


def mmr_rerank(scored, prev_embeddings, lam, embed_dim=512):
    """Re-rank a scored candidate pool with MMR diversity penalty.

    `scored`           list of (raw_score, SceneClip), pre-sorted descending
                        — only the top MMR_POOL are passed in
    `prev_embeddings`  numpy array (n_recent, embed_dim), embeddings of clips
                        previously selected (last MMR_RECENT_WINDOW); empty
                        for the first phrase
    `lam`              MMR_LAMBDA

    Score normalization is per-phrase min-max within the pool — without this,
    `mmr_score = score - lam * cosine` is brittle because the raw scoring
    function produces values whose range varies wildly per phrase (a phrase
    where all candidates score 0.6–0.8 sees a different effective λ than one
    where they score 1.5–4.0).

    Returns (re_sorted_scored, diagnostics) where re_sorted_scored is a list
    of (mmr_score, clip) and diagnostics is a list of dicts for logging.
    """
    if not scored:
        return scored, []

    raw_scores = np.array([s for s, _ in scored], dtype=np.float64)
    s_min, s_max = raw_scores.min(), raw_scores.max()
    if s_max - s_min < 1e-9:
        # degenerate: all candidates score the same — no normalization possible
        norm_scores = np.zeros_like(raw_scores)
    else:
        norm_scores = (raw_scores - s_min) / (s_max - s_min)

    # Build candidate embedding matrix. Clips without sidecars get a zero vector
    # which will produce cosine 0 vs any prev → no penalty, no boost (graceful
    # fallback for the missing-embedding case).
    cand_emb = np.zeros((len(scored), embed_dim), dtype=np.float64)
    for i, (_, clip) in enumerate(scored):
        if clip.clip_embedding is not None:
            cand_emb[i] = clip.clip_embedding
    cand_emb = _l2_normalize(cand_emb)

    if prev_embeddings.shape[0] == 0:
        max_sim = np.zeros(len(scored))
    else:
        prev = _l2_normalize(prev_embeddings.astype(np.float64))
        sims = cand_emb @ prev.T            # (n_candidates, n_recent)
        max_sim = sims.max(axis=1)

    mmr_scores = norm_scores - lam * max_sim

    # Diagnostics (one per candidate; caller decides whether to record)
    diagnostics = [
        {
            "raw_score": float(raw_scores[i]),
            "norm_score": float(norm_scores[i]),
            "max_recent_cosine": float(max_sim[i]),
            "mmr_score": float(mmr_scores[i]),
            "source_name": scored[i][1].source_name,
            "scene_index": scored[i][1].scene_index,
        }
        for i in range(len(scored))
    ]

    # Re-sort by mmr_score descending; preserve (mmr_score, clip) shape so the
    # downstream weighted-random sampler can use the same code path.
    order = np.argsort(-mmr_scores)
    re_sorted = [(float(mmr_scores[i]), scored[i][1]) for i in order]
    return re_sorted, diagnostics


# ─── Perceptual Diversity Metrics (Phase D) ────────────────────────────────

def compute_perceptual_diversity(selections, scenes):
    """Measure perceptual diversity over the final ordered clip selection.

    Operates after select_clips() has returned. The metrics are what the
    viewer actually experiences (post-MMR if Phase C is active, post-
    weighted-random in all cases). Per the audit-driven acceptance bar,
    we track close-pair *counts* at multiple cosine thresholds rather than
    a single mean — repeats are about specific pairs, not aggregate.

    Returns a dict suitable for printing and for committing to
    bench/perceptual_baselines.json.
    """
    # Map (source_name, scene_index) -> SceneClip so we can recover embeddings
    # from selection dicts. (clip_start no longer identifies a scene — window
    # mode varies the start offset within the scene per reuse.)
    lookup = {(s.source_name, s.scene_index): s for s in scenes}

    chrono = []
    n_excluded = 0
    for sel in selections:
        key = (sel["clip_source_name"], sel.get("scene_index"))
        clip = lookup.get(key)
        if clip is None or clip.clip_embedding is None:
            n_excluded += 1
            chrono.append(None)
        else:
            chrono.append(clip.clip_embedding)

    def cos(a, b):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    consec_sims = []
    for i in range(1, len(chrono)):
        if chrono[i] is not None and chrono[i-1] is not None:
            consec_sims.append(cos(chrono[i], chrono[i-1]))

    def count_pairs(window, thresholds):
        counts = {t: 0 for t in thresholds}
        for i in range(len(chrono)):
            if chrono[i] is None:
                continue
            for j in range(max(0, i - window), i):
                if chrono[j] is None:
                    continue
                sim = cos(chrono[i], chrono[j])
                for t in thresholds:
                    if sim >= t:
                        counts[t] += 1
        return counts

    return {
        "n_selections": len(selections),
        "n_excluded": n_excluded,
        "n_consec_pairs": len(consec_sims),
        "consec_mean": float(np.mean(consec_sims)) if consec_sims else 0.0,
        "consec_median": float(np.median(consec_sims)) if consec_sims else 0.0,
        "consec_p90": float(np.percentile(consec_sims, 90)) if consec_sims else 0.0,
        "consec_p95": float(np.percentile(consec_sims, 95)) if consec_sims else 0.0,
        "close_pairs_w5": count_pairs(5, [0.85, 0.80, 0.75]),
        "close_pairs_w30": count_pairs(30, [0.90, 0.85]),
    }


# ─── Clip Selection ────────────────────────────────────────────────────────

def merge_phrases_adaptive(phrase_features):
    """Merge adjacent low-energy, stable phrases into longer holds.
    High-energy sections keep short phrases (fast cuts).
    Low/sustained sections get merged (let the clip breathe).
    """
    if not phrase_features:
        return phrase_features

    merged = [phrase_features[0]]

    for phrase in phrase_features[1:]:
        prev = merged[-1]
        prev_dur = prev.duration_sec
        combined_dur = prev_dur + phrase.duration_sec

        # Conditions for merging:
        # 1. Both phrases are below energy threshold
        # 2. Energy is stable between them (not a transition)
        # 3. Combined duration doesn't exceed max
        # 4. Same track (don't merge across track boundaries)
        should_merge = (
            prev.energy < MERGE_ENERGY_THRESHOLD
            and phrase.energy < MERGE_ENERGY_THRESHOLD
            and abs(prev.energy - phrase.energy) < MERGE_ENERGY_DELTA
            and combined_dur <= MAX_MERGED_DURATION
            and prev.track_title == phrase.track_title
            and prev.energy_shape in ("sustain", "decay")
        )

        if should_merge:
            # Extend the previous phrase
            merged[-1] = PhraseFeatures(
                start_sec=prev.start_sec,
                end_sec=phrase.end_sec,
                duration_sec=combined_dur,
                energy=(prev.energy * prev_dur + phrase.energy * phrase.duration_sec) / combined_dur,
                energy_peak=max(prev.energy_peak, phrase.energy_peak),
                energy_shape="sustain",
                bass_ratio=(prev.bass_ratio + phrase.bass_ratio) / 2,
                brightness_audio=(prev.brightness_audio + phrase.brightness_audio) / 2,
                percussive_ratio=(prev.percussive_ratio + phrase.percussive_ratio) / 2,
                bpm=(prev.bpm + phrase.bpm) / 2,
                track_title=prev.track_title,
            )
        else:
            merged.append(phrase)

    return merged


def _beatlock_cycle(natural_cycle_sec, bpm, beats=None, start_sec=0.0, n_frames=0):
    """Fit a loop/pingpong cycle to an integer beat count.

    Returns (speed, cycle_frames) — playback rate (setpts=PTS/speed) that
    makes the cycle land on the beat grid, and that retimed cycle's frame
    count — or None when nothing fits the beat-lock speed band.

    With only a bpm, candidates are k·beat_sec cycles and the one nearest
    native speed wins. When the actual beat array is provided, candidates
    also include cycles anchored to the real beat positions, and each is
    scored by the mean distance of its seams (start + j·cycle) to the
    actual beats — the grid wobbles off any fixed lattice, so optimizing
    against real beats beats trusting the average period.
    """
    if not (BEATLOOP_BPM_MIN <= bpm <= BEATLOOP_BPM_MAX) or natural_cycle_sec <= 0:
        return None
    beat_sec = 60.0 / bpm
    k0 = int(round(natural_cycle_sec / beat_sec))

    bt = None
    idx0 = 0
    if beats is not None and len(beats) and n_frames > 0:
        bt = np.asarray(beats, dtype=np.float64)
        idx0 = int(np.searchsorted(bt, start_sec - 1e-3))

    candidates = set()
    for k in (k0 - 1, k0, k0 + 1):
        if k < 1:
            continue
        candidates.add(int(round(k * beat_sec * FPS)))
        # Anchor variant: first seam exactly on the k-th actual beat.
        if bt is not None and idx0 + k < len(bt):
            candidates.add(int(round((bt[idx0 + k] - start_sec) * FPS)))

    best = None  # (score, |speed-1|, speed, cf)
    for cf in candidates:
        if cf < 2:
            continue
        speed = natural_cycle_sec / (cf / FPS)
        if not (BEATLOOP_SPEED_MIN <= speed <= BEATLOOP_SPEED_MAX):
            continue
        if bt is not None and n_frames // cf > 0:
            seams = [start_sec + j * cf / FPS for j in range(1, n_frames // cf + 1)]
            dists = []
            for s in seams:
                i = int(np.searchsorted(bt, s))
                d = min((abs(bt[j] - s) for j in (i - 1, i) if 0 <= j < len(bt)),
                        default=0.0)
                dists.append(d)
            score = float(np.mean(dists))
        else:
            score = 0.0  # no visible seams (or no grid): any in-band cf is fine
        key = (score, abs(speed - 1.0))
        if best is None or key < best[:2]:
            best = (*key, speed, cf)
    if best is None:
        return None
    return best[2], best[3]


def _snap_to_beat(t, beat_grid):
    """Nearest beat to t, or t itself if no grid is available."""
    if beat_grid is None or not len(beat_grid):
        return t
    idx = np.searchsorted(beat_grid, t)
    best = t, np.inf
    for j in (idx - 1, idx):
        if 0 <= j < len(beat_grid) and abs(beat_grid[j] - t) < best[1]:
            best = float(beat_grid[j]), abs(beat_grid[j] - t)
    return best[0]


def select_clips(scenes, phrase_features, style_hints=None, beat_times=None):
    """Select clips with hard variety enforcement and adaptive diversity."""
    if style_hints is None:
        style_hints = {}
    beat_grid = (np.asarray(beat_times, dtype=np.float64)
                 if beat_times is not None and len(beat_times) else None)

    selections = []
    selected_clips = []      # SceneClip objects in selection order (for MMR)
    recent_sources = []      # hard block window (source-level)
    recent_scenes = []       # hard block window (scene-level)
    source_use_count = {}    # global usage tracking
    scene_use_count = {}     # (source_name, scene_index) → uses; identical frames decay fast

    # Frame-exact timeline anchor: clip i always ends at the frame nearest
    # (phrase_i.end_sec - timeline_start). Per-clip rounding cannot accumulate.
    timeline_start = phrase_features[0].start_sec if phrase_features else 0.0
    frames_emitted = 0

    # Truncate MMR diagnostic log at start of each run so it doesn't accumulate
    # across multiple --set invocations in a batch.
    if MMR_LAMBDA > 0:
        MMR_DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        MMR_DIAG_PATH.write_text("")

    total_sources = len(set(s.source_name for s in scenes))
    total_phrases = len(phrase_features)
    # Cap: no source can exceed this many uses
    max_source_uses = max(3, int(total_phrases / max(total_sources, 1) * MAX_SOURCE_USAGE_MULT))

    for i, phrase in enumerate(phrase_features):
        scored = []
        for clip in scenes:
            # Hard block: perceptual near-duplicate (Phase A).
            # near_dup-flagged clips are excluded from the strict pool the same
            # way as VARIETY_WINDOW — both are perceptual-diversity constraints.
            if "near_dup" in clip.cull_flags:
                continue

            # Hard block: skip if source was used recently
            if clip.source_name in recent_sources:
                continue

            # Hard block: skip exact scene if used recently
            scene_key = f"{clip.source_name}:{clip.scene_index}"
            if scene_key in recent_scenes:
                continue

            # Hard block: skip if source hit usage ceiling
            use_count = source_use_count.get(clip.source_name, 0)
            if use_count >= max_source_uses:
                continue

            s = score_scene(clip, phrase, style_hints)

            # Steeper soft penalty for overused sources
            if use_count > 0:
                s *= 1.0 / (1.0 + REUSE_PENALTY * use_count)

            # Per-scene penalty on top — REUSE_PENALTY is keyed by source, so
            # without this a top-scoring scene legally recurred every
            # SCENE_VARIETY_WINDOW phrases showing identical frames.
            scene_uses = scene_use_count.get((clip.source_name, clip.scene_index), 0)
            if scene_uses > 0:
                s *= 1.0 / (1.0 + SCENE_REUSE_PENALTY * scene_uses)

            scored.append((s, clip))

        if not scored:
            # Fallback 1: relax perceptual-diversity constraints (VARIETY_WINDOW
            # AND near_dup hard skip) together. Keep scene dedup + usage ceiling.
            for clip in scenes:
                scene_key = f"{clip.source_name}:{clip.scene_index}"
                if scene_key in recent_scenes:
                    continue
                use_count = source_use_count.get(clip.source_name, 0)
                if use_count >= max_source_uses:
                    continue
                s = score_scene(clip, phrase, style_hints)
                scored.append((s, clip))

        if not scored:
            # Fallback 2: drop scene dedup too. Last meaningful constraint
            # (usage ceiling) still applies.
            for clip in scenes:
                use_count = source_use_count.get(clip.source_name, 0)
                if use_count >= max_source_uses:
                    continue
                s = score_scene(clip, phrase, style_hints)
                scored.append((s, clip))

        if not scored:
            # Last resort: fully relaxed (including usage ceiling). Should
            # essentially never fire on a healthy corpus.
            for clip in scenes:
                s = score_scene(clip, phrase, style_hints)
                scored.append((s, clip))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ── MMR diversity re-rank (Phase C) ─────────────────────────────────
        # Take a wider top-MMR_POOL, then re-rank by score - λ * max_cosine to
        # the last MMR_RECENT_WINDOW selected clips. The final weighted-random
        # draw happens over the TOP_CANDIDATES of the re-ranked list.
        pool = scored[:min(MMR_POOL, len(scored))]
        if MMR_LAMBDA > 0 and pool:
            # Build the recent-embedding matrix (may be empty for first phrase)
            recent_clips = selected_clips[-MMR_RECENT_WINDOW:]
            recent_embs = [c.clip_embedding for c in recent_clips
                           if c.clip_embedding is not None]
            if recent_embs:
                prev_matrix = np.array(recent_embs, dtype=np.float64)
            else:
                prev_matrix = np.zeros((0, 512), dtype=np.float64)

            embed_dim = (recent_embs[0].shape[0] if recent_embs
                         else (pool[0][1].clip_embedding.shape[0]
                               if pool[0][1].clip_embedding is not None else 512))
            pool, mmr_diag = mmr_rerank(pool, prev_matrix, MMR_LAMBDA, embed_dim)

            # Diagnostic logging for the first MMR_DIAGNOSTIC_PHRASES phrases.
            if i < MMR_DIAGNOSTIC_PHRASES:
                with open(MMR_DIAG_PATH, "a") as f:
                    for k, diag in enumerate(mmr_diag):
                        f.write(
                            f"phrase={i} cand={k} src={diag['source_name'][:30]} "
                            f"scene={diag['scene_index']} "
                            f"raw={diag['raw_score']:.3f} norm={diag['norm_score']:.3f} "
                            f"max_recent={diag['max_recent_cosine']:.3f} "
                            f"mmr={diag['mmr_score']:.3f}\n"
                        )

        # Wider random pool for more diversity (operates on MMR-reranked list
        # when MMR is active; on raw-sorted list when MMR_LAMBDA == 0)
        top_n = min(TOP_CANDIDATES, len(pool))
        top_scores = [s for s, _ in pool[:top_n]]
        if top_scores:
            weights = np.array(top_scores)
            # Ensure positive weights — MMR scores can go negative when the
            # diversity penalty exceeds the normalized score.
            weights = weights - weights.min() + 0.01
            weights = weights / weights.sum()
            chosen_idx = np.random.choice(top_n, p=weights)
        else:
            chosen_idx = 0

        best_score, best_clip = pool[chosen_idx]

        # ── Frame-exact render plan ──────────────────────────────────────
        # The clip must end at the frame nearest the phrase's end on the
        # audio timeline. v2.0 truncated clips to scene length (most scenes
        # are shorter than their phrase), which desynced every later cut.
        # Snap the planned end to the beat grid first: a near-no-op for
        # unmerged 2.2.0 phrases (already beat times), but protects merged
        # boundaries and pre-2.2.0 JSONs from landing between beats.
        end_frame = int(round((_snap_to_beat(phrase.end_sec, beat_grid) - timeline_start) * FPS))
        n_frames = max(end_frame - frames_emitted, FPS)  # never under 1s
        target_dur = n_frames / FPS
        frames_emitted += n_frames
        # Audio-timeline instant where this clip begins in the render —
        # the reference for beat-locked seam placement.
        clip_audio_start = timeline_start + (frames_emitted - n_frames) / FPS

        scene_dur = best_clip.duration_sec
        render_mode = "window"
        speed = 1.0
        clip_start = best_clip.start_sec
        src_duration = target_dur
        cycle_frames = None

        # Beat period for beat-locked cycles: prefer the LOCAL beat grid's
        # median IOI over phrase.bpm (a windowed average) — locking to the
        # average drifts seams off the actual beats when tempo ramps.
        local_bpm = phrase.bpm
        if beat_grid is not None and len(beat_grid):
            i0 = int(np.searchsorted(beat_grid, phrase.start_sec))
            i1 = int(np.searchsorted(beat_grid, phrase.end_sec))
            if i1 - i0 >= 4:
                local_ioi = float(np.median(np.diff(beat_grid[i0:i1 + 1])))
                if local_ioi > 0:
                    local_bpm = 60.0 / local_ioi

        if scene_dur >= target_dur * SPEED_FIT_MAX:
            # Long scene: cut a phrase-length window. Vary the offset so a
            # reused scene shows different footage, not its opening frames.
            max_off = scene_dur - target_dur - 0.1
            if max_off > 0.5:
                clip_start += float(np.random.uniform(0, max_off))
        elif scene_dur >= target_dur * SPEED_FIT_MIN:
            # Near fit: play the whole scene, speed-adjusted to land exactly
            # (slow-mo below 1.0, speed-up above).
            render_mode = "speedfit"
            speed = scene_dur / target_dur
            src_duration = scene_dur
        elif best_clip.loopable:
            # Detected seamless loop: repeat, beat-locked when a cycle of
            # k beats fits the speed band (seams land on beats).
            render_mode = "loop"
            src_duration = scene_dur
            fit = _beatlock_cycle(scene_dur, local_bpm, beat_grid,
                                  clip_audio_start, n_frames)
            if fit:
                render_mode = "beatloop"
                speed, cycle_frames = fit
        elif scene_dur <= PINGPONG_MAX_SCENE:
            # Forward-reverse cycling fills any phrase from any short scene
            # with no visible seam. Full cycle = forward + reverse.
            render_mode = "pingpong"
            src_duration = scene_dur
            fit = _beatlock_cycle(2 * scene_dur, local_bpm, beat_grid,
                                  clip_audio_start, n_frames)
            if fit:
                render_mode = "beatpingpong"
                speed, cycle_frames = fit
        else:
            # Mid-length scene, too long to reverse in RAM: plain loop
            # (jump cut at the loop point — rare; scenes 15s–0.7×phrase).
            # Beat-locking matters most here: the jump cut becomes a cut
            # ON the beat instead of a mid-bar glitch.
            render_mode = "loop"
            src_duration = scene_dur
            fit = _beatlock_cycle(scene_dur, local_bpm, beat_grid,
                                  clip_audio_start, n_frames)
            if fit:
                render_mode = "beatloop"
                speed, cycle_frames = fit

        selections.append({
            "phrase_index": i,
            "audio_start": phrase.start_sec,
            "audio_end": phrase.end_sec,
            "audio_duration": phrase.duration_sec,
            "clip_source": best_clip.source_path,
            "clip_source_name": best_clip.source_name,
            "scene_index": best_clip.scene_index,
            "sub_index": best_clip.sub_index,
            "clip_start": round(clip_start, 3),
            "clip_duration": round(target_dur, 4),
            "n_frames": n_frames,
            "render_mode": render_mode,
            "speed": round(speed, 4),
            "src_duration": round(src_duration, 3),
            "cycle_frames": cycle_frames,
            "clip_motion": round(best_clip.motion_mean, 2),
            "clip_brightness": round(best_clip.brightness_mean, 3),
            "clip_color_temp": round(best_clip.color_temperature, 3),
            "match_score": round(best_score, 3),
            "audio_energy": phrase.energy,
            "track_title": phrase.track_title,
            "clip_quality": round(best_clip.quality_score, 2),
            "clip_cull_flags": best_clip.cull_flags,
        })

        # Update source tracking
        recent_sources.append(best_clip.source_name)
        if len(recent_sources) > VARIETY_WINDOW:
            recent_sources.pop(0)

        # Update scene tracking
        scene_key = f"{best_clip.source_name}:{best_clip.scene_index}"
        recent_scenes.append(scene_key)
        if len(recent_scenes) > SCENE_VARIETY_WINDOW:
            recent_scenes.pop(0)

        # MMR history (Phase C): the clip object itself, for cosine vs candidates.
        # No window cap here — the slicing happens at use time with [-MMR_RECENT_WINDOW:].
        selected_clips.append(best_clip)

        source_use_count[best_clip.source_name] = source_use_count.get(best_clip.source_name, 0) + 1
        skey = (best_clip.source_name, best_clip.scene_index)
        scene_use_count[skey] = scene_use_count.get(skey, 0) + 1

    return selections


# ─── Progress Helpers ─────────────────────────────────────────────────────

def fmt_time(seconds):
    """Format seconds as h:mm:ss or m:ss."""
    if seconds < 0:
        return "--:--"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bar(pct, width=20):
    """Render a progress bar."""
    filled = int(width * min(pct, 1.0))
    return "█" * filled + "░" * (width - filled)


def fmt_size(bytes_val):
    """Format bytes as human-readable size."""
    if bytes_val < 1024 * 1024:
        return f"{bytes_val / 1024:.0f} KB"
    elif bytes_val < 1024 * 1024 * 1024:
        return f"{bytes_val / (1024 * 1024):.1f} MB"
    else:
        return f"{bytes_val / (1024 * 1024 * 1024):.2f} GB"


# ─── Video Assembly ────────────────────────────────────────────────────────

BASE_VF = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
           "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1")
ENC_ARGS = ["-c:v", "libx264", "-crf", "20", "-preset", "fast", "-an"]


def render_black_filler(output_path, n_frames):
    """Black filler clip — a failed extraction must still occupy its frames,
    otherwise every cut after it shifts off the audio timeline."""
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", f"color=c=black:s=1920x1080:r={FPS}",
        "-frames:v", str(n_frames), *ENC_ARGS, str(output_path)
    ], capture_output=True, timeout=120)
    return output_path.exists() and output_path.stat().st_size > 0


def extract_clip(source_path, start_sec, output_path, n_frames,
                 render_mode="window", speed=1.0, src_duration=None,
                 cycle_frames=None):
    """Render exactly n_frames at FPS from a source scene.

    window       — cut n_frames straight from start_sec (scene ≥ phrase)
    speedfit     — play the whole scene speed-adjusted via setpts to land exactly
    loop         — repeat the scene start-to-start (loopable-flagged or too long
                   to reverse)
    pingpong     — forward+reverse cycle repeated to fill (seamless for any scene)
    beatloop /   — loop/pingpong with the cycle retimed (setpts, constant rate)
    beatpingpong   to exactly cycle_frames = k beats, so every seam lands on a
                   beat. The cycle file is built frame-exact (-frames:v + tpad),
                   so seams can't drift across repetitions.

    All modes are frame-counted (-frames:v), never wall-clock (-t output):
    the sync contract is "rendered frames == planned frames". tpad clones
    the last frame as insurance against the source running dry a frame early.
    """
    if render_mode == "window":
        cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-i", source_path,
               "-vf", f"{BASE_VF},fps={FPS},tpad=stop_mode=clone:stop=8",
               "-frames:v", str(n_frames), *ENC_ARGS, str(output_path)]
        subprocess.run(cmd, capture_output=True, timeout=300)

    elif render_mode == "speedfit":
        # -t must be an INPUT option (before -i): it bounds how much source is
        # read. As an output option it would cap the result at src_duration,
        # silently undoing the setpts stretch.
        cmd = ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(src_duration),
               "-i", source_path,
               "-vf", f"{BASE_VF},setpts=PTS/{speed:.6f},fps={FPS},"
                      f"tpad=stop_mode=clone:stop=8",
               "-frames:v", str(n_frames), *ENC_ARGS, str(output_path)]
        subprocess.run(cmd, capture_output=True, timeout=300)

    else:  # loop / pingpong (± beat-lock) — two-step via an intermediate cycle file
        pingpong = render_mode in ("pingpong", "beatpingpong")
        beat_locked = render_mode in ("beatloop", "beatpingpong") and cycle_frames
        # setpts before fps: retime the source, then resample to CFR.
        retime = f"setpts=PTS/{speed:.6f}," if beat_locked and abs(speed - 1.0) > 1e-6 else ""
        cycle = output_path.with_suffix(".cycle.mp4")
        frame_cap = ["-frames:v", str(cycle_frames)] if beat_locked else []
        if pingpong:
            # trim=start_frame=1 drops the reversed half's first frame
            # — it duplicates the forward half's last frame, which
            # stutters at the seam and gets dropped (with a timestamp
            # gap) by the later concat-copy.
            graph = (f"[0:v]{BASE_VF},{retime}fps={FPS},split[f][b];"
                     f"[b]reverse,trim=start_frame=1,setpts=PTS-STARTPTS[r];"
                     f"[f][r]concat=n=2:v=1[v]")
            out_label = "[v]"
            if beat_locked:
                graph += ";[v]tpad=stop_mode=clone:stop=8[vo]"
                out_label = "[vo]"
            cmd1 = ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(src_duration),
                    "-i", source_path,
                    "-filter_complex", graph, "-map", out_label,
                    *frame_cap, *ENC_ARGS, str(cycle)]
        else:
            vf = f"{BASE_VF},{retime}fps={FPS}"
            if beat_locked:
                vf += ",tpad=stop_mode=clone:stop=8"
            cmd1 = ["ffmpeg", "-y", "-ss", str(start_sec), "-t", str(src_duration),
                    "-i", source_path,
                    "-vf", vf,
                    *frame_cap, *ENC_ARGS, str(cycle)]
        subprocess.run(cmd1, capture_output=True, timeout=300)
        if not cycle.exists() or cycle.stat().st_size < 1000:
            return False

        if beat_locked:
            per_cycle = cycle_frames
        else:
            # Conservative cycle length (could be ±2 frames after fps rounding):
            # underestimate so we always loop *enough*; surplus is trimmed.
            per_pass = max(int(src_duration * FPS) - 2, 1)
            per_cycle = per_pass * (2 if pingpong else 1)
        loops = math.ceil(n_frames / per_cycle)  # stream_loop N = N+1 plays
        cmd2 = ["ffmpeg", "-y", "-stream_loop", str(loops), "-i", str(cycle),
                "-frames:v", str(n_frames), *ENC_ARGS, str(output_path)]
        subprocess.run(cmd2, capture_output=True, timeout=300)
        cycle.unlink(missing_ok=True)

    return output_path.exists() and output_path.stat().st_size > 1000


def assemble_video(selections, output_path, audio_path, output_dir=None, global_state=None):
    """Assemble clips into final video with audio."""
    if output_dir is None:
        output_dir = output_path.parent / "mv2_build"
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    n_clips = len(selections)
    print(f"\n    Extracting {n_clips} clips")
    print(f"    {'─' * 68}")

    clip_paths = []
    failed = 0
    timings = []
    extract_start = time.time()

    for i, sel in enumerate(selections):
        clip_path = clip_dir / f"clip_{i:04d}.mp4"
        clip_t0 = time.time()

        ok = extract_clip(
            sel["clip_source"], sel["clip_start"], clip_path,
            n_frames=sel["n_frames"],
            render_mode=sel.get("render_mode", "window"),
            speed=sel.get("speed", 1.0),
            src_duration=sel.get("src_duration"),
            cycle_frames=sel.get("cycle_frames"),
        )
        if not ok:
            # Frame-exact contract: a failed clip still occupies its frames
            # (black filler) so later cuts stay on the audio timeline.
            ok = render_black_filler(clip_path, sel["n_frames"])
            failed += 1
        clip_elapsed = time.time() - clip_t0
        timings.append(clip_elapsed)

        if ok:
            clip_paths.append((clip_path, sel))

        # Update global clip counter if in batch mode
        if global_state:
            global_state["clips_done"] += 1

        # Progress line every 5 clips
        if (i + 1) % 5 == 0 or i == n_clips - 1:
            pct = (i + 1) / n_clips
            avg_t = sum(timings[-20:]) / len(timings[-20:])
            remaining = avg_t * (n_clips - i - 1)
            elapsed = time.time() - extract_start
            src_short = sel["clip_source_name"][:25]

            # Global progress
            g_info = ""
            if global_state:
                g_pct = global_state["clips_done"] / global_state["clips_total"]
                g_remain = avg_t * (global_state["clips_total"] - global_state["clips_done"])
                g_info = f"  all:{fmt_bar(g_pct, 12)} {global_state['clips_done']}/{global_state['clips_total']} ETA:{fmt_time(g_remain)}"

            status = "✓" if ok else "✗"
            print(f"    {fmt_bar(pct)} {i+1:>4}/{n_clips}"
                  f"  {clip_elapsed:>4.1f}s  {status}"
                  f"  ETA:{fmt_time(remaining)}"
                  f"{g_info}", flush=True)

    total_extract = time.time() - extract_start
    print(f"    {'─' * 68}")
    print(f"    Extracted: {len(clip_paths)}/{n_clips}"
          f"  ({failed} failed → black filler)  in {fmt_time(total_extract)}")

    if not clip_paths:
        print("    ERROR: No clips extracted!")
        return

    # ── Frame-count verification ────────────────────────────────────────
    # The sync contract is "rendered frames == planned frames" per clip; any
    # mismatch shifts every later cut off the audio timeline, so check all.
    print(f"    Verifying frame counts...", end=" ", flush=True)
    drift = 0
    bad = []
    for path, sel in clip_paths:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True)
        try:
            n = int(probe.stdout.strip())
        except ValueError:
            bad.append((path.name, "unreadable"))
            continue
        if n != sel["n_frames"]:
            bad.append((path.name, f"{n} frames, planned {sel['n_frames']}"))
            drift += n - sel["n_frames"]
    if bad:
        print(f"{len(bad)} clips off-plan (net drift {drift:+d} frames = {drift/FPS:+.2f}s)")
        for name, msg in bad[:8]:
            print(f"      {name}: {msg}")
    else:
        print(f"all {len(clip_paths)} clips frame-exact")

    # Build concat list
    concat_file = output_dir / "concat.txt"
    with open(concat_file, 'w') as f:
        for path, _ in clip_paths:
            f.write(f"file '{path.resolve()}'\n")

    # Concatenate. Duration accounting uses planned frames — verified above.
    video_duration = sum(sel["n_frames"] for _, sel in clip_paths) / FPS
    print(f"\n    Concatenating {len(clip_paths)} clips ({fmt_time(video_duration)})...",
          end=" ", flush=True)
    concat_t0 = time.time()
    concat_video = output_dir / "concat_raw.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_file), "-c", "copy",
        "-movflags", "+faststart", str(concat_video)
    ], capture_output=True, timeout=3600)
    print(f"done ({time.time() - concat_t0:.0f}s)")

    # Add audio
    audio_start = selections[0]["audio_start"]
    print(f"    Adding audio (from {fmt_time(audio_start)}, duration {fmt_time(video_duration)})...",
          end=" ", flush=True)
    audio_t0 = time.time()
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(concat_video),
        "-ss", str(audio_start),
        "-i", str(audio_path),
        "-t", str(video_duration),
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a", "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ], capture_output=True, timeout=3600)
    print(f"done ({time.time() - audio_t0:.0f}s)")

    # Cleanup
    for path, _ in clip_paths:
        path.unlink(missing_ok=True)
    concat_video.unlink(missing_ok=True)
    concat_file.unlink(missing_ok=True)
    if clip_dir.exists():
        try:
            clip_dir.rmdir()
        except OSError:
            pass


# ─── Main ─────────────────────────────────────────────────────────────────

def run_set(set_name, set_config, args, set_idx=1, total_sets=1, global_state=None):
    """Run the assembler for a single DJ set."""
    analysis_path = BASE_DIR / "sets" / set_config["analysis"]
    audio_path = set_config["audio"]
    output_dir = BASE_DIR / "sets" / f"{set_name}_mv_build"
    output_path = Path(args.output) if args.output else BASE_DIR / "sets" / set_config["output_name"]

    if not analysis_path.exists():
        print(f"\n  ERROR: Analysis file not found: {analysis_path}")
        return False, 0
    if not audio_path.exists():
        print(f"\n  ERROR: Audio file not found: {audio_path}")
        return False, 0

    set_start = time.time()

    # ── Set banner ──
    print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  [{set_idx}/{total_sets}]  {set_name}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if global_state and total_sets > 1:
        g_pct = global_state["sets_done"] / total_sets
        print(f"  Batch: {fmt_bar(g_pct, 25)} {global_state['sets_done']}/{total_sets} sets"
              f"  elapsed: {fmt_time(time.time() - global_state['batch_start'])}")

    # ── 1. Build scene database with text filtering ──
    print(f"\n  [1/5] Building scene database...", end=" ", flush=True)
    t0 = time.time()
    text_seconds = load_text_flags()
    text_sources = len(set(k for k in text_seconds))
    scenes = build_scene_database(text_seconds,
                                  salvage=not getattr(args, "no_salvage", False))
    n_sources = len(set(s.source_name for s in scenes))
    print(f"{len(scenes)} scenes from {n_sources} videos ({time.time()-t0:.1f}s)")
    if text_sources:
        print(f"    Text filter active: {text_sources} flagged sources, scenes with English text excluded")

    motions = [s.motion_mean for s in scenes]
    durs = [s.duration_sec for s in scenes]
    print(f"    Motion: {min(motions):.1f}–{max(motions):.1f} (mean {np.mean(motions):.1f})"
          f"  |  Duration: {min(durs):.1f}–{max(durs):.1f}s (median {np.median(durs):.1f}s)"
          f"  |  Semantic: {sum(1 for s in scenes if s.has_semantic)}")

    # ── 2. Load audio analysis ──
    print(f"\n  [2/5] Loading audio analysis...", end=" ", flush=True)
    data = json.loads(analysis_path.read_text())
    phrase_key = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[args.phrase_bars]
    phrases = data["phrases"][phrase_key]

    if args.segment:
        seg_start, seg_end = args.segment
        phrases = [p for p in phrases if p["start_sec"] >= seg_start and p["end_sec"] <= seg_end]
        print(f"{len(phrases)} phrases in {seg_start:.0f}–{seg_end:.0f}s")
    else:
        # Compute total audio duration from phrases
        audio_dur = phrases[-1]["end_sec"] if phrases else 0
        print(f"{len(phrases)} phrases ({args.phrase_bars}-bar), {fmt_time(audio_dur)}")

    # ── 3. Extract phrase features + adaptive merging ──
    print(f"  [3/5] Extracting phrase features...", end=" ", flush=True)
    t0 = time.time()
    phrase_features = extract_phrase_features(data, phrases)
    pre_merge = len(phrase_features)
    phrase_features = merge_phrases_adaptive(phrase_features)
    post_merge = len(phrase_features)
    print(f"{pre_merge} phrases → {post_merge} after adaptive merge ({time.time()-t0:.1f}s)")

    energies = [p.energy for p in phrase_features]
    durations = [p.duration_sec for p in phrase_features]
    print(f"    Energy: {min(energies):.2f}–{max(energies):.2f} (mean {np.mean(energies):.2f})")
    print(f"    Phrase durations: {min(durations):.1f}–{max(durations):.1f}s"
          f" (median {np.median(durations):.1f}s)")
    if pre_merge > post_merge:
        print(f"    Merged {pre_merge - post_merge} low-energy phrases into longer holds")

    # ── 3b. Load lyrics if available ──
    beat_times = data.get("beats", {}).get("times_sec", [])
    lyrics_path = BASE_DIR / "sets" / f"{set_name}.lyrics.json"
    if lyrics_path.exists():
        lyrics_data = json.loads(lyrics_path.read_text())

        # Move phrase boundaries off sung words onto word-clear beats before
        # any phrase-keyed lookups happen (word timings from Whisper).
        phrase_features, n_moved, n_vmerged = adjust_cuts_for_vocals(
            phrase_features, lyrics_data.get("segments", []), beat_times)
        if n_moved or n_vmerged:
            print(f"    Vocal-aware cuts: {n_moved} boundaries moved off sung words, "
                  f"{n_vmerged} merged away (continuous vocals)")

        phrase_lyrics = lyrics_data.get("phrase_lyrics", {}).get(
            {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[args.phrase_bars], {})
        lyrics_applied = 0
        for pf in phrase_features:
            # Find keywords for this phrase index
            kw = phrase_lyrics.get(str(pf.start_sec), [])
            if not kw:
                # Try matching by phrase index from the original timeline
                for idx_str, kw_list in phrase_lyrics.items():
                    # Check if any phrase overlaps
                    try:
                        idx = int(idx_str)
                        # Look up original phrase timing
                        if idx < len(phrases):
                            p = phrases[idx]
                            if (p["start_sec"] < pf.end_sec and
                                    p["end_sec"] > pf.start_sec):
                                kw = kw_list
                                break
                    except (ValueError, IndexError):
                        continue
            if kw:
                pf.lyrics_keywords = kw
                lyrics_applied += 1
        total_kw = len(lyrics_data.get("all_keywords", []))
        print(f"    Lyrics loaded: {total_kw} keywords, {lyrics_applied}/{len(phrase_features)} phrases have lyrics")
    else:
        print(f"    No lyrics file found (run extract_lyrics.py --set {set_name} to add)")

    # ── 3c. Encode lyrics with CLIP for semantic matching ──
    any_has_clip_emb = any(c.clip_embedding is not None for c in scenes)  # was: scene_pool (undefined in this scope)
    phrases_with_lyrics = [pf for pf in phrase_features if pf.lyrics_keywords]
    if any_has_clip_emb and phrases_with_lyrics:
        try:
            import open_clip
            import torch
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=device)
            clip_model.eval()
            tokenizer = open_clip.get_tokenizer("ViT-B-32")

            encoded = 0
            for pf in phrases_with_lyrics:
                text = " ".join(pf.lyrics_keywords[:20])
                tokens = tokenizer([text]).to(device)
                with torch.no_grad():
                    feat = clip_model.encode_text(tokens)
                    feat = feat / feat.norm(dim=-1, keepdim=True)
                pf.clip_text_embedding = feat.cpu().numpy()[0].astype(np.float32)
                encoded += 1
            print(f"    CLIP text embeddings: {encoded} phrases encoded")

            # Map lyric keywords onto the canonical tag vocabulary so the
            # lyric↔tag dimension matches synonyms ("ocean" ↔ "sea"), not
            # just exact strings. Two deliberate restrictions (calibrated on
            # the 5-set corpus — see PR #13):
            #  - mapping-ONLY: a lyric word maps only if it ever appeared as
            #    a corpus tag. Free CLIP-cosine mapping of arbitrary words is
            #    junk-prone ('tryna' → 'dynamic' at 0.95).
            #  - DF filter: canonical terms tagging more than
            #    CANON_MATCH_MAX_DF of the corpus carry no signal in this
            #    corpus ('abstract', 'dynamic') and are excluded from
            #    lyric matching.
            vocab_path = BASE_DIR / "canonical_tags.json"
            if vocab_path.exists():
                vocab = json.loads(vocab_path.read_text())
                kw_map = dict(vocab.get("mapping", {}))
                df = {}
                for sc in scenes:
                    for t in set(sc.canonical_tags):
                        df[t] = df.get(t, 0) + 1
                selective = {t for t in vocab["terms"]
                             if df.get(t, 0) / max(len(scenes), 1) <= CANON_MATCH_MAX_DF}
                mapped_phrases = 0
                for pf in phrases_with_lyrics:
                    canon = sorted({kw_map[kw.lower()] for kw in pf.lyrics_keywords
                                    if kw_map.get(kw.lower()) in selective})
                    if canon:
                        pf.canonical_keywords = canon
                        mapped_phrases += 1
                print(f"    Canonical lyric keywords: {mapped_phrases} phrases mapped "
                      f"({len(selective)}/{len(vocab['terms'])} selective terms, "
                      f"DF ≤ {CANON_MATCH_MAX_DF:.0%})")

            del clip_model  # free memory
        except ImportError:
            print(f"    CLIP text encoding skipped (open_clip not installed)")

    # ── 4. Match scenes to phrases ──
    style_hints = set_config.get("style_hints", {})
    if style_hints:
        print(f"\n  [4/5] Matching scenes to phrases (style hints active)...", end=" ", flush=True)
    else:
        print(f"\n  [4/5] Matching scenes to phrases...", end=" ", flush=True)
    t0 = time.time()
    selections = select_clips(scenes, phrase_features, style_hints, beat_times=beat_times)
    print(f"done ({time.time()-t0:.1f}s)")

    # Quality of the selection — observability for tuning QUALITY_PENALTY_WEIGHT.
    flagged_sel = [s for s in selections if s.get("clip_cull_flags")]
    if any(s.get("clip_quality", 1.0) < 1.0 for s in selections):
        mean_q = np.mean([s.get("clip_quality", 1.0) for s in selections])
        print(f"    Selection quality: mean {mean_q:.2f}, {len(flagged_sel)} flagged clips kept")
        for s in flagged_sel[:10]:
            q = s.get("clip_quality", 1.0)
            pen = -(1.0 - q) * QUALITY_PENALTY_WEIGHT
            print(f"      phrase {s['phrase_index']}: quality={q:.2f} penalty={pen:+.2f} "
                  f"flags={s['clip_cull_flags']}")

    source_counts = {}
    for s in selections:
        name = s["clip_source_name"][:35]
        source_counts[name] = source_counts.get(name, 0) + 1
    max_possible = max(3, int(len(selections) / max(n_sources, 1) * MAX_SOURCE_USAGE_MULT))
    print(f"    Sources: {len(source_counts)}/{n_sources} used (cap: {max_possible}/source)")
    print(f"    Top sources:")
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1])[:8]:
        bar = "▓" * min(count, 30)
        print(f"      {name:<37} {count:>3}  {bar}")

    scores = [s["match_score"] for s in selections]
    print(f"    Scores: mean={np.mean(scores):.2f}  min={min(scores):.2f}  max={max(scores):.2f}")

    mode_counts = {}
    for s in selections:
        mode_counts[s["render_mode"]] = mode_counts.get(s["render_mode"], 0) + 1
    distinct_scenes = len(set((s["clip_source_name"], s["scene_index"]) for s in selections))
    speeds = [s["speed"] for s in selections
              if s["render_mode"] in ("speedfit", "beatloop", "beatpingpong")
              and abs(s["speed"] - 1.0) > 1e-6]
    speed_info = (f"  |  retimed range {min(speeds):.2f}x–{max(speeds):.2f}x"
                  if speeds else "")
    print(f"    Render modes: "
          + "  ".join(f"{m}:{c}" for m, c in sorted(mode_counts.items(), key=lambda x: -x[1]))
          + speed_info)
    print(f"    Distinct scenes: {distinct_scenes}/{len(selections)}")

    # ── 4b. Perceptual diversity logger (Phase D) ──
    # Operates on the FINAL ordered selection (post-weighted-random) so it
    # measures what the viewer actually experiences, not the candidate pool.
    pd_metrics = compute_perceptual_diversity(selections, scenes)
    print(f"\n  [4b/5] Perceptual diversity")
    print(f"    Consecutive cosine: mean={pd_metrics['consec_mean']:.3f}  "
          f"median={pd_metrics['consec_median']:.3f}  "
          f"(n={pd_metrics['n_consec_pairs']}"
          + (f", {pd_metrics['n_excluded']} excluded for missing embeddings)"
             if pd_metrics['n_excluded'] else ")"))
    print(f"    Close pairs within window 5:")
    for t in [0.85, 0.80, 0.75]:
        n = pd_metrics["close_pairs_w5"].get(t, 0)
        print(f"      cosine ≥ {t:.2f}: {n:>4} pairs")
    print(f"    Close pairs within window 30:")
    for t in [0.90, 0.85]:
        n = pd_metrics["close_pairs_w30"].get(t, 0)
        print(f"      cosine ≥ {t:.2f}: {n:>4} pairs")

    # Save plan
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "selection_plan_v2.json"
    with open(plan_path, 'w') as f:
        json.dump(selections, f, indent=2)

    # Update global clip count for this set
    if global_state:
        global_state["current_set_clips"] = len(selections)

    # ── 5. Assemble video ──
    print(f"\n  [5/5] Assembling video ({len(selections)} clips)")
    t0 = time.time()
    assemble_video(selections, output_path, audio_path, output_dir, global_state)
    elapsed = time.time() - t0

    # ── Result ──
    if output_path.exists():
        size = output_path.stat().st_size
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(output_path)],
            capture_output=True
        )
        dur = "?"
        if probe.returncode == 0:
            d = float(json.loads(probe.stdout).get("format", {}).get("duration", 0))
            dur = f"{int(d//60)}:{int(d%60):02d}"

        set_elapsed = time.time() - set_start

        # ── Optional post-render text check ──
        if getattr(args, "verify_text", False):
            print(f"\n  [6/5] Post-render text check...")
            rc = subprocess.run(
                [sys.executable, str(BASE_DIR / "scripts" / "check_render_text.py"),
                 str(output_path), "--plan", str(plan_path)]).returncode
            if rc != 0:
                print(f"  ⚠ English text found in render — see report above "
                      f"(re-run checker with --update-flags to exclude the sources)")

        print(f"\n  ╔═══════════════════════════════════════════════════════════════════╗")
        print(f"  ║  DONE: {set_name:<58}║")
        print(f"  ║  Duration: {dur:<8}  Size: {fmt_size(size):<10}  Render: {fmt_time(set_elapsed):<14}║")
        print(f"  ║  → {str(output_path.name):<63}║")
        print(f"  ╚═══════════════════════════════════════════════════════════════════╝")
        return True, len(selections)
    else:
        print(f"\n  ERROR: Output not created for {set_name}!")
        return False, len(selections)


def main():
    parser = argparse.ArgumentParser(description="Music video assembler v2")
    parser.add_argument("--set", type=str, choices=list(SET_CONFIGS.keys()),
                        help="DJ set to render (omit for all)")
    parser.add_argument("--all", action="store_true", help="Render all sets")
    parser.add_argument("--segment", nargs=2, type=float, metavar=("START", "END"),
                        help="Time segment in seconds")
    parser.add_argument("--phrase-bars", type=int, default=4, choices=[4, 8, 16],
                        help="Phrase length (default: 4 bars)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output video path (auto-named if omitted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verify-text", action="store_true",
                        help="After rendering, re-check the output for lingering "
                             "English text (scripts/check_render_text.py)")
    parser.add_argument("--no-salvage", action="store_true",
                        help="Discard partially text-flagged scenes whole instead "
                             "of salvaging clean sub-ranges (pre-salvage behavior)")

    args = parser.parse_args()

    if args.set:
        sets_to_run = {args.set: SET_CONFIGS[args.set]}
    elif args.all or args.output is None:
        sets_to_run = SET_CONFIGS
    else:
        sets_to_run = {"blue-sky-genesis-2025": SET_CONFIGS["blue-sky-genesis-2025"]}

    total_sets = len(sets_to_run)

    # ── Pre-scan: count total clips across all sets for global ETA ──
    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  MUSIC VIDEO ASSEMBLER v2")
    print(f"  {total_sets} set{'s' if total_sets > 1 else ''} to render")
    print(f"  Text filtering: enabled (English text excluded)")
    print(f"  ════════════════════════════════════════════════════════════════════")

    if total_sets > 1:
        print(f"\n  Planning batch render:")
        total_phrases = 0
        for name, cfg in sets_to_run.items():
            ap = BASE_DIR / "sets" / cfg["analysis"]
            if ap.exists():
                d = json.loads(ap.read_text())
                pk = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[args.phrase_bars]
                n = len(d["phrases"][pk])
                dur = d["phrases"][pk][-1]["end_sec"] if d["phrases"][pk] else 0
                total_phrases += n
                print(f"    {name:<30}  {n:>4} clips  {fmt_time(dur)}")
            else:
                print(f"    {name:<30}  MISSING")

        est_per_clip = 3.5  # rough seconds per clip extraction
        est_total = total_phrases * est_per_clip
        print(f"\n  Total: ~{total_phrases} clips")
        print(f"  Estimated render time: {fmt_time(est_total)}")

    # ── Global state for cross-set progress tracking ──
    global_state = {
        "batch_start": time.time(),
        "sets_done": 0,
        "clips_done": 0,
        "clips_total": 0,  # will be updated per-set
        "current_set_clips": 0,
    }

    # Count total clips (re-scan, quick)
    for name, cfg in sets_to_run.items():
        ap = BASE_DIR / "sets" / cfg["analysis"]
        if ap.exists():
            d = json.loads(ap.read_text())
            pk = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[args.phrase_bars]
            global_state["clips_total"] += len(d["phrases"][pk])

    results = {}
    for set_idx, (set_name, set_config) in enumerate(sets_to_run.items(), 1):
        np.random.seed(args.seed)
        ok, n_clips = run_set(set_name, set_config, args,
                              set_idx=set_idx, total_sets=total_sets,
                              global_state=global_state)
        results[set_name] = ok
        global_state["sets_done"] += 1

    # ── Final summary ──
    batch_elapsed = time.time() - global_state["batch_start"]
    n_ok = sum(results.values())

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  BATCH COMPLETE")
    print(f"  ════════════════════════════════════════════════════════════════════")
    print(f"  {n_ok}/{len(results)} sets rendered in {fmt_time(batch_elapsed)}")
    print(f"  {global_state['clips_done']} total clips processed")
    print()
    for name, ok in results.items():
        icon = "✓" if ok else "✗"
        out_name = SET_CONFIGS[name]["output_name"]
        out_path = BASE_DIR / "sets" / out_name
        if ok and out_path.exists():
            size = fmt_size(out_path.stat().st_size)
            print(f"  {icon} {name:<32} {size:>8}  → {out_name}")
        elif ok:
            print(f"  {icon} {name:<32} rendered (custom --output path)")
        else:
            print(f"  {icon} {name:<32} {'FAILED':>8}")
    print(f"\n  Output directory: {BASE_DIR / 'sets'}/")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
