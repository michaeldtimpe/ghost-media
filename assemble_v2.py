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
SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")
SETS_DIR = Path("/Volumes/archive/3000/3100/sets")

# Set definitions: analysis file → audio file → output name
SET_CONFIGS = {
    "blue-sky-genesis-2025": {
        "analysis": "Mtimpe MIX1 02DEC MASTER MP3 V6.deep-analysis.json",
        "audio": SETS_DIR / "blue-sky-genesis-2025" / "Mtimpe MIX1 02DEC MASTER MP3  V6  .mp3",
        "output_name": "blue-sky-genesis-2025_music_video.mp4",
        "style_hints": {},  # empty = no bias, pure audio-reactive
    },
    "boxing-day-2025": {
        "analysis": "Mtimpe MIX 30OCT MASTER WAV v5.deep-analysis.json",
        "audio": SETS_DIR / "boxing-day-2025" / "Mtimpe MIX 30OCT MASTER MP3 v5.mp3",
        "output_name": "boxing-day-2025_music_video.mp4",
        "style_hints": {},
    },
    "cheerleader-exodus-2025": {
        "analysis": "Mtimpe MIX2 10DEC MASTER WAV v3.deep-analysis.json",
        "audio": SETS_DIR / "cheerleader-exodus-2025" / "Mtimpe MIX2 10DEC MASTER MP3 v3.mp3",
        "output_name": "cheerleader-exodus-2025_music_video.mp4",
        "style_hints": {},
    },
    "waiting-to-begin-2024": {
        "analysis": "arche august dj set rev 4.deep-analysis.json",
        "audio": SETS_DIR / "waiting-to-begin-2024" / "arche august dj set rev 4.mp3",
        "output_name": "waiting-to-begin-2024_music_video.mp4",
        "style_hints": {},
    },
    "will-call-2025": {
        "analysis": "Mtimpe MIX 09JUN MASTER WAV v4.deep-analysis.json",
        "audio": SETS_DIR / "will-call-2025" / "Mtimpe MIX 09JUN MASTER MP3 v4.mp3",
        "output_name": "will-call-2025_music_video.mp4",
        "style_hints": {},
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
CROSSFADE_FRAMES = 8       # frames of crossfade between clips
TOP_CANDIDATES = 20        # random pool size for weighted selection
REUSE_PENALTY = 0.5        # steeper logarithmic penalty per reuse
MAX_SOURCE_USAGE_MULT = 2.0  # cap any source at (total_phrases / n_sources) * this
QUALITY_PENALTY_WEIGHT = 3.0  # soft penalty for dead scenes: -(1-quality)*this (see flag_quality.py)

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
    """Find the actual source video file."""
    original_path = Path(file_info.get("path", ""))
    if original_path.exists():
        return str(original_path)
    # Try source dir — first by prefix match, then by sanitized fuzzy match
    sanitized_target = _sanitize_for_match(original_path.stem)
    for candidate in SOURCE_DIR.iterdir():
        if original_path.stem[:25] in candidate.stem or candidate.stem[:25] in original_path.stem:
            return str(candidate)
    # Fuzzy match: strip all special chars and compare
    for candidate in SOURCE_DIR.iterdir():
        sanitized_candidate = _sanitize_for_match(candidate.stem)
        # Check if one is a substring of the other (handles truncation + sanitization)
        if len(sanitized_target) > 10 and len(sanitized_candidate) > 10:
            if sanitized_target[:25] in sanitized_candidate or sanitized_candidate[:25] in sanitized_target:
                return str(candidate)
    return str(original_path)  # return anyway, extraction will fail gracefully


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
            # Store under both the source stem and the text_flag stem for matching
            text_seconds[source_stem] = flagged
            text_seconds[tf_stem] = flagged

    return text_seconds


def scene_has_text(clip_source_name, start_sec, end_sec, text_seconds):
    """Check if a scene overlaps with any English text seconds."""
    if not text_seconds:
        return False
    # Try to match source name against text flag keys
    for key, flagged_secs in text_seconds.items():
        if key in clip_source_name or clip_source_name in key:
            # Check if any second in the scene range is flagged
            for sec in range(int(start_sec), int(end_sec) + 1):
                if sec in flagged_secs:
                    return True
    return False


def build_scene_database(text_seconds=None):
    """Build scene database from all enriched files, merging all data sources."""
    scenes = []
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

            # Motion from optical flow
            motion_vals = get_range(motion_tl, start, end, "mean_motion")
            motion_peaks = get_range(motion_tl, start, end, "max_motion")

            # Brightness
            bright_vals = get_range(brightness_tl, start, end, "brightness")
            contrast_vals = get_range(brightness_tl, start, end, "contrast")

            # Colors
            colors = get_colors_at(color_tl, start, end)

            # Semantic (from enrichment or scene-level)
            semantic = s.get("semantic", {}) or fa_by_scene.get(si, {})
            has_semantic = bool(semantic)
            mood = semantic.get("mood", {}) if semantic else {}
            style = semantic.get("visual_style", "") if semantic else ""
            tags = semantic.get("content_tags", []) if semantic else []

            # Skip scenes with English text
            if text_seconds and scene_has_text(source_name, start, end, text_seconds):
                continue

            clip = SceneClip(
                source_name=source_name,
                source_path=source_path,
                scene_index=si,
                start_sec=start,
                end_sec=end,
                duration_sec=dur,
                motion_mean=float(np.mean(motion_vals)) if motion_vals else 0,
                motion_peak=float(np.max(motion_peaks)) if motion_peaks else 0,
                brightness_mean=float(np.mean(bright_vals)) if bright_vals else 0.3,
                contrast_mean=float(np.mean(contrast_vals)) if contrast_vals else 0.3,
                dominant_colors=colors,
                color_temperature=compute_color_temperature(colors),
                color_saturation=compute_color_saturation(colors),
                visual_style=style,
                mood_energy=mood.get("energy", ""),
                content_tags=tags,
                has_semantic=has_semantic,
                loopable=s.get("loopable", False),
                loop_similarity=s.get("loop_similarity", 0.0),
                enriched_stem=src_stem,
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
    lyrics_keywords: list = field(default_factory=list)  # keywords from vocals
    clip_text_embedding: np.ndarray = None  # CLIP text encoding of lyrics


def extract_phrase_features(data, phrases):
    """Extract audio features per phrase."""
    multiband = data["multiband_energy"]
    hpss = data["hpss_timeline"]
    spectral = data["spectral_timeline"]
    bpm_tl = data["bpm_timeline"]
    tracks = data.get("tracks", [])

    results = []
    for phrase in phrases:
        start, end = phrase["start_sec"], phrase["end_sec"]

        mb = [m for m in multiband if start <= m["time_sec"] < end]
        hp = [h for h in hpss if start <= h["time_sec"] < end]
        sp = [s for s in spectral if start <= s["time_sec"] < end]
        bp = [b for b in bpm_tl if start <= b["time_sec"] < end]

        if not mb:
            continue

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

    # 1. Motion-energy match (visual motion should match audio energy)
    clip_motion_norm = min(clip.motion_mean / 15.0, 1.0)
    motion_match = 1.0 - abs(clip_motion_norm - target_energy)

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

    # 4. Duration suitability
    target_dur = 3.0 + (1.0 - target_energy) * 12.0
    dur_ratio = clip.duration_sec / target_dur
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
    #    Match lyric keywords against clip content_tags and visual description keywords
    lyrics_bonus = 0.0
    if phrase.lyrics_keywords and clip.has_semantic and clip.content_tags:
        lyric_set = set(kw.lower() for kw in phrase.lyrics_keywords)
        tag_set = set(tag.lower() for tag in clip.content_tags)
        # Count matching keywords
        matches = lyric_set & tag_set
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

    # Weighted combination
    score = (
        motion_match * 1.0 +
        bright_match * 0.8 +
        temp_match * 0.7 +
        dur_match * 0.9 +
        sat_match * 0.5 +
        contrast_match * 0.5 +
        semantic_bonus +
        style_bonus +
        lyrics_bonus +
        loop_bonus +
        clip_sim_bonus +
        quality_penalty
    )

    return score


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


def select_clips(scenes, phrase_features, style_hints=None):
    """Select clips with hard variety enforcement and adaptive diversity."""
    if style_hints is None:
        style_hints = {}

    selections = []
    recent_sources = []      # hard block window (source-level)
    recent_scenes = []       # hard block window (scene-level)
    source_use_count = {}    # global usage tracking

    total_sources = len(set(s.source_name for s in scenes))
    total_phrases = len(phrase_features)
    # Cap: no source can exceed this many uses
    max_source_uses = max(3, int(total_phrases / max(total_sources, 1) * MAX_SOURCE_USAGE_MULT))

    for i, phrase in enumerate(phrase_features):
        scored = []
        for clip in scenes:
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

            scored.append((s, clip))

        if not scored:
            # Fallback: relax variety constraint but keep scene dedup
            for clip in scenes:
                scene_key = f"{clip.source_name}:{clip.scene_index}"
                if scene_key in recent_scenes:
                    continue
                s = score_scene(clip, phrase, style_hints)
                scored.append((s, clip))

        if not scored:
            # Last resort: fully relaxed
            for clip in scenes:
                s = score_scene(clip, phrase, style_hints)
                scored.append((s, clip))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Wider random pool for more diversity
        top_n = min(TOP_CANDIDATES, len(scored))
        top_scores = [s for s, _ in scored[:top_n]]
        if top_scores:
            weights = np.array(top_scores)
            # Ensure positive weights
            weights = weights - weights.min() + 0.01
            weights = weights / weights.sum()
            chosen_idx = np.random.choice(top_n, p=weights)
        else:
            chosen_idx = 0

        best_score, best_clip = scored[chosen_idx]

        # Determine clip duration: use phrase duration but cap to scene length
        # If scene is loopable and phrase is longer, loop it instead of truncating
        needs_loop = False
        loop_count = 0
        if best_clip.loopable and phrase.duration_sec > best_clip.duration_sec:
            clip_duration = phrase.duration_sec
            loop_count = math.ceil(phrase.duration_sec / best_clip.duration_sec) - 1
            needs_loop = True
        else:
            clip_duration = min(phrase.duration_sec, best_clip.duration_sec - 0.2)
        clip_duration = max(clip_duration, 1.0)

        # Start at scene start (natural cut point)
        clip_start = best_clip.start_sec

        selections.append({
            "phrase_index": i,
            "audio_start": phrase.start_sec,
            "audio_end": phrase.end_sec,
            "audio_duration": phrase.duration_sec,
            "clip_source": best_clip.source_path,
            "clip_source_name": best_clip.source_name,
            "clip_start": clip_start,
            "clip_duration": clip_duration,
            "clip_motion": round(best_clip.motion_mean, 2),
            "clip_brightness": round(best_clip.brightness_mean, 3),
            "clip_color_temp": round(best_clip.color_temperature, 3),
            "match_score": round(best_score, 3),
            "audio_energy": phrase.energy,
            "track_title": phrase.track_title,
            "loop_count": loop_count,
            "scene_duration": best_clip.duration_sec if needs_loop else 0,
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

        source_use_count[best_clip.source_name] = source_use_count.get(best_clip.source_name, 0) + 1

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

def extract_clip(source_path, start_sec, duration_sec, output_path,
                 loop_count=0, scene_duration=0):
    """Extract clip, scale to 1080p. Optionally loop for loopable scenes."""
    vf = ("scale=1920:1080:force_original_aspect_ratio=decrease,"
          "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,setsar=1")

    if loop_count > 0 and scene_duration > 0:
        # Two-step: extract scene, then loop it
        scene_clip = output_path.with_suffix(".scene.mp4")
        cmd_scene = [
            "ffmpeg", "-y", "-ss", str(start_sec), "-i", source_path,
            "-t", str(scene_duration + 0.2),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-an", "-r", "30", str(scene_clip)
        ]
        subprocess.run(cmd_scene, capture_output=True, timeout=120)
        if not scene_clip.exists():
            return False

        cmd_loop = [
            "ffmpeg", "-y",
            "-stream_loop", str(loop_count),
            "-i", str(scene_clip),
            "-t", str(duration_sec + 0.5),
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-an", str(output_path)
        ]
        subprocess.run(cmd_loop, capture_output=True, timeout=120)
        scene_clip.unlink(missing_ok=True)
    else:
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", source_path,
            "-t", str(duration_sec + 0.5),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "20", "-preset", "fast",
            "-an", "-r", "30",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)

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
            sel["clip_source"], sel["clip_start"],
            sel["clip_duration"], clip_path,
            loop_count=sel.get("loop_count", 0),
            scene_duration=sel.get("scene_duration", 0),
        )
        clip_elapsed = time.time() - clip_t0
        timings.append(clip_elapsed)

        if ok:
            clip_paths.append((clip_path, sel["clip_duration"]))
        else:
            failed += 1

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
          f"  ({failed} failed)  in {fmt_time(total_extract)}")

    if not clip_paths:
        print("    ERROR: No clips extracted!")
        return

    # Build concat list
    concat_file = output_dir / "concat.txt"
    with open(concat_file, 'w') as f:
        for path, dur in clip_paths:
            f.write(f"file '{path.resolve()}'\n")

    # Concatenate
    video_duration = sum(dur for _, dur in clip_paths)
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
    scenes = build_scene_database(text_seconds)
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
    lyrics_path = BASE_DIR / "sets" / f"{set_name}.lyrics.json"
    if lyrics_path.exists():
        lyrics_data = json.loads(lyrics_path.read_text())
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
    selections = select_clips(scenes, phrase_features, style_hints)
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
        else:
            print(f"  {icon} {name:<32} {'FAILED':>8}")
    print(f"\n  Output directory: {BASE_DIR / 'sets'}/")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
