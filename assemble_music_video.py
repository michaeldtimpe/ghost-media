#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MUSIC VIDEO ASSEMBLER
  Matches video clips from enriched library to DJ set audio features
  and assembles a beat-synced music video
═══════════════════════════════════════════════════════════════════════════════

Pipeline:
  1. Load enriched video library → build clip database
  2. Load deep audio analysis → extract phrase-level features
  3. For each musical phrase, score all clips by mood/energy/style match
  4. Select clips with variety constraints (no immediate repeats)
  5. Extract clip segments from source videos via ffmpeg
  6. Assemble final video with crossfades at phrase boundaries

Usage:
    python assemble_music_video.py [--segment START END] [--phrase-bars 8]
"""

import json
import math
import os
import subprocess
import sys
import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

# ─── Configuration ─────────────────────────────────────────────────────────

ENRICHED_DIR = Path(__file__).parent / "enriched"
ANALYSIS_PATH = Path(__file__).parent / "sets" / "Mtimpe MIX1 02DEC MASTER MP3 V6.deep-analysis.json"
AUDIO_PATH = Path("/Volumes/archive/3000/3100/sets/blue-sky-genesis-2025/Mtimpe MIX1 02DEC MASTER MP3  V6  .mp3")
import media_paths
SOURCE_DIR = media_paths.FOOTAGE_ROOT / "raw"
OUTPUT_DIR = Path(__file__).parent / "sets" / "music_video_build"
FINAL_OUTPUT = Path(__file__).parent / "sets" / "music_video.mp4"

CROSSFADE_SEC = 0.5  # crossfade duration between clips


# ─── Clip Database ─────────────────────────────────────────────────────────

@dataclass
class ClipInfo:
    """A single analyzed frame/moment from a source video."""
    source_video: str       # filename of source video
    source_path: str        # full path
    time_sec: float         # timestamp in source video
    duration_sec: float     # total source video duration
    scene_index: int
    visual_style: str
    energy: str             # calm, building, moderate, intense, chaotic
    mood_tone: str
    implied_motion: str
    dominant_colors: list
    color_temperature: str
    color_relationship: str
    content_tags: list
    visual_description: str
    # Derived scores (0-1)
    energy_score: float = 0.0
    brightness_score: float = 0.0
    motion_score: float = 0.0


ENERGY_MAP = {"calm": 0.1, "building": 0.4, "moderate": 0.5, "intense": 0.8, "chaotic": 1.0}
MOTION_MAP = {
    "none": 0.0, "drift": 0.2, "pan_left": 0.3, "pan_right": 0.3,
    "pulse": 0.4, "zoom_in": 0.5, "zoom_out": 0.5,
    "rotation_cw": 0.6, "rotation_ccw": 0.6, "spiral": 0.8, "complex": 0.9,
}
WARM_COLORS = {"red", "orange", "yellow", "gold", "amber", "warm", "brown", "tan", "cream", "beige"}
COOL_COLORS = {"blue", "cyan", "teal", "purple", "violet", "indigo", "green", "ice"}

# Styles that work well with different audio moods
STYLE_ENERGY_AFFINITY = {
    "psychedelic": 0.7, "glitch": 0.8, "particle": 0.6, "fractal": 0.5,
    "abstract": 0.5, "geometric": 0.6, "motion_graphics": 0.5,
    "cinematic": 0.3, "photorealistic": 0.2, "minimalist": 0.1,
    "organic": 0.3, "liquid": 0.4, "mixed": 0.5,
}


def build_clip_database():
    """Load all enriched files and build a searchable clip database."""
    clips = []
    files = sorted(ENRICHED_DIR.glob("*.enriched.json"))

    for f in files:
        data = json.loads(f.read_text())
        file_info = data.get("file", {})
        source_path = file_info.get("path", "")
        source_name = file_info.get("name", f.stem.replace(".enriched", ""))
        duration = data.get("metadata", {}).get("duration_sec", 0)

        # Check source exists
        sp = Path(source_path)
        if not sp.exists():
            # Try source dir
            for ext in ['.mp4', '.mov', '.mkv', '.avi', '.webm']:
                candidate = SOURCE_DIR / (sp.stem + ext)
                if candidate.exists():
                    source_path = str(candidate)
                    break
                # Fuzzy match
                for sf in media_paths.iter_footage().values():
                    if sf.suffix.lower() == ext and sp.stem[:20] in sf.stem:
                        source_path = str(sf)
                        break

        fas = data.get("frame_analyses", [])
        for fa in fas:
            analysis = fa.get("analysis", {})
            if not analysis:
                continue

            mood = analysis.get("mood", {})
            palette = analysis.get("color_palette", {})
            composition = analysis.get("composition", {})
            tags = analysis.get("content_tags", [])

            # Skip pure black frames
            if all(c.lower() == "black" for c in palette.get("dominant_colors", ["black"])):
                if "black" in str(tags).lower() and len(tags) <= 2:
                    continue

            energy_label = mood.get("energy", "moderate")
            motion_label = composition.get("implied_motion", "none")

            clip = ClipInfo(
                source_video=source_name,
                source_path=source_path,
                time_sec=fa.get("time_sec", 0),
                duration_sec=duration,
                scene_index=fa.get("scene_index", 0),
                visual_style=analysis.get("visual_style", "mixed"),
                energy=energy_label,
                mood_tone=mood.get("tone", ""),
                implied_motion=motion_label,
                dominant_colors=palette.get("dominant_colors", []),
                color_temperature=palette.get("temperature", "neutral"),
                color_relationship=palette.get("color_relationship", ""),
                content_tags=tags,
                visual_description=analysis.get("visual_description", ""),
                energy_score=ENERGY_MAP.get(energy_label, 0.5),
                motion_score=MOTION_MAP.get(motion_label, 0.3),
            )

            # Brightness score from colors
            bright_colors = {"white", "yellow", "neon", "bright", "vibrant"}
            dark_colors = {"black", "dark", "shadow"}
            color_str = " ".join(clip.dominant_colors).lower()
            if any(c in color_str for c in bright_colors):
                clip.brightness_score = 0.8
            elif any(c in color_str for c in dark_colors):
                clip.brightness_score = 0.2
            else:
                clip.brightness_score = 0.5

            clips.append(clip)

    return clips


# ─── Audio Feature Extraction for Matching ─────────────────────────────────

@dataclass
class PhraseFeatures:
    """Audio features for a musical phrase, used for clip matching."""
    start_sec: float
    end_sec: float
    duration_sec: float
    energy_mean: float      # 0-1 normalized
    energy_peak: float
    energy_shape: str       # building, decaying, sustain, dynamic
    bpm: float
    spectral_centroid: float
    spectral_brightness: str  # dark, moderate, bright, very_bright
    bass_energy: float
    high_energy: float
    percussive_ratio: float  # 0-1, higher = more percussive
    dominant_chroma: str
    track_title: str


def extract_phrase_features(data, phrases):
    """Extract audio features for each phrase from the deep analysis."""
    results = []

    multiband = data["multiband_energy"]
    hpss = data["hpss_timeline"]
    spectral = data["spectral_timeline"]
    chroma = data["chroma_timeline"]
    bpm_tl = data["bpm_timeline"]
    tracks = data.get("tracks", [])

    for phrase in phrases:
        start = phrase["start_sec"]
        end = phrase["end_sec"]

        # Multiband in range
        mb_range = [m for m in multiband if start <= m["time_sec"] < end]
        hp_range = [h for h in hpss if start <= h["time_sec"] < end]
        sp_range = [s for s in spectral if start <= s["time_sec"] < end]
        ch_range = [c for c in chroma if start <= c["time_sec"] < end]
        bpm_range = [b for b in bpm_tl if start <= b["time_sec"] < end]

        if not mb_range:
            continue

        rms_vals = [m["total_rms"] for m in mb_range]
        bass_vals = [m["bass"] + m["sub_bass"] for m in mb_range]
        high_vals = [m["presence"] + m["brilliance"] for m in mb_range]

        # Normalize energy to 0-1 (typical max ~0.5)
        e_mean = np.mean(rms_vals)
        e_norm = min(e_mean / 0.45, 1.0)
        e_peak = min(max(rms_vals) / 0.5, 1.0)

        # Percussive ratio
        if hp_range:
            harm = np.mean([h["harmonic"] for h in hp_range])
            perc = np.mean([h["percussive"] for h in hp_range])
            perc_ratio = perc / (harm + perc + 1e-10)
        else:
            perc_ratio = 0.5

        # Spectral
        if sp_range:
            centroid = np.mean([s["centroid_hz"] for s in sp_range])
        else:
            centroid = 2000

        if centroid < 1500:
            brightness = "dark"
        elif centroid < 3000:
            brightness = "moderate"
        elif centroid < 5000:
            brightness = "bright"
        else:
            brightness = "very_bright"

        # BPM
        bpm = np.mean([b["bpm"] for b in bpm_range]) if bpm_range else 128

        # Dominant chroma
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if ch_range:
            chroma_avg = {pc: np.mean([c[pc] for c in ch_range]) for pc in pitch_classes}
            dom_chroma = max(chroma_avg, key=chroma_avg.get)
        else:
            dom_chroma = "C"

        # Track at this time
        track_title = ""
        for tr in tracks:
            if tr["start_sec"] <= start < tr["end_sec"]:
                track_title = tr["title"]
                break

        results.append(PhraseFeatures(
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            energy_mean=round(e_norm, 4),
            energy_peak=round(e_peak, 4),
            energy_shape=phrase.get("energy_shape", "sustain"),
            bpm=round(bpm, 1),
            spectral_centroid=round(centroid, 1),
            spectral_brightness=brightness,
            bass_energy=round(min(np.mean(bass_vals) / 0.3, 1.0), 4),
            high_energy=round(min(np.mean(high_vals) / 0.1, 1.0), 4),
            percussive_ratio=round(perc_ratio, 4),
            dominant_chroma=dom_chroma,
            track_title=track_title,
        ))

    return results


# ─── Clip Matching ─────────────────────────────────────────────────────────

def score_clip(clip: ClipInfo, phrase: PhraseFeatures) -> float:
    """Score how well a clip matches a musical phrase. Higher = better match."""
    score = 0.0

    # 1. Energy match (most important) — weight: 3.0
    energy_diff = abs(clip.energy_score - phrase.energy_mean)
    score += (1.0 - energy_diff) * 3.0

    # 2. Motion matches energy — weight: 1.5
    # High energy audio → high motion visuals
    motion_diff = abs(clip.motion_score - phrase.energy_mean)
    score += (1.0 - motion_diff) * 1.5

    # 3. Color temperature matches spectral character — weight: 1.0
    # Warm audio (low centroid) → warm colors, bright audio → cool colors
    if phrase.spectral_brightness in ("dark", "moderate"):
        if clip.color_temperature == "warm":
            score += 1.0
        elif clip.color_temperature == "neutral":
            score += 0.5
    else:
        if clip.color_temperature == "cool":
            score += 1.0
        elif clip.color_temperature == "neutral":
            score += 0.5

    # 4. Visual style affinity — weight: 1.0
    style_energy = STYLE_ENERGY_AFFINITY.get(clip.visual_style, 0.5)
    style_match = 1.0 - abs(style_energy - phrase.energy_mean)
    score += style_match * 1.0

    # 5. Percussive audio → geometric/glitch visuals — weight: 0.8
    if phrase.percussive_ratio > 0.5:
        if clip.visual_style in ("geometric", "glitch", "particle", "motion_graphics"):
            score += 0.8
    else:
        if clip.visual_style in ("liquid", "organic", "cinematic", "photorealistic"):
            score += 0.8

    # 6. Bass-heavy → dark/saturated visuals — weight: 0.5
    if phrase.bass_energy > 0.6:
        if clip.brightness_score < 0.4:
            score += 0.5
        if clip.visual_style in ("abstract", "psychedelic", "fractal"):
            score += 0.3

    # 7. Content tag bonus — weight: 0.3
    tag_str = " ".join(clip.content_tags).lower()
    if phrase.energy_mean > 0.7 and any(w in tag_str for w in ["neon", "vibrant", "dynamic", "energy"]):
        score += 0.3
    if phrase.energy_mean < 0.3 and any(w in tag_str for w in ["serene", "calm", "minimal", "quiet"]):
        score += 0.3

    return score


def select_clips(clips, phrase_features, recent_window=4):
    """Select the best clip for each phrase with variety constraints."""
    selections = []
    recent_sources = []  # track recently used source videos

    for i, phrase in enumerate(phrase_features):
        # Score all clips
        scored = []
        for clip in clips:
            s = score_clip(clip, phrase)

            # Penalize recently used sources
            if clip.source_video in recent_sources:
                recency = len(recent_sources) - recent_sources.index(clip.source_video)
                s *= 0.3 * (recency / len(recent_sources))

            scored.append((s, clip))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Pick the best
        best_score, best_clip = scored[0]

        # Determine how long to show this clip
        # Use the phrase duration but don't exceed available video
        clip_duration = phrase.duration_sec
        # Make sure we don't go past the end of the source video
        max_available = best_clip.duration_sec - best_clip.time_sec
        clip_duration = min(clip_duration, max_available - 0.5)
        clip_duration = max(clip_duration, 1.0)

        selections.append({
            "phrase_index": i,
            "audio_start": phrase.start_sec,
            "audio_end": phrase.end_sec,
            "audio_duration": phrase.duration_sec,
            "clip_source": best_clip.source_path,
            "clip_source_name": best_clip.source_video,
            "clip_start": best_clip.time_sec,
            "clip_duration": clip_duration,
            "clip_style": best_clip.visual_style,
            "clip_energy": best_clip.energy,
            "clip_motion": best_clip.implied_motion,
            "match_score": round(best_score, 3),
            "audio_energy": phrase.energy_mean,
            "audio_brightness": phrase.spectral_brightness,
            "track_title": phrase.track_title,
        })

        # Update recent sources
        recent_sources.append(best_clip.source_video)
        if len(recent_sources) > recent_window:
            recent_sources.pop(0)

    return selections


# ─── Video Assembly ────────────────────────────────────────────────────────

def extract_clip(source_path, start_sec, duration_sec, output_path, target_res="1920:1080"):
    """Extract a clip segment from a source video, scaling to target resolution."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-i", source_path,
        "-t", str(duration_sec + CROSSFADE_SEC),  # extra for crossfade
        "-vf", f"scale={target_res}:force_original_aspect_ratio=decrease,"
               f"pad={target_res}:(ow-iw)/2:(oh-ih)/2:black,"
               f"setsar=1",
        "-c:v", "libx264", "-crf", "20", "-preset", "fast",
        "-an",  # no audio for clips
        "-r", "30",
        str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return output_path.exists()


def assemble_video(selections, output_path, audio_path):
    """Assemble all clips into final video with audio."""
    clip_dir = OUTPUT_DIR / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Extracting {len(selections)} clips...")
    clip_paths = []
    failed = 0

    for i, sel in enumerate(selections):
        clip_path = clip_dir / f"clip_{i:04d}.mp4"

        if i % 20 == 0:
            pct = i / len(selections) * 100
            print(f"    [{i}/{len(selections)}] {pct:.0f}%...", flush=True)

        ok = extract_clip(
            sel["clip_source"],
            sel["clip_start"],
            sel["clip_duration"],
            clip_path,
        )

        if ok:
            clip_paths.append((clip_path, sel["clip_duration"]))
        else:
            failed += 1

    print(f"    Extracted: {len(clip_paths)}, Failed: {failed}")

    if not clip_paths:
        print("  ERROR: No clips extracted!")
        return

    # Build concat list
    concat_file = OUTPUT_DIR / "concat.txt"
    with open(concat_file, 'w') as f:
        for path, dur in clip_paths:
            f.write(f"file '{path.resolve()}'\n")

    # Concatenate clips (copy codec — all clips are already same format)
    print(f"  Concatenating clips...")
    concat_video = OUTPUT_DIR / "concat_raw.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        "-movflags", "+faststart",
        str(concat_video)
    ], capture_output=True, timeout=3600)

    # Get audio segment matching the video duration
    video_duration = sum(dur for _, dur in clip_paths)
    audio_start = selections[0]["audio_start"]

    # Add audio
    print(f"  Adding audio (from {audio_start:.1f}s, {video_duration:.1f}s)...")
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(concat_video),
        "-ss", str(audio_start),
        "-i", str(audio_path),
        "-t", str(video_duration),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path)
    ], capture_output=True, timeout=3600)

    # Cleanup
    print(f"  Cleaning up...")
    for path, _ in clip_paths:
        path.unlink()
    concat_video.unlink()
    concat_file.unlink()
    clip_dir.rmdir()


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Assemble music video from clip library")
    parser.add_argument("--segment", nargs=2, type=float, metavar=("START", "END"),
                        help="Only render a time segment (in seconds)")
    parser.add_argument("--phrase-bars", type=int, default=8, choices=[4, 8, 16],
                        help="Phrase length for clip switching (default: 8 bars)")
    parser.add_argument("-o", "--output", type=str, help="Output video path")

    args = parser.parse_args()

    if args.output:
        final_output = Path(args.output)
    else:
        final_output = FINAL_OUTPUT

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  MUSIC VIDEO ASSEMBLER")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    # 1. Build clip database
    print(f"  [1/5] Building clip database...", end=" ", flush=True)
    t0 = time.time()
    clips = build_clip_database()
    print(f"{len(clips)} clips from {len(set(c.source_video for c in clips))} videos ({time.time()-t0:.1f}s)")

    # Show database summary
    styles = {}
    for c in clips:
        styles[c.visual_style] = styles.get(c.visual_style, 0) + 1
    print(f"    Styles: {dict(sorted(styles.items(), key=lambda x: -x[1]))}")

    energies = {}
    for c in clips:
        energies[c.energy] = energies.get(c.energy, 0) + 1
    print(f"    Energy levels: {dict(sorted(energies.items(), key=lambda x: -x[1]))}")

    # 2. Load audio analysis
    print(f"\n  [2/5] Loading audio analysis...", end=" ", flush=True)
    data = json.loads(ANALYSIS_PATH.read_text())

    # Select phrase size
    phrase_key = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[args.phrase_bars]
    phrases = data["phrases"][phrase_key]

    # Filter to segment if specified
    if args.segment:
        seg_start, seg_end = args.segment
        phrases = [p for p in phrases if p["start_sec"] >= seg_start and p["end_sec"] <= seg_end]
        print(f"{len(phrases)} phrases in segment {seg_start:.0f}-{seg_end:.0f}s")
    else:
        print(f"{len(phrases)} phrases ({args.phrase_bars}-bar)")

    # 3. Extract phrase features
    print(f"  [3/5] Extracting phrase features...", end=" ", flush=True)
    t0 = time.time()
    phrase_features = extract_phrase_features(data, phrases)
    print(f"{len(phrase_features)} phrases ({time.time()-t0:.1f}s)")

    # 4. Match clips to phrases
    print(f"  [4/5] Matching clips to phrases...", end=" ", flush=True)
    t0 = time.time()
    selections = select_clips(clips, phrase_features)
    print(f"done ({time.time()-t0:.1f}s)")

    # Show selection summary
    source_counts = {}
    for s in selections:
        name = s["clip_source_name"][:30]
        source_counts[name] = source_counts.get(name, 0) + 1
    print(f"    Sources used: {len(source_counts)} unique videos")
    print(f"    Top sources:")
    for name, count in sorted(source_counts.items(), key=lambda x: -x[1])[:8]:
        print(f"      {name:<32} {count} clips")

    # Show a few selections
    print(f"\n    Sample selections:")
    for s in selections[:6]:
        print(f"      {s['audio_start']/60:5.1f}m  E:{s['audio_energy']:.2f}"
              f"  → {s['clip_source_name'][:35]:35} @{s['clip_start']:.0f}s"
              f"  style:{s['clip_style']:15} score:{s['match_score']:.2f}")

    # Save selection plan
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan_path = OUTPUT_DIR / "selection_plan.json"
    with open(plan_path, 'w') as f:
        json.dump(selections, f, indent=2)
    print(f"\n    Plan saved: {plan_path}")

    # 5. Assemble video
    print(f"\n  [5/5] Assembling video...")
    t0 = time.time()
    assemble_video(selections, final_output, AUDIO_PATH)
    elapsed = time.time() - t0

    if final_output.exists():
        size_mb = final_output.stat().st_size / (1024 * 1024)
        # Get duration
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(final_output)],
            capture_output=True
        )
        dur = "?"
        if probe.returncode == 0:
            fmt = json.loads(probe.stdout).get("format", {})
            d = float(fmt.get("duration", 0))
            dur = f"{int(d//60)}:{int(d%60):02d}"

        print(f"\n  ═══════════════════════════════════════════════════════════")
        print(f"  MUSIC VIDEO COMPLETE")
        print(f"  Output: {final_output}")
        print(f"  Duration: {dur}")
        print(f"  Size: {size_mb:.1f} MB")
        print(f"  Assembly time: {elapsed:.0f}s")
        print(f"  ═══════════════════════════════════════════════════════════\n")
    else:
        print(f"\n  ERROR: Output file not created!")


if __name__ == "__main__":
    main()
