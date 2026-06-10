#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  DJ SET ANALYZER
  Windowed audio analysis with tracklist-aware segmentation
  Designed for driving live visuals from pre-analyzed audio
═══════════════════════════════════════════════════════════════════════════════

Produces a time-series analysis of a DJ set with:
- Per-track metadata from a tracklist file
- Windowed BPM tracking (local tempo changes)
- Beat grid with per-beat energy
- Energy / spectral / mood curves at configurable resolution
- Transition detection between tracks
- Beat-aligned phrase boundaries (4-bar, 8-bar, 16-bar)

Usage:
    python analyze_dj_set.py <audio_file> --tracklist <tracklist.txt> -o <output.json>
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf


# ─── Tracklist Parser ──────────────────────────────────────────────────────

def parse_tracklist(filepath):
    """Parse tracklist with timestamps. Handles [MM:SS] and [HH:MM:SS] formats."""
    tracks = []
    with open(filepath, 'r') as f:
        lines = f.readlines()

    ts_pattern = re.compile(
        r'^\*{0,3}\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)',
    )

    for line in lines:
        line = line.strip()
        m = ts_pattern.match(line)
        if not m:
            continue
        ts_str, title = m.group(1), m.group(2).strip()
        parts = ts_str.split(':')
        if len(parts) == 3:
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        else:
            secs = int(parts[0]) * 60 + int(parts[1])
        tracks.append({"start_sec": secs, "title": title})

    return tracks


def build_track_segments(tracks, total_duration):
    """Build segments with start/end times from tracklist."""
    segments = []
    for i, t in enumerate(tracks):
        end = tracks[i + 1]["start_sec"] if i + 1 < len(tracks) else total_duration
        segments.append({
            "track_index": i,
            "title": t["title"],
            "start_sec": t["start_sec"],
            "end_sec": round(end, 3),
            "duration_sec": round(end - t["start_sec"], 3),
        })
    return segments


# ─── Windowed BPM ──────────────────────────────────────────────────────────

def estimate_windowed_bpm(y, sr, window_sec=8, hop_sec=2):
    """Estimate BPM in sliding windows across the track."""
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = len(y)

    results = []
    pos = 0
    while pos + window_samples <= total_samples:
        segment = y[pos:pos + window_samples]
        time_center = (pos + window_samples / 2) / sr

        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

        results.append({
            "time_sec": round(time_center, 3),
            "bpm": round(bpm, 2),
        })
        pos += hop_samples

    return results


# ─── Per-Beat Energy ───────────────────────────────────────────────────────

def compute_beat_energy(y, sr, beat_times):
    """Compute RMS energy at each beat position using a short window around the beat."""
    half_window = int(0.05 * sr)  # 50ms window
    energies = []
    for bt in beat_times:
        center = int(bt * sr)
        start = max(0, center - half_window)
        end = min(len(y), center + half_window)
        segment = y[start:end]
        rms = float(np.sqrt(np.mean(segment ** 2)))
        energies.append(round(rms, 6))
    return energies


# ─── Phrase Detection ──────────────────────────────────────────────────────

def detect_phrases(beat_times, beats_per_bar=4, phrase_bars=8):
    """Group beats into musical phrases (default: 8-bar = 32-beat phrases)."""
    beats_per_phrase = beats_per_bar * phrase_bars
    phrases = []
    for i in range(0, len(beat_times), beats_per_phrase):
        chunk = beat_times[i:i + beats_per_phrase]
        if len(chunk) < beats_per_bar:
            break
        phrases.append({
            "phrase_index": len(phrases),
            "start_sec": round(chunk[0], 4),
            "end_sec": round(chunk[-1], 4),
            "beat_count": len(chunk),
            "bars": len(chunk) // beats_per_bar,
        })
    return phrases


# ─── Transition Detection ─────────────────────────────────────────────────

def detect_transitions(energy_timeline, bpm_timeline, track_segments):
    """Detect transition zones around track boundaries using energy/BPM changes."""
    transitions = []
    for i in range(len(track_segments) - 1):
        boundary = track_segments[i]["end_sec"]
        # Look at a window around the boundary
        window = 15  # seconds each side
        t_start = boundary - window
        t_end = boundary + window

        # Energy delta across boundary
        pre_energy = [e["rms"] for e in energy_timeline
                      if t_start <= e["time_sec"] < boundary]
        post_energy = [e["rms"] for e in energy_timeline
                       if boundary <= e["time_sec"] < t_end]

        pre_avg = np.mean(pre_energy) if pre_energy else 0
        post_avg = np.mean(post_energy) if post_energy else 0

        # BPM delta
        pre_bpm = [b["bpm"] for b in bpm_timeline
                    if t_start <= b["time_sec"] < boundary]
        post_bpm = [b["bpm"] for b in bpm_timeline
                     if boundary <= b["time_sec"] < t_end]

        pre_bpm_avg = np.mean(pre_bpm) if pre_bpm else 0
        post_bpm_avg = np.mean(post_bpm) if post_bpm else 0

        transitions.append({
            "from_track": track_segments[i]["title"],
            "to_track": track_segments[i + 1]["title"],
            "boundary_sec": boundary,
            "energy_before": round(float(pre_avg), 6),
            "energy_after": round(float(post_avg), 6),
            "energy_delta": round(float(post_avg - pre_avg), 6),
            "bpm_before": round(float(pre_bpm_avg), 2),
            "bpm_after": round(float(post_bpm_avg), 2),
            "bpm_delta": round(float(post_bpm_avg - pre_bpm_avg), 2),
            "type": _classify_transition(pre_avg, post_avg, pre_bpm_avg, post_bpm_avg),
        })

    return transitions


def _classify_transition(pre_e, post_e, pre_bpm, post_bpm):
    e_ratio = post_e / (pre_e + 1e-10)
    bpm_diff = abs(post_bpm - pre_bpm)

    if e_ratio > 1.5:
        return "build_drop"
    elif e_ratio < 0.6:
        return "breakdown"
    elif bpm_diff > 5:
        return "tempo_shift"
    else:
        return "smooth_blend"


# ─── High-Resolution Energy Timeline ──────────────────────────────────────

def compute_energy_timeline(y, sr, fps=4):
    """Compute energy, spectral centroid, and spectral flatness at given fps."""
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Downsample to target fps
    native_fps = sr / hop_length
    step = max(1, int(native_fps / fps))

    timeline = []
    for i in range(0, len(times), step):
        timeline.append({
            "time_sec": round(float(times[i]), 3),
            "rms": round(float(rms[i]), 6),
            "spectral_centroid_hz": round(float(centroid[i]), 1),
            "spectral_flatness": round(float(flatness[i]), 6),
            "spectral_rolloff_hz": round(float(rolloff[i]), 1),
        })

    return timeline


# ─── Per-Track Analysis ───────────────────────────────────────────────────

def analyze_track_segment(y, sr, segment, beat_times, bpm_timeline, energy_timeline):
    """Compute per-track summary stats."""
    start, end = segment["start_sec"], segment["end_sec"]

    # Beats in this segment
    seg_beats = [b for b in beat_times if start <= b < end]

    # BPM in this segment
    seg_bpm = [b["bpm"] for b in bpm_timeline if start <= b["time_sec"] < end]

    # Energy in this segment
    seg_energy = [e for e in energy_timeline if start <= e["time_sec"] < end]
    seg_rms = [e["rms"] for e in seg_energy]
    seg_centroid = [e["spectral_centroid_hz"] for e in seg_energy]

    segment["beat_count"] = len(seg_beats)
    segment["bpm"] = {
        "mean": round(float(np.mean(seg_bpm)), 2) if seg_bpm else 0,
        "min": round(float(np.min(seg_bpm)), 2) if seg_bpm else 0,
        "max": round(float(np.max(seg_bpm)), 2) if seg_bpm else 0,
        "std": round(float(np.std(seg_bpm)), 2) if seg_bpm else 0,
    }
    segment["energy"] = {
        "mean": round(float(np.mean(seg_rms)), 6) if seg_rms else 0,
        "peak": round(float(np.max(seg_rms)), 6) if seg_rms else 0,
        "min": round(float(np.min(seg_rms)), 6) if seg_rms else 0,
    }
    segment["spectral"] = {
        "centroid_mean_hz": round(float(np.mean(seg_centroid)), 1) if seg_centroid else 0,
        "brightness": _classify_brightness(np.mean(seg_centroid)) if seg_centroid else "unknown",
    }

    return segment


def _classify_brightness(centroid_mean):
    if centroid_mean < 1500:
        return "dark"
    elif centroid_mean < 3000:
        return "moderate"
    elif centroid_mean < 5000:
        return "bright"
    else:
        return "very_bright"


# ─── Main Pipeline ────────────────────────────────────────────────────────

def analyze_dj_set(audio_path, tracklist_path=None, sr=22050, energy_fps=4,
                   bpm_window=8, bpm_hop=2):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"  ERROR: File not found: {audio_path}")
        sys.exit(1)

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  DJ SET ANALYZER")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    # ── Load audio ──
    print(f"  Loading: {audio_path.name}")
    t0 = time.time()
    y, sr = librosa.load(str(audio_path), sr=sr, mono=True)
    duration = float(len(y) / sr)
    print(f"  Duration: {duration / 60:.1f} min | SR: {sr} | Load: {time.time() - t0:.1f}s")

    # ── Parse tracklist ──
    tracks = []
    track_segments = []
    if tracklist_path:
        print(f"  Tracklist: {Path(tracklist_path).name}")
        tracks = parse_tracklist(tracklist_path)
        track_segments = build_track_segments(tracks, duration)
        print(f"  Tracks: {len(track_segments)}")

    # ── Global beat tracking ──
    print(f"  Detecting beats...", end=" ", flush=True)
    t0 = time.time()
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    global_bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    print(f"{len(beat_times)} beats, global BPM: {global_bpm:.1f} ({time.time() - t0:.1f}s)")

    # ── Windowed BPM ──
    print(f"  Computing windowed BPM ({bpm_window}s window, {bpm_hop}s hop)...", end=" ", flush=True)
    t0 = time.time()
    bpm_timeline = estimate_windowed_bpm(y, sr, window_sec=bpm_window, hop_sec=bpm_hop)
    bpm_values = [b["bpm"] for b in bpm_timeline]
    print(f"range: {min(bpm_values):.0f}-{max(bpm_values):.0f} BPM ({time.time() - t0:.1f}s)")

    # ── Energy / spectral timeline ──
    print(f"  Computing energy timeline ({energy_fps} fps)...", end=" ", flush=True)
    t0 = time.time()
    energy_timeline = compute_energy_timeline(y, sr, fps=energy_fps)
    print(f"{len(energy_timeline)} samples ({time.time() - t0:.1f}s)")

    # ── Beat energy ──
    print(f"  Computing per-beat energy...", end=" ", flush=True)
    t0 = time.time()
    beat_energy = compute_beat_energy(y, sr, beat_times)
    print(f"done ({time.time() - t0:.1f}s)")

    # ── Phrases ──
    print(f"  Detecting phrases (8-bar)...", end=" ", flush=True)
    phrases_8bar = detect_phrases(beat_times, beats_per_bar=4, phrase_bars=8)
    phrases_4bar = detect_phrases(beat_times, beats_per_bar=4, phrase_bars=4)
    print(f"{len(phrases_8bar)} phrases (8-bar), {len(phrases_4bar)} (4-bar)")

    # ── Per-track analysis ──
    if track_segments:
        print(f"  Analyzing per-track features...", end=" ", flush=True)
        t0 = time.time()
        for seg in track_segments:
            analyze_track_segment(y, sr, seg, beat_times, bpm_timeline, energy_timeline)
        print(f"done ({time.time() - t0:.1f}s)")

    # ── Transitions ──
    transitions = []
    if track_segments:
        print(f"  Detecting transitions...", end=" ", flush=True)
        transitions = detect_transitions(energy_timeline, bpm_timeline, track_segments)
        types = {}
        for t in transitions:
            types[t["type"]] = types.get(t["type"], 0) + 1
        print(f"{len(transitions)} transitions ({', '.join(f'{v} {k}' for k, v in types.items())})")

    # ── Key detection ──
    print(f"  Detecting key...", end=" ", flush=True)
    t0 = time.time()
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best_corr, best_key, best_mode = -1, "C", "major"
    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
            corr = float(np.corrcoef(rolled, profile)[0, 1])
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, pitch_classes[i], mode
    print(f"{best_key} {best_mode} (confidence: {best_corr:.3f}) ({time.time() - t0:.1f}s)")

    # ── Assemble output ──
    result = {
        "schema_version": "1.0.0",
        "analyzer": "dj-set-analyzer",
        "file": {
            "path": str(audio_path.resolve()),
            "name": audio_path.name,
            "format": audio_path.suffix.lstrip('.').lower(),
            "duration_sec": round(duration, 3),
            "sample_rate": sr,
        },
        "global_tempo": {
            "bpm": round(global_bpm, 2),
            "beat_count": len(beat_times),
        },
        "key": {
            "key": best_key,
            "mode": best_mode,
            "label": f"{best_key} {best_mode}",
            "confidence": round(best_corr, 4),
        },
        "tracks": track_segments,
        "transitions": transitions,
        "beats": {
            "times_sec": [round(t, 4) for t in beat_times],
            "energy": beat_energy,
        },
        "bpm_timeline": bpm_timeline,
        "energy_timeline": energy_timeline,
        "phrases": {
            "eight_bar": phrases_8bar,
            "four_bar": phrases_4bar,
        },
    }

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  ANALYSIS COMPLETE")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze a DJ set for live visuals")
    parser.add_argument("input", help="Path to audio file")
    parser.add_argument("--tracklist", "-t", help="Path to tracklist timestamps file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    parser.add_argument("--energy-fps", type=int, default=4, help="Energy timeline samples per second (default: 4)")
    parser.add_argument("--bpm-window", type=int, default=8, help="BPM estimation window in seconds (default: 8)")
    parser.add_argument("--bpm-hop", type=int, default=2, help="BPM estimation hop in seconds (default: 2)")

    args = parser.parse_args()

    result = analyze_dj_set(
        args.input,
        tracklist_path=args.tracklist,
        sr=args.sr,
        energy_fps=args.energy_fps,
        bpm_window=args.bpm_window,
        bpm_hop=args.bpm_hop,
    )

    output_path = args.output or str(Path(args.input).with_suffix('.dj-analysis.json'))

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  Output: {output_path}")
    size_kb = Path(output_path).stat().st_size / 1024
    print(f"  Size: {size_kb:.0f} KB\n")


if __name__ == "__main__":
    main()
