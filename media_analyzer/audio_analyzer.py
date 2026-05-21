"""
Audio analysis module.

Analyzes music/audio files and produces structured JSON with:
- BPM / tempo estimation
- Beat timestamps
- Song section segmentation (verse, chorus, bridge, etc.)
- Energy / loudness curve over time
- Spectral features (brightness, warmth)
- Key and scale detection
- Mood descriptors
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import librosa
import soundfile as sf


def load_audio(filepath: str, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load any audio format via soundfile/librosa."""
    y, sr_orig = librosa.load(filepath, sr=sr, mono=True)
    return y, sr


def estimate_tempo_and_beats(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Estimate BPM and beat positions."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # tempo may be an array in newer librosa versions
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo = float(tempo)

    return {
        "bpm": round(tempo, 2),
        "beat_count": len(beat_times),
        "beat_times_sec": [round(t, 4) for t in beat_times],
    }


def analyze_energy(y: np.ndarray, sr: int, hop_length: int = 512) -> dict[str, Any]:
    """Compute RMS energy curve and loudness statistics."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Downsample to ~2 points per second for manageable output
    target_fps = 2
    step = max(1, int(sr / hop_length / target_fps))
    sampled_times = times[::step].tolist()
    sampled_rms = rms[::step].tolist()

    return {
        "sample_rate_hz": target_fps,
        "times_sec": [round(t, 3) for t in sampled_times],
        "rms_energy": [round(float(v), 6) for v in sampled_rms],
        "peak_energy": round(float(np.max(rms)), 6),
        "mean_energy": round(float(np.mean(rms)), 6),
        "dynamic_range_db": round(float(20 * np.log10(np.max(rms) / (np.min(rms) + 1e-10))), 2),
    }


def analyze_spectral(y: np.ndarray, sr: int, hop_length: int = 512) -> dict[str, Any]:
    """Spectral centroid (brightness) and rolloff (warmth indicator)."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]

    return {
        "spectral_centroid_mean_hz": round(float(np.mean(centroid)), 2),
        "spectral_centroid_std_hz": round(float(np.std(centroid)), 2),
        "spectral_rolloff_mean_hz": round(float(np.mean(rolloff)), 2),
        "spectral_bandwidth_mean_hz": round(float(np.mean(bandwidth)), 2),
        "zero_crossing_rate_mean": round(float(np.mean(zcr)), 6),
        "brightness": _classify_brightness(np.mean(centroid)),
        "warmth": _classify_warmth(np.mean(rolloff), sr),
    }


def _classify_brightness(centroid_mean: float) -> str:
    if centroid_mean < 1500:
        return "dark"
    elif centroid_mean < 3000:
        return "moderate"
    elif centroid_mean < 5000:
        return "bright"
    else:
        return "very_bright"


def _classify_warmth(rolloff_mean: float, sr: int) -> str:
    ratio = rolloff_mean / (sr / 2)
    if ratio < 0.5:
        return "very_warm"
    elif ratio < 0.7:
        return "warm"
    elif ratio < 0.85:
        return "neutral"
    else:
        return "cold"


def detect_key(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Estimate musical key using chroma features."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Major and minor profiles (Krumhansl-Kessler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    best_corr = -1
    best_key = "C"
    best_mode = "major"

    for i in range(12):
        rolled = np.roll(chroma_mean, -i)
        corr_maj = float(np.corrcoef(rolled, major_profile)[0, 1])
        corr_min = float(np.corrcoef(rolled, minor_profile)[0, 1])

        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = pitch_classes[i]
            best_mode = "major"
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = pitch_classes[i]
            best_mode = "minor"

    # Chroma distribution for downstream use
    chroma_dist = {pitch_classes[i]: round(float(chroma_mean[i]), 4) for i in range(12)}

    return {
        "key": best_key,
        "mode": best_mode,
        "key_label": f"{best_key} {best_mode}",
        "confidence": round(best_corr, 4),
        "chroma_distribution": chroma_dist,
    }


def segment_sections(y: np.ndarray, sr: int) -> dict[str, Any]:
    """
    Detect structural sections using self-similarity and spectral clustering.
    Labels are numeric (section_0, section_1, ...) since automatic semantic
    labeling (verse/chorus) requires a trained model.
    """
    # Compute features for segmentation
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    features = np.vstack([mfcc, chroma])

    # Structural segmentation via recurrence/novelty
    try:
        bound_frames = librosa.segment.agglomerative(features, k=None)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr).tolist()
    except Exception:
        # Fallback: use novelty-based segmentation
        novelty = librosa.onset.onset_strength(y=y, sr=sr)
        # Find peaks in novelty as section boundaries
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(novelty, distance=sr // 512 * 8, prominence=np.std(novelty))
        bound_times = librosa.frames_to_time(peaks, sr=sr).tolist()

    duration = float(len(y) / sr)

    # Build section list
    boundaries = [0.0] + bound_times + [duration]
    boundaries = sorted(set(boundaries))

    sections = []
    for i in range(len(boundaries) - 1):
        sections.append({
            "label": f"section_{i}",
            "start_sec": round(boundaries[i], 3),
            "end_sec": round(boundaries[i + 1], 3),
            "duration_sec": round(boundaries[i + 1] - boundaries[i], 3),
        })

    return {
        "section_count": len(sections),
        "boundary_times_sec": [round(t, 3) for t in boundaries],
        "sections": sections,
    }


def infer_mood(energy: dict, spectral: dict, key_info: dict, tempo: float) -> dict[str, Any]:
    """
    Heuristic mood inference based on acoustic features.
    Returns mood tags and valence/arousal estimates.
    """
    tags = []
    valence = 0.5  # 0=negative, 1=positive
    arousal = 0.5  # 0=calm, 1=energetic

    # Tempo contribution
    if tempo < 80:
        tags.append("slow")
        arousal -= 0.2
    elif tempo < 120:
        tags.append("moderate_tempo")
    elif tempo < 150:
        tags.append("upbeat")
        arousal += 0.15
    else:
        tags.append("fast")
        arousal += 0.3

    # Energy contribution
    mean_e = energy["mean_energy"]
    if mean_e < 0.05:
        tags.append("quiet")
        arousal -= 0.15
    elif mean_e > 0.15:
        tags.append("loud")
        arousal += 0.15

    # Brightness
    brightness = spectral["brightness"]
    if brightness in ("dark",):
        tags.append("dark")
        valence -= 0.15
    elif brightness in ("bright", "very_bright"):
        tags.append("bright")
        valence += 0.1

    # Key mode
    if key_info["mode"] == "minor":
        tags.append("melancholic")
        valence -= 0.2
    else:
        tags.append("uplifting")
        valence += 0.15

    # Dynamic range
    if energy["dynamic_range_db"] > 30:
        tags.append("dynamic")
    elif energy["dynamic_range_db"] < 10:
        tags.append("compressed")

    # Clamp
    valence = round(max(0, min(1, valence)), 3)
    arousal = round(max(0, min(1, arousal)), 3)

    # Top-level mood label
    if valence > 0.6 and arousal > 0.6:
        mood = "energetic_positive"
    elif valence > 0.6 and arousal <= 0.6:
        mood = "calm_positive"
    elif valence <= 0.4 and arousal > 0.6:
        mood = "intense_dark"
    elif valence <= 0.4 and arousal <= 0.4:
        mood = "somber"
    else:
        mood = "neutral"

    return {
        "primary_mood": mood,
        "tags": tags,
        "valence": valence,
        "arousal": arousal,
    }


def analyze_audio(filepath: str, sr: int = 22050, quiet: bool = False) -> dict[str, Any]:
    """Full audio analysis pipeline."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    def _log(msg: str):
        if not quiet:
            print(msg)

    _log(f"Loading: {filepath}")
    y, sr = load_audio(filepath, sr=sr)
    duration = float(len(y) / sr)
    _log(f"  Duration: {duration:.1f}s | Sample rate: {sr}")

    _log("  Analyzing tempo & beats...")
    tempo_info = estimate_tempo_and_beats(y, sr)

    _log("  Analyzing energy...")
    energy_info = analyze_energy(y, sr)

    _log("  Analyzing spectral features...")
    spectral_info = analyze_spectral(y, sr)

    _log("  Detecting key...")
    key_info = detect_key(y, sr)

    _log("  Segmenting sections...")
    section_info = segment_sections(y, sr)

    _log("  Inferring mood...")
    mood_info = infer_mood(energy_info, spectral_info, key_info, tempo_info["bpm"])

    result = {
        "schema_version": "1.0.0",
        "analyzer": "media-analyzer-audio",
        "file": {
            "path": str(path.resolve()),
            "name": path.name,
            "format": path.suffix.lstrip('.').lower(),
            "duration_sec": round(duration, 3),
            "sample_rate": sr,
        },
        "tempo": tempo_info,
        "energy": energy_info,
        "spectral": spectral_info,
        "key": key_info,
        "sections": section_info,
        "mood": mood_info,
    }

    _log("  Done.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze audio files for ML-friendly JSON output")
    parser.add_argument("input", help="Path to audio file (mp3, wav, flac, ogg, m4a, aiff, etc.)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <input>.analysis.json)")
    parser.add_argument("--sr", type=int, default=22050, help="Sample rate for analysis (default: 22050)")
    parser.add_argument("--pretty", action="store_true", default=True, help="Pretty-print JSON (default: true)")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output")

    args = parser.parse_args()

    result = analyze_audio(args.input, sr=args.sr)

    output_path = args.output or (args.input + ".analysis.json")
    indent = None if args.compact else 2

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=indent)

    print(f"\nAnalysis written to: {output_path}")


if __name__ == "__main__":
    main()
