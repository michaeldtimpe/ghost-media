#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  DEEP DJ SET ANALYZER
  High-resolution, multi-dimensional audio analysis for driving live visuals
═══════════════════════════════════════════════════════════════════════════════

Produces:
  - Multi-band energy (7 frequency bands at 8 fps)
  - Harmonic-percussive separation energy curves
  - Onset detection with strength envelope
  - Chromagram timeline (12 pitch classes)
  - Spectral flux / novelty curve
  - Windowed BPM with confidence
  - Per-beat features (energy, spectral centroid, band ratios)
  - Per-track detailed analysis with tracklist alignment
  - Transition characterization
  - Phrase structure (4-bar, 8-bar, 16-bar)
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import librosa
import scipy.signal


# ─── Configuration ─────────────────────────────────────────────────────────

SAMPLE_RATE = 22050
HOP_LENGTH = 512
TIMELINE_FPS = 8  # samples per second for timelines
BEAT_WINDOW_MS = 50  # ms around each beat for energy

# Frequency bands for multi-band analysis
BANDS = {
    "sub_bass":   (20, 60),
    "bass":       (60, 250),
    "low_mid":    (250, 500),
    "mid":        (500, 2000),
    "high_mid":   (2000, 4000),
    "presence":   (4000, 8000),
    "brilliance": (8000, 11025),
}


# ─── Tracklist Parser ──────────────────────────────────────────────────────

def parse_tracklist(filepath):
    tracks = []
    ts_pattern = re.compile(r'^\*{0,3}\s*\[(\d{1,2}:\d{2}(?::\d{2})?)\]\s*(.+)')
    with open(filepath, 'r') as f:
        for line in f:
            m = ts_pattern.match(line.strip())
            if not m:
                continue
            ts_str, title = m.group(1), m.group(2).strip()
            parts = ts_str.split(':')
            secs = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2]) if len(parts) == 3 \
                else int(parts[0]) * 60 + int(parts[1])
            tracks.append({"start_sec": secs, "title": title})
    return tracks


def build_track_segments(tracks, total_duration):
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


# ─── Multi-Band Energy ─────────────────────────────────────────────────────

def compute_multiband_energy(y, sr, hop_length, fps):
    """Compute energy in each frequency band using STFT."""
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    n_frames = S.shape[1]
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # Compute energy per band
    band_energy = {}
    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_power = np.sum(S[mask, :] ** 2, axis=0)
        band_rms = np.sqrt(band_power / max(1, np.sum(mask)))
        band_energy[band_name] = band_rms

    # Total energy
    total_rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]

    # Downsample to target fps
    native_fps = sr / hop_length
    step = max(1, int(native_fps / fps))

    timeline = []
    for i in range(0, len(times), step):
        entry = {"time_sec": round(float(times[i]), 4)}
        entry["total_rms"] = round(float(total_rms[i]), 6)
        for band_name in BANDS:
            entry[band_name] = round(float(band_energy[band_name][i]), 6)
        timeline.append(entry)

    return timeline, S, freqs


# ─── Harmonic-Percussive Separation ───────────────────────────────────────

def compute_hpss_energy(y, sr, hop_length, fps):
    """Separate harmonic and percussive components, compute energy curves."""
    y_harm, y_perc = librosa.effects.hpss(y)

    rms_harm = librosa.feature.rms(y=y_harm, hop_length=hop_length)[0]
    rms_perc = librosa.feature.rms(y=y_perc, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms_harm)), sr=sr, hop_length=hop_length)

    native_fps = sr / hop_length
    step = max(1, int(native_fps / fps))

    timeline = []
    for i in range(0, len(times), step):
        timeline.append({
            "time_sec": round(float(times[i]), 4),
            "harmonic": round(float(rms_harm[i]), 6),
            "percussive": round(float(rms_perc[i]), 6),
            "hp_ratio": round(float(rms_harm[i] / (rms_perc[i] + 1e-10)), 4),
        })

    return timeline, y_harm, y_perc


# ─── Onset Detection ──────────────────────────────────────────────────────

def detect_onsets(y, sr, hop_length, fps):
    """Detect onsets and compute onset strength envelope."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length,
        onset_envelope=onset_env, backtrack=True
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length).tolist()

    # Onset strength envelope at target fps
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    native_fps = sr / hop_length
    step = max(1, int(native_fps / fps))

    envelope = []
    for i in range(0, len(times), step):
        envelope.append({
            "time_sec": round(float(times[i]), 4),
            "strength": round(float(onset_env[i]), 6),
        })

    return {
        "onset_count": len(onset_times),
        "onset_times_sec": [round(t, 4) for t in onset_times],
        "onset_rate_per_sec": round(len(onset_times) / (len(y) / sr), 3),
        "strength_envelope": envelope,
    }


# ─── Chromagram ────────────────────────────────────────────────────────────

def compute_chromagram(y, sr, hop_length, fps):
    """Compute chromagram timeline for key/harmony tracking."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    native_fps = sr / hop_length
    step = max(1, int(native_fps / fps))

    timeline = []
    for i in range(0, chroma.shape[1], step):
        entry = {"time_sec": round(float(times[i]), 4)}
        for j, pc in enumerate(pitch_classes):
            entry[pc] = round(float(chroma[j, i]), 4)
        # Dominant pitch class at this moment
        entry["dominant"] = pitch_classes[int(np.argmax(chroma[:, i]))]
        timeline.append(entry)

    return timeline


# ─── Spectral Features Timeline ───────────────────────────────────────────

def compute_spectral_timeline(y, sr, hop_length, fps):
    """Compute spectral centroid, bandwidth, flatness, rolloff, flux."""
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)

    # Spectral flux
    S = np.abs(librosa.stft(y, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    flux = np.concatenate([[0], flux])

    times = librosa.frames_to_time(np.arange(len(centroid)), sr=sr, hop_length=hop_length)
    native_fps = sr / hop_length
    step = max(1, int(native_fps / fps))

    timeline = []
    for i in range(0, len(times), step):
        timeline.append({
            "time_sec": round(float(times[i]), 4),
            "centroid_hz": round(float(centroid[i]), 1),
            "bandwidth_hz": round(float(bandwidth[i]), 1),
            "flatness": round(float(flatness[i]), 6),
            "rolloff_hz": round(float(rolloff[i]), 1),
            "flux": round(float(flux[i]), 4),
            "contrast_mean": round(float(np.mean(contrast[:, i])), 4),
        })

    return timeline


# ─── Windowed BPM with Confidence ─────────────────────────────────────────

def estimate_windowed_bpm(y, sr, window_sec=8, hop_sec=2):
    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    results = []

    pos = 0
    while pos + window_samples <= len(y):
        segment = y[pos:pos + window_samples]
        time_center = (pos + window_samples / 2) / sr

        onset_env = librosa.onset.onset_strength(y=segment, sr=sr)

        # Get tempo with prior to help track continuous changes
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
        bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)

        # Tempogram for confidence
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        ac = np.mean(tg, axis=1)
        confidence = float(np.max(ac[1:])) / (float(np.mean(ac[1:])) + 1e-10)

        results.append({
            "time_sec": round(time_center, 3),
            "bpm": round(bpm, 2),
            "confidence": round(min(confidence, 10.0), 3),
        })
        pos += hop_samples

    return results


# ─── Beat-Grid Stability ──────────────────────────────────────────────────

# Thresholds for the three checks. Calibrated against a 5-set corpus
# (Phase 5b backfill, 2026-05-28): see audio_field_audit.md and lessons.md
# for the per-set distribution and the rationale behind the chosen numbers.
IOI_FRACTIONAL_TOLERANCE = 0.15       # ±15% of median beat interval
IOI_OUTLIER_PCT_WARN = 4.0            # corpus median sat near 2-3%; trips at 4% catch true jitter, not transitions
OCTAVE_DOUBLING_PCT_WARN = 10.0       # corpus max <2%; threshold has clean headroom
OCTAVE_DOUBLING_RUN_WARN = 3          # 60% of corpus trips; intentional — these are real librosa-lock events worth flagging
METRONOMIC_DEVIATION_WARN_SEC = 60.0  # corpus range 23-54s for healthy sets (intentional tempo ramping); warn only on extreme outliers


def validate_beat_grid(beat_times, bpm_timeline, global_bpm):
    """Three observability checks on librosa's beat output.

    NOT a verdict on tracker quality — `metronomic_deviation` measures
    deviation from a constant-tempo extrapolation, which is non-zero for
    healthy sets with intentional tempo ramps. Use as one signal among
    several, not a pass/fail.

    Non-blocking: returns a dict; the caller decides whether to warn.
    """
    bt = np.asarray(beat_times, dtype=np.float64)
    n_beats = len(bt)

    # ── Check A: IOI outlier rate ─────────────────────────────────────────
    # Fractional tolerance avoids the np.std(iois) pitfall (the outliers
    # inflate std, loosening the threshold against themselves).
    if n_beats > 2:
        iois = np.diff(bt)
        median_ioi = float(np.median(iois))
        tolerance = IOI_FRACTIONAL_TOLERANCE * median_ioi
        ioi_outliers = int(np.sum(np.abs(iois - median_ioi) > tolerance))
        ioi_outlier_pct = 100.0 * ioi_outliers / len(iois)
    else:
        median_ioi = 0.0
        ioi_outliers = 0
        ioi_outlier_pct = 0.0

    # ── Check B: octave-doubling rate + max run length ────────────────────
    # A 3-window run (24s at the analyzer's 8s window step) of locked-doubled
    # tempo is qualitatively worse than 3 scattered windows; report both.
    bpm_vals = np.array([b["bpm"] for b in bpm_timeline], dtype=np.float64)
    if len(bpm_vals) and global_bpm > 0:
        ratios = bpm_vals / global_bpm
        doubling_mask = ((ratios >= 0.45) & (ratios <= 0.55)) | \
                        ((ratios >= 1.9) & (ratios <= 2.1))
        octave_doubling_pct = 100.0 * float(np.mean(doubling_mask))
        # Longest run of consecutive True
        run_max = 0
        cur = 0
        for v in doubling_mask:
            if v:
                cur += 1
                if cur > run_max:
                    run_max = cur
            else:
                cur = 0
        octave_doubling_run_max = int(run_max)
    else:
        octave_doubling_pct = 0.0
        octave_doubling_run_max = 0

    # ── Check C: deviation from constant-tempo extrapolation ──────────────
    # NOT "tracker drift": this is the distance between observed beats and
    # an idealized metronomic grid built from global_bpm. Healthy DJ sets
    # with intentional tempo ramps will register non-zero deviation here.
    # Invalid if Check B flagged octave-locking (expected grid is built on
    # wrong-tempo base; deviation becomes meaningless).
    if n_beats > 1 and global_bpm > 0:
        expected_times = bt[0] + np.arange(n_beats) * (60.0 / global_bpm)
        deviation = bt - expected_times
        metronomic_deviation_max_sec = float(np.max(np.abs(deviation)))
        metronomic_deviation_final_sec = float(deviation[-1])
    else:
        metronomic_deviation_max_sec = 0.0
        metronomic_deviation_final_sec = 0.0

    metronomic_deviation_valid = octave_doubling_pct <= OCTAVE_DOUBLING_PCT_WARN

    result = {
        "ioi_median_sec": round(median_ioi, 5),
        "ioi_outlier_count": ioi_outliers,
        "ioi_outlier_pct": round(ioi_outlier_pct, 3),
        "octave_doubling_pct": round(octave_doubling_pct, 3),
        "octave_doubling_run_max": octave_doubling_run_max,
        "metronomic_deviation_max_sec": round(metronomic_deviation_max_sec, 3),
        "metronomic_deviation_final_sec": round(metronomic_deviation_final_sec, 3),
        "metronomic_deviation_valid": metronomic_deviation_valid,
    }

    # ── Console warnings ───────────────────────────────────────────────────
    if ioi_outlier_pct > IOI_OUTLIER_PCT_WARN:
        print(f"    ⚠ beat-grid: IOI outliers {ioi_outlier_pct:.1f}% > {IOI_OUTLIER_PCT_WARN}% threshold")
    if octave_doubling_pct > OCTAVE_DOUBLING_PCT_WARN:
        print(f"    ⚠ beat-grid: octave-doubling {octave_doubling_pct:.1f}% > {OCTAVE_DOUBLING_PCT_WARN}% (windowed BPM locked at 0.5× or 2× global)")
    if octave_doubling_run_max > OCTAVE_DOUBLING_RUN_WARN:
        print(f"    ⚠ beat-grid: octave-doubling run of {octave_doubling_run_max} windows (~{octave_doubling_run_max*8}s of locked-wrong tempo)")
    if metronomic_deviation_valid and metronomic_deviation_max_sec > METRONOMIC_DEVIATION_WARN_SEC:
        print(f"    ⚠ beat-grid: metronomic deviation {metronomic_deviation_max_sec:.2f}s > {METRONOMIC_DEVIATION_WARN_SEC}s (set diverges from constant-tempo extrapolation; could be intentional tempo ramp or tracker failure)")

    return result


# ─── BPM-Timeline Repair ───────────────────────────────────────────────────

# Rolling-median window for octave-lock detection, in timeline entries
# (2s hop → 31 entries ≈ 62s). Wide enough that the documented 7–15-window
# locked runs (lessons.md) stay a minority of the window and get caught.
BPM_REPAIR_MEDIAN_WINDOW = 31
# Confidence multiplier for folded windows: the BPM value is corrected but
# the tracker was confused there, so downstream consumers (assembler
# dur_match weighting) should distrust these windows.
OCTAVE_CONFIDENCE_PENALTY = 0.25


def repair_bpm_timeline(bpm_timeline):
    """Fold half/double-tempo locked windows back onto the local tempo.

    librosa's tempogram locks at 0.5×/2× during DJ transitions (3/5 sets had
    56–120s runs; lessons.md). Each window's BPM is compared to a rolling
    median; ratios in the octave bands fold by 2×/0.5× and keep a penalized
    confidence. Mutates in place; returns (timeline, corrected_times).
    """
    if not bpm_timeline:
        return bpm_timeline, []
    bpms = np.array([b["bpm"] for b in bpm_timeline], dtype=np.float64)
    n = len(bpms)
    half = BPM_REPAIR_MEDIAN_WINDOW // 2
    corrected_times = []
    for i, entry in enumerate(bpm_timeline):
        local_median = float(np.median(bpms[max(0, i - half):min(n, i + half + 1)]))
        if local_median <= 0:
            continue
        ratio = entry["bpm"] / local_median
        if 0.45 <= ratio <= 0.55:
            factor = 2.0
        elif 1.9 <= ratio <= 2.1:
            factor = 0.5
        else:
            continue
        entry["bpm"] = round(entry["bpm"] * factor, 2)
        entry["confidence"] = round(entry["confidence"] * OCTAVE_CONFIDENCE_PENALTY, 3)
        corrected_times.append(entry["time_sec"])
    return bpm_timeline, corrected_times


# ─── Downbeat / Phrase-Anchor Estimation ──────────────────────────────────

def estimate_downbeats(beat_times, multiband_timeline, beats_per_bar=4):
    """Estimate which beat phase carries the downbeats via bass-arrival voting.

    Scores each phase offset by the mean positive delta of (sub_bass + bass)
    energy arriving at its beats — kick re-entry lands on bar/phrase starts.
    HEURISTIC: on the 5-set corpus the candidate signals (bass arrival, |ΔRMS|,
    brilliance arrival) disagree on phase, so this is persisted for
    observability and audit only; phrase anchoring stays opt-in
    (--anchor-phrases) until validated by ear. See lessons.md.
    """
    bt = np.asarray(beat_times, dtype=np.float64)
    result = {
        "estimator": "bass-arrival-heuristic",
        "bar_offset": 0,
        "phrase_offset_16": 0,
        "bar_margin": 0.0,
        "phrase_margin": 0.0,
        "downbeats_sec": [round(float(t), 4) for t in bt[::beats_per_bar]],
    }
    if len(bt) < beats_per_bar * 8 or not multiband_timeline:
        return result

    times = np.array([m["time_sec"] for m in multiband_timeline], dtype=np.float64)
    bass = np.array([m["sub_bass"] + m["bass"] for m in multiband_timeline], dtype=np.float64)
    beat_bass = np.interp(bt, times, bass)
    arrival = np.concatenate([[0.0], np.clip(np.diff(beat_bass), 0.0, None)])

    def phase_scores(period):
        scores = np.array([arrival[p::period].mean() for p in range(period)])
        best = int(np.argmax(scores))
        median = float(np.median(scores))
        margin = (float(scores[best]) - median) / (median + 1e-10)
        return best, margin, scores

    bar_offset, bar_margin, _ = phase_scores(beats_per_bar)
    phrase_offset, phrase_margin, _ = phase_scores(beats_per_bar * 4)
    result.update({
        "bar_offset": bar_offset,
        "phrase_offset_16": phrase_offset,
        "bar_margin": round(bar_margin, 4),
        "phrase_margin": round(phrase_margin, 4),
        "downbeats_sec": [round(float(t), 4) for t in bt[bar_offset::beats_per_bar]],
    })
    return result


# ─── Beat Grid with Features ──────────────────────────────────────────────

def compute_beat_features(y, sr, beat_times, S, freqs, hop_length):
    """Compute per-beat features: energy, spectral centroid, band ratios."""
    half_win = int(BEAT_WINDOW_MS / 1000 * sr)
    results = []

    for bt in beat_times:
        center = int(bt * sr)
        start = max(0, center - half_win)
        end = min(len(y), center + half_win)
        seg = y[start:end]

        rms = float(np.sqrt(np.mean(seg ** 2)))

        # Spectral centroid for this beat
        frame_idx = int(bt * sr / hop_length)
        frame_idx = min(frame_idx, S.shape[1] - 1)
        spectrum = S[:, frame_idx]

        weighted_freq = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

        # Band ratios
        low_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = freqs >= 4000
        total_power = np.sum(spectrum ** 2) + 1e-10
        low_ratio = float(np.sum(spectrum[low_mask] ** 2) / total_power)
        mid_ratio = float(np.sum(spectrum[mid_mask] ** 2) / total_power)
        high_ratio = float(np.sum(spectrum[high_mask] ** 2) / total_power)

        results.append({
            "time_sec": round(bt, 4),
            "energy": round(rms, 6),
            "centroid_hz": round(float(weighted_freq), 1),
            "low_ratio": round(low_ratio, 4),
            "mid_ratio": round(mid_ratio, 4),
            "high_ratio": round(high_ratio, 4),
        })

    return results


# ─── Phrase Detection ──────────────────────────────────────────────────────

def detect_phrases(beat_times, beat_features, beats_per_bar=4, anchor_offset=0):
    """Detect 4-bar, 8-bar, and 16-bar phrases with per-phrase energy stats.

    Schema 2.2.0: end_sec is the start of the NEXT phrase (the beat after the
    chunk's last beat), so [start_sec, end_sec) spans the full musical phrase.
    Pre-2.2.0 set end_sec to the last beat's timestamp, undershooting by one
    beat — every assembler cut landed one beat early (cut-alignment audit:
    median cut→next-phrase gap = exactly one beat period on all 5 sets).

    anchor_offset shifts the chunk grid to start at a downbeat estimate
    (beats before the anchor form a leading stub phrase). Default 0: the
    downbeat estimators disagree on this corpus (see lessons.md), so
    anchoring is opt-in via --anchor-phrases until validated by ear.
    """
    median_ioi = float(np.median(np.diff(beat_times))) if len(beat_times) > 2 else 0.5
    phrases = {}
    for phrase_bars, label in [(4, "four_bar"), (8, "eight_bar"), (16, "sixteen_bar")]:
        beats_per_phrase = beats_per_bar * phrase_bars
        starts = list(range(anchor_offset, len(beat_times), beats_per_phrase))
        if anchor_offset >= beats_per_bar:
            starts.insert(0, 0)  # leading stub phrase before the first anchor
        elif anchor_offset > 0 and starts:
            starts[0] = 0  # fold a sub-bar lead-in into the first phrase
        phrase_list = []
        for n, i in enumerate(starts):
            stop = starts[n + 1] if n + 1 < len(starts) else i + beats_per_phrase
            chunk_times = beat_times[i:stop]
            chunk_feats = beat_features[i:stop]
            if len(chunk_times) < beats_per_bar:
                break
            energies = [f["energy"] for f in chunk_feats]
            # End at the next beat after the chunk — the true phrase span.
            next_beat_idx = i + len(chunk_times)
            end_sec = (beat_times[next_beat_idx] if next_beat_idx < len(beat_times)
                       else chunk_times[-1] + median_ioi)
            phrase_list.append({
                "phrase_index": len(phrase_list),
                "start_sec": round(chunk_times[0], 4),
                "end_sec": round(end_sec, 4),
                "beat_count": len(chunk_times),
                "bars": len(chunk_times) // beats_per_bar,
                "energy_mean": round(float(np.mean(energies)), 6),
                "energy_peak": round(float(np.max(energies)), 6),
                "energy_shape": _classify_energy_shape(energies),
            })
        phrases[label] = phrase_list
    return phrases


def _classify_energy_shape(energies):
    """Classify the energy contour of a phrase."""
    if len(energies) < 4:
        return "flat"
    first_half = np.mean(energies[:len(energies)//2])
    second_half = np.mean(energies[len(energies)//2:])
    ratio = second_half / (first_half + 1e-10)
    if ratio > 1.3:
        return "building"
    elif ratio < 0.7:
        return "decaying"
    elif np.std(energies) / (np.mean(energies) + 1e-10) > 0.5:
        return "dynamic"
    else:
        return "sustain"


# ─── Per-Track Deep Analysis ──────────────────────────────────────────────

def analyze_tracks_deep(track_segments, beat_times, beat_features, bpm_timeline,
                        multiband_timeline, hpss_timeline, spectral_timeline,
                        chroma_timeline):
    """Compute per-track summary stats from all feature timelines."""
    for seg in track_segments:
        start, end = seg["start_sec"], seg["end_sec"]

        # Beats
        seg_beats = [(bt, bf) for bt, bf in zip(beat_times, beat_features)
                     if start <= bt < end]
        seg_beat_energies = [bf["energy"] for _, bf in seg_beats]
        seg_beat_centroids = [bf["centroid_hz"] for _, bf in seg_beats]

        # BPM
        seg_bpm = [b["bpm"] for b in bpm_timeline if start <= b["time_sec"] < end]

        # Multiband
        seg_mb = [m for m in multiband_timeline if start <= m["time_sec"] < end]

        # HPSS
        seg_hp = [h for h in hpss_timeline if start <= h["time_sec"] < end]

        # Spectral
        seg_sp = [s for s in spectral_timeline if start <= s["time_sec"] < end]

        # Chroma - find dominant key for this track
        seg_ch = [c for c in chroma_timeline if start <= c["time_sec"] < end]
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        seg["beat_count"] = len(seg_beats)
        seg["bpm"] = {
            "mean": round(float(np.mean(seg_bpm)), 2) if seg_bpm else 0,
            "min": round(float(np.min(seg_bpm)), 2) if seg_bpm else 0,
            "max": round(float(np.max(seg_bpm)), 2) if seg_bpm else 0,
            "std": round(float(np.std(seg_bpm)), 2) if seg_bpm else 0,
        }
        seg["energy"] = {
            "mean": round(float(np.mean(seg_beat_energies)), 6) if seg_beat_energies else 0,
            "peak": round(float(np.max(seg_beat_energies)), 6) if seg_beat_energies else 0,
            "std": round(float(np.std(seg_beat_energies)), 6) if seg_beat_energies else 0,
        }

        # Band energy averages
        if seg_mb:
            seg["bands"] = {}
            for band in BANDS:
                vals = [m[band] for m in seg_mb]
                seg["bands"][band] = {
                    "mean": round(float(np.mean(vals)), 6),
                    "peak": round(float(np.max(vals)), 6),
                }

        # Harmonic-percussive balance
        if seg_hp:
            harm_vals = [h["harmonic"] for h in seg_hp]
            perc_vals = [h["percussive"] for h in seg_hp]
            seg["harmonic_percussive"] = {
                "harmonic_mean": round(float(np.mean(harm_vals)), 6),
                "percussive_mean": round(float(np.mean(perc_vals)), 6),
                "balance": "harmonic" if np.mean(harm_vals) > np.mean(perc_vals) * 1.3
                    else "percussive" if np.mean(perc_vals) > np.mean(harm_vals) * 1.3
                    else "balanced",
            }

        # Spectral character
        if seg_sp:
            centroids = [s["centroid_hz"] for s in seg_sp]
            flatnesses = [s["flatness"] for s in seg_sp]
            fluxes = [s["flux"] for s in seg_sp]
            seg["spectral"] = {
                "centroid_mean_hz": round(float(np.mean(centroids)), 1),
                "centroid_std_hz": round(float(np.std(centroids)), 1),
                "flatness_mean": round(float(np.mean(flatnesses)), 6),
                "flux_mean": round(float(np.mean(fluxes)), 4),
                "flux_peak": round(float(np.max(fluxes)), 4),
                "brightness": _classify_brightness(np.mean(centroids)),
                "texture": "noisy" if np.mean(flatnesses) > 0.1 else "tonal",
            }

        # Dominant chroma
        if seg_ch:
            chroma_avgs = {pc: np.mean([c[pc] for c in seg_ch]) for pc in pitch_classes}
            dom_pc = max(chroma_avgs, key=chroma_avgs.get)
            seg["chroma_dominant"] = dom_pc

    return track_segments


def _classify_brightness(centroid_mean):
    if centroid_mean < 1500:
        return "dark"
    elif centroid_mean < 3000:
        return "moderate"
    elif centroid_mean < 5000:
        return "bright"
    return "very_bright"


# ─── Transition Analysis ──────────────────────────────────────────────────

def analyze_transitions(track_segments, multiband_timeline, hpss_timeline,
                        bpm_timeline, spectral_timeline):
    transitions = []
    for i in range(len(track_segments) - 1):
        boundary = track_segments[i]["end_sec"]
        window = 15

        def get_window(timeline, key, t_start, t_end):
            return [e[key] for e in timeline if t_start <= e["time_sec"] < t_end]

        pre_e = get_window(multiband_timeline, "total_rms", boundary - window, boundary)
        post_e = get_window(multiband_timeline, "total_rms", boundary, boundary + window)
        pre_bass = get_window(multiband_timeline, "bass", boundary - window, boundary)
        post_bass = get_window(multiband_timeline, "bass", boundary, boundary + window)
        pre_hp = get_window(hpss_timeline, "hp_ratio", boundary - window, boundary)
        post_hp = get_window(hpss_timeline, "hp_ratio", boundary, boundary + window)
        pre_bpm = get_window(bpm_timeline, "bpm", boundary - window, boundary)
        post_bpm = get_window(bpm_timeline, "bpm", boundary, boundary + window)
        pre_flux = get_window(spectral_timeline, "flux", boundary - window, boundary)
        post_flux = get_window(spectral_timeline, "flux", boundary, boundary + window)

        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.0

        pre_e_avg, post_e_avg = safe_mean(pre_e), safe_mean(post_e)
        pre_bass_avg, post_bass_avg = safe_mean(pre_bass), safe_mean(post_bass)
        pre_bpm_avg, post_bpm_avg = safe_mean(pre_bpm), safe_mean(post_bpm)

        transitions.append({
            "from_track": track_segments[i]["title"],
            "to_track": track_segments[i + 1]["title"],
            "boundary_sec": boundary,
            "energy_before": round(pre_e_avg, 6),
            "energy_after": round(post_e_avg, 6),
            "energy_delta": round(post_e_avg - pre_e_avg, 6),
            "bass_before": round(pre_bass_avg, 6),
            "bass_after": round(post_bass_avg, 6),
            "bass_delta": round(post_bass_avg - pre_bass_avg, 6),
            "bpm_before": round(pre_bpm_avg, 2),
            "bpm_after": round(post_bpm_avg, 2),
            "bpm_delta": round(post_bpm_avg - pre_bpm_avg, 2),
            "hp_ratio_before": round(safe_mean(pre_hp), 4),
            "hp_ratio_after": round(safe_mean(post_hp), 4),
            "flux_before": round(safe_mean(pre_flux), 4),
            "flux_after": round(safe_mean(post_flux), 4),
            "type": _classify_transition(
                pre_e_avg, post_e_avg, pre_bpm_avg, post_bpm_avg,
                pre_bass_avg, post_bass_avg
            ),
        })

    return transitions


def _classify_transition(pre_e, post_e, pre_bpm, post_bpm, pre_bass, post_bass):
    e_ratio = post_e / (pre_e + 1e-10)
    bpm_diff = abs(post_bpm - pre_bpm)
    bass_ratio = post_bass / (pre_bass + 1e-10)

    if e_ratio > 1.5 and bass_ratio > 1.3:
        return "bass_drop"
    elif e_ratio > 1.4:
        return "build_drop"
    elif e_ratio < 0.6:
        return "breakdown"
    elif bass_ratio > 1.5:
        return "bass_swell"
    elif bass_ratio < 0.5:
        return "bass_cut"
    elif bpm_diff > 5:
        return "tempo_shift"
    else:
        return "smooth_blend"


# ─── Key Detection (Global + Windowed) ────────────────────────────────────

def detect_key_windowed(y, sr, window_sec=30, hop_sec=10):
    """Detect key in sliding windows for key-change tracking."""
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    window_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    results = []

    pos = 0
    while pos + window_samples <= len(y):
        segment = y[pos:pos + window_samples]
        time_center = (pos + window_samples / 2) / sr

        chroma = librosa.feature.chroma_cqt(y=segment, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        best_corr, best_key, best_mode = -1, "C", "major"
        for i in range(12):
            rolled = np.roll(chroma_mean, -i)
            for profile, mode in [(major_profile, "major"), (minor_profile, "minor")]:
                corr = float(np.corrcoef(rolled, profile)[0, 1])
                if corr > best_corr:
                    best_corr, best_key, best_mode = corr, pitch_classes[i], mode

        results.append({
            "time_sec": round(time_center, 3),
            "key": best_key,
            "mode": best_mode,
            "label": f"{best_key} {best_mode}",
            "confidence": round(best_corr, 4),
        })
        pos += hop_samples

    return results


# ─── Main Pipeline ─────────────────────────────────────────────────────────

def analyze_deep(audio_path, tracklist_path=None, anchor_phrases=False):
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"  ERROR: File not found: {audio_path}")
        sys.exit(1)

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  DEEP DJ SET ANALYZER")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    # ── Load ──
    print(f"  Loading: {audio_path.name}")
    t0 = time.time()
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = float(len(y) / sr)
    print(f"  Duration: {duration / 60:.1f} min | SR: {sr} | ({time.time() - t0:.1f}s)")

    # ── Tracklist ──
    track_segments = []
    if tracklist_path:
        tracks = parse_tracklist(tracklist_path)
        track_segments = build_track_segments(tracks, duration)
        print(f"  Tracklist: {len(track_segments)} tracks")

    # ── Beat detection ──
    print(f"  [1/10] Beat detection...", end=" ", flush=True)
    t0 = time.time()
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    global_bpm = float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)
    print(f"{len(beat_times)} beats, {global_bpm:.1f} BPM ({time.time() - t0:.1f}s)")

    # ── Windowed BPM ──
    print(f"  [2/10] Windowed BPM...", end=" ", flush=True)
    t0 = time.time()
    bpm_timeline = estimate_windowed_bpm(y, sr, window_sec=8, hop_sec=2)
    bpms = [b["bpm"] for b in bpm_timeline]
    print(f"{min(bpms):.0f}-{max(bpms):.0f} BPM ({time.time() - t0:.1f}s)")

    # ── Multi-band energy ──
    print(f"  [3/10] Multi-band energy ({TIMELINE_FPS} fps)...", end=" ", flush=True)
    t0 = time.time()
    multiband_timeline, S, freqs = compute_multiband_energy(y, sr, HOP_LENGTH, TIMELINE_FPS)
    print(f"{len(multiband_timeline)} samples ({time.time() - t0:.1f}s)")

    # ── HPSS ──
    print(f"  [4/10] Harmonic-percussive separation...", end=" ", flush=True)
    t0 = time.time()
    hpss_timeline, y_harm, y_perc = compute_hpss_energy(y, sr, HOP_LENGTH, TIMELINE_FPS)
    print(f"done ({time.time() - t0:.1f}s)")

    # ── Onsets ──
    print(f"  [5/10] Onset detection...", end=" ", flush=True)
    t0 = time.time()
    onsets = detect_onsets(y, sr, HOP_LENGTH, TIMELINE_FPS)
    print(f"{onsets['onset_count']} onsets ({time.time() - t0:.1f}s)")

    # ── Chromagram ──
    print(f"  [6/10] Chromagram...", end=" ", flush=True)
    t0 = time.time()
    chroma_timeline = compute_chromagram(y, sr, HOP_LENGTH, TIMELINE_FPS)
    print(f"{len(chroma_timeline)} samples ({time.time() - t0:.1f}s)")

    # ── Spectral features ──
    print(f"  [7/10] Spectral features...", end=" ", flush=True)
    t0 = time.time()
    spectral_timeline = compute_spectral_timeline(y, sr, HOP_LENGTH, TIMELINE_FPS)
    print(f"{len(spectral_timeline)} samples ({time.time() - t0:.1f}s)")

    # ── Per-beat features ──
    print(f"  [8/10] Per-beat features...", end=" ", flush=True)
    t0 = time.time()
    beat_features = compute_beat_features(y, sr, beat_times, S, freqs, HOP_LENGTH)
    print(f"{len(beat_features)} beats ({time.time() - t0:.1f}s)")

    # ── Downbeat / phrase-anchor estimation ──
    downbeats = estimate_downbeats(beat_times, multiband_timeline)
    anchor_offset = 0
    if anchor_phrases:
        anchor_offset = downbeats["phrase_offset_16"]
        print(f"  Anchoring phrases at beat {anchor_offset} "
              f"(phrase_margin {downbeats['phrase_margin']:.2f})")

    # ── Phrases ──
    print(f"  [9/10] Phrase structure...", end=" ", flush=True)
    t0 = time.time()
    phrases = detect_phrases(beat_times, beat_features, anchor_offset=anchor_offset)
    counts = {k: len(v) for k, v in phrases.items()}
    print(f"{counts} ({time.time() - t0:.1f}s)")

    # ── Key detection ──
    print(f"  [10/10] Key detection (windowed)...", end=" ", flush=True)
    t0 = time.time()
    key_timeline = detect_key_windowed(y, sr, window_sec=30, hop_sec=10)
    print(f"{len(key_timeline)} windows ({time.time() - t0:.1f}s)")

    # ── Beat-grid stability checks ──
    # Validate the RAW timeline first (observability of librosa's behavior),
    # then repair octave locks so persisted BPM + confidence are trustworthy.
    beat_quality = validate_beat_grid(beat_times, bpm_timeline, global_bpm)
    bpm_timeline, octave_corrected = repair_bpm_timeline(bpm_timeline)
    beat_quality["octave_corrected_windows"] = len(octave_corrected)
    beat_quality["octave_corrected_times_sec"] = octave_corrected
    if octave_corrected:
        print(f"    repaired {len(octave_corrected)} octave-locked BPM windows "
              f"(~{len(octave_corrected) * 2}s), confidence penalized ×{OCTAVE_CONFIDENCE_PENALTY}")

    # ── Per-track analysis ──
    transitions = []
    if track_segments:
        print(f"  Per-track analysis...", end=" ", flush=True)
        t0 = time.time()
        track_segments = analyze_tracks_deep(
            track_segments, beat_times, beat_features, bpm_timeline,
            multiband_timeline, hpss_timeline, spectral_timeline, chroma_timeline
        )
        transitions = analyze_transitions(
            track_segments, multiband_timeline, hpss_timeline,
            bpm_timeline, spectral_timeline
        )
        print(f"done ({time.time() - t0:.1f}s)")

    # ── Global key ──
    # Most common key across windows
    key_counts = {}
    for k in key_timeline:
        label = k["label"]
        key_counts[label] = key_counts.get(label, 0) + 1
    global_key = max(key_counts, key=key_counts.get) if key_counts else "unknown"

    # ── Persisted-output pruning (Phase 4) ──
    # Internal analyzers (analyze_tracks_deep, analyze_transitions, beat
    # aggregates) consume the full per-item field sets above. Only the
    # JSON we ship needs to be slim. Drops are per audio_field_audit.md
    # conservative drop list: zero-reader fields removed; latent signals
    # (presence, brilliance, key_timeline, transitions) preserved.
    persisted_multiband = [
        {"time_sec": m["time_sec"],
         "total_rms": m["total_rms"],
         "sub_bass": m["sub_bass"],
         "bass": m["bass"],
         "presence": m["presence"],
         "brilliance": m["brilliance"]}
        for m in multiband_timeline
    ]
    persisted_hpss = [
        {"time_sec": h["time_sec"],
         "harmonic": h["harmonic"],
         "percussive": h["percussive"]}
        for h in hpss_timeline
    ]
    persisted_spectral = [
        {"time_sec": s["time_sec"],
         "centroid_hz": s["centroid_hz"]}
        for s in spectral_timeline
    ]

    # ── Assemble ──
    result = {
        "schema_version": "2.2.0",
        "analyzer": "dj-set-analyzer-deep",
        "file": {
            "path": str(audio_path.resolve()),
            "name": audio_path.name,
            "format": audio_path.suffix.lstrip('.').lower(),
            "duration_sec": round(duration, 3),
            "sample_rate": sr,
        },
        "global": {
            "bpm": round(global_bpm, 2),
            "bpm_range": [round(min(bpms), 2), round(max(bpms), 2)],
            "beat_count": len(beat_times),
            "key": global_key,
            "onset_count": onsets["onset_count"],
            "onset_rate_per_sec": onsets["onset_rate_per_sec"],
        },
        "beat_quality": beat_quality,
        "tracks": track_segments,
        "transitions": transitions,
        "beats": {
            "times_sec": [round(t, 4) for t in beat_times],
            # beats.features per-beat array dropped (Phase 4): consumed only by
            # analyze_tracks_deep as transient input, aggregated into per-track
            # stats and never read again.
            # 2.2.0: heuristic downbeat estimate (observability/audit only —
            # see estimate_downbeats; phrase anchoring is opt-in).
            "downbeats_sec": downbeats["downbeats_sec"],
            "downbeat_estimator": downbeats["estimator"],
            "downbeat_bar_offset": downbeats["bar_offset"],
            "downbeat_bar_margin": downbeats["bar_margin"],
            "phrase_anchor_offset": anchor_offset,
            "phrase_anchor_candidate_16": downbeats["phrase_offset_16"],
            "phrase_anchor_margin": downbeats["phrase_margin"],
        },
        "bpm_timeline": bpm_timeline,
        "multiband_energy": persisted_multiband,
        "hpss_timeline": persisted_hpss,
        "onsets": {
            "count": onsets["onset_count"],
            "times_sec": onsets["onset_times_sec"],
            # onsets.strength_envelope dropped (Phase 4): ~1.7 MB per set,
            # zero readers. The flat per-onset times list above is what gets
            # consumed (Phase 2 onset_density).
        },
        # chroma_timeline dropped (Phase 4): ~27% of file, persisted+aggregate
        # only. chroma_dominant per track is preserved on each tracks[] entry
        # via analyze_tracks_deep; full timeline is recomputable in ~10s/set
        # from audio if future key-aware sequencing needs it.
        "spectral_timeline": persisted_spectral,
        "key_timeline": key_timeline,
        "phrases": phrases,
    }

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  DEEP ANALYSIS COMPLETE")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    return result


def main():
    parser = argparse.ArgumentParser(description="Deep DJ set analysis for live visuals")
    parser.add_argument("input", help="Path to audio file")
    parser.add_argument("--tracklist", "-t", help="Path to tracklist timestamps file")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--anchor-phrases", action="store_true",
                        help="Anchor the phrase grid at the estimated downbeat "
                             "(heuristic — validate by ear before trusting)")

    args = parser.parse_args()
    result = analyze_deep(args.input, tracklist_path=args.tracklist,
                          anchor_phrases=args.anchor_phrases)

    output_path = args.output or str(Path(args.input).with_suffix('.deep-analysis.json'))
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.1f} MB\n")


if __name__ == "__main__":
    main()
