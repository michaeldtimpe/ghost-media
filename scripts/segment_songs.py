"""Auto-detect song boundaries in a DJ-set deep-analysis JSON and write `tracks`.

DJ sets are sequences of songs, but `analyze_dj_set_deep.py` only populates
`tracks` when given an external tracklist. This standalone post-process reads an
existing `*.deep-analysis.json` and infers song boundaries from signals already
in the file — NO audio re-analysis — then writes a `tracks[]` list the assembler
joins to phrases by time-overlap (`assemble_v2.py:724,814-818`).

Boundaries come from three novelty signals fused on a 1 Hz grid:
  - key change   (key_timeline, confidence-weighted, stabilized)
  - BPM step     (bpm_timeline, rolling-median fractional jump) — supporting
  - energy dip→rebuild + spectral-centroid shift (the classic DJ transition)

The planner only needs segmentation good enough for STABLE SONG IDENTITIES, not
broadcast-quality track detection. We deliberately bias toward FEWER/LONGER
songs: over-segmentation chops a song's identity (worse) while under-segmentation
degrades gracefully to fewer/longer songs. Boundaries are snapped to 16-bar
phrase starts so they coincide with real cut points (keeps the assembler's
frame-exact / on-beat contract intact — this script never touches `phrases`).

Usage:
    python scripts/segment_songs.py <analysis.json> [--dry-run]
    python scripts/segment_songs.py <analysis.json> --min-song-sec 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from scipy.signal import find_peaks

# ─── Tuning (biased toward fewer/longer songs) ──────────────────────────────
MIN_SONG_SEC = 120.0       # no song shorter than this (merge through; biased long)
MAX_SONGS = 30             # hard ceiling; keep only the most prominent boundaries
KEY_CONF_MIN = 0.55        # ignore key labels below this confidence
BPM_STEP_FRAC = 0.06       # rolling-median fractional jump that counts as a step
ENERGY_DIP_FRAC = 0.25     # trough must drop this fraction below local pre-level
REBUILD_WINDOW_SEC = 30.0  # energy must rebuild within this window after a trough
W_KEY, W_BPM, W_ENERGY = 1.0, 0.8, 1.2   # fused novelty weights (energy highest)
PROM_K = 0.75              # peak prominence = mean + PROM_K * std of the curve
EXPECTED_SONG_SEC = 240.0  # nominal song length for the sanity warning only


def _grid(duration: float) -> np.ndarray:
    return np.arange(0.0, float(duration) + 1.0, 1.0)


def _smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average; win in samples (odd)."""
    if win <= 1:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def _norm(x: np.ndarray) -> np.ndarray:
    """Clip outliers at p98 then scale to [0,1]; robust to spikes."""
    if not np.any(x):
        return x
    hi = np.percentile(x, 98)
    if hi <= 0:
        return np.zeros_like(x)
    return np.clip(x / hi, 0.0, 1.0)


def _key_novelty(data, grid):
    """Confidence-weighted impulses where the stabilized dominant key flips."""
    kt = data.get("key_timeline", [])
    nov = np.zeros_like(grid)
    if len(kt) < 3:
        return nov
    times = np.array([k["time_sec"] for k in kt])
    labels = [k.get("label", "") for k in kt]
    confs = np.array([k.get("confidence", 0.0) for k in kt])
    # Stabilize: running mode over a ±3-window span (key estimate is jittery).
    stab = []
    for i in range(len(labels)):
        lo, hi = max(0, i - 3), min(len(labels), i + 4)
        window = [labels[j] for j in range(lo, hi) if confs[j] >= KEY_CONF_MIN]
        stab.append(Counter(window).most_common(1)[0][0] if window else labels[i])
    for i in range(1, len(stab)):
        if stab[i] and stab[i] != stab[i - 1] and confs[i] >= KEY_CONF_MIN:
            sec = int(round(times[i]))
            if 0 <= sec < len(nov):
                nov[sec] = max(nov[sec], float(min(confs[i], 1.0)))
    return _smooth(nov, 7)  # widen impulses into bumps for peak-picking


def _bpm_novelty(data, grid):
    """Relative jump in rolling-median BPM (supporting signal)."""
    bt = data.get("bpm_timeline", [])
    nov = np.zeros_like(grid)
    if len(bt) < 5:
        return nov
    t = np.array([b["time_sec"] for b in bt])
    bpm = np.array([b["bpm"] for b in bt])
    bpm_1hz = np.interp(grid, t, bpm)
    med = _smooth(bpm_1hz, 31)  # ~±15s rolling median proxy
    W = 15
    for i in range(W, len(med) - W):
        before, after = med[i - W], med[i + W]
        if before > 0 and abs(after - before) / before >= BPM_STEP_FRAC:
            nov[i] = abs(after - before) / before
    return _smooth(nov, 7)


def _energy_novelty(data, grid):
    """Dip→rebuild in total_rms + spectral-centroid shift (the DJ transition)."""
    me = data.get("multiband_energy", [])
    sp = data.get("spectral_timeline", [])
    nov = np.zeros_like(grid)
    if len(me) < 10:
        return nov
    t = np.array([m["time_sec"] for m in me])
    rms = np.array([m.get("total_rms", 0.0) for m in me])
    rms_1hz = _smooth(np.interp(grid, t, rms), 3)
    W = int(REBUILD_WINDOW_SEC)
    for i in range(W, len(rms_1hz) - W):
        pre = np.max(rms_1hz[i - W:i])
        post = np.max(rms_1hz[i:i + W])
        trough = rms_1hz[i]
        if pre > 0 and (pre - trough) / pre >= ENERGY_DIP_FRAC \
                and (post - trough) / max(pre, 1e-9) >= ENERGY_DIP_FRAC \
                and trough <= rms_1hz[i - 1] and trough <= rms_1hz[i + 1]:
            nov[i] = (pre - trough) / pre
    # Spectral-centroid shift (filter sweeps move it hard at transitions).
    if sp:
        ts = np.array([s["time_sec"] for s in sp])
        cen = np.array([s.get("centroid_hz", 0.0) for s in sp])
        cen_1hz = _smooth(np.interp(grid, ts, cen), 15)
        d = np.abs(np.gradient(cen_1hz))
        nov = nov + 0.5 * _norm(d)
    return _smooth(nov, 5)


def detect_song_boundaries(data, *, min_song_sec=MIN_SONG_SEC):
    """Return interior boundary times (sec), pre-snap, biased to fewer/longer."""
    duration = float(data["multiband_energy"][-1]["time_sec"])
    grid = _grid(duration)
    fused = (W_KEY * _norm(_key_novelty(data, grid))
             + W_BPM * _norm(_bpm_novelty(data, grid))
             + W_ENERGY * _norm(_energy_novelty(data, grid)))
    prom = float(np.mean(fused) + PROM_K * np.std(fused))
    peaks, props = find_peaks(fused, distance=min_song_sec, prominence=max(prom, 1e-6))
    return [float(grid[p]) for p in peaks], fused, props


def _phrase_starts(data, grid_bars=16):
    key = {16: "sixteen_bar", 8: "eight_bar", 4: "four_bar"}[grid_bars]
    return np.array([p["start_sec"] for p in data["phrases"][key]])


def _snap_to_phrase(boundaries, phrase_starts):
    snapped = []
    for b in boundaries:
        snapped.append(float(phrase_starts[int(np.argmin(np.abs(phrase_starts - b)))]))
    # de-dup boundaries that snapped to the same phrase
    return sorted(set(snapped))


def _enforce_min_length(boundaries, novelty_at, duration, min_song_sec):
    """Greedily drop the weakest boundary until all segments >= min_song_sec."""
    bounds = [0.0] + sorted(boundaries) + [duration]
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        for i in range(1, len(bounds) - 1):
            if bounds[i] - bounds[i - 1] < min_song_sec or bounds[i + 1] - bounds[i] < min_song_sec:
                bounds.pop(i)  # drop this interior boundary (weakest-first ordering below)
                changed = True
                break
    return bounds[1:-1]


def _build_tracks(boundaries, data, prefix="track"):
    duration = float(data["multiband_energy"][-1]["time_sec"])
    edges = [0.0] + sorted(boundaries) + [duration]
    me = data.get("multiband_energy", [])
    kt = data.get("key_timeline", [])
    bt = data.get("bpm_timeline", [])
    me_t = np.array([m["time_sec"] for m in me]) if me else np.array([0.0])
    me_rms = np.array([m.get("total_rms", 0.0) for m in me]) if me else np.array([0.0])
    tracks = []
    for i in range(len(edges) - 1):
        s, e = edges[i], edges[i + 1]
        mask = (me_t >= s) & (me_t < e)
        seg_rms = me_rms[mask] if mask.any() else np.array([0.0])
        bpm_seg = [b["bpm"] for b in bt if s <= b["time_sec"] < e]
        key_seg = [k.get("label", "") for k in kt
                   if s <= k["time_sec"] < e and k.get("confidence", 0) >= KEY_CONF_MIN]
        tracks.append({
            "track_index": i,
            "title": f"{prefix}_{i + 1:02d}",
            "start_sec": round(s, 3),
            "end_sec": round(e, 3),
            "duration_sec": round(e - s, 3),
            "bpm": {"mean": round(float(np.mean(bpm_seg)), 2)} if bpm_seg else {},
            "energy": {"mean": round(float(np.mean(seg_rms)), 4),
                       "peak": round(float(np.max(seg_rms)), 4)},
            "key": Counter(key_seg).most_common(1)[0][0] if key_seg else "",
            "auto_segmented": True,
        })
    return tracks


def segment_songs(data, *, phrase_grid=16, min_song_sec=MIN_SONG_SEC, max_songs=MAX_SONGS):
    duration = float(data["multiband_energy"][-1]["time_sec"])
    raw, fused, props = detect_song_boundaries(data, min_song_sec=min_song_sec)
    # Order boundaries by prominence so min-length merging drops weakest first.
    proms = props.get("prominences", np.ones(len(raw)))
    ordered = [b for _, b in sorted(zip(proms, raw), reverse=True)]
    # Cap to max_songs-1 interior boundaries (keep most prominent).
    if len(ordered) > max_songs - 1:
        ordered = ordered[:max_songs - 1]
    phrase_starts = _phrase_starts(data, phrase_grid)
    snapped = _snap_to_phrase(ordered, phrase_starts)
    kept = _enforce_min_length(snapped, None, duration, min_song_sec)
    tracks = _build_tracks(kept, data)
    return tracks, fused


def _validate(n_songs, duration):
    expected = duration / EXPECTED_SONG_SEC
    if n_songs < max(2, expected * 0.4):
        print(f"  ⚠ under-segmented? {n_songs} songs for {duration/60:.0f} min "
              f"(expected ~{expected:.0f}) — likely beatmatched transitions.")
    elif n_songs > expected * 2.5:
        print(f"  ⚠ over-segmented? {n_songs} songs for {duration/60:.0f} min "
              f"(expected ~{expected:.0f}).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_json")
    ap.add_argument("--phrase-grid", type=int, default=16, choices=[4, 8, 16])
    ap.add_argument("--min-song-sec", type=float, default=MIN_SONG_SEC)
    ap.add_argument("--max-songs", type=int, default=MAX_SONGS)
    ap.add_argument("--dry-run", action="store_true", help="print boundaries, don't write")
    ap.add_argument("--title-prefix", default="track")
    args = ap.parse_args()

    path = args.analysis_json
    data = json.loads(open(path).read())
    if "multiband_energy" not in data or "phrases" not in data:
        print(f"  ERROR: {path} missing multiband_energy/phrases (not a deep-analysis?)")
        return 1
    duration = float(data["multiband_energy"][-1]["time_sec"])

    tracks, _ = segment_songs(data, phrase_grid=args.phrase_grid,
                              min_song_sec=args.min_song_sec, max_songs=args.max_songs)
    print(f"  {os.path.basename(path)}: {len(tracks)} songs over {duration/60:.1f} min")
    for t in tracks:
        print(f"    {t['title']}  {t['start_sec']:>8.1f}–{t['end_sec']:<8.1f} "
              f"({t['duration_sec']/60:.1f} min)  key={t['key']}  "
              f"bpm={t['bpm'].get('mean','?')}")
    _validate(len(tracks), duration)

    if args.dry_run:
        print("  (dry-run: not written)")
        return 0

    data["tracks"] = tracks
    data["song_segmentation"] = {
        "method": "auto-novelty",
        "params": {"phrase_grid": args.phrase_grid, "min_song_sec": args.min_song_sec,
                   "weights": {"key": W_KEY, "bpm": W_BPM, "energy": W_ENERGY}},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)
    print(f"  ✓ wrote {len(tracks)} tracks → {os.path.basename(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
