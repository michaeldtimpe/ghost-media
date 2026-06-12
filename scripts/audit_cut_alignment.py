"""Cut-alignment audit: measure how far cuts land from beats and lyric boundaries.

Reconstructs the frame-exact render plan's cut times (the same arithmetic as
assemble_v2.select_clips: cumulative n_frames anchored at the first phrase) and
reports, per cut:

  - distance to the nearest beat in beats.times_sec (ms and frames @ FPS)
  - distance to the nearest downbeat in beats.downbeats_sec (n/a until the
    schema carries downbeats)
  - whether the cut falls inside a Whisper lyric segment, and inside a
    word span from segments[].words[] (mid-word = cut audibly bisects a word)
  - the gap between the cut and the next phrase's start_sec (exposes the
    detect_phrases end_sec undershoot, ~1 beat)

Cut times depend only on phrase boundaries, not on which clips are selected,
so no scene database, embeddings, or mounted footage are needed.

Two modes:
  fresh (default)  recompute phrase features from the set's .deep-analysis.json
  --plan PATH      reconstruct cut times from an existing selection_plan_v2.json
                   (audio_start of row 0 + cumulative n_frames)

Output: cut_alignment_audit.txt at repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Import the assembler's helpers so phrase extraction matches production exactly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import assemble_v2 as av  # noqa: E402

# A word is "bisected" only if the cut is strictly inside its span with this
# much clearance from either edge — cuts within the pad of a word edge are
# perceptually on the boundary, not mid-word.
WORD_EDGE_PAD_SEC = 0.05

# Histogram bucket edges for cut-to-beat distance, in ms. 17ms ≈ half a frame
# at 30fps (the best the frame grid can ever do), 33ms = one frame.
BEAT_DIST_BUCKETS_MS = [17, 33, 50, 100, 250]


def phrase_boundaries_fresh(set_name: str, phrase_bars: int) -> tuple[list, dict]:
    """Recompute merged + vocal-adjusted phrase features the same way run_set
    does; return (phrase_features, analysis_data)."""
    set_config = av.SET_CONFIGS[set_name]
    analysis_path = av.BASE_DIR / "sets" / set_config["analysis"]
    if not analysis_path.exists():
        raise SystemExit(f"Analysis file missing: {analysis_path}")
    data = json.loads(analysis_path.read_text())
    phrase_key = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[phrase_bars]
    phrases = data["phrases"][phrase_key]
    phrase_features = av.extract_phrase_features(data, phrases)
    phrase_features = av.merge_phrases_adaptive(phrase_features)
    lyrics_path = av.BASE_DIR / "sets" / f"{set_name}.lyrics.json"
    if lyrics_path.exists():
        lyrics_data = json.loads(lyrics_path.read_text())
        phrase_features, _, _ = av.adjust_cuts_for_vocals(
            phrase_features, lyrics_data.get("segments", []),
            data.get("beats", {}).get("times_sec", []))
    return phrase_features, data


def cut_times_from_phrases(phrase_features, beat_times) -> tuple[list[float], list[float]]:
    """Replicate select_clips' frame-exact plan arithmetic (incl. beat snap).
    Returns (cut_times, next_phrase_starts): the audio-timeline instant of each
    interior cut (clip i → clip i+1) and the start_sec of the phrase that
    begins at that cut."""
    beat_grid = np.asarray(beat_times, dtype=np.float64) if len(beat_times) else None
    timeline_start = phrase_features[0].start_sec
    frames_emitted = 0
    cuts, next_starts = [], []
    for i, phrase in enumerate(phrase_features):
        end_frame = int(round((av._snap_to_beat(phrase.end_sec, beat_grid)
                               - timeline_start) * av.FPS))
        n_frames = max(end_frame - frames_emitted, av.FPS)  # never under 1s
        frames_emitted += n_frames
        if i < len(phrase_features) - 1:  # interior boundary only
            cuts.append(timeline_start + frames_emitted / av.FPS)
            next_starts.append(phrase_features[i + 1].start_sec)
    return cuts, next_starts


def cut_times_from_plan(plan_path: Path) -> tuple[list[float], list[float]]:
    """Reconstruct cut times from a persisted selection_plan_v2.json."""
    rows = json.loads(plan_path.read_text())
    if not rows:
        raise SystemExit(f"Empty plan: {plan_path}")
    timeline_start = rows[0]["audio_start"]
    frames_emitted = 0
    cuts, next_starts = [], []
    for i, row in enumerate(rows):
        frames_emitted += row["n_frames"]
        if i < len(rows) - 1:
            cuts.append(timeline_start + frames_emitted / av.FPS)
            next_starts.append(rows[i + 1]["audio_start"])
    return cuts, next_starts


def load_lyric_spans(set_name: str) -> tuple[list, list]:
    """Return (segment_spans, word_spans) from sets/<set>.lyrics.json, or
    ([], []) if no lyrics file exists."""
    lyrics_path = av.BASE_DIR / "sets" / f"{set_name}.lyrics.json"
    if not lyrics_path.exists():
        return [], []
    data = json.loads(lyrics_path.read_text())
    seg_spans, word_spans = [], []
    for seg in data.get("segments", []):
        seg_spans.append((seg["start_sec"], seg["end_sec"], seg.get("text", "")))
        for w in seg.get("words", []):
            word_spans.append((w["start"], w["end"], w.get("word", "")))
    return seg_spans, word_spans


def nearest_distance(t: float, sorted_times: np.ndarray) -> float:
    idx = np.searchsorted(sorted_times, t)
    best = np.inf
    for j in (idx - 1, idx):
        if 0 <= j < len(sorted_times):
            best = min(best, abs(t - sorted_times[j]))
    return best


def span_hit(t: float, spans: list, pad: float = 0.0) -> tuple[bool, str]:
    """True if t is strictly inside a span, shrunk by pad on each side."""
    for start, end, label in spans:
        if start + pad < t < end - pad:
            return True, label
    return False, ""


def audit_set(set_name: str, phrase_bars: int, plan_path: Path | None) -> dict:
    if plan_path is not None:
        cuts, next_starts = cut_times_from_plan(plan_path)
        # Beats still come from the set's analysis JSON.
        _, data = phrase_boundaries_fresh(set_name, phrase_bars)
    else:
        phrase_features, data = phrase_boundaries_fresh(set_name, phrase_bars)
        cuts, next_starts = cut_times_from_phrases(
            phrase_features, data.get("beats", {}).get("times_sec", []))

    beat_times = np.array(data.get("beats", {}).get("times_sec", []), dtype=np.float64)
    downbeats = np.array(data.get("beats", {}).get("downbeats_sec", []), dtype=np.float64)
    seg_spans, word_spans = load_lyric_spans(set_name)

    beat_dists_ms, downbeat_dists_ms, phrase_gaps_ms = [], [], []
    mid_segment, mid_word, word_examples = 0, 0, []
    for cut, nxt in zip(cuts, next_starts):
        if len(beat_times):
            beat_dists_ms.append(nearest_distance(cut, beat_times) * 1000.0)
        if len(downbeats):
            downbeat_dists_ms.append(nearest_distance(cut, downbeats) * 1000.0)
        phrase_gaps_ms.append((nxt - cut) * 1000.0)
        hit_seg, _ = span_hit(cut, seg_spans)
        mid_segment += hit_seg
        hit_word, word = span_hit(cut, word_spans, pad=WORD_EDGE_PAD_SEC)
        mid_word += hit_word
        if hit_word and len(word_examples) < 8:
            word_examples.append((cut, word))

    return {
        "set": set_name,
        "n_cuts": len(cuts),
        "beat_dists_ms": np.array(beat_dists_ms),
        "downbeat_dists_ms": np.array(downbeat_dists_ms),
        "phrase_gaps_ms": np.array(phrase_gaps_ms),
        "mid_segment": mid_segment,
        "mid_word": mid_word,
        "word_examples": word_examples,
        "has_lyrics": bool(seg_spans),
        "has_downbeats": bool(len(downbeats)),
        "n_words": len(word_spans),
    }


def bucket_counts(dists_ms: np.ndarray) -> list[tuple[str, int]]:
    rows, prev = [], 0.0
    for edge in BEAT_DIST_BUCKETS_MS:
        rows.append((f"≤{edge}ms", int(np.sum((dists_ms > prev) & (dists_ms <= edge)))))
        prev = edge
    rows.append((f">{prev:.0f}ms", int(np.sum(dists_ms > prev))))
    return rows


def emit_report(results: list[dict], out_path: Path) -> None:
    lines = ["# Cut-Alignment Audit", ""]
    for r in results:
        bd = r["beat_dists_ms"]
        lines.append("─" * 72)
        lines.append(f"## {r['set']}  ({r['n_cuts']} cuts)")
        if len(bd):
            frames = bd / (1000.0 / av.FPS)
            lines.append(f"  cut-to-beat:     median {np.median(bd):6.1f}ms ({np.median(frames):.1f} frames)   "
                         f"p90 {np.percentile(bd, 90):6.1f}ms   max {bd.max():6.1f}ms")
            for label, count in bucket_counts(bd):
                bar = "#" * int(round(40 * count / max(r["n_cuts"], 1)))
                lines.append(f"    {label:>7}: {count:>5}  {bar}")
        else:
            lines.append("  cut-to-beat:     n/a (no beats.times_sec in analysis)")
        if r["has_downbeats"]:
            db = r["downbeat_dists_ms"]
            lines.append(f"  cut-to-downbeat: median {np.median(db):6.1f}ms   p90 {np.percentile(db, 90):6.1f}ms")
        else:
            lines.append("  cut-to-downbeat: n/a (no beats.downbeats_sec — pre-2.2.0 schema)")
        pg = r["phrase_gaps_ms"]
        lines.append(f"  cut→next-phrase gap: median {np.median(pg):6.1f}ms   p90 {np.percentile(pg, 90):6.1f}ms   "
                     f"(undershoot: cut lands this far before the next phrase begins)")
        if r["has_lyrics"]:
            lines.append(f"  mid-segment cuts: {r['mid_segment']}/{r['n_cuts']}    "
                         f"mid-word cuts: {r['mid_word']}/{r['n_cuts']}  ({r['n_words']} word spans)")
            for cut, word in r["word_examples"]:
                lines.append(f"    cut @ {cut:9.3f}s bisects word {word!r}")
        else:
            lines.append("  lyric checks: n/a (no .lyrics.json for this set)")
        med_beat = f"{np.median(bd):.0f}ms" if len(bd) else "n/a"
        lines.append(f"  verdict: median cut-to-beat {med_beat}, "
                     f"mid-word cuts {r['mid_word']}/{r['n_cuts']}")
        lines.append("")

    # Aggregate verdict across sets
    all_bd = np.concatenate([r["beat_dists_ms"] for r in results if len(r["beat_dists_ms"])])
    total_cuts = sum(r["n_cuts"] for r in results)
    total_midword = sum(r["mid_word"] for r in results)
    all_pg = np.concatenate([r["phrase_gaps_ms"] for r in results])
    lines.append("─" * 72)
    if len(all_bd):
        lines.append(f"## ALL SETS: median cut-to-beat {np.median(all_bd):.1f}ms "
                     f"({np.median(all_bd) / (1000.0 / av.FPS):.1f} frames), "
                     f"median cut→next-phrase gap {np.median(all_pg):.0f}ms, "
                     f"mid-word cuts {total_midword}/{total_cuts}")
        lines.append("   note: a gap of ~one beat period means cuts land on a beat but one beat")
        lines.append("   EARLY relative to the musical phrase boundary (detect_phrases undershoot).")
        lines.append(f"   acceptance after Phase 1: gap ≤ {1000.0 / av.FPS:.0f}ms (1 frame), mid-word ≈ 0")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nreport written to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", default="all",
                   help="set name from SET_CONFIGS, or 'all' (default)")
    p.add_argument("--phrase-bars", type=int, default=4, choices=[4, 8, 16])
    p.add_argument("--plan", type=Path, default=None,
                   help="reconstruct cuts from an existing selection_plan_v2.json "
                        "(single --set required)")
    p.add_argument("--output", default="cut_alignment_audit.txt")
    args = p.parse_args()

    if args.plan and args.set == "all":
        raise SystemExit("--plan requires a single --set")
    set_names = list(av.SET_CONFIGS) if args.set == "all" else [args.set]

    results = []
    for name in set_names:
        print(f"auditing {name}...", flush=True)
        r = audit_set(name, args.phrase_bars, args.plan)
        results.append(r)
        print(f"  {r['n_cuts']} cuts, median cut-to-beat "
              f"{np.median(r['beat_dists_ms']):.1f}ms, mid-word {r['mid_word']}"
              if len(r["beat_dists_ms"]) else f"  {r['n_cuts']} cuts (no beats)")

    emit_report(results, Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
