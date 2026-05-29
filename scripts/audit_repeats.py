"""Phase 0 forensic audit: classify visible-repeat substrate in assembler output.

Runs the same selection pipeline as `assemble_v2.run_set` up to (but not including)
ffmpeg extraction. Then post-processes the chronological selection sequence to
identify three substrates of repetition:

  0a — literal (source, scene_index) repeats within SCENE_VARIETY_WINDOW=30.
       If this fires, there is a bug in the constraint code at assemble_v2.py:765.
  0b — within-source perceptual: same source_name, within 5 phrases, CLIP cosine >= 0.95.
  0c — cross-source perceptual: different source_name, within 5 phrases, CLIP cosine >= 0.90.

Single pass over the selection list, all three substrates audited simultaneously.

Output: phase0_repeat_audit.txt at repo root.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import the assembler's existing helpers so we can re-run select_clips faithfully.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import assemble_v2 as av  # noqa: E402

WITHIN_PHRASES = 5
WITHIN_SOURCE_CLIP_SIM_THRESHOLD = 0.95
CROSS_SOURCE_CLIP_SIM_THRESHOLD = 0.90

# Wider-net histogram parameters (separate from the substrate audit)
HIST_WINDOWS = [1, 3, 5, 10, 30]            # how far back to look
HIST_THRESHOLDS = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def run_selection_only(set_name: str, args_namespace) -> tuple[list, dict]:
    """Re-run the assembler's selection pipeline through select_clips; return
    (selections, scene_lookup) where scene_lookup maps (source_name, start_sec)
    to the SceneClip object so we can recover embeddings + scene_index later.
    """
    set_config = av.SET_CONFIGS[set_name]
    analysis_path = av.BASE_DIR / "sets" / set_config["analysis"]
    if not analysis_path.exists():
        raise SystemExit(f"Analysis file missing: {analysis_path}")

    print(f"  loading scene database...", flush=True)
    text_seconds = av.load_text_flags()
    scenes = av.build_scene_database(text_seconds)

    scene_lookup = {(s.source_name, round(s.start_sec, 4)): s for s in scenes}

    print(f"  loading audio analysis...", flush=True)
    data = json.loads(analysis_path.read_text())
    phrase_key = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[args_namespace.phrase_bars]
    phrases = data["phrases"][phrase_key]

    phrase_features = av.extract_phrase_features(data, phrases)
    phrase_features = av.merge_phrases_adaptive(phrase_features)

    # Skip the lyrics CLIP encoding — the audit doesn't need lyric-text bonuses;
    # the selection's diversity behavior is unaffected by their absence (clips
    # without clip_text_embedding just get 0 from that scoring term).
    print(f"  running select_clips...", flush=True)
    style_hints = set_config.get("style_hints", {})
    np.random.seed(getattr(args_namespace, "seed", 42))
    selections = av.select_clips(scenes, phrase_features, style_hints)
    return selections, scene_lookup


def audit(selections: list, scene_lookup: dict) -> dict:
    """Single-pass audit of all three substrates."""
    findings = {
        "n_selections": len(selections),
        "n_with_embeddings": 0,
        "n_without_embeddings": 0,
        "0a": {"count": 0, "examples": []},
        "0b": {"count": 0, "examples": []},
        "0c": {"count": 0, "examples": []},
    }

    # Build an enriched chronological list: (phrase_idx, source, start_sec, embedding)
    chrono = []
    for i, sel in enumerate(selections):
        key = (sel["clip_source_name"], round(sel["clip_start"], 4))
        scene = scene_lookup.get(key)
        emb = scene.clip_embedding if scene is not None else None
        scene_index = scene.scene_index if scene is not None else None
        chrono.append({
            "phrase_idx": i,
            "source": sel["clip_source_name"],
            "scene_index": scene_index,
            "clip_start": sel["clip_start"],
            "embedding": emb,
        })
        if emb is not None:
            findings["n_with_embeddings"] += 1
        else:
            findings["n_without_embeddings"] += 1

    # ── 0a: literal (source, scene_index) repeats within SCENE_VARIETY_WINDOW ──
    window_0a = av.SCENE_VARIETY_WINDOW
    for i, cur in enumerate(chrono):
        if cur["scene_index"] is None:
            continue
        cur_key = (cur["source"], cur["scene_index"])
        for j in range(max(0, i - window_0a), i):
            prev = chrono[j]
            if prev["scene_index"] is None:
                continue
            if (prev["source"], prev["scene_index"]) == cur_key:
                findings["0a"]["count"] += 1
                if len(findings["0a"]["examples"]) < 8:
                    findings["0a"]["examples"].append({
                        "prev_phrase": prev["phrase_idx"],
                        "cur_phrase": cur["phrase_idx"],
                        "delta_phrases": i - j,
                        "source": cur["source"],
                        "scene_index": cur["scene_index"],
                    })

    # ── 0b + 0c: perceptual repeats within WITHIN_PHRASES ──
    for i, cur in enumerate(chrono):
        if cur["embedding"] is None:
            continue
        for j in range(max(0, i - WITHIN_PHRASES), i):
            prev = chrono[j]
            if prev["embedding"] is None:
                continue
            sim = cosine(cur["embedding"], prev["embedding"])
            if cur["source"] == prev["source"]:
                # Skip pairs where scene_index is identical (already counted in 0a)
                if cur["scene_index"] == prev["scene_index"]:
                    continue
                if sim >= WITHIN_SOURCE_CLIP_SIM_THRESHOLD:
                    findings["0b"]["count"] += 1
                    if len(findings["0b"]["examples"]) < 10:
                        findings["0b"]["examples"].append({
                            "prev_phrase": prev["phrase_idx"],
                            "cur_phrase": cur["phrase_idx"],
                            "delta_phrases": i - j,
                            "source": cur["source"],
                            "prev_scene_index": prev["scene_index"],
                            "cur_scene_index": cur["scene_index"],
                            "cosine": round(sim, 4),
                        })
            else:
                if sim >= CROSS_SOURCE_CLIP_SIM_THRESHOLD:
                    findings["0c"]["count"] += 1
                    if len(findings["0c"]["examples"]) < 10:
                        findings["0c"]["examples"].append({
                            "prev_phrase": prev["phrase_idx"],
                            "cur_phrase": cur["phrase_idx"],
                            "delta_phrases": i - j,
                            "prev_source": prev["source"],
                            "cur_source": cur["source"],
                            "prev_scene_index": prev["scene_index"],
                            "cur_scene_index": cur["scene_index"],
                            "cosine": round(sim, 4),
                        })

    # ── Distribution histogram across multiple windows and thresholds ──
    # This shows the broader landscape so the substrate counts above can be
    # interpreted in context. Reports n_pairs at each (window, threshold) cell.
    hist = {}
    consec_sims = []  # consecutive-pair cosines for the distribution gauge
    for i, cur in enumerate(chrono):
        if cur["embedding"] is None:
            continue
        if i > 0 and chrono[i-1]["embedding"] is not None:
            consec_sims.append(cosine(cur["embedding"], chrono[i-1]["embedding"]))
        for w in HIST_WINDOWS:
            for j in range(max(0, i - w), i):
                prev = chrono[j]
                if prev["embedding"] is None:
                    continue
                sim = cosine(cur["embedding"], prev["embedding"])
                for t in HIST_THRESHOLDS:
                    if sim >= t:
                        hist[(w, t)] = hist.get((w, t), 0) + 1
    findings["histogram"] = hist
    findings["consec_sims"] = consec_sims

    return findings


def emit_report(findings: dict, out_path: Path, set_name: str) -> None:
    lines = []
    lines.append(f"# Phase 0 — Forensic Repeat Audit")
    lines.append(f"set: {set_name}")
    lines.append(f"selections: {findings['n_selections']}")
    lines.append(f"  with embeddings:    {findings['n_with_embeddings']}")
    lines.append(f"  without embeddings: {findings['n_without_embeddings']}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"## 0a — literal (source, scene_index) repeats within SCENE_VARIETY_WINDOW={av.SCENE_VARIETY_WINDOW}")
    lines.append(f"count: {findings['0a']['count']}")
    if findings['0a']['count'] == 0:
        lines.append("  → no constraint-code bug; literal repeats are blocked by the window")
    else:
        lines.append("  → CONSTRAINT CODE BUG — literal repeats happening within the window")
    for ex in findings['0a']['examples']:
        lines.append(f"  phrase {ex['prev_phrase']:>4} ↔ phrase {ex['cur_phrase']:>4}  Δ={ex['delta_phrases']:>2}  "
                     f"source={ex['source'][:50]}  scene_index={ex['scene_index']}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"## 0b — within-source perceptual (same source, within {WITHIN_PHRASES} phrases, cosine ≥ {WITHIN_SOURCE_CLIP_SIM_THRESHOLD})")
    lines.append(f"count: {findings['0b']['count']}")
    for ex in findings['0b']['examples']:
        lines.append(f"  phrase {ex['prev_phrase']:>4} ↔ phrase {ex['cur_phrase']:>4}  Δ={ex['delta_phrases']:>2}  "
                     f"cosine={ex['cosine']:.3f}  source={ex['source'][:40]}  "
                     f"scenes={ex['prev_scene_index']}→{ex['cur_scene_index']}")
    lines.append("")
    lines.append("─" * 72)
    lines.append("")
    lines.append(f"## 0c — cross-source perceptual (different sources, within {WITHIN_PHRASES} phrases, cosine ≥ {CROSS_SOURCE_CLIP_SIM_THRESHOLD})")
    lines.append(f"count: {findings['0c']['count']}")
    for ex in findings['0c']['examples']:
        lines.append(f"  phrase {ex['prev_phrase']:>4} ↔ phrase {ex['cur_phrase']:>4}  Δ={ex['delta_phrases']:>2}  "
                     f"cosine={ex['cosine']:.3f}")
        lines.append(f"    prev: {ex['prev_source'][:50]}  scene={ex['prev_scene_index']}")
        lines.append(f"    cur:  {ex['cur_source'][:50]}  scene={ex['cur_scene_index']}")

    # ── Distribution histogram ──
    lines.append("")
    lines.append("─" * 72)
    lines.append("")
    lines.append("## Cross-pair similarity distribution (any source) — pair counts")
    lines.append("Reads as: within window W phrases, this many pairs have cosine ≥ T")
    lines.append("")
    hdr = "  window |" + "".join(f" t≥{t:>4}" for t in HIST_THRESHOLDS)
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for w in HIST_WINDOWS:
        row = f"  {w:>5}  |"
        for t in HIST_THRESHOLDS:
            count = findings["histogram"].get((w, t), 0)
            row += f"  {count:>5}"
        lines.append(row)
    lines.append("")
    # Consecutive-pair stats (mean + percentile)
    consec = findings["consec_sims"]
    if consec:
        arr = np.array(consec)
        lines.append(f"## Consecutive-pair cosine distribution (n={len(arr)})")
        lines.append(f"  mean   = {arr.mean():.4f}")
        lines.append(f"  median = {np.median(arr):.4f}")
        lines.append(f"  p10/p25/p50/p75/p90/p95 = "
                     f"{np.percentile(arr, 10):.3f} / "
                     f"{np.percentile(arr, 25):.3f} / "
                     f"{np.percentile(arr, 50):.3f} / "
                     f"{np.percentile(arr, 75):.3f} / "
                     f"{np.percentile(arr, 90):.3f} / "
                     f"{np.percentile(arr, 95):.3f}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nreport written to {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", default="waiting-to-begin-2024")
    p.add_argument("--phrase-bars", type=int, default=4, choices=[4, 8, 16])
    p.add_argument("--segment", nargs=2, type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="phase0_repeat_audit.txt")
    args = p.parse_args()

    selections, scene_lookup = run_selection_only(args.set, args)
    findings = audit(selections, scene_lookup)
    emit_report(findings, Path(args.output), args.set)
    print(f"\nsummary: 0a={findings['0a']['count']}  0b={findings['0b']['count']}  0c={findings['0c']['count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
