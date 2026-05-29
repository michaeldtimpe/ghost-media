#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SCENE QUALITY / CULL PASS
  Flag dead scenes (black, blown-out, frozen, flat, near-duplicate) → sidecar.
═══════════════════════════════════════════════════════════════════════════════

The blog's lesson: cull "non-recordings" before the editor sees them. ghost-media
had no quality filtering at all — black/blown/frozen scenes flowed straight into
the candidate pool. This pass computes a per-scene quality score from data that
ALREADY exists in the enriched timelines (brightness / motion / colors) — no
video re-decoding — and writes a grep-able `<src>.quality.json` sidecar next to
the CLIP-embedding sidecars. The assembler turns the score into a soft penalty
(see assemble_v2.py), so a dead scene sinks but is never hard-excluded.

Design notes:
  - Brightness & contrast are 0–1 normalized and stable across the corpus, so the
    black / blown-out / flat thresholds are absolute.
  - Optical-flow MAGNITUDE varies hugely between videos (mean ~0.006 in one file,
    ~3.8 in another), so "frozen" only fires when the per-scene PEAK motion is
    essentially zero — a true freeze, not merely slow/minimal footage.
  - Near-duplicate uses the existing CLIP embeddings, compared WITHIN a source
    only (lingering static shots, repeated intros) — no destructive cross-corpus
    dedup.

Usage:
    python flag_quality.py                 # all enriched files
    python flag_quality.py --video Tame    # filter by substring
    python flag_quality.py --skip-existing # skip files that already have a sidecar
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent
ENRICHED_DIR = BASE_DIR / "enriched"

MODEL_VERSION = "1.0"

# ─── Thresholds (tunable) ────────────────────────────────────────────────────
BLACK_BRIGHTNESS_P95 = 0.05   # 95th-pct brightness below this → effectively black
BLOWN_BRIGHTNESS_P5 = 0.95    # 5th-pct brightness above this  → clipped white
FROZEN_MAX_MOTION = 0.05      # peak motion below this → a true freeze/still
FLAT_CONTRAST = 0.04          # mean contrast below this → near-uniform, low-info
NEAR_DUP_SIM = 0.97           # cosine ≥ this vs an earlier kept scene → duplicate
                              # (was 0.985 — too permissive, only caught byte-
                              # identical scenes. 0.97 catches perceptually-near-
                              # duplicates while not over-culling middle-rich
                              # sources. 0.95 was probed and produced 9 sources
                              # with >40% flagged; 0.97 gives 8 with a less
                              # severe distribution — see Phase 0 audit notes.)

# ─── Penalties → sub-scores ──────────────────────────────────────────────────
# black/blown/frozen are genuinely unusable → strong. low_info (flat frame) is
# low-value → moderate. near_dup is NOT bad quality (a loop is usable footage);
# the assembler's reuse penalty + variety windows already handle repetition, so
# near_dup is only a gentle nudge here.
STRUCTURAL_SCORE = 0.1        # technical_score when black/blown/frozen
LOW_INFO_PENALTY = 0.5
NEAR_DUP_PENALTY = 0.25


def _in_range(timeline, start, end, key):
    return [e[key] for e in timeline
            if start <= e.get("time_sec", -1) < end and key in e]


def _clamp01(x):
    return max(0.0, min(1.0, x))


def compute_scene_metrics(scene, brightness_tl, motion_tl):
    """Return (metrics dict, flags list) for one scene from its timeline slices."""
    start = scene.get("start_sec", 0)
    end = scene.get("end_sec", start)

    bright = _in_range(brightness_tl, start, end, "brightness")
    contrast = _in_range(brightness_tl, start, end, "contrast")
    motion_max = _in_range(motion_tl, start, end, "max_motion")
    motion_mean = _in_range(motion_tl, start, end, "mean_motion")

    metrics, flags = {}, []

    if bright:
        p5, p95 = float(np.percentile(bright, 5)), float(np.percentile(bright, 95))
        metrics["brightness_p5"] = round(p5, 4)
        metrics["brightness_p95"] = round(p95, 4)
        if p95 < BLACK_BRIGHTNESS_P95:
            flags.append("black")
        if p5 > BLOWN_BRIGHTNESS_P5:
            flags.append("blown_out")

    if contrast:
        cmean = float(np.mean(contrast))
        metrics["contrast_mean"] = round(cmean, 4)
        if cmean < FLAT_CONTRAST:
            flags.append("low_info")

    if motion_max:
        mmax = float(np.max(motion_max))
        metrics["motion_max"] = round(mmax, 4)
        metrics["motion_mean"] = round(float(np.mean(motion_mean)), 4) if motion_mean else 0.0
        metrics["motion_std"] = round(float(np.std(motion_mean)), 4) if motion_mean else 0.0
        if mmax < FROZEN_MAX_MOTION:
            flags.append("frozen")

    return metrics, flags


def detect_near_dups(scene_indices, embeddings):
    """Flag scenes whose CLIP vector matches an earlier KEPT scene (same source).

    Walking in scene order and only comparing against already-kept (non-dup)
    scenes means a long static run flags every scene after the first.
    """
    dup = set()
    kept_vecs = []  # list of 1-D normalized vectors
    for si in scene_indices:
        v = embeddings.get(si)
        if v is None:
            continue
        if kept_vecs:
            sims = np.dot(np.stack(kept_vecs), v)  # vectors are L2-normalized
            if float(np.max(sims)) >= NEAR_DUP_SIM:
                dup.add(si)
                continue
        kept_vecs.append(v)
    return dup


def score_scene(flags):
    """Combine flags into technical / editorial / overall quality sub-scores."""
    structural = any(f in flags for f in ("black", "blown_out", "frozen"))
    technical = STRUCTURAL_SCORE if structural else 1.0

    editorial = 1.0
    if "low_info" in flags:
        editorial -= LOW_INFO_PENALTY
    if "near_dup" in flags:
        editorial -= NEAR_DUP_PENALTY
    editorial = _clamp01(editorial)

    quality = min(technical, editorial)
    return round(quality, 3), round(technical, 3), round(editorial, 3)


def process_file(enriched_path):
    data = json.loads(enriched_path.read_text())
    scenes = data.get("scenes", {})
    scene_list = scenes.get("scenes", []) if isinstance(scenes, dict) else scenes
    brightness_tl = data.get("brightness", {}).get("timeline", [])
    motion_tl = data.get("motion", {}).get("timeline", [])

    # Load CLIP embeddings for near-dup (optional).
    name = enriched_path.name.replace(".enriched.json", "")
    emb_path = enriched_path.with_name(f"{name}.clip_embeddings.json")
    embeddings = {}
    if emb_path.exists():
        for se in json.loads(emb_path.read_text()).get("scenes", []):
            embeddings[se["scene_index"]] = np.asarray(se["embedding"], dtype=np.float32)

    per_scene = []
    for scene in scene_list:
        metrics, flags = compute_scene_metrics(scene, brightness_tl, motion_tl)
        per_scene.append({"scene_index": scene.get("scene_index"),
                          "flags": flags, "metrics": metrics})

    if embeddings:
        dup = detect_near_dups([s["scene_index"] for s in per_scene], embeddings)
        for s in per_scene:
            if s["scene_index"] in dup:
                s["flags"].append("near_dup")

    out_scenes, n_flagged = [], 0
    for s in per_scene:
        q, tech, ed = score_scene(s["flags"])
        if s["flags"]:
            n_flagged += 1
        out_scenes.append({
            "scene_index": s["scene_index"],
            "quality_score": q,
            "technical_score": tech,
            "editorial_score": ed,
            "flags": s["flags"],
            "metrics": s["metrics"],
        })

    output = {
        "model_version": MODEL_VERSION,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": enriched_path.name,
        "thresholds": {
            "black_brightness_p95": BLACK_BRIGHTNESS_P95,
            "blown_brightness_p5": BLOWN_BRIGHTNESS_P5,
            "frozen_max_motion": FROZEN_MAX_MOTION,
            "flat_contrast": FLAT_CONTRAST,
            "near_dup_sim": NEAR_DUP_SIM,
        },
        "scenes": out_scenes,
    }
    out_path = enriched_path.with_name(f"{name}.quality.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False))
    return len(out_scenes), n_flagged


def main():
    parser = argparse.ArgumentParser(description="Flag dead scenes from enriched timelines")
    parser.add_argument("--video", nargs="*", help="Filter by enriched filename substring")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have a .quality.json sidecar")
    args = parser.parse_args()

    files = sorted(ENRICHED_DIR.glob("*.enriched.json"))
    if args.video:
        files = [f for f in files if any(v.lower() in f.name.lower() for v in args.video)]
    if args.skip_existing:
        files = [f for f in files
                 if not f.with_name(f.name.replace(".enriched.json", ".quality.json")).exists()]

    print(f"\n  {'═' * 68}")
    print(f"  SCENE QUALITY PASS  ·  {len(files)} files")
    print(f"  {'═' * 68}")

    grand_scenes = grand_flagged = 0
    for f in files:
        n, flagged = process_file(f)
        grand_scenes += n
        grand_flagged += flagged
        pct = (flagged / n * 100) if n else 0
        print(f"    {f.name.replace('.enriched.json','')[:52]:<54} "
              f"{flagged:>5}/{n:<5} flagged ({pct:4.1f}%)")

    pct = (grand_flagged / grand_scenes * 100) if grand_scenes else 0
    print(f"\n  {'═' * 68}")
    print(f"  DONE — {grand_flagged}/{grand_scenes} scenes flagged ({pct:.1f}%)")
    print(f"  Sidecars: {ENRICHED_DIR}/*.quality.json")
    print(f"  {'═' * 68}\n")


if __name__ == "__main__":
    main()
