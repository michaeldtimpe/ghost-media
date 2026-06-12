"""Semantic-consistency audit: flag scenes whose VLM semantics contradict
measured features. JSON-only — no video decoding, no VLM.

Checks per scene (enriched/*.enriched.json):
  energy-vs-motion   mood.energy "intense"/"chaotic" on a bottom-quintile
                     motion scene, or "calm" on a top-quintile one
  missing-semantic   scene never inherited a semantic from the sampling plan

Output: a console table per check + audit_semantics_worklist.json — a scoped
worklist for re-enrichment (`enrich_analyses.py --reenrich-flagged --video
<name>` per lessons.md: NEVER run it unscoped).

Motion quintiles are corpus-relative (same percentile-rank construction the
assembler uses), so the check is scale-free across heterogeneous sources.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
ENRICHED_DIR = BASE_DIR / "enriched"
WORKLIST_PATH = BASE_DIR / "audit_semantics_worklist.json"

LOW_RANK = 0.2    # bottom quintile of corpus motion
HIGH_RANK = 0.8   # top quintile


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rows", type=int, default=12,
                   help="example rows to print per check")
    args = p.parse_args()

    rows = []  # (video, scene_index, motion_mean, energy, has_semantic)
    for f in sorted(ENRICHED_DIR.glob("*.enriched.json")):
        data = json.loads(f.read_text())
        name = f.name.replace(".enriched.json", "")
        motion_tl = data.get("motion", {}).get("timeline", [])
        sd = data.get("scenes", {})
        scene_list = sd.get("scenes", []) if isinstance(sd, dict) else sd
        fa_by_scene = {fa.get("scene_index"): fa["analysis"]
                       for fa in data.get("frame_analyses", [])
                       if fa.get("scene_index") is not None and "analysis" in fa}
        for s in scene_list:
            start, end = s.get("start_sec", 0), s.get("end_sec", 0)
            sem = s.get("semantic") or fa_by_scene.get(s.get("scene_index"))
            vals = [e["mean_motion"] for e in motion_tl
                    if start <= e.get("time_sec", 0) < end]
            motion = float(np.mean(vals)) if vals else 0.0
            energy = (sem or {}).get("mood", {}).get("energy", "")
            rows.append((name, s.get("scene_index"), motion, energy, bool(sem)))

    motions = np.array([r[2] for r in rows])
    ranks = motions.argsort().argsort().astype(np.float64) / max(len(motions) - 1, 1)

    intense_static, calm_busy, missing = [], [], []
    for (name, si, motion, energy, has_sem), rank in zip(rows, ranks):
        if not has_sem:
            missing.append((name, si))
        elif energy in ("intense", "chaotic") and rank < LOW_RANK:
            intense_static.append((name, si, energy, rank))
        elif energy == "calm" and rank > HIGH_RANK:
            calm_busy.append((name, si, energy, rank))

    def report(label, found, fmt):
        print(f"\n## {label}: {len(found)}")
        for row in found[:args.max_rows]:
            print("  " + fmt(row))
        if len(found) > args.max_rows:
            print(f"  ... +{len(found) - args.max_rows} more")

    print(f"Scenes audited: {len(rows)}")
    report("intense/chaotic mood on near-static scene (motion rank < 0.2)",
           intense_static,
           lambda r: f"{r[0][:48]:48} scene={r[1]:>5} {r[2]:>8} rank={r[3]:.2f}")
    report("calm mood on top-quintile-motion scene (rank > 0.8)",
           calm_busy,
           lambda r: f"{r[0][:48]:48} scene={r[1]:>5} {r[2]:>8} rank={r[3]:.2f}")
    report("scene without any semantic", missing,
           lambda r: f"{r[0][:48]:48} scene={r[1]}")

    by_video = {}
    for name, si, *_ in intense_static + calm_busy:
        by_video.setdefault(name, {"contradictory_scenes": []})[
            "contradictory_scenes"].append(si)
    WORKLIST_PATH.write_text(json.dumps({
        "checks": {"intense_static": len(intense_static),
                   "calm_busy": len(calm_busy),
                   "missing_semantic": len(missing)},
        "reenrich_worklist": by_video,
        "note": "scope re-enrichment: enrich_analyses.py --reenrich-flagged "
                "--video <name> (never unscoped — lessons.md)",
    }, indent=2))
    print(f"\nworklist → {WORKLIST_PATH.name} "
          f"({len(by_video)} videos with contradictions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
