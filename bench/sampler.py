#!/usr/bin/env python3
"""
Adaptive frame sampler — the fix for the prior pipeline's 2.6% first-hand coverage.

Instead of ~1 frame / 30s propagated across a whole video, this samples several
candidate frames per scene (more where the image moves), CLIP-embeds them, clusters
them into DISTINCT VISUAL STATES, and emits one representative (the cluster medoid —
a real frame) per state. Every scene with visual content gets at least one
representative, so nothing inherits a distant neighbour's tags.

Output: one `enriched/<display_stem>.sampling_plan.json` per video — the single
source of truth for "which frames get described", consumed by both the bake-off
runner and the pilot rescan (`enrich_analyses.py --sampling-plan`).

Candidate count per scene scales with duration and PER-VIDEO-NORMALIZED motion
(raw optical-flow magnitude isn't comparable across videos — lessons.md). Clustering
is greedy + time-ordered at DISTINCT_THRESHOLD (looser than near-dup 0.985 — we want
genuinely different content). For low-variance / ambient / loop videos that yield too
few states, the threshold steps down (THRESHOLD_FALLBACK) so internal variation still
splits.
"""

import json
import math
from pathlib import Path

import numpy as np

from bench import config
from bench.util import atomic_write_json, fmt_bar
from clip_utils import CLIP_MODEL, CLIP_PRETRAINED, encode_images
from generate_clip_embeddings import extract_frame


# ─── Scene feature helpers ───────────────────────────────────────────────────

def _scene_list(analysis):
    sc = analysis.get("scenes", {})
    return sc.get("scenes", []) if isinstance(sc, dict) else (sc or [])


def _motion_timeline(analysis):
    mo = analysis.get("motion", {})
    return mo.get("timeline", []) if isinstance(mo, dict) else (mo or [])


def _scene_motion(scene, timeline):
    """Mean optical-flow motion within a scene's window (0.0 if no samples)."""
    s, e = scene.get("start_sec", 0.0), scene.get("end_sec", 0.0)
    vals = [t.get("mean_motion", 0.0) for t in timeline if s <= t.get("time_sec", -1) <= e]
    return float(np.mean(vals)) if vals else 0.0


def _motion_percentiles(scenes, timeline):
    """Per-video percentile rank (0..1) of each scene's mean motion."""
    raw = np.array([_scene_motion(s, timeline) for s in scenes], dtype=float)
    n = len(raw)
    if n <= 1:
        return np.full(n, 0.5)
    order = np.argsort(raw, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n)
    return ranks / (n - 1)


def n_candidates(duration, m_pct):
    """Candidates for a scene: scales with duration, boosted by motion percentile."""
    n = math.ceil(duration / config.BASE_CADENCE_SEC * (0.5 + float(m_pct)))
    return int(max(config.MIN_CANDIDATES, min(config.MAX_CANDIDATES, n)))


def candidate_times(scene, n):
    """`n` evenly spaced sample times inside the scene (with small edge margins)."""
    s, e = scene.get("start_sec", 0.0), scene.get("end_sec", 0.0)
    if e <= s:
        return [s]
    margin = min(0.2, (e - s) * 0.1)
    s, e = s + margin, e - margin
    if n == 1:
        return [round((s + e) / 2, 3)]
    return [round(s + (e - s) * i / (n - 1), 3) for i in range(n)]


# ─── Clustering (greedy, time-ordered, cosine) ───────────────────────────────

def cluster_embeddings(embs, threshold):
    """Greedy time-ordered grouping. `embs`: list of L2-normalized vectors.

    A frame joins the existing cluster whose centroid it's most similar to, if that
    similarity >= threshold; otherwise it starts a new cluster. Returns a list of
    clusters, each a list of member indices (into `embs`).
    """
    clusters = []          # each: {"members": [idx...], "centroid": vec}
    for i, e in enumerate(embs):
        best, best_sim = None, -1.0
        for c in clusters:
            sim = float(np.dot(e, c["centroid"]))
            if sim > best_sim:
                best, best_sim = c, sim
        if best is not None and best_sim >= threshold:
            best["members"].append(i)
            m = np.mean([embs[j] for j in best["members"]], axis=0)
            best["centroid"] = m / (np.linalg.norm(m) + 1e-9)
        else:
            clusters.append({"members": [i], "centroid": np.asarray(e, dtype=float)})
    return [c["members"] for c in clusters]


def _medoid(members, embs):
    """Index (within `members`) of the frame most central to the cluster."""
    if len(members) == 1:
        return members[0]
    sub = np.stack([embs[j] for j in members])
    sims = sub @ sub.T
    return members[int(np.argmax(sims.sum(axis=1)))]


# ─── Plan building ───────────────────────────────────────────────────────────

def _subsample_scenes(scenes, max_states):
    """Cap scenes processed for the bench (most scenes → 1 state). Even spacing."""
    if max_states is None or len(scenes) <= max_states:
        return scenes
    step = len(scenes) / max_states
    return [scenes[int(i * step)] for i in range(max_states)]


def build_plan_for_video(rec, work_dir, max_states=None, force=False):
    """Build (and persist) the sampling plan for one registry record.

    Returns a stats dict for the histogram, or None if skipped (no source / cached).
    """
    stem = rec["display_stem"]
    out_path = config.ENRICHED_DIR / f"{stem}.sampling_plan.json"
    if out_path.exists() and not force:
        data = json.loads(out_path.read_text())
        return {**data.get("totals", {}), "display_stem": stem, "cached": True}
    if not rec.get("source_resolved"):
        return None

    analysis = json.loads(Path(rec["analysis_path"]).read_text())
    duration = analysis.get("metadata", {}).get("duration_sec", 0.0) or 0.0
    scenes = [s for s in _scene_list(analysis) if s.get("duration_sec", 0) >= 0.5]
    scenes = _subsample_scenes(scenes, max_states)
    if not scenes:
        return None
    timeline = _motion_timeline(analysis)
    m_pcts = _motion_percentiles(scenes, timeline)

    vid_dir = Path(work_dir) / rec["key"]
    vid_dir.mkdir(parents=True, exist_ok=True)
    src = rec["source_path"]

    # 1) extract + embed candidates per scene
    per_scene = []   # {scene_index, times, paths, embs}
    for scene, mp in zip(scenes, m_pcts):
        idx = scene.get("scene_index")
        nc = n_candidates(scene.get("duration_sec", 0.0), mp)
        times, paths = [], []
        for t in candidate_times(scene, nc):
            fp = vid_dir / f"s{idx:06d}_t{t:09.3f}.jpg"
            if fp.exists() or extract_frame(src, t, fp):
                times.append(t)
                paths.append(fp)
        if not paths:
            continue
        embs = [np.asarray(v, dtype=float) for v in encode_images([str(p) for p in paths])]
        per_scene.append({"scene_index": idx, "times": times, "paths": paths, "embs": embs})

    # 2) cluster per scene, with a video-level low-variance fallback
    def cluster_all(threshold):
        out, total = [], 0
        for ps in per_scene:
            groups = cluster_embeddings(ps["embs"], threshold)
            out.append(groups)
            total += len(groups)
        return out, total

    threshold = config.DISTINCT_THRESHOLD
    groups_per_scene, total_states = cluster_all(threshold)
    minutes = max(duration / 60.0, 1e-6)
    used_fallback = False
    for thr in config.THRESHOLD_FALLBACK:
        if total_states / minutes >= config.MIN_STATES_PER_MIN:
            break
        if thr >= threshold:
            continue
        threshold = thr
        groups_per_scene, total_states = cluster_all(threshold)
        used_fallback = True

    # 3) emit representatives (medoids); drop non-representative candidate frames
    plan_scenes = []
    keep_paths = set()
    for ps, groups in zip(per_scene, groups_per_scene):
        reps = []
        for cid, members in enumerate(groups):
            mi = _medoid(members, ps["embs"])
            keep_paths.add(ps["paths"][mi])
            reps.append({
                "time_sec": ps["times"][mi],
                "cluster_id": cid,
                "cluster_size": len(members),
                "frame_path": str(ps["paths"][mi].relative_to(config.BASE_DIR)),
                "embedding": [round(float(x), 6) for x in ps["embs"][mi]],
            })
        plan_scenes.append({
            "scene_index": ps["scene_index"],
            "candidates_examined": len(ps["paths"]),
            "representatives": reps,
        })
    for ps in per_scene:
        for p in ps["paths"]:
            if p not in keep_paths:
                try:
                    p.unlink()
                except OSError:
                    pass

    totals = {
        "scenes": len(plan_scenes),
        "candidates": sum(s["candidates_examined"] for s in plan_scenes),
        "distinct_states": total_states,
        "duration_sec": round(duration, 1),
        "states_per_min": round(total_states / minutes, 2),
        "final_threshold": threshold,
        "used_fallback": used_fallback,
    }
    plan = {
        "schema_version": "1.0.0",
        "key": rec["key"],
        "display_stem": stem,
        "source_path": src,
        "clip_model": f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
        "distinct_threshold": threshold,
        "scenes": plan_scenes,
        "totals": totals,
    }
    atomic_write_json(out_path, plan)
    return {**totals, "display_stem": stem, "cached": False}


def build_plans(registry, work_dir=None, max_states=None, force=False, filt=None):
    """Build plans for every video in the registry; return per-video stats list."""
    from clip_utils import load_clip
    load_clip(verbose=True)   # fail loudly here if CLIP/torch is broken (lessons.md)
    work_dir = work_dir or config.FRAMES_DIR
    if max_states is None:
        max_states = config.MAX_STATES_PER_VIDEO_BENCH
    stats = []
    items = sorted(registry.items())
    for i, (stem, rec) in enumerate(items, 1):
        if filt and filt.lower() not in stem.lower():
            continue
        print(f"  [{i}/{len(items)}] planning {stem[:55]} …", flush=True)
        st = build_plan_for_video(rec, work_dir, max_states=max_states, force=force)
        if st is None:
            print(f"      ✗ skipped (no source / no scenes)")
            continue
        tag = "cached" if st.get("cached") else (
            f"thr={st['final_threshold']}" + (" (fallback)" if st.get("used_fallback") else ""))
        print(f"      states={st['distinct_states']:>4}  scenes={st['scenes']:>4}  "
              f"cand={st['candidates']:>4}  {st['states_per_min']}/min  {tag}")
        stats.append(st)
    return stats


def print_histogram(stats):
    """Pre-run histogram so undersampling on long low-motion videos is caught early."""
    if not stats:
        print("\n  (no plans)")
        return
    print(f"\n  {'─'*72}\n  SAMPLING PLAN HISTOGRAM\n  {'─'*72}")
    print(f"  {'video':<40s} {'states':>7s} {'st/min':>7s} {'states/scene':>13s}")
    max_states = max(s["distinct_states"] for s in stats) or 1
    for s in stats:
        ratio = s["distinct_states"] / max(1, s["scenes"])
        bar = fmt_bar(s["distinct_states"] / max_states, 16)
        flag = " ⚠low" if s["states_per_min"] < config.MIN_STATES_PER_MIN else ""
        print(f"  {s['display_stem'][:40]:<40s} {s['distinct_states']:>7d} "
              f"{s['states_per_min']:>7.2f} {ratio:>10.2f}   {bar}{flag}")
    tot = sum(s["distinct_states"] for s in stats)
    print(f"  {'─'*72}\n  TOTAL distinct states to describe: {tot}  "
          f"(prior pipeline saw ~513 across the whole corpus)\n")
