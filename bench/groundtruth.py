#!/usr/bin/env python3
"""
Ground-truth stores for the bake-off, kept as two SEPARATE concerns:

  text_labels.json     — primary: human y/n on whether a representative frame has
                         readable English text. Scores every engine's
                         has_english_text + the EAST/OCR incumbent (baseline).
  desc_spotcheck.json  — small human 1-5 sanity check that the objective CLIP
                         round-trip ranking matches human judgement (not scored).

`propose_text_frames` builds a balanced labeling queue: it uses the existing
text_flags (the incumbent) only to PROPOSE likely-text frames so the human sees a
mix of text/no-text rather than 95% blanks — the human label is the truth, the
incumbent never defines it. Non-Latin script does NOT count as English (matches
the prompt + EAST/OCR), which the labeler instructions restate.
"""

import json

from bench import config
from bench.util import atomic_write_json, load_json


def _text_seconds_for(stem):
    """Per-second English-text flags for a video from text_flags/ (incumbent)."""
    p = config.TEXT_FLAGS_DIR / f"{stem}.text_flags.json"
    if not p.exists():
        return set()
    data = load_json(p)
    if data.get("status") != "complete":
        return set()
    return {int(s) for s, info in data.get("flags", {}).items()
            if info.get("has_english_text")}


def propose_text_frames(registry, target_per_video=None, target_positive_frac=0.35):
    """Build a labeling queue across the pilot, balanced toward ~1/3 likely-positives."""
    queue = []
    for stem, rec in sorted(registry.items()):
        plan_p = config.ENRICHED_DIR / f"{stem}.sampling_plan.json"
        if not plan_p.exists():
            continue
        plan = load_json(plan_p)
        flagged = _text_seconds_for(stem)
        reps = []
        for s in plan["scenes"]:
            for rep in s["representatives"]:
                proposed = int(rep["time_sec"]) in flagged or \
                    (int(rep["time_sec"]) - 1) in flagged or (int(rep["time_sec"]) + 1) in flagged
                reps.append({
                    "key": rec["key"], "display_stem": stem,
                    "scene_index": s["scene_index"], "cluster_id": rep["cluster_id"],
                    "time_sec": rep["time_sec"], "frame_path": rep["frame_path"],
                    "proposed_text": proposed,
                })
        pos = [r for r in reps if r["proposed_text"]]
        neg = [r for r in reps if not r["proposed_text"]]
        if target_per_video:
            n_pos = min(len(pos), int(target_per_video * target_positive_frac))
            n_neg = min(len(neg), target_per_video - n_pos)
            # even spacing so we don't cluster at the start
            def even(lst, k):
                if k <= 0 or not lst:
                    return []
                step = len(lst) / k
                return [lst[int(i * step)] for i in range(k)]
            reps = even(pos, n_pos) + even(neg, n_neg)
        queue.extend(reps)
    return queue


def labels_path():
    return config.GROUNDTRUTH_DIR / "text_labels.json"


def load_labels():
    p = labels_path()
    if not p.exists():
        return {"labels": []}
    return load_json(p)


def labeled_ids(store):
    return {(r["key"], r["scene_index"], r["cluster_id"]) for r in store.get("labels", [])}


def add_label(store, rec, has_text, text=""):
    store["labels"].append({
        "key": rec["key"], "display_stem": rec["display_stem"],
        "scene_index": rec["scene_index"], "cluster_id": rec["cluster_id"],
        "time_sec": rec["time_sec"], "frame_path": rec["frame_path"],
        "has_english_text": bool(has_text), "text": text,
    })


def save_labels(store):
    atomic_write_json(labels_path(), store)


def stats(store):
    labels = store.get("labels", [])
    pos = sum(1 for r in labels if r["has_english_text"])
    return {"total": len(labels), "positives": pos, "negatives": len(labels) - pos}
