#!/usr/bin/env python3
"""
Objective, engine-agnostic metrics for the bake-off. All run on the SAME
representative frames (from the sampling plans), so engines are compared
apples-to-apples. These functions are pure (numpy in, dict out) and unit-tested in
smoke_test.py; the runner/report wire them to real engine outputs.

Metric families (the composite rank uses a, a-bis, c, d — never the LLM-judge):
  (a)     reconstruction round-trip: does a description retrieve its OWN frame?
  (a-bis) adjacent-state discriminability: are neighbouring states described
          differently (anti generic-but-CLIP-friendly captions)?
  (c)     coverage + tag entropy/dup-rate vs the baseline
  (d)     text presence precision/recall/F1 (+ structured-output non-compliance)
  (e)     cost / latency, with an M5-projected column
"""

import math
from collections import Counter

import numpy as np

from bench import config


def _norm(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n else v


# ─── (a) Reconstruction round-trip / discriminability ────────────────────────

def retrieval_metrics(query_embs, query_keys, pool_embs, pool_keys, target_idx,
                      within=False):
    """Rank each query's target frame within a pool of candidate image embeddings.

    query_embs : (m, d) text (or, for the ceiling, image) embeddings of described frames
    pool_embs  : (N, d) image embeddings of ALL representative frames
    target_idx : (m,)   index into the pool for each query's own frame
    within     : restrict each query's pool to frames from the same video key

    Returns recall@1, recall@5, MRR, median_rank, n.
    """
    if len(query_embs) == 0:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0,
                "median_rank": None, "n": 0}
    Q = np.stack([_norm(q) for q in query_embs])
    P = np.stack([_norm(p) for p in pool_embs])
    pool_keys = np.asarray(pool_keys)
    ranks = []
    for i in range(len(Q)):
        sims = P @ Q[i]
        if within:
            mask = pool_keys == query_keys[i]
        else:
            mask = np.ones(len(P), dtype=bool)
        tgt = target_idx[i]
        # rank = 1 + (# of masked candidates strictly more similar than the target)
        tgt_sim = sims[tgt]
        more = np.sum((sims > tgt_sim) & mask)
        ranks.append(int(more) + 1)
    ranks = np.array(ranks)
    return {
        "recall_at_1": float(np.mean(ranks == 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "n": int(len(ranks)),
    }


# ─── (a-bis) Adjacent-state discriminability ─────────────────────────────────

def adjacent_discriminability(per_video_text_embs):
    """Distinctness of descriptions of temporally adjacent distinct states.

    per_video_text_embs: dict key -> list of text embeddings in time order.
    Returns mean (1 - cosine) over adjacent pairs and the semantic-collapse rate
    (fraction of adjacent pairs whose descriptions are near-identical).
    """
    dists, collapses, pairs = [], 0, 0
    for embs in per_video_text_embs.values():
        for a, b in zip(embs, embs[1:]):
            cos = float(np.dot(_norm(a), _norm(b)))
            dists.append(1.0 - cos)
            pairs += 1
            if cos >= config.ADJACENT_COLLAPSE_SIM:
                collapses += 1
    return {
        "adjacent_discriminability": float(np.mean(dists)) if dists else 0.0,
        "semantic_collapse_rate": (collapses / pairs) if pairs else 0.0,
        "adjacent_pairs": pairs,
    }


# ─── (c) Coverage + tag stats ────────────────────────────────────────────────

def coverage(states_present, states_described):
    return {
        "states_present": int(states_present),
        "states_described": int(states_described),
        "coverage": (states_described / states_present) if states_present else 0.0,
    }


def tag_stats(tag_lists):
    """Shannon entropy + duplicate rate over all content_tags an engine produced."""
    flat = [t.strip().lower() for tags in tag_lists for t in (tags or []) if str(t).strip()]
    total = len(flat)
    if not total:
        return {"tag_entropy": 0.0, "tag_dup_rate": 0.0, "unique_tags": 0, "total_tags": 0}
    counts = Counter(flat)
    ent = -sum((n / total) * math.log2(n / total) for n in counts.values())
    return {
        "tag_entropy": float(ent),
        "tag_dup_rate": float(1.0 - len(counts) / total),
        "unique_tags": len(counts),
        "total_tags": total,
    }


# ─── (d) Text presence precision/recall/F1 ───────────────────────────────────

def text_prf(preds, labels):
    """preds/labels: dict id -> bool. Scored over the intersection of ids."""
    tp = fp = fn = tn = 0
    for fid, y in labels.items():
        if fid not in preds:
            continue
        p = preds[fid]
        if p and y:
            tp += 1
        elif p and not y:
            fp += 1
        elif (not p) and y:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_scored": tp + fp + fn + tn,
    }


# ─── Schema stability (production reliability, per review) ───────────────────

def schema_stats(frames):
    """JSON adherence + enum-snap rate across an engine's frames.

    Reviewers stressed that for a tagging pipeline, schema stability + low
    hallucination matter MORE than vivid captions. json_adherence = fraction of
    frames that parsed into usable structured output; enum_snap_rate = mean number
    of fields normalize_analysis had to correct per frame (lower = more compliant).
    """
    n = len(frames)
    if not n:
        return {"json_adherence": 0.0, "enum_snap_rate": 0.0, "n": 0}
    ok = sum(1 for f in frames if f.get("parse_ok", not f.get("error")))
    snaps = sum(int(f.get("n_snaps", 0)) for f in frames)
    return {"json_adherence": ok / n, "enum_snap_rate": snaps / n, "n": n}


# ─── (e) Cost / latency ──────────────────────────────────────────────────────

def cost_stats(elapseds, n_engine_videos=None):
    arr = [float(e) for e in elapseds if e]
    if not arr:
        return {"sec_per_frame": 0.0, "frames_per_min": 0.0,
                "est_corpus_hrs_m1": 0.0, "est_corpus_hrs_m5": 0.0, "n": 0}
    spf = sum(arr) / len(arr)
    hrs_m1 = spf * config.FULL_CORPUS_SCENES / 3600.0
    return {
        "sec_per_frame": spf,
        "frames_per_min": 60.0 / spf if spf else 0.0,
        "est_corpus_hrs_m1": hrs_m1,
        "est_corpus_hrs_m5": hrs_m1 / config.M5_SPEEDUP_VS_M1,
        "n": len(arr),
    }


# ─── Composite (CLIP-anchored objective metrics only) ────────────────────────

def composite_score(scoreboard):
    """Weighted blend of objective metrics in [0,1]-ish. Judge is NEVER included.

    Uses within-video Recall@1, coverage, text-F1, and adjacent discriminability,
    per config.COMPOSITE_WEIGHTS. Missing components contribute 0.
    """
    w = config.COMPOSITE_WEIGHTS
    parts = {
        "recall_at_1_within": scoreboard.get("roundtrip_within", {}).get("recall_at_1", 0.0),
        "coverage": scoreboard.get("coverage", {}).get("coverage", 0.0),
        "text_f1": scoreboard.get("text", {}).get("f1", 0.0),
        "adjacent_discriminability": scoreboard.get("adjacent", {}).get(
            "adjacent_discriminability", 0.0),
    }
    return float(sum(w.get(k, 0.0) * v for k, v in parts.items()))


# ─── Orchestrator ────────────────────────────────────────────────────────────

def score_engine(engine_name, frames, plan_states_present, *, is_ceiling=False,
                 text_labels=None, encode_text_fn=None):
    """Compute a full scoreboard for one engine.

    frames: list of dicts, one per representative the engine processed:
      {key, scene_index, cluster_id, time_sec, img_emb (list/np),
       description (str), content_tags (list), has_english_text (bool),
       noncompliant (bool), elapsed (float), error (str|None)}
    plan_states_present: total representatives present in the plans for this engine's
      videos (coverage denominator).
    is_ceiling: use each frame's image embedding as its own query (CLIP image->image).
    text_labels: optional dict frame-id -> bool ground truth (id = f"{key}:{scene}:{cluster}").
    encode_text_fn: callable(str)->vector (clip_utils.encode_text); required unless ceiling.
    """
    described = [f for f in frames if not f.get("error") and (f.get("description") or "").strip()]
    img_embs = [_norm(f["img_emb"]) for f in described]

    # text (query) embeddings
    if is_ceiling:
        query_embs = img_embs
    else:
        if encode_text_fn is None:
            raise ValueError("encode_text_fn required for non-ceiling engines")
        cache = {}
        query_embs = []
        for f in described:
            d = f["description"].strip()
            if d not in cache:
                cache[d] = _norm(encode_text_fn(d))
            query_embs.append(cache[d])

    keys = [f["key"] for f in described]
    target_idx = np.arange(len(described))   # query i's own frame is pool item i

    sb = {"engine": engine_name, "n_described": len(described), "n_frames": len(frames)}
    sb["roundtrip_global"] = retrieval_metrics(query_embs, keys, img_embs, keys,
                                               target_idx, within=False)
    sb["roundtrip_within"] = retrieval_metrics(query_embs, keys, img_embs, keys,
                                               target_idx, within=True)

    # adjacent discriminability (skip for ceiling — image embeddings, not prose)
    if not is_ceiling:
        per_video = {}
        order = sorted(range(len(described)),
                       key=lambda i: (described[i]["key"], described[i]["scene_index"],
                                      described[i]["cluster_id"]))
        for i in order:
            per_video.setdefault(described[i]["key"], []).append(query_embs[i])
        sb["adjacent"] = adjacent_discriminability(per_video)
    else:
        sb["adjacent"] = {"adjacent_discriminability": 0.0, "semantic_collapse_rate": 0.0,
                          "adjacent_pairs": 0}

    sb["coverage"] = coverage(plan_states_present, len(described))
    sb["tags"] = tag_stats([f.get("content_tags", []) for f in frames])

    # text P/R/F1 + non-compliance (skip for ceiling)
    if not is_ceiling and text_labels:
        preds = {f"{f['key']}:{f['scene_index']}:{f['cluster_id']}":
                 bool(f.get("has_english_text")) for f in frames}
        sb["text"] = text_prf(preds, text_labels)
        nc = sum(1 for f in frames if f.get("noncompliant"))
        sb["text"]["noncompliance_rate"] = nc / len(frames) if frames else 0.0
    else:
        sb["text"] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_scored": 0,
                      "noncompliance_rate": 0.0}

    sb["schema"] = schema_stats(frames) if not is_ceiling else {
        "json_adherence": 1.0, "enum_snap_rate": 0.0, "n": len(frames)}
    sb["cost"] = cost_stats([f.get("elapsed", 0.0) for f in frames])
    sb["composite"] = composite_score(sb)
    return sb
