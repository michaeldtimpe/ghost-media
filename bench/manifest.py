#!/usr/bin/env python3
"""
Frozen run manifest. Every bake-off run records exactly what produced its numbers
— prompt text, per-engine model + temperature, CLIP model identity, sampler
thresholds, the sampling-plan checksum, and the code commit SHA — so that a
re-run weeks later can attribute any delta to the model, prompt, embeddings,
clustering, or sampling rather than guessing. Results reference a manifest by id.
"""

import hashlib
import json
from datetime import datetime

from bench import config, util
from bench.util import atomic_write_json


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sampling_plan_checksum(registry, plans_dir=None):
    """Stable checksum over the sampling plans in play (which frames get described)."""
    plans_dir = plans_dir or config.ENRICHED_DIR
    h = hashlib.sha256()
    for stem in sorted(registry):
        p = plans_dir / f"{stem}.sampling_plan.json"
        if p.exists():
            data = json.loads(p.read_text())
            # checksum the (scene_index, time_sec, cluster_id) triples — the
            # identity of "what we describe" — not the embeddings (float noise).
            reps = [(s["scene_index"], r["time_sec"], r["cluster_id"])
                    for s in data.get("scenes", []) for r in s.get("representatives", [])]
            h.update(json.dumps([stem, reps], sort_keys=True).encode())
    return h.hexdigest()[:16]


def build(prompt_text, engine_names, registry, clip_model="ViT-B-32/openai",
          extra=None):
    """Assemble (and id) a manifest dict for this run."""
    engines = []
    for n in engine_names:
        e = config.ENGINES.get(n, {})
        engines.append({
            "name": n,
            "backend": e.get("backend"),
            "model": e.get("model"),
            "synthetic": e.get("synthetic"),
            "temp": 0.0,
        })
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "commit_sha": util.git_commit_sha(),
        "prompt_sha": _sha(prompt_text),
        "prompt_text": prompt_text,
        "clip_model": clip_model,
        "engines": engines,
        "videos": [{"display_stem": s, "key": r["key"]} for s, r in sorted(registry.items())],
        "thresholds": {
            "base_cadence_sec": config.BASE_CADENCE_SEC,
            "min_candidates": config.MIN_CANDIDATES,
            "max_candidates": config.MAX_CANDIDATES,
            "distinct_threshold": config.DISTINCT_THRESHOLD,
            "threshold_fallback": config.THRESHOLD_FALLBACK,
            "min_states_per_min": config.MIN_STATES_PER_MIN,
            "adjacent_collapse_sim": config.ADJACENT_COLLAPSE_SIM,
        },
        "composite_weights": config.COMPOSITE_WEIGHTS,
        "sampling_plan_checksum": sampling_plan_checksum(registry),
    }
    if extra:
        manifest.update(extra)
    manifest["manifest_id"] = _sha(json.dumps(manifest, sort_keys=True))[:16]
    return manifest


def write(manifest, results_dir=None):
    results_dir = results_dir or config.RESULTS_DIR
    path = results_dir / "manifests" / f"{manifest['manifest_id']}.json"
    atomic_write_json(path, manifest)
    return path
