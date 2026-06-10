#!/usr/bin/env python3
"""
Bake-off runner. Builds the sampling plans once, then runs each engine over the
IDENTICAL representative frames, scores them with bench.metrics (CLIP-anchored),
and optionally runs the bench.judge audit. Resumable + pausable like the existing
drivers (atomic writes; per-frame skip on resume).

Engines:
  real (claude-cli / ollama / mlx) — one vision call per representative
  baseline                         — the CURRENT system's effective per-scene
                                     semantics (scene.semantic or frame_analyses[si],
                                     else empty) + EAST/OCR text flags. Honest floor.
  clip-ceiling                     — image->image retrieval (discriminability ceiling)
"""

import json
import time
from pathlib import Path

from bench import config, manifest as manifest_mod, metrics as M
from bench.util import atomic_write_json, fmt_time, load_json
from vision_schema import SCENE_PROMPT, normalize_analysis
from enrich_analyses import try_parse_json
from vision_backends import get_backend


# ─── State ───────────────────────────────────────────────────────────────────

def _state_path():
    return config.RESULTS_DIR / "state.json"


def load_state():
    p = _state_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"status": "not_started", "paused": False, "errors": []}


def save_state(state):
    atomic_write_json(_state_path(), state)


def _raw_path(engine):
    return config.RESULTS_DIR / "raw" / engine


def _plan_path(stem):
    return config.ENRICHED_DIR / f"{stem}.sampling_plan.json"


# ─── Field extraction from a model response ──────────────────────────────────

def _fields(parsed):
    """(description, tags, style, has_text, noncompliant, n_snaps) from a parsed analysis.

    n_snaps = number of fields normalize_analysis had to correct (enum snaps / list
    coercions / bool fixes) — a per-frame schema-instability signal.
    """
    if not isinstance(parsed, dict):
        return "", [], None, False, False, 0
    clean, notes = normalize_analysis(parsed)
    nc = bool(notes.get("has_english_text", {}).get("noncompliant"))
    return (clean.get("visual_description", "") or "",
            clean.get("content_tags", []) or [],
            clean.get("visual_style"),
            bool(clean.get("has_english_text", False)),
            nc, len(notes))


# ─── Real VLM engine ─────────────────────────────────────────────────────────

def run_vlm_engine(engine_name, registry, save_every=5):
    cfg = config.ENGINES[engine_name]
    backend = get_backend(cfg["backend"], cfg["model"])
    ok, msg = backend.health_check()
    print(f"\n  ── engine {engine_name} ({backend.label}) — {msg}")
    if not ok:
        print(f"     ✗ unavailable; skipping {engine_name}")
        return False

    out_dir = _raw_path(engine_name)
    for stem, rec in sorted(registry.items()):
        plan_p = _plan_path(stem)
        if not plan_p.exists():
            continue
        plan = load_json(plan_p)
        raw_p = out_dir / f"{rec['key']}.json"
        existing = load_json(raw_p) if raw_p.exists() else {"frames": []}
        done = {(f["scene_index"], f["cluster_id"]) for f in existing["frames"]}
        frames = existing["frames"]

        reps = [(s["scene_index"], r) for s in plan["scenes"] for r in s["representatives"]]
        todo = [(si, r) for si, r in reps if (si, r["cluster_id"]) not in done]
        if not todo:
            continue
        print(f"     {stem[:48]:<48s} {len(todo)} frames", flush=True)
        t0 = time.time()
        for n, (si, rep) in enumerate(todo, 1):
            st = load_state()
            if st.get("paused"):
                atomic_write_json(raw_p, {"engine": engine_name, "key": rec["key"],
                                          "display_stem": stem, "frames": frames})
                print("     ⏸  paused"); return False
            fpath = config.BASE_DIR / rep["frame_path"]
            text, elapsed, error = backend.analyze_frame(str(fpath), SCENE_PROMPT)
            desc, tags, style = "", [], None
            has_text = noncompliant = False
            n_snaps, parse_ok = 0, False
            if not error:
                parsed = try_parse_json(text)
                parse_ok = parsed is not None
                desc, tags, style, has_text, noncompliant, n_snaps = _fields(parsed)
                if not desc:
                    error = "unparsed/empty"
            frames.append({
                "scene_index": si, "cluster_id": rep["cluster_id"],
                "time_sec": rep["time_sec"], "frame_path": rep["frame_path"],
                "description": desc, "content_tags": tags, "visual_style": style,
                "has_english_text": has_text, "noncompliant": noncompliant,
                "parse_ok": parse_ok, "n_snaps": n_snaps,
                "elapsed": round(elapsed, 2), "error": error,
            })
            if n % save_every == 0:
                atomic_write_json(raw_p, {"engine": engine_name, "key": rec["key"],
                                          "display_stem": stem, "frames": frames})
        atomic_write_json(raw_p, {"engine": engine_name, "key": rec["key"],
                                  "display_stem": stem, "frames": frames})
        print(f"        done {len(todo)} in {fmt_time(time.time()-t0)}")
    return True


# ─── Baseline (current system) ───────────────────────────────────────────────

def _scene_bounds(analysis):
    sc = analysis.get("scenes", {})
    lst = sc.get("scenes", []) if isinstance(sc, dict) else sc
    return {s.get("scene_index"): (s.get("start_sec", 0), s.get("end_sec", 0)) for s in lst}


def run_baseline(registry):
    """Mirror the assembler's effective per-scene semantics + EAST/OCR text flags."""
    import assemble_v2 as A
    text_seconds = A.load_text_flags()
    out_dir = _raw_path("baseline")
    print("\n  ── engine baseline (current system: enriched semantics + EAST/OCR text)")
    for stem, rec in sorted(registry.items()):
        plan_p = _plan_path(stem)
        ef = config.ENRICHED_DIR / f"{stem}.enriched.json"
        if not plan_p.exists() or not ef.exists():
            continue
        plan = load_json(plan_p)
        data = load_json(ef)
        source_name = data.get("file", {}).get("name", stem)
        bounds = _scene_bounds(data)
        scene_sem = {}
        sc = data.get("scenes", {})
        for s in (sc.get("scenes", []) if isinstance(sc, dict) else sc):
            if s.get("semantic"):
                scene_sem[s.get("scene_index")] = s["semantic"]
        fa_by_scene = {fa.get("scene_index"): fa["analysis"]
                       for fa in data.get("frame_analyses", []) if "analysis" in fa}

        frames = []
        for s in plan["scenes"]:
            si = s["scene_index"]
            sem = scene_sem.get(si) or fa_by_scene.get(si) or {}
            start, end = bounds.get(si, (0, 0))
            has_text = bool(text_seconds) and A.scene_has_text(source_name, start, end, text_seconds)
            for rep in s["representatives"]:
                desc = sem.get("visual_description", "") if isinstance(sem, dict) else ""
                frames.append({
                    "scene_index": si, "cluster_id": rep["cluster_id"],
                    "time_sec": rep["time_sec"], "frame_path": rep["frame_path"],
                    "description": desc,
                    "content_tags": sem.get("content_tags", []) if isinstance(sem, dict) else [],
                    "visual_style": sem.get("visual_style") if isinstance(sem, dict) else None,
                    "has_english_text": has_text, "noncompliant": False,
                    "parse_ok": bool(desc), "n_snaps": 0,
                    "elapsed": 0.0, "error": None if desc else "no baseline semantic",
                })
        atomic_write_json(out_dir / f"{rec['key']}.json",
                          {"engine": "baseline", "key": rec["key"],
                           "display_stem": stem, "frames": frames})
    return True


def run_ceiling(registry):
    """CLIP image->image: each rep's own image is the query (discriminability ceiling)."""
    out_dir = _raw_path("clip-ceiling")
    print("\n  ── engine clip-ceiling (image→image retrieval upper bound)")
    for stem, rec in sorted(registry.items()):
        plan_p = _plan_path(stem)
        if not plan_p.exists():
            continue
        plan = load_json(plan_p)
        frames = []
        for s in plan["scenes"]:
            for rep in s["representatives"]:
                frames.append({
                    "scene_index": s["scene_index"], "cluster_id": rep["cluster_id"],
                    "time_sec": rep["time_sec"], "frame_path": rep["frame_path"],
                    "description": "<image-query>", "content_tags": [],
                    "visual_style": None, "has_english_text": False,
                    "noncompliant": False, "parse_ok": True, "n_snaps": 0,
                    "elapsed": 0.0, "error": None,
                })
        atomic_write_json(out_dir / f"{rec['key']}.json",
                          {"engine": "clip-ceiling", "key": rec["key"],
                           "display_stem": stem, "frames": frames})
    return True


# ─── Scoring ─────────────────────────────────────────────────────────────────

def _load_text_labels():
    p = config.GROUNDTRUTH_DIR / "text_labels.json"
    if not p.exists():
        return None
    labels = {}
    for r in json.loads(p.read_text()).get("labels", []):
        if "key" in r and "scene_index" in r and "cluster_id" in r:
            labels[f"{r['key']}:{r['scene_index']}:{r['cluster_id']}"] = bool(r["has_english_text"])
    return labels or None


def score_all(registry, engines):
    from clip_utils import encode_text, load_clip
    load_clip()
    text_labels = _load_text_labels()

    # plan embeddings, indexed by (key, scene_index, cluster_id)
    emb_index, present_by_key = {}, {}
    for stem, rec in registry.items():
        p = _plan_path(stem)
        if not p.exists():
            continue
        plan = load_json(p)
        n = 0
        for s in plan["scenes"]:
            for rep in s["representatives"]:
                emb_index[(rec["key"], s["scene_index"], rep["cluster_id"])] = rep["embedding"]
                n += 1
        present_by_key[rec["key"]] = n

    boards = []
    for eng in engines:
        raw_dir = _raw_path(eng)
        if not raw_dir.exists():
            continue
        frames, present = [], 0
        seen_keys = set()
        for rec in registry.values():
            rp = raw_dir / f"{rec['key']}.json"
            if not rp.exists():
                continue
            seen_keys.add(rec["key"])
            for f in load_json(rp)["frames"]:
                emb = emb_index.get((rec["key"], f["scene_index"], f["cluster_id"]))
                if emb is None:
                    continue
                frames.append({**f, "key": rec["key"], "img_emb": emb})
        present = sum(present_by_key.get(k, 0) for k in seen_keys)
        if not frames:
            continue
        is_ceiling = (eng == "clip-ceiling")
        sb = M.score_engine(eng, frames, present, is_ceiling=is_ceiling,
                            text_labels=text_labels,
                            encode_text_fn=None if is_ceiling else encode_text)
        atomic_write_json(config.RESULTS_DIR / "scores" / f"{eng}.scoreboard.json", sb)
        boards.append(sb)

    boards.sort(key=lambda b: b.get("composite", 0), reverse=True)
    atomic_write_json(config.RESULTS_DIR / "scoreboard.json",
                      {"ranked": boards, "weights": config.COMPOSITE_WEIGHTS})
    return boards


def run_judge(registry, engines, sample_n=40):
    from bench.judge import judge_engine
    backend = get_backend(config.JUDGE_BACKEND, config.JUDGE_MODEL)
    ok, msg = backend.health_check()
    print(f"\n  ── judge ({backend.label}) — {msg}")
    if not ok:
        print("     ✗ judge unavailable; skipping audit")
        return
    for eng in engines:
        if eng in ("clip-ceiling",):
            continue
        raw_dir = _raw_path(eng)
        frames = []
        for rec in registry.values():
            rp = raw_dir / f"{rec['key']}.json"
            if rp.exists():
                for f in load_json(rp)["frames"]:
                    frames.append({**f, "key": rec["key"],
                                   "frame_path": str(config.BASE_DIR / f["frame_path"])})
        if not frames:
            continue
        print(f"     judging {eng} ({min(sample_n, len(frames))} frames)…", flush=True)
        result = judge_engine(backend, frames, sample_n=sample_n)
        atomic_write_json(config.RESULTS_DIR / "judge" / f"{eng}.json", result)
        print(f"        {result['aggregate']}")


# ─── Orchestration ───────────────────────────────────────────────────────────

def run(engines, registry, do_judge=False, allow_metered=False, judge_n=40):
    config.assert_billing_safe(engines, allow_metered=allow_metered, will_judge=do_judge)
    state = load_state()
    state.update({"status": "running", "paused": False})
    save_state(state)

    man = manifest_mod.build(SCENE_PROMPT, engines, registry)
    manifest_mod.write(man)
    print(f"\n  manifest {man['manifest_id']} (commit {man['commit_sha'][:8]}, "
          f"plan-checksum {man['sampling_plan_checksum']})")

    for eng in engines:
        cfg = config.ENGINES.get(eng, {})
        if cfg.get("synthetic") == "baseline":
            run_baseline(registry)
        elif cfg.get("synthetic") == "clip-ceiling":
            run_ceiling(registry)
        else:
            run_vlm_engine(eng, registry)

    print("\n  scoring…")
    boards = score_all(registry, engines)
    if do_judge:
        run_judge(registry, engines, sample_n=judge_n)
    state = load_state(); state["status"] = "complete"; save_state(state)
    return boards


def health(engines):
    print("\n  BACKEND HEALTH")
    print(f"  {'─'*60}")
    for eng in engines:
        cfg = config.ENGINES.get(eng, {})
        if cfg.get("synthetic"):
            print(f"  {eng:<16s} synthetic — always available")
            continue
        b = get_backend(cfg["backend"], cfg["model"])
        ok, msg = b.health_check()
        print(f"  {eng:<16s} {'✓' if ok else '✗'} {b.label} — {msg}")
    import os
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("\n  ⚠ ANTHROPIC_API_KEY is SET — claude-cli/judge would bill the metered API.")
        print("    unset it to use the subscription, or pass --allow-metered.")
