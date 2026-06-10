#!/usr/bin/env python3
"""Hybrid engine synthesizer for the bake-off.

Composes "virtual engines" from existing per-engine raw outputs under a declarative
spec. The synthesized raw files have the same schema as a real engine's, so the
existing scoring pipeline (`bench.runner.score_all` + `bench.metrics.score_engine`)
consumes them unchanged.

Two modes:

  trigger-replace   — full frame swap from `escalation` whenever a per-frame
                      predicate fires (otherwise keep `base`'s frame).
  override          — per-field swap or compound op (union / majority-vote)
                      while keeping the rest of `base`'s frame.

Spec format (`bench/hybrids/<name>.spec.json`):

  trigger-replace:
    {"name": "...", "mode": "trigger-replace", "base": "<eng>", "escalation": "<eng>",
     "trigger": {"any_of": [{"field":"parse_ok","op":"==","value":false}, ...]}}

  override:
    {"name": "...", "mode": "override", "base": "<eng>",
     "overrides": {"visual_description": {"from": "<eng>"},
                   "content_tags":     {"op": "union", "with": "<eng>"},
                   "has_english_text": {"op": "majority-vote",
                                         "from": ["<eng1>","<eng2>",...]}}}

`elapsed` is summed across contributing engines so the scoreboard's `s/frm` column
reflects real cost-if-cascaded (or full cost if both engines always run in override).
"""

import glob
import json
import random
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench import config

SPECS_DIR = Path(__file__).parent / "hybrids"


@dataclass
class HybridSpec:
    name: str
    mode: str
    base: str
    escalation: Optional[str] = None
    trigger: Optional[dict] = None
    overrides: Dict[str, dict] = field(default_factory=dict)
    description: str = ""


def load_specs() -> List[HybridSpec]:
    SPECS_DIR.mkdir(exist_ok=True)
    return [HybridSpec(**json.loads(Path(fp).read_text()))
            for fp in sorted(SPECS_DIR.glob("*.spec.json"))]


def register_hybrids() -> None:
    """Insert hybrid names into config.ENGINES so the bench's `_engines()` validation
    and `score_all` accept them naturally. score_all only cares that a raw dir exists;
    `runner.run()`'s dispatch never sees synthetic='hybrid' since you don't `run` them."""
    for s in load_specs():
        if s.name in config.ENGINES:
            continue
        config.ENGINES[s.name] = {"synthetic": "hybrid", "spec_name": s.name}


# ─── Predicate DSL ───────────────────────────────────────────────────────────

def _eval_predicate(frame: dict, pred: dict) -> bool:
    if "any_of" in pred:
        return any(_eval_predicate(frame, p) for p in pred["any_of"])
    if "all_of" in pred:
        return all(_eval_predicate(frame, p) for p in pred["all_of"])
    f = pred["field"]
    op = pred["op"]
    val = pred.get("value")
    if f == "description_len":
        v = len((frame.get("description") or ""))
    elif f == "error":
        v = bool(frame.get("error"))
    elif f == "parse_ok":
        v = bool(frame.get("parse_ok"))
    else:
        v = frame.get(f)
    if op == "==":     return v == val
    if op == "!=":     return v != val
    if op == ">=":     return v is not None and v >= val
    if op == "<=":     return v is not None and v <= val
    if op == ">":      return v is not None and v > val
    if op == "<":      return v is not None and v < val
    if op == "truthy": return bool(v)
    if op == "falsy":  return not bool(v)
    raise ValueError(f"unknown predicate op {op!r}")


# ─── Raw loaders ─────────────────────────────────────────────────────────────

def _load_raw_indexed(engine: str) -> Dict[Tuple[str, int, int], dict]:
    """Returns {(key, scene_index, cluster_id): frame_dict} for an engine."""
    out: Dict[Tuple[str, int, int], dict] = {}
    for fp in glob.glob(str(config.RESULTS_DIR / "raw" / engine / "*.json")):
        d = json.load(open(fp))
        for f in d["frames"]:
            out[(d["key"], f["scene_index"], f["cluster_id"])] = f
    return out


def _load_raw_grouped(engine: str) -> Dict[str, dict]:
    """Returns {key: full-engine-output-dict (engine,key,display_stem,frames)}."""
    out: Dict[str, dict] = {}
    for fp in glob.glob(str(config.RESULTS_DIR / "raw" / engine / "*.json")):
        d = json.load(open(fp))
        out[d["key"]] = d
    return out


# ─── Compose ────────────────────────────────────────────────────────────────

def compose(spec: HybridSpec) -> Tuple[Dict[str, dict], dict]:
    """Returns (per-video composed records, stats dict)."""
    base_grouped = _load_raw_grouped(spec.base)
    if not base_grouped:
        raise RuntimeError(f"no raw found for base engine {spec.base!r}")

    esc_indexed = _load_raw_indexed(spec.escalation) if spec.escalation else {}

    # Pre-load any engines referenced in overrides
    override_indexed: Dict[str, Dict[Tuple[str, int, int], dict]] = {}
    for op_spec in spec.overrides.values():
        src = op_spec.get("from")
        if isinstance(src, str):
            override_indexed.setdefault(src, _load_raw_indexed(src))
        elif isinstance(src, list):
            for e in src:
                override_indexed.setdefault(e, _load_raw_indexed(e))
        if "with" in op_spec:
            override_indexed.setdefault(op_spec["with"], _load_raw_indexed(op_spec["with"]))

    out: Dict[str, dict] = {}
    total_frames = 0
    trigger_fires = 0
    for key, dat in base_grouped.items():
        new_frames = []
        for bf in dat["frames"]:
            total_frames += 1
            si, cl = bf["scene_index"], bf["cluster_id"]

            if spec.mode == "trigger-replace":
                if spec.trigger and _eval_predicate(bf, spec.trigger):
                    ef = esc_indexed.get((dat["key"], si, cl))
                    if ef is not None:
                        trigger_fires += 1
                        nf = dict(ef)
                        nf["elapsed"] = round(
                            (bf.get("elapsed") or 0) + (ef.get("elapsed") or 0), 2)
                        new_frames.append(nf)
                        continue
                new_frames.append(dict(bf))

            elif spec.mode == "override":
                nf = dict(bf)
                extra_el = 0.0
                for fld, op_spec in spec.overrides.items():
                    op = op_spec.get("op", "swap")
                    if op == "swap":
                        src = op_spec["from"]
                        src_frame = override_indexed[src].get((dat["key"], si, cl))
                        if src_frame is not None:
                            nf[fld] = src_frame.get(fld)
                            if src != spec.base:
                                extra_el += src_frame.get("elapsed") or 0
                    elif op == "union":
                        with_eng = op_spec["with"]
                        src_frame = override_indexed[with_eng].get((dat["key"], si, cl))
                        if src_frame is not None:
                            base_list = bf.get(fld) or []
                            other = src_frame.get(fld) or []
                            # dedupe preserving order; lowercase for tag-uniqueness
                            seen = set()
                            merged = []
                            for t in base_list + other:
                                k = (t or "").strip().lower()
                                if k and k not in seen:
                                    seen.add(k); merged.append(t)
                            nf[fld] = merged
                            if with_eng != spec.base:
                                extra_el += src_frame.get("elapsed") or 0
                    elif op == "majority-vote":
                        engines = op_spec["from"]
                        votes = []
                        for e in engines:
                            ef2 = override_indexed[e].get((dat["key"], si, cl))
                            if ef2 is not None:
                                votes.append(bool(ef2.get(fld)))
                        if votes:
                            yes = sum(votes); no = len(votes) - yes
                            # tie → False (conservative)
                            nf[fld] = yes > no
                        # voter cost: sum non-base voters' elapsed
                        for e in engines:
                            if e == spec.base:
                                continue
                            ef2 = override_indexed[e].get((dat["key"], si, cl))
                            if ef2 is not None:
                                extra_el += ef2.get("elapsed") or 0
                    else:
                        raise ValueError(f"unknown override op {op!r}")
                nf["elapsed"] = round((bf.get("elapsed") or 0) + extra_el, 2)
                new_frames.append(nf)

            else:
                raise ValueError(f"unknown mode {spec.mode!r}")

        out[dat["key"]] = {
            "engine": spec.name,
            "key": dat["key"],
            "display_stem": dat.get("display_stem"),
            "frames": new_frames,
        }

    stats = {"total_frames": total_frames, "trigger_fires": trigger_fires}
    return out, stats


def write_hybrid_raw(spec_name: str, composed: Dict[str, dict]) -> Path:
    out_dir = config.RESULTS_DIR / "raw" / spec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean stale files (a rebuild fully replaces)
    for old in out_dir.glob("*.json"):
        old.unlink()
    for key, dat in composed.items():
        (out_dir / f"{key}.json").write_text(json.dumps(dat))
    return out_dir


def build_all() -> None:
    register_hybrids()
    specs = load_specs()
    if not specs:
        print(f"  no specs found in {SPECS_DIR}")
        return
    print(f"\n  building {len(specs)} hybrid(s)…")
    for spec in specs:
        composed, stats = compose(spec)
        write_hybrid_raw(spec.name, composed)
        if spec.mode == "trigger-replace":
            fr = stats['trigger_fires']
            rate = f"  fires={fr}/{stats['total_frames']} ({fr/max(1,stats['total_frames'])*100:.1f}%)"
        else:
            rate = f"  frames={stats['total_frames']} (override)"
        print(f"    ✓ {spec.name:32s} mode={spec.mode:18s}{rate}")
    print(f"\n  wrote raw to {config.RESULTS_DIR / 'raw' / '<hybrid-name>'}/")


# ─── Real cascade-cost probe ────────────────────────────────────────────────

def _ollama_unload_all() -> None:
    """Stop every model currently loaded by the Ollama server (pre/post-run hygiene)."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=5) as r:
            data = json.loads(r.read())
    except Exception:
        return
    for m in data.get("models", []):
        try:
            subprocess.run(["ollama", "stop", m["name"]],
                           capture_output=True, timeout=15)
        except Exception:
            pass


def cascade_run(spec_name: str, n_sample: int = 30, seed: int = 42) -> dict:
    """Actually invoke base + conditional escalation on n_sample random frames.

    The synthetic test assumes "we already had both engines' outputs" — this
    probe gives the real wall-clock cost when only one engine runs per frame
    (with escalation when the trigger fires). Pre/post-flight unloads ollama
    models so timing isn't contaminated by stale loaded models and so the GPU
    is freed when we're done.
    """
    register_hybrids()
    specs = {s.name: s for s in load_specs()}
    if spec_name not in specs:
        raise SystemExit(f"unknown spec {spec_name!r}; available: {list(specs)}")
    spec = specs[spec_name]
    if spec.mode != "trigger-replace":
        raise SystemExit(
            f"cascade-cost probe only supports mode=trigger-replace (got {spec.mode!r})")

    # Lazy imports to avoid pulling torch/mlx for build/list paths
    from vision_backends import get_backend
    from vision_schema import SCENE_PROMPT
    from enrich_analyses import try_parse_json
    from bench.runner import _fields

    print(f"\n  cascade-cost probe: {spec_name}")
    print(f"    base={spec.base}  escalation={spec.escalation}  n={n_sample}")
    print(f"    pre-flight: unloading any loaded ollama models…")
    _ollama_unload_all()

    # Gather frame paths from all pilot videos (sampling plans)
    reg = config.pilot_registry()
    all_frames: List[str] = []
    for stem in reg:
        plan_p = config.ENRICHED_DIR / f"{stem}.sampling_plan.json"
        plan = json.loads(plan_p.read_text())
        for s in plan["scenes"]:
            for r in s["representatives"]:
                all_frames.append(str(config.BASE_DIR / r["frame_path"]))
    rng = random.Random(seed); rng.shuffle(all_frames)
    sample = all_frames[:n_sample]

    base_cfg = config.ENGINES[spec.base]
    base_backend = get_backend(base_cfg["backend"], base_cfg["model"])
    esc_cfg = config.ENGINES[spec.escalation]
    esc_backend = get_backend(esc_cfg["backend"], esc_cfg["model"])

    results = []
    print(f"    running cascade on {len(sample)} frames…")
    t_run_start = time.time()
    for i, fp in enumerate(sample):
        text, base_el, err = base_backend.analyze_frame(fp, SCENE_PROMPT)
        parsed = try_parse_json(text or "") if text else None
        d_, t_, st_, ht, nc, ns = _fields(parsed)
        base_frame = {
            "description": d_, "n_snaps": ns, "parse_ok": parsed is not None,
            "has_english_text": ht, "visual_style": st_, "error": err,
        }
        triggered = bool(spec.trigger and _eval_predicate(base_frame, spec.trigger))
        esc_el = 0.0
        if triggered:
            _, esc_el, _ = esc_backend.analyze_frame(fp, SCENE_PROMPT)
        total = base_el + esc_el
        results.append({
            "frame": Path(fp).name,
            "base_el": round(base_el, 2),
            "triggered": triggered,
            "esc_el": round(esc_el, 2),
            "total": round(total, 2),
        })
        flag = f"TRIG (esc={esc_el:.2f}s)" if triggered else "no-trig"
        print(f"      [{i+1:2d}/{len(sample)}] {Path(fp).name[:42]:42s} "
              f"base={base_el:5.2f}s  {flag:<18s}  total={total:5.2f}s", flush=True)
    wall = time.time() - t_run_start

    print(f"\n    post-flight: unloading models…")
    _ollama_unload_all()

    n = len(results)
    n_trig = sum(1 for r in results if r["triggered"])
    totals = sorted(r["total"] for r in results)
    bases = [r["base_el"] for r in results]
    triggered_escs = [r["esc_el"] for r in results if r["triggered"]]
    aggregate = {
        "spec": spec_name,
        "base": spec.base, "escalation": spec.escalation,
        "n_frames": n,
        "trigger_fire_rate": round(n_trig / n, 3),
        "n_triggered": n_trig,
        "mean_total_s": round(sum(totals) / n, 2),
        "median_total_s": totals[n // 2],
        "p90_total_s": totals[int(n * 0.9)],
        "min_total_s": min(totals),
        "max_total_s": max(totals),
        "mean_base_s": round(sum(bases) / n, 2),
        "mean_esc_s_when_triggered":
            round(sum(triggered_escs) / max(1, len(triggered_escs)), 2),
        "wall_clock_s": round(wall, 1),
    }
    out_dir = config.RESULTS_DIR / "cost"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_p = out_dir / f"{spec_name}.json"
    out_p.write_text(json.dumps({"aggregate": aggregate, "details": results}, indent=2))
    print(f"\n    AGGREGATE:")
    print(f"      trigger fire rate:    {aggregate['trigger_fire_rate']*100:.1f}% ({n_trig}/{n})")
    print(f"      mean base s/frm:      {aggregate['mean_base_s']}")
    print(f"      mean esc s (when triggered): {aggregate['mean_esc_s_when_triggered']}")
    print(f"      mean total s/frm:     {aggregate['mean_total_s']}")
    print(f"      median total:         {aggregate['median_total_s']}")
    print(f"      p90 total:            {aggregate['p90_total_s']}")
    print(f"      wall clock:           {aggregate['wall_clock_s']}s")
    print(f"      saved to {out_p}")
    return aggregate
