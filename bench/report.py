#!/usr/bin/env python3
"""
Scoreboard renderer. The ranked table is CLIP-anchored (composite from objective
metrics only); the LLM-judge appears in a SEPARATE audit block so it can never be
mistaken for part of the rank. `dump_video` prints engines' descriptions of the
same frames side-by-side for eyeballing.
"""

import json

from bench import config
from bench.util import fmt_time, load_json


def _g(sb, *path, default=0.0):
    cur = sb
    for p in path:
        cur = cur.get(p, {}) if isinstance(cur, dict) else {}
    return cur if cur not in ({}, None) else default


def render(results_dir=None):
    results_dir = results_dir or config.RESULTS_DIR
    sp = results_dir / "scoreboard.json"
    if not sp.exists():
        print("  No scoreboard yet — run `bench_run.py run` first.")
        return
    data = load_json(sp)
    boards = data.get("ranked", [])

    print(f"\n  {'═'*100}")
    print(f"  BAKE-OFF SCOREBOARD   (composite = CLIP-anchored objective metrics; judge is audit-only)")
    print(f"  weights: {data.get('weights')}")
    print(f"  {'═'*100}")
    hdr = (f"  {'engine':<15s}{'comp':>6s}{'R@1in':>7s}{'MRRin':>7s}"
           f"{'adjD':>6s}{'clps':>6s}{'cover':>7s}{'tEnt':>6s}{'dup':>6s}"
           f"{'txtF1':>7s}{'jAdh':>6s}{'snap':>6s}{'ncmp':>6s}{'s/frm':>7s}{'M5h':>7s}")
    print(hdr)
    print(f"  {'─'*104}")
    for sb in boards:
        rt_in = sb.get("roundtrip_within", {})
        adj, cov, tags = sb.get("adjacent", {}), sb.get("coverage", {}), sb.get("tags", {})
        txt, cost, sch = sb.get("text", {}), sb.get("cost", {}), sb.get("schema", {})
        print(f"  {sb['engine']:<15s}"
              f"{sb.get('composite',0):>6.3f}"
              f"{rt_in.get('recall_at_1',0):>7.3f}"
              f"{rt_in.get('mrr',0):>7.3f}"
              f"{adj.get('adjacent_discriminability',0):>6.2f}"
              f"{adj.get('semantic_collapse_rate',0):>6.2f}"
              f"{cov.get('coverage',0):>7.2f}"
              f"{tags.get('tag_entropy',0):>6.2f}"
              f"{tags.get('tag_dup_rate',0):>6.2f}"
              f"{txt.get('f1',0):>7.2f}"
              f"{sch.get('json_adherence',0):>6.2f}"
              f"{sch.get('enum_snap_rate',0):>6.2f}"
              f"{txt.get('noncompliance_rate',0):>6.2f}"
              f"{cost.get('sec_per_frame',0):>7.1f}"
              f"{cost.get('est_corpus_hrs_m5',0):>7.1f}")
    print(f"  {'─'*104}")
    print("  R@1in=Recall@1 within-video · adjD=adjacent-state distinctness · clps=semantic-collapse")
    print("  cover=states-described/present · tEnt=tag entropy · dup=dup-tag rate · txtF1=text F1")
    print("  jAdh=JSON adherence · snap=enum-snaps/frame · ncmp=text non-compliance (jAdh/snap/ncmp = "
          "production reliability)")
    print("  s/frm=sec/frame · M5h=est. hours to rescan full corpus on M5 (projected)")

    # legend on floor/ceiling
    names = {b["engine"] for b in boards}
    if "baseline" in names:
        print("\n  baseline = the CURRENT system (floor). clip-ceiling = image→image upper bound.")
    _render_judge(results_dir, boards)


def _render_judge(results_dir, boards):
    jdir = results_dir / "judge"
    if not jdir.exists() or not any(jdir.glob("*.json")):
        return
    print(f"\n  {'─'*70}\n  LLM-JUDGE AUDIT (diagnostic only — NOT in the composite)\n  {'─'*70}")
    print(f"  {'engine':<14s}{'acc':>6s}{'spec':>6s}{'halu':>6s}{'undr':>6s}{'txt✓':>6s}{'n':>5s}")
    for sb in boards:
        jp = jdir / f"{sb['engine']}.json"
        if not jp.exists():
            continue
        a = load_json(jp).get("aggregate", {})
        def f(x): return f"{x:>6.2f}" if isinstance(x, (int, float)) else f"{'—':>6s}"
        print(f"  {sb['engine']:<14s}{f(a.get('accuracy'))}{f(a.get('specificity'))}"
              f"{f(a.get('hallucination'))}{f(a.get('under_description'))}"
              f"{f(a.get('text_flag_accuracy'))}{a.get('n_judged',0):>5d}")
    print("  acc=accuracy spec=specificity halu=hallucination(5=none) undr=under-description(5=full) "
          "txt✓=text-flag accuracy")
    print("  If the judge ranking disagrees with the composite above, investigate — do not average.")


def dump_video(registry, video_substr, engines, results_dir=None):
    """Side-by-side: each engine's description of the same representative frames."""
    results_dir = results_dir or config.RESULTS_DIR
    match = [(s, r) for s, r in registry.items() if video_substr.lower() in s.lower()]
    if not match:
        print(f"  no pilot video matching {video_substr!r}")
        return
    stem, rec = match[0]
    print(f"\n  {'═'*90}\n  {stem}\n  {'═'*90}")
    # load each engine's frames keyed by (scene,cluster)
    eng_frames = {}
    for eng in engines:
        rp = results_dir / "raw" / eng / f"{rec['key']}.json"
        if rp.exists():
            eng_frames[eng] = {(f["scene_index"], f["cluster_id"]): f
                               for f in load_json(rp)["frames"]}
    plan = load_json(config.ENRICHED_DIR / f"{stem}.sampling_plan.json")
    shown = 0
    for s in plan["scenes"]:
        for rep in s["representatives"]:
            kk = (s["scene_index"], rep["cluster_id"])
            print(f"\n  ── scene {s['scene_index']} state {rep['cluster_id']} @ {rep['time_sec']}s")
            for eng in engines:
                f = eng_frames.get(eng, {}).get(kk)
                if not f:
                    continue
                d = (f.get("description") or "").strip() or f"[{f.get('error','—')}]"
                txt = " [TEXT]" if f.get("has_english_text") else ""
                print(f"     {eng:<12s}{txt} {d[:140]}")
            shown += 1
            if shown >= 12:
                print("\n  … (truncated)")
                return
