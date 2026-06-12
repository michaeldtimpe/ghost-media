#!/usr/bin/env python3
"""
Smoke test — fast, dependency-light checks for the feature interactions that the
full render can't easily exercise. Run: `python3 smoke_test.py`

Covers:
  - score_scene with the lyrics × CLIP-embedding path active (the scoring branch
    near the `scene_pool` NameError that used to crash lyric+CLIP runs)
  - select_clips over a tiny mixed database (one scene with a CLIP vector + lyrics,
    one without) returning one selection per phrase
  - the quality soft-penalty math
  - vision_schema.normalize_analysis enum snapping / list coercion
"""

import sys

import numpy as np

import assemble_v2 as a
from vision_schema import normalize_analysis


def _clip(**kw):
    base = dict(
        source_name="src", source_path="src.mp4", scene_index=0,
        start_sec=0.0, end_sec=5.0, duration_sec=5.0,
        motion_mean=5.0, motion_peak=8.0, motion_std=1.0,
        brightness_mean=0.5, contrast_mean=0.3,
        dominant_colors=[], color_temperature=0.5, color_saturation=0.4,
        visual_style="abstract", mood_energy="moderate", content_tags=[],
        has_semantic=True,
    )
    base.update(kw)
    return a.SceneClip(**base)


def _phrase(**kw):
    base = dict(
        start_sec=0.0, end_sec=5.0, duration_sec=5.0, energy=0.6, energy_peak=0.7,
        energy_shape="sustain", bass_ratio=0.5, brightness_audio=0.5,
        percussive_ratio=0.4, bpm=120.0, track_title="t", lyrics_keywords=[],
    )
    base.update(kw)
    return a.PhraseFeatures(**base)


def check(name, cond):
    print(f"  {'✓' if cond else '✗'} {name}")
    if not cond:
        raise AssertionError(name)


def main():
    print("\n  SMOKE TEST")
    print("  " + "─" * 60)

    rng = np.random.default_rng(0)
    vec = rng.standard_normal(512).astype(np.float32)
    vec /= np.linalg.norm(vec)

    # 1. Lyrics × CLIP path: phrase carries lyric keywords AND a text embedding,
    #    one scene carries matching tags AND a visual embedding.
    s_rich = _clip(content_tags=["fire", "sky"], clip_embedding=vec)
    s_plain = _clip(scene_index=1, source_name="src2", clip_embedding=None,
                    content_tags=[])
    ph = _phrase(lyrics_keywords=["fire", "sky"], clip_text_embedding=vec)
    sr = a.score_scene(s_rich, ph)
    sp = a.score_scene(s_plain, ph)
    check("score_scene runs with lyrics+CLIP active", isinstance(sr, float))
    check("rich scene (lyric+CLIP match) outscores plain", sr > sp)

    # 2. select_clips over a mixed database, multiple phrases.
    sels = a.select_clips([s_rich, s_plain], [ph, _phrase(start_sec=5, end_sec=10)])
    check("select_clips returns one selection per phrase", len(sels) == 2)
    check("selections carry quality fields",
          all("clip_quality" in s and "clip_cull_flags" in s for s in sels))

    # 3. Quality soft-penalty math: a dead scene scores exactly the penalty below.
    clean = _clip()
    dead = _clip(quality_score=0.1, cull_flags=["black"])
    delta = a.score_scene(dead, ph) - a.score_scene(clean, ph)
    expected = -(1.0 - 0.1) * a.QUALITY_PENALTY_WEIGHT
    check(f"quality penalty == {expected:+.2f}", abs(delta - expected) < 1e-6)

    # 4. Schema normalizer snaps junk enums and coerces lists.
    clean_an, notes = normalize_analysis({
        "visual_style": "TRIPPY", "mood": {"energy": "super"},
        "content_tags": "fire, sky, love",
    })
    check("invalid visual_style → unclear", clean_an["visual_style"] == "unclear")
    check("invalid mood.energy → unclear", clean_an["mood"]["energy"] == "unclear")
    check("comma-string tags → list", clean_an["content_tags"] == ["fire", "sky", "love"])
    check("corrections recorded in _validation", "_validation" in clean_an and bool(notes))

    cut_timing_checks()
    bench_checks(rng)

    print("  " + "─" * 60)
    print("  ALL CHECKS PASSED\n")


def cut_timing_checks():
    """Beat snapping + vocal-aware boundary adjustment (cut-timing fixes)."""
    print("\n  CUT TIMING")
    print("  " + "─" * 60)

    # _snap_to_beat: nearest beat, both directions, and graceful no-grid.
    grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    check("snap down", a._snap_to_beat(0.71, grid) == 0.5)
    check("snap up", a._snap_to_beat(0.79, grid) == 1.0)
    check("snap exact is identity", a._snap_to_beat(1.5, grid) == 1.5)
    check("snap past last beat clamps", a._snap_to_beat(9.9, grid) == 2.0)
    check("no grid → unchanged", a._snap_to_beat(0.71, None) == 0.71)

    # _schema_tuple gating
    check("schema 2.1.0 < (2,2)", a._schema_tuple({"schema_version": "2.1.0"}) < (2, 2))
    check("schema 2.2.0 >= (2,2)", a._schema_tuple({"schema_version": "2.2.0"}) >= (2, 2))
    check("garbage schema → (0,0)", a._schema_tuple({"schema_version": "?"}) == (0, 0))

    # adjust_cuts_for_vocals: 120 BPM beat grid, two 8s phrases with the
    # boundary at 8.0s landing inside the word "fire" (7.9–8.2s).
    beats = [round(i * 0.5, 4) for i in range(40)]
    segs = [{"start_sec": 7.0, "end_sec": 9.0, "words": [
        {"word": "set", "start": 7.0, "end": 7.3},
        {"word": "fire", "start": 7.9, "end": 8.2},
    ]}]
    p1, p2 = _phrase(start_sec=0.0, end_sec=8.0, duration_sec=8.0), \
        _phrase(start_sec=8.0, end_sec=16.0, duration_sec=8.0)
    pfs, moved, merged = a.adjust_cuts_for_vocals([p1, p2], segs, beats)
    check("mid-word boundary moved", moved == 1 and merged == 0 and len(pfs) == 2)
    check("moved to a word-clear beat", p1.end_sec in (7.0, 7.5, 9.0)
          and not (7.9 - a.VOCAL_WORD_PAD_SEC < p1.end_sec < 8.2 + a.VOCAL_WORD_PAD_SEC))
    check("end/start stay a matched pair", p1.end_sec == p2.start_sec)
    check("durations updated", abs(p1.duration_sec - p1.end_sec) < 1e-9
          and abs(p2.duration_sec - (16.0 - p2.start_sec)) < 1e-9)

    # Continuous vocals across every candidate beat → merge the boundary away
    # (combined 16s ≤ VOCAL_MERGE_MAX_SEC).
    dense = [{"start_sec": 6.0, "end_sec": 10.0, "words": [
        {"word": f"w{i}", "start": 6.0 + i * 0.4, "end": 6.39 + i * 0.4}
        for i in range(10)
    ]}]
    p1, p2 = _phrase(start_sec=0.0, end_sec=8.0, duration_sec=8.0), \
        _phrase(start_sec=8.0, end_sec=16.0, duration_sec=8.0)
    pfs, moved, merged = a.adjust_cuts_for_vocals([p1, p2], dense, beats)
    check("continuous vocals → boundary merged away",
          merged == 1 and len(pfs) == 1
          and pfs[0].start_sec == 0.0 and pfs[0].end_sec == 16.0
          and abs(pfs[0].duration_sec - 16.0) < 1e-9)

    # Continuous vocals but merge would exceed the cap → boundary stays put.
    p1, p2 = _phrase(start_sec=0.0, end_sec=8.0, duration_sec=8.0), \
        _phrase(start_sec=8.0, end_sec=8.0 + a.VOCAL_MERGE_MAX_SEC,
                duration_sec=a.VOCAL_MERGE_MAX_SEC)
    pfs, moved, merged = a.adjust_cuts_for_vocals([p1, p2], dense, beats)
    check("over-cap merge skipped → boundary unmoved",
          merged == 0 and len(pfs) == 2 and pfs[0].end_sec == 8.0)

    # Boundary clear of any word → untouched.
    p1, p2 = _phrase(start_sec=0.0, end_sec=4.0, duration_sec=4.0), \
        _phrase(start_sec=4.0, end_sec=16.0, duration_sec=12.0)
    pfs, moved, merged = a.adjust_cuts_for_vocals([p1, p2], segs, beats)
    check("word-clear boundary untouched",
          moved == 0 and merged == 0 and pfs[0].end_sec == 4.0)


def bench_checks(rng):
    """Bake-off harness unit checks (skip the torch-dependent parts if unavailable)."""
    print("\n  BENCH HARNESS")
    print("  " + "─" * 60)

    # keys: determinism + collision abort
    from bench import keys
    check("canonical_key is deterministic + path-sensitive",
          keys.canonical_key("/a/x.mp4") == keys.canonical_key("/a/x.mp4")
          != keys.canonical_key("/a/y.mp4"))
    try:
        keys.build_registry([("A.analysis.json", "/s/a.mp4"),
                             ("A.analysis.json", "/s/b.mp4")])
        check("stem collision aborts", False)
    except keys.StemCollisionError:
        check("stem collision aborts", True)

    # schema: folded-in text-detection bool
    from vision_schema import _coerce_bool, normalize_analysis
    check("_coerce_bool compliant", _coerce_bool("yes") == (True, True)
          and _coerce_bool(False) == (False, True))
    check("_coerce_bool flags non-compliance", _coerce_bool("there is text") == (False, False))
    clean, notes = normalize_analysis({"has_english_text": "true"})
    check("has_english_text coerced to bool", clean["has_english_text"] is True)
    clean, notes = normalize_analysis({"has_english_text": "a whole sentence"})
    check("non-bool text flag → False + noncompliant",
          clean["has_english_text"] is False and notes["has_english_text"]["noncompliant"])

    # metrics (numpy-only)
    from bench import metrics as Mx
    d = 16
    imgs = [Mx._norm(rng.standard_normal(d)) for _ in range(6)]
    kk = ["A", "A", "A", "B", "B", "B"]
    import numpy as _np
    perfect = Mx.retrieval_metrics(imgs, kk, imgs, kk, _np.arange(6), within=True)
    check("retrieval: perfect query → R@1=1.0", perfect["recall_at_1"] == 1.0)
    garbage = [Mx._norm(_np.ones(d)) for _ in range(6)]
    g = Mx.retrieval_metrics(garbage, kk, imgs, kk, _np.arange(6), within=True)
    check("retrieval: identical queries → R@1<1.0", g["recall_at_1"] < 1.0)
    adj_same = Mx.adjacent_discriminability({"A": [imgs[0], imgs[0]]})
    adj_diff = Mx.adjacent_discriminability({"A": [imgs[0], imgs[1]]})
    check("adjacent: identical → collapse=1.0", adj_same["semantic_collapse_rate"] == 1.0)
    check("adjacent: distinct > identical",
          adj_diff["adjacent_discriminability"] > adj_same["adjacent_discriminability"])
    prf = Mx.text_prf({"a": True, "b": False}, {"a": True, "b": True})
    check("text_prf tp/fn", prf["tp"] == 1 and prf["fn"] == 1)
    same = Mx.tag_stats([["abstract"]] * 8)
    uniq = Mx.tag_stats([[f"t{i}"] for i in range(8)])
    check("tag entropy: unique > repeated", uniq["tag_entropy"] > same["tag_entropy"])
    sb = {"roundtrip_within": {"recall_at_1": 0.8}, "coverage": {"coverage": 0.9},
          "text": {"f1": 0.7}, "adjacent": {"adjacent_discriminability": 0.5}}
    check("composite blend", abs(Mx.composite_score(sb)
          - (0.4 * 0.8 + 0.25 * 0.9 + 0.2 * 0.7 + 0.15 * 0.5)) < 1e-9)

    # sampler clustering (needs torch via clip_utils; skip if unavailable)
    try:
        from bench import sampler
    except Exception as e:
        print(f"  ~ sampler clustering skipped (torch unavailable: {str(e)[:40]})")
        return
    a = _np.array([1.0, 0.0, 0.0]); b = _np.array([0.0, 1.0, 0.0])
    groups = sampler.cluster_embeddings([a, a.copy(), b], threshold=0.9)
    check("cluster: 2 distinct + 1 dup → 2 clusters", len(groups) == 2)
    check("medoid returns a member index",
          sampler._medoid([0, 1, 2], [a, a.copy(), b]) in (0, 1, 2))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n  FAILED: {e}\n")
        sys.exit(1)
