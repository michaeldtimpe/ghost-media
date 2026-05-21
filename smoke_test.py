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
        motion_mean=5.0, motion_peak=8.0, brightness_mean=0.5, contrast_mean=0.3,
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

    print("  " + "─" * 60)
    print("  ALL CHECKS PASSED\n")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n  FAILED: {e}\n")
        sys.exit(1)
