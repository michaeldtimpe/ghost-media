#!/usr/bin/env python3
"""
LLM-judge faithfulness audit — DIAGNOSTIC ONLY.

A Claude judge (via the subscription CLI) looks at the ACTUAL frame plus a blinded
candidate description and rates accuracy / specificity / hallucination, and crucially
**penalizes under-description** as harshly as hallucination (VLMs learn that short,
vague captions dodge being caught — which is exactly what we're trying to eliminate).

This signal is NEVER folded into the composite rank (bench/metrics.composite_score).
It exists to (1) surface hallucination/vagueness patterns and (2) corroborate the
objective CLIP rank. If the judge and the objective metrics disagree, that's a flag to
investigate — not an average. Because Claude is also a contestant, results are reported
with and without Claude-authored outputs (the runner blinds + randomizes engine order).
"""

import json
import random

from enrich_analyses import try_parse_json

JUDGE_PROMPT = """You are auditing a single analyst's description of the attached video
frame for a VJ/footage-retrieval system. You can SEE the frame. Judge ONLY against what
is actually visible.

Analyst's description:
\"\"\"{description}\"\"\"
Analyst also claims has_english_text = {has_text}.

Score on a 0-5 integer scale and return ONLY this JSON:
{{
  "accuracy": 0-5,          // are the stated objects/colors/motion actually present?
  "specificity": 0-5,       // does it capture what makes THIS frame distinct? 5 = vivid+precise
  "hallucination": 0-5,     // 5 = nothing invented; 0 = major invented content
  "under_description": 0-5, // 5 = appropriately complete; 0 = vacuous/generic ("abstract colorful")
  "text_flag_correct": true|false,  // is the has_english_text claim right for this frame?
  "rationale": "one sentence"
}}

Penalize a vague/generic/short description in BOTH specificity AND under_description even
if nothing is strictly wrong — vagueness is a failure, not a safe choice.
Return ONLY valid JSON, no markdown."""


def judge_frame(backend, frame_path, description, has_text):
    """Return (parsed_dict_or_None, error_or_None)."""
    prompt = JUDGE_PROMPT.format(description=(description or "").strip(),
                                 has_text=str(bool(has_text)).lower())
    text, _elapsed, error = backend.analyze_frame(frame_path, prompt)
    if error:
        return None, error
    parsed = try_parse_json(text)
    if not isinstance(parsed, dict):
        return None, "unparseable judge response"
    return parsed, None


def judge_engine(backend, frames, sample_n=40, seed=42):
    """Audit a random sample of an engine's described frames.

    frames: same shape the runner produces (needs frame_path, description,
    has_english_text). Returns an aggregate dict + the per-frame raw judgements.
    """
    described = [f for f in frames
                 if not f.get("error") and (f.get("description") or "").strip()
                 and f.get("frame_path")]
    rng = random.Random(seed)
    rng.shuffle(described)
    sample = described[:sample_n]

    rows, errors = [], 0
    for f in sample:
        parsed, err = judge_frame(backend, f["frame_path"], f.get("description"),
                                  f.get("has_english_text"))
        if err:
            errors += 1
            continue
        rows.append({
            "key": f["key"], "scene_index": f["scene_index"],
            "cluster_id": f["cluster_id"],
            **{k: parsed.get(k) for k in
               ("accuracy", "specificity", "hallucination", "under_description",
                "text_flag_correct", "rationale")},
        })

    def _mean(field):
        vals = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        return round(sum(vals) / len(vals), 2) if vals else None

    agg = {
        "n_judged": len(rows),
        "n_errors": errors,
        "accuracy": _mean("accuracy"),
        "specificity": _mean("specificity"),
        "hallucination": _mean("hallucination"),
        "under_description": _mean("under_description"),
        "text_flag_accuracy": (
            round(sum(1 for r in rows if r.get("text_flag_correct")) / len(rows), 2)
            if rows else None),
    }
    return {"aggregate": agg, "rows": rows}
