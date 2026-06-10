"""Capture perceptual-diversity baselines for the validation sets.

Runs the assembler's selection pipeline (no ffmpeg render) on each set in
SET_CONFIGS, computes the Phase D metrics, writes the results to
bench/perceptual_baselines.json. Used to anchor Phase F's acceptance criteria
against a real distribution rather than an assumed one.

Run BEFORE Phase C (MMR) lands so the baselines reflect post-A pre-MMR state.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import assemble_v2 as av  # noqa: E402

DEFAULT_SETS = [
    "waiting-to-begin-2024",
    "cheerleader-exodus-2025",
    "boxing-day-2025",
]


def selection_for(set_name: str, phrase_bars: int = 4, seed: int = 42):
    set_config = av.SET_CONFIGS[set_name]
    analysis_path = av.BASE_DIR / "sets" / set_config["analysis"]
    if not analysis_path.exists():
        raise SystemExit(f"Analysis missing for {set_name}: {analysis_path}")

    text_seconds = av.load_text_flags()
    scenes = av.build_scene_database(text_seconds)
    data = json.loads(analysis_path.read_text())
    phrase_key = {4: "four_bar", 8: "eight_bar", 16: "sixteen_bar"}[phrase_bars]
    phrases = data["phrases"][phrase_key]
    phrase_features = av.extract_phrase_features(data, phrases)
    phrase_features = av.merge_phrases_adaptive(phrase_features)
    style_hints = set_config.get("style_hints", {})
    np.random.seed(seed)
    selections = av.select_clips(scenes, phrase_features, style_hints)
    return selections, scenes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    p.add_argument("--output", default="bench/perceptual_baselines.json")
    p.add_argument("--phrase-bars", type=int, default=4, choices=[4, 8, 16])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default="phase_A_baseline",
                   help="label for this capture (becomes a top-level key)")
    args = p.parse_args()

    # Try to capture the current git sha for provenance
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        sha = "unknown"

    capture = {
        "schema_version": "1.0",
        "label": args.label,
        "assembler_commit": sha,
        "phrase_bars": args.phrase_bars,
        "seed": args.seed,
        "sets": {},
    }

    for set_name in args.sets:
        print(f"\n=== {set_name} ===", flush=True)
        try:
            selections, scenes = selection_for(set_name, args.phrase_bars, args.seed)
        except SystemExit as e:
            print(f"  SKIPPED: {e}")
            continue
        metrics = av.compute_perceptual_diversity(selections, scenes)
        # Source distribution
        source_counts = {}
        for s in selections:
            name = s["clip_source_name"]
            source_counts[name] = source_counts.get(name, 0) + 1
        metrics["distinct_sources_used"] = len(source_counts)
        metrics["max_single_source_count"] = max(source_counts.values()) if source_counts else 0
        capture["sets"][set_name] = metrics
        print(f"  selections      = {metrics['n_selections']}")
        print(f"  consec mean/med = {metrics['consec_mean']:.4f} / {metrics['consec_median']:.4f}")
        print(f"  consec p90/p95  = {metrics['consec_p90']:.4f} / {metrics['consec_p95']:.4f}")
        print(f"  pairs ≥0.85 w5  = {metrics['close_pairs_w5'].get(0.85, 0)}")
        print(f"  pairs ≥0.80 w5  = {metrics['close_pairs_w5'].get(0.80, 0)}")
        print(f"  pairs ≥0.75 w5  = {metrics['close_pairs_w5'].get(0.75, 0)}")
        print(f"  pairs ≥0.90 w30 = {metrics['close_pairs_w30'].get(0.90, 0)}")
        print(f"  pairs ≥0.85 w30 = {metrics['close_pairs_w30'].get(0.85, 0)}")
        print(f"  sources used    = {metrics['distinct_sources_used']}")
        print(f"  max source ct   = {metrics['max_single_source_count']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # JSON keys must be strings — convert numeric thresholds in close_pairs_* dicts
    def stringify_keys(d):
        if isinstance(d, dict):
            return {str(k): stringify_keys(v) for k, v in d.items()}
        return d
    capture = stringify_keys(capture)

    out.write_text(json.dumps(capture, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
