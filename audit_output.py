#!/usr/bin/env python3
"""
Audit all analysis JSON files and produce a compact summary.

Usage:
    python audit_output.py ~/Downloads/ghost-media

Outputs:
    ~/Downloads/ghost-media/_audit_summary.json
"""

import json
import sys
import os
from pathlib import Path
from collections import Counter


def audit_file(fp: str) -> dict:
    """Extract quality metrics from one analysis JSON."""
    with open(fp) as f:
        d = json.load(f)

    file_info = d.get("file", {})
    meta = d.get("metadata", {})
    vid = meta.get("video", {})
    proc = d.get("_processing", {})

    duration = meta.get("duration_sec", 0)
    dur_min = duration / 60 if duration > 0 else 0

    # Scenes
    scenes_data = d.get("scenes", {})
    scene_list = scenes_data.get("scenes", [])
    scene_count = scenes_data.get("scene_count", len(scene_list))

    scene_durs = [s["duration_sec"] for s in scene_list] if scene_list else []
    sub_half_sec = sum(1 for dur in scene_durs if dur < 0.5)
    sub_one_sec = sum(1 for dur in scene_durs if dur < 1.0)
    cuts_per_min = scene_count / dur_min if dur_min > 0 else 0

    # Noise ratio: what % of scenes are under 0.5s (likely false positives)
    noise_ratio = sub_half_sec / scene_count if scene_count > 0 else 0

    # Scene quality score (0-1, higher = better)
    # Penalize: too many cuts/min, high noise ratio, too few scenes
    scene_quality = 1.0
    if cuts_per_min > 30:
        scene_quality -= min(0.5, (cuts_per_min - 30) / 100)
    if noise_ratio > 0.3:
        scene_quality -= min(0.4, noise_ratio)
    if scene_count < 3 and duration > 60:
        scene_quality -= 0.3
    scene_quality = max(0, scene_quality)

    # Colors
    colors = d.get("colors", {})
    color_samples = colors.get("sample_count", 0)
    color_interval = colors.get("sample_interval_sec", 0)
    color_timeline = colors.get("timeline", [])
    color_last_t = color_timeline[-1]["time_sec"] if color_timeline else 0
    color_coverage = color_last_t / duration if duration > 0 else 0
    color_per_min = color_samples / dur_min if dur_min > 0 else 0

    # Brightness
    bright = d.get("brightness", {})
    bright_samples = bright.get("sample_count", 0)
    bright_interval = bright.get("sample_interval_sec", 0)
    bright_timeline = bright.get("timeline", [])
    bright_last_t = bright_timeline[-1]["time_sec"] if bright_timeline else 0
    bright_coverage = bright_last_t / duration if duration > 0 else 0
    bright_per_min = bright_samples / dur_min if dur_min > 0 else 0

    # Motion
    motion = d.get("motion")
    has_motion = motion is not None and motion.get("sample_count", 0) > 0
    motion_samples = motion.get("sample_count", 0) if motion else 0
    motion_interval = motion.get("sample_interval_sec", 0) if motion else 0
    motion_level = motion.get("motion_level", "n/a") if motion else "skipped"
    motion_mean = motion.get("overall_mean_motion", 0) if motion else 0
    motion_per_min = motion_samples / dur_min if (dur_min > 0 and motion) else 0

    if has_motion:
        motion_timeline = motion.get("timeline", [])
        motion_last_t = motion_timeline[-1]["time_sec"] if motion_timeline else 0
        motion_coverage = motion_last_t / duration if duration > 0 else 0
    else:
        motion_coverage = 0

    # Data density score (0-1)
    # Ideal: >= 30 color/min, >= 60 bright/min, >= 60 motion/min
    density_score = min(1.0, (
        min(1.0, color_per_min / 30) * 0.3 +
        min(1.0, bright_per_min / 60) * 0.3 +
        (min(1.0, motion_per_min / 60) if has_motion else 0) * 0.4
    ))

    # Overall quality (0-1)
    motion_penalty = 0.3 if not has_motion else 0
    coverage_penalty = max(0, 0.2 * (1 - min(color_coverage, bright_coverage)))
    overall = max(0, min(1.0,
        scene_quality * 0.35 +
        density_score * 0.35 +
        (1.0 - motion_penalty) * 0.2 +
        min(color_coverage, bright_coverage) * 0.1
    ))

    # Determine issues
    issues = []
    if not has_motion:
        issues.append("no_motion")
    if noise_ratio > 0.3:
        issues.append(f"noisy_scenes({noise_ratio:.0%})")
    if cuts_per_min > 40:
        issues.append(f"excessive_cuts({cuts_per_min:.0f}/min)")
    if color_coverage < 0.95:
        issues.append(f"color_gap({color_coverage:.0%})")
    if bright_coverage < 0.95:
        issues.append(f"brightness_gap({bright_coverage:.0%})")
    if color_per_min < 5:
        issues.append(f"sparse_color({color_per_min:.1f}/min)")
    if bright_per_min < 10:
        issues.append(f"sparse_brightness({bright_per_min:.1f}/min)")

    # Recommendation
    if overall >= 0.7 and not issues:
        recommendation = "good"
    elif overall >= 0.5 and len(issues) <= 1:
        recommendation = "acceptable"
    else:
        recommendation = "reprocess"

    # What specifically to redo
    reprocess_flags = []
    if not has_motion:
        reprocess_flags.append("add_motion")
    if noise_ratio > 0.3 or cuts_per_min > 40:
        reprocess_flags.append("raise_scene_threshold")
    if color_per_min < 5:
        reprocess_flags.append("finer_color_sampling")
    if bright_per_min < 10:
        reprocess_flags.append("finer_brightness_sampling")

    return {
        "file": file_info.get("name", Path(fp).stem),
        "source_path": file_info.get("path", ""),
        "duration_sec": round(duration, 1),
        "duration_min": round(dur_min, 1),
        "resolution": f"{vid.get('width', '?')}x{vid.get('height', '?')}",
        "fps": vid.get("fps", 0),
        "tier": proc.get("tier", "unknown"),
        "processing_sec": proc.get("elapsed_sec", 0),

        "scene_count": scene_count,
        "cuts_per_min": round(cuts_per_min, 1),
        "scene_noise_ratio": round(noise_ratio, 3),
        "scene_quality": round(scene_quality, 3),

        "color_samples": color_samples,
        "color_interval": color_interval,
        "color_per_min": round(color_per_min, 1),
        "color_coverage": round(color_coverage, 3),

        "bright_samples": bright_samples,
        "bright_interval": bright_interval,
        "bright_per_min": round(bright_per_min, 1),
        "bright_coverage": round(bright_coverage, 3),

        "has_motion": has_motion,
        "motion_samples": motion_samples,
        "motion_interval": motion_interval,
        "motion_level": motion_level,
        "motion_mean": round(motion_mean, 4),
        "motion_per_min": round(motion_per_min, 1),
        "motion_coverage": round(motion_coverage, 3),

        "density_score": round(density_score, 3),
        "overall_score": round(overall, 3),
        "issues": issues,
        "recommendation": recommendation,
        "reprocess_flags": reprocess_flags,
    }


def main():
    if len(sys.argv) < 2:
        analysis_dir = os.path.expanduser("~/Downloads/ghost-media")
    else:
        analysis_dir = sys.argv[1]

    analysis_path = Path(analysis_dir)
    if not analysis_path.is_dir():
        print(f"Error: {analysis_dir} not found")
        sys.exit(1)

    # Find all analysis JSONs (exclude manifest/audit files)
    json_files = sorted([
        str(p) for p in analysis_path.glob("*.json")
        if not p.name.startswith("_")
    ])

    if not json_files:
        print(f"No analysis files found in {analysis_dir}")
        sys.exit(1)

    print(f"Auditing {len(json_files)} analysis files...\n")

    results = []
    errors = []
    for fp in json_files:
        try:
            r = audit_file(fp)
            results.append(r)
        except Exception as e:
            errors.append({"file": Path(fp).name, "error": str(e)})
            print(f"  ✗ {Path(fp).name}: {e}")

    # Categorize
    good = [r for r in results if r["recommendation"] == "good"]
    acceptable = [r for r in results if r["recommendation"] == "acceptable"]
    reprocess = [r for r in results if r["recommendation"] == "reprocess"]

    # Print summary
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    print(f"{BOLD}{'═' * 78}{RESET}")
    print(f"  {BOLD}ANALYSIS QUALITY AUDIT{RESET}")
    print(f"  {len(results)} files analyzed, {len(errors)} errors")
    print(f"{BOLD}{'═' * 78}{RESET}")

    print(f"\n  {GREEN}Good ({len(good)}):{RESET} No issues, ready to use")
    print(f"  {YELLOW}Acceptable ({len(acceptable)}):{RESET} Minor gaps, usable as-is")
    print(f"  {RED}Reprocess ({len(reprocess)}):{RESET} Significant quality issues\n")

    if reprocess:
        print(f"  {BOLD}Files needing reprocessing:{RESET}")
        print(f"  {'─' * 74}")
        for r in sorted(reprocess, key=lambda x: x["overall_score"]):
            dur_str = f"{r['duration_min']:.0f}m"
            score = r["overall_score"]
            issues = ", ".join(r["issues"])
            flags = ", ".join(r["reprocess_flags"])
            print(f"  {RED}●{RESET} {r['file'][:50]}")
            print(f"    {dur_str} | score: {score:.2f} | issues: {issues}")
            print(f"    fix: {flags}")
            print()

    if acceptable:
        print(f"  {BOLD}Acceptable files:{RESET}")
        print(f"  {'─' * 74}")
        for r in sorted(acceptable, key=lambda x: x["overall_score"]):
            dur_str = f"{r['duration_min']:.0f}m"
            issues = ", ".join(r["issues"]) if r["issues"] else "minor"
            print(f"  {YELLOW}●{RESET} {r['file'][:55]}  ({dur_str}, {issues})")
        print()

    if good:
        print(f"  {BOLD}Good files:{RESET}")
        print(f"  {'─' * 74}")
        for r in sorted(good, key=lambda x: -x["overall_score"]):
            dur_str = f"{r['duration_min']:.0f}m"
            print(f"  {GREEN}●{RESET} {r['file'][:55]}  ({dur_str}, score: {r['overall_score']:.2f})")
        print()

    # Save full audit
    output = {
        "audit_version": "1.0.0",
        "total_files": len(results),
        "good": len(good),
        "acceptable": len(acceptable),
        "reprocess": len(reprocess),
        "errors": len(errors),
        "files": results,
        "error_files": errors,
    }

    out_path = analysis_path / "_audit_summary.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  Full audit saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()