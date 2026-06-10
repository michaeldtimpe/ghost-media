#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SECOND PASS — Targeted Reprocessing
  Based on audit of first-pass output
═══════════════════════════════════════════════════════════════════════════════

Reprocesses 16 files that need improvement:

  GROUP A — Full reprocess (3 files, ~655 min)
    Need: motion analysis + higher scene threshold
    These had unusable scene data (85-99 cuts/min) and no motion.

  GROUP B — Add motion only (3 files, ~361 min)
    Scene data is fine, just missing motion analysis.

  GROUP C — Scene threshold fix (10 files, ~16 min)
    Short files with good motion but overly sensitive scene detection.
    These are all fast-cut motion graphics / reels — high cuts/min is
    partially real, but the threshold can be raised to reduce noise.

Usage:
    python reprocess_pass2.py
    python reprocess_pass2.py --dry-run
    python reprocess_pass2.py --group a         # only the 3 worst files
    python reprocess_pass2.py --group b         # only missing-motion files
    python reprocess_pass2.py --group c         # only scene-threshold fixes
    python reprocess_pass2.py --resume          # skip already-reprocessed
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─── Defaults ────────────────────────────────────────────────────────────────

SOURCE_DIR = "/Volumes/archive/3000/3100/visuals/raw visuals footage"
OUTPUT_DIR = os.path.expanduser("~/Downloads/ghost-media")

# ─── ANSI ────────────────────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

# ─── File Configurations ─────────────────────────────────────────────────────
# Each file gets hand-tuned parameters based on the audit results.

REPROCESS_MANIFEST = [
    # ═══ GROUP A: Full reprocess — motion + scene fix ═══
    {
        "group": "a",
        "file": "7 Hours Visual In Full HD _ https___visual_randes_ee _ VFX Background Video.mp4",
        "reason": "83% scene noise (37644 scenes @ 85 cuts/min), no motion",
        "params": {
            # This is a 7-hour VFX loop — needs very high threshold.
            # The original threshold was 15 (ultra-coarse tier).
            # VFX flashes cause constant false cuts, so raise aggressively.
            "scene_threshold": 70.0,
            "color_interval": 8.0,       # Keep same as first pass (adequate)
            "motion_interval": 8.0,      # Very coarse — 7 hours is a lot
            "brightness_interval": 4.0,  # Keep same
            "n_colors": 3,
            "skip_motion": False,        # KEY CHANGE: add motion
        },
        "workers": 1,
        "est_minutes": 60,
    },
    {
        "group": "a",
        "file": "Experience the THRILL of Flying with This 2 Hour FPV Drone Compilation-b78b89.mp4",
        "reason": "99 cuts/min (12681 scenes), no motion — drone footage needs motion data",
        "params": {
            # FPV drone footage: constant camera movement triggers scene changes.
            # Raise threshold significantly. Motion is the key data for this content.
            "scene_threshold": 65.0,
            "color_interval": 4.0,       # Upgrade from 8s (more useful)
            "motion_interval": 2.0,      # Denser motion — this is drone footage
            "brightness_interval": 3.0,
            "n_colors": 3,
            "skip_motion": False,
        },
        "workers": 1,
        "est_minutes": 35,
    },
    {
        "group": "a",
        "file": "Pretty Lights __ A Color Map of the Sun __ The Visual Project.mp4",
        "reason": "54 cuts/min (4591 scenes), no motion — music visual project",
        "params": {
            # Music visual project — the high cut rate is partially real
            # (it's a music video compilation) but 54/min still has noise.
            "scene_threshold": 50.0,
            "color_interval": 3.0,       # Keep similar (was 5s)
            "motion_interval": 2.0,
            "brightness_interval": 2.0,
            "n_colors": 4,
            "skip_motion": False,
        },
        "workers": 1,
        "est_minutes": 25,
    },

    # ═══ GROUP B: Add motion only (scene data is acceptable) ═══
    {
        "group": "b",
        "file": "Flying Through Portals Compilation [4K] I Trippy Visuals while high on Mushrooms & LSD.mp4",
        "reason": "No motion (13 cuts/min is fine for this content)",
        "params": {
            "scene_threshold": 20.0,     # Keep same — scenes looked good
            "color_interval": 8.0,       # Keep same
            "motion_interval": 3.0,      # 3-hour file, moderate density
            "brightness_interval": 4.0,  # Keep same
            "n_colors": 3,
            "skip_motion": False,
        },
        "workers": 1,
        "est_minutes": 40,
    },
    {
        "group": "b",
        "file": "Mega 4K VJ Loop_ 2 Hours of Non-Stop Dynamic Party & Nightclub Visuals.mp4",
        "reason": "No motion (17 cuts/min is fine for VJ content)",
        "params": {
            "scene_threshold": 20.0,
            "color_interval": 8.0,
            "motion_interval": 3.0,
            "brightness_interval": 4.0,
            "n_colors": 3,
            "skip_motion": False,
        },
        "workers": 1,
        "est_minutes": 30,
    },
    {
        "group": "b",
        "file": "Cinematic Drone Compilation - One Hour of Amazing FPV Drone Flying - 4K-8bf298.mp4",
        "reason": "No motion — drone footage, motion is critical",
        "params": {
            "scene_threshold": 25.0,     # Keep same — 6 cuts/min was fine
            "color_interval": 3.0,       # Denser than first pass (was 5s)
            "motion_interval": 1.5,      # Dense motion — only 1 hour, 60fps
            "brightness_interval": 2.0,
            "n_colors": 4,
            "skip_motion": False,
        },
        "workers": 1,
        "est_minutes": 20,
    },

    # ═══ GROUP C: Scene threshold fix only (short files, fast) ═══
    {
        "group": "c",
        "file": "isshin REEL 2019.mp4",
        "reason": "66 cuts/min — motion graphics reel, raise threshold",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,   # Can parallelize with other small files
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2020.mp4",
        "reason": "64 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2021.webm",
        "reason": "63 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2022-d557b.webm",
        "reason": "66 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2022.webm",
        "reason": "66 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2023.webm",
        "reason": "57 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2024-35e329.webm",
        "reason": "65 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "isshin REEL 2024.webm",
        "reason": "65 cuts/min",
        "params": {
            "scene_threshold": 55.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "Disengaging.mp4",
        "reason": "59 cuts/min",
        "params": {
            "scene_threshold": 50.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
    {
        "group": "c",
        "file": "モーショングラフィックス _Glancent_.webm",
        "reason": "42 cuts/min",
        "params": {
            "scene_threshold": 45.0,
            "color_interval": 0.5,
            "motion_interval": 0.25,
            "brightness_interval": 0.25,
            "n_colors": 5,
            "skip_motion": False,
        },
        "workers": 8,
        "est_minutes": 1,
    },
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    elif s < 3600:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec}s"
    else:
        h, rem = divmod(int(s), 3600)
        m, _ = divmod(rem, 60)
        return f"{h}h {m}m"


def fmt_eta(seconds: float) -> str:
    target = datetime.now() + timedelta(seconds=seconds)
    return target.strftime("%-I:%M %p")


def get_output_path(output_dir: str, filename: str) -> str:
    stem = Path(filename).stem
    safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in stem).strip()[:120]
    return str(Path(output_dir) / f"{safe}.analysis.json")


def is_already_reprocessed(output_path: str) -> bool:
    """Check if file was already reprocessed (has pass2 marker)."""
    try:
        with open(output_path) as f:
            data = json.load(f)
        proc = data.get("_processing", {})
        return proc.get("pass") == 2
    except Exception:
        return False


def analyze_one(source_dir: str, filename: str, output_path: str,
                params: dict) -> dict:
    """Analyze one file with specific parameters."""
    t0 = time.time()
    filepath = os.path.join(source_dir, filename)

    try:
        if not os.path.exists(filepath):
            return {"name": filename, "status": "error",
                    "elapsed": 0, "error": f"File not found: {filepath}"}

        from media_analyzer.video_analyzer import analyze_video

        result = analyze_video(
            filepath,
            scene_threshold=params["scene_threshold"],
            color_interval=params["color_interval"],
            motion_interval=params["motion_interval"],
            brightness_interval=params["brightness_interval"],
            n_colors=params["n_colors"],
            skip_motion=params["skip_motion"],
            quiet=True,
        )

        elapsed = time.time() - t0
        result["_processing"] = {
            "pass": 2,
            "elapsed_sec": round(elapsed, 2),
            "motion_analyzed": not params["skip_motion"],
            "scene_threshold": params["scene_threshold"],
            "timestamp": datetime.now().isoformat(),
        }

        # Back up the original before overwriting
        if os.path.exists(output_path):
            backup = output_path.replace(".analysis.json", ".pass1.analysis.json")
            if not os.path.exists(backup):
                os.rename(output_path, backup)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return {
            "name": filename, "status": "ok",
            "elapsed": elapsed, "output": output_path,
        }

    except Exception as e:
        return {
            "name": filename, "status": "error",
            "elapsed": time.time() - t0, "error": str(e),
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Second-pass reprocessing with per-file tuned parameters"
    )
    parser.add_argument("--source", default=SOURCE_DIR, help="Source directory")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--group", choices=["a", "b", "c"],
                        help="Only process one group")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files already reprocessed (pass 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show plan without processing")

    args = parser.parse_args()

    # Filter manifest
    manifest = REPROCESS_MANIFEST
    if args.group:
        manifest = [m for m in manifest if m["group"] == args.group]

    if not manifest:
        print("Nothing to process.")
        sys.exit(0)

    # Organize by group
    groups = {}
    for item in manifest:
        g = item["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(item)

    # Banner
    print()
    print(f"{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  SECOND PASS — Targeted Reprocessing{RESET}")
    print(f"  {DIM}Reprocessing files identified by audit{RESET}")
    print(f"{BOLD}{'═' * 78}{RESET}")

    total_est = sum(m["est_minutes"] for m in manifest)
    group_labels = {
        "a": f"{RED}A: Full reprocess{RESET}",
        "b": f"{YELLOW}B: Add motion{RESET}",
        "c": f"{GREEN}C: Fix scenes{RESET}",
    }

    print(f"\n  {BOLD}{'Group':<30} {'Files':>6} {'Est. Time':>12}{RESET}")
    print(f"  {'─' * 52}")
    for g in ["a", "b", "c"]:
        if g not in groups:
            continue
        items = groups[g]
        est = sum(m["est_minutes"] for m in items)
        print(f"  {group_labels[g]:<44} {len(items):>6} {est:>9} min")
    print(f"  {'─' * 52}")
    print(f"  {'TOTAL':<30} {len(manifest):>6} {total_est:>9} min")
    print(f"  {DIM}Estimated finish: ~{fmt_eta(total_est * 60)}{RESET}")

    # Detail per file
    for g in ["a", "b", "c"]:
        if g not in groups:
            continue
        items = groups[g]
        print(f"\n  {group_labels[g]}:")
        for item in items:
            p = item["params"]
            motion_str = f"motion @ {p['motion_interval']}s" if not p["skip_motion"] else "no motion"
            print(f"    • {item['file'][:55]}")
            print(f"      scene_thresh={p['scene_threshold']} | {motion_str} | color @ {p['color_interval']}s")

    if args.dry_run:
        print(f"\n{DIM}  (Dry run — nothing processed){RESET}\n")
        sys.exit(0)

    print(f"\n  {BOLD}Starting in 3 seconds...{RESET} (Ctrl+C to cancel)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(0)

    # Process
    os.makedirs(args.output, exist_ok=True)
    all_results = []
    global_start = time.time()

    try:
        for g in ["c", "b", "a"]:  # Fast stuff first
            if g not in groups:
                continue
            items = groups[g]

            print(f"\n{'─' * 78}")
            print(f"  {group_labels[g]} ({len(items)} files)")
            print(f"{'─' * 78}")

            if g == "c":
                # Small files: batch in parallel
                completed = 0
                with ProcessPoolExecutor(max_workers=8) as executor:
                    futures = {}
                    for item in items:
                        out_path = get_output_path(args.output, item["file"])
                        if args.resume and is_already_reprocessed(out_path):
                            print(f"  {DIM}⊘ Skipping (already pass 2): {item['file'][:50]}{RESET}")
                            continue
                        future = executor.submit(
                            analyze_one, args.source, item["file"],
                            out_path, item["params"]
                        )
                        futures[future] = item

                    for future in as_completed(futures):
                        completed += 1
                        item = futures[future]
                        try:
                            r = future.result()
                        except Exception as e:
                            r = {"name": item["file"], "status": "error",
                                 "elapsed": 0, "error": str(e)}
                        all_results.append(r)

                        icon = f"{GREEN}✓{RESET}" if r["status"] == "ok" else f"{RED}✗{RESET}"
                        print(f"  {icon} {fmt_dur(r['elapsed']):>8}  {r['name'][:55]}")
            else:
                # Large files: sequential
                for i, item in enumerate(items, 1):
                    out_path = get_output_path(args.output, item["file"])
                    if args.resume and is_already_reprocessed(out_path):
                        print(f"  {DIM}⊘ Skipping (already pass 2): {item['file'][:50]}{RESET}")
                        continue

                    print(f"\n  ▸ [{i}/{len(items)}] {item['file'][:55]}")
                    print(f"    {item['reason']}")

                    r = analyze_one(args.source, item["file"], out_path, item["params"])
                    all_results.append(r)

                    icon = f"{GREEN}✓{RESET}" if r["status"] == "ok" else f"{RED}✗{RESET}"
                    print(f"    {icon} {fmt_dur(r['elapsed'])}", end="")

                    if r.get("error"):
                        print(f" — {RED}{r['error'][:60]}{RESET}")
                    else:
                        # ETA for remaining in this group
                        remaining = len(items) - i
                        if remaining > 0:
                            avg = sum(rr["elapsed"] for rr in all_results if rr["status"] == "ok") / max(1, sum(1 for rr in all_results if rr["status"] == "ok"))
                            print(f"  {DIM}(~{fmt_eta(remaining * avg)} for group){RESET}")
                        else:
                            print()

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Interrupted — saving progress...{RESET}")
        print(f"  Use {BOLD}--resume{RESET} to continue.\n")

    # Summary
    global_elapsed = time.time() - global_start
    ok = sum(1 for r in all_results if r["status"] == "ok")
    errors = sum(1 for r in all_results if r["status"] == "error")

    print()
    print(f"{BOLD}{'═' * 78}{RESET}")
    print(f"  {BOLD}{GREEN}PASS 2 COMPLETE{RESET}")
    print(f"  {ok} reprocessed · {errors} errors")
    print(f"  Total time: {fmt_dur(global_elapsed)}")
    print(f"  Finished: {datetime.now().strftime('%-I:%M %p')}")
    print(f"\n  Original files backed up as *.pass1.analysis.json")
    print(f"  Output: {args.output}")
    print(f"{BOLD}{'═' * 78}{RESET}")
    print()


if __name__ == "__main__":
    main()
