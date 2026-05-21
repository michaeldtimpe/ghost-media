#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  VISUALS LIBRARY ANALYZER
  Tailored for: M1 MacBook Pro Max · 64 GB RAM
  Target:       /Volumes/archive/3000/3100/visuals/raw visuals footage
  Output:       ~/Downloads/ghost-media/
═══════════════════════════════════════════════════════════════════════════════

Processes 57 video files (~44 GB) using a tiered parallelism strategy:

  Tier 1 — Small (<100 MB):   27 files → 8 parallel workers, full analysis
  Tier 2 — Medium (100-500 MB): 18 files → 4 parallel, full analysis
  Tier 3 — Large (500 MB-2 GB): 5 files → 2 parallel, moderate sampling
  Tier 4 — Huge (>2 GB):      7 files → 1 sequential, coarse sampling,
                                          motion skipped on files > 1 hour

Usage:
    python analyze_visuals_library.py
    python analyze_visuals_library.py --dry-run
    python analyze_visuals_library.py --source /path/to/folder --output ~/Desktop/out
    python analyze_visuals_library.py --tier small     # only process one tier
    python analyze_visuals_library.py --resume          # skip already-analyzed files
"""

import json
import os
import sys
import time
import shutil
import argparse
import subprocess
import resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

# ─── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_SOURCE = "/Volumes/archive/3000/3100/visuals/raw visuals footage"
DEFAULT_OUTPUT = os.path.expanduser("~/Downloads/ghost-media")

VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wmv', '.flv', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.ogv', '.ts', '.mts',
}

# ─── Tier Configuration (tuned for M1 Max 64GB) ─────────────────────────────

TIERS = {
    "small": {
        "label": "Small (< 100 MB)",
        "max_bytes": 100 * 1024**2,
        "workers": 8,
        "color_interval": 0.5,
        "motion_interval": 0.25,
        "brightness_interval": 0.25,
        "scene_threshold": 30.0,
        "n_colors": 5,
        "skip_motion": False,
    },
    "medium": {
        "label": "Medium (100-500 MB)",
        "max_bytes": 500 * 1024**2,
        "workers": 4,
        "color_interval": 1.0,
        "motion_interval": 0.5,
        "brightness_interval": 0.5,
        "scene_threshold": 28.0,
        "n_colors": 4,
        "skip_motion": False,
    },
    "large": {
        "label": "Large (500 MB - 2 GB)",
        "max_bytes": 2 * 1024**3,
        "workers": 2,
        "color_interval": 2.0,
        "motion_interval": 1.0,
        "brightness_interval": 1.0,
        "scene_threshold": 25.0,
        "n_colors": 3,
        "skip_motion": False,
    },
    "huge": {
        "label": "Huge (> 2 GB)",
        "max_bytes": float('inf'),
        "workers": 1,
        "color_interval": 4.0,
        "motion_interval": 2.0,
        "brightness_interval": 2.0,
        "scene_threshold": 20.0,
        "n_colors": 3,
        "skip_motion": False,  # Overridden per-file based on duration
    },
}

# Duration above which we skip motion on huge files (seconds)
MOTION_SKIP_DURATION = 3600  # 1 hour

# If a huge file is over this, use ultra-coarse sampling
ULTRA_COARSE_DURATION = 7200  # 2 hours


# ─── Formatting Helpers ──────────────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RESET = "\033[0m"


def fmt_size(n: int) -> str:
    for u in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def fmt_dur(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    elif s < 3600:
        m, sec = divmod(int(s), 60)
        return f"{m}m {sec}s"
    else:
        h, rem = divmod(int(s), 3600)
        m, sec = divmod(rem, 60)
        return f"{h}h {m}m"


def fmt_eta(seconds: float) -> str:
    if seconds <= 0:
        return "done"
    target = datetime.now() + timedelta(seconds=seconds)
    return target.strftime("%-I:%M %p")


def progress_bar(current: int, total: int, width: int = 30) -> str:
    pct = current / total if total > 0 else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {pct * 100:5.1f}%"


def get_rss_mb() -> float:
    """Current process RSS in MB (macOS/Linux)."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # On macOS, ru_maxrss is in bytes; on Linux it's in KB
        if sys.platform == 'darwin':
            return ru.ru_maxrss / (1024 * 1024)
        else:
            return ru.ru_maxrss / 1024
    except Exception:
        return 0.0


# ─── File Discovery ──────────────────────────────────────────────────────────

def probe_duration(filepath: str) -> Optional[float]:
    """Fast duration via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", filepath],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            return float(json.loads(r.stdout).get("format", {}).get("duration", 0))
    except Exception:
        pass
    return None


def discover_files(source_dir: str) -> list[dict]:
    """Scan directory and gather file metadata."""
    src = Path(source_dir)
    if not src.is_dir():
        print(f"{RED}Error:{RESET} '{source_dir}' is not a directory or is not mounted.")
        print(f"       Make sure your NAS is mounted at the expected path.")
        sys.exit(1)

    files = []
    print(f"{DIM}Scanning {source_dir}...{RESET}")

    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if p.name.startswith('.'):
            continue

        size = p.stat().st_size
        files.append({
            "path": str(p),
            "name": p.name,
            "size": size,
            "duration": None,   # Filled in next pass
            "tier": None,       # Assigned after sizing
        })

    return files


def assign_tiers(files: list[dict]) -> dict[str, list[dict]]:
    """Assign each file to a processing tier."""
    tiered = {k: [] for k in TIERS}

    for f in files:
        size = f["size"]
        if size < TIERS["small"]["max_bytes"]:
            f["tier"] = "small"
        elif size < TIERS["medium"]["max_bytes"]:
            f["tier"] = "medium"
        elif size < TIERS["large"]["max_bytes"]:
            f["tier"] = "large"
        else:
            f["tier"] = "huge"
        tiered[f["tier"]].append(f)

    # Sort each tier by size (smallest first)
    for tier_files in tiered.values():
        tier_files.sort(key=lambda f: f["size"])

    return tiered


def probe_durations(files: list[dict]):
    """Probe durations — only for large/huge files where it matters for params."""
    to_probe = [f for f in files if f["tier"] in ("large", "huge")]
    if not to_probe:
        return

    print(f"{DIM}Probing durations for {len(to_probe)} large files...{RESET}")
    for f in to_probe:
        f["duration"] = probe_duration(f["path"])
        dur_str = fmt_dur(f["duration"]) if f["duration"] else "?"
        print(f"  {dur_str:>8}  {f['name'][:65]}")


def get_output_path(output_dir: str, filepath: str, tier: str) -> str:
    """Generate output JSON path for a file."""
    stem = Path(filepath).stem
    safe = "".join(c if c.isalnum() or c in ' -_' else '_' for c in stem).strip()[:120]
    return str(Path(output_dir) / f"{safe}.analysis.json")


def is_already_analyzed(output_path: str) -> bool:
    """Check if a valid analysis file already exists."""
    try:
        if not os.path.exists(output_path):
            return False
        with open(output_path) as f:
            data = json.load(f)
        return data.get("schema_version") == "1.0.0"
    except Exception:
        return False


# ─── Analysis Worker ─────────────────────────────────────────────────────────

def analyze_one(filepath: str, output_path: str, tier: str,
                duration: Optional[float]) -> dict:
    """
    Analyze a single video file. Runs in a worker process.
    Returns a result dict (never raises).
    """
    t0 = time.time()
    name = Path(filepath).name

    try:
        from media_analyzer.video_analyzer import analyze_video
        from media_analyzer.audio_analyzer import analyze_audio as _analyze_audio

        cfg = TIERS[tier].copy()

        # Per-file overrides for huge files based on actual duration
        if tier == "huge" and duration:
            if duration > ULTRA_COARSE_DURATION:
                # Ultra-long: very coarse, no motion
                cfg["color_interval"] = 8.0
                cfg["motion_interval"] = 5.0
                cfg["brightness_interval"] = 4.0
                cfg["scene_threshold"] = 15.0
                cfg["skip_motion"] = True
            elif duration > MOTION_SKIP_DURATION:
                # Over 1 hour: skip motion
                cfg["color_interval"] = 5.0
                cfg["motion_interval"] = 3.0
                cfg["brightness_interval"] = 3.0
                cfg["skip_motion"] = True

        result = analyze_video(
            filepath,
            scene_threshold=cfg["scene_threshold"],
            color_interval=cfg["color_interval"],
            motion_interval=cfg["motion_interval"],
            brightness_interval=cfg["brightness_interval"],
            n_colors=cfg["n_colors"],
            skip_motion=cfg["skip_motion"],
            quiet=True,
        )

        # Add processing metadata
        elapsed = time.time() - t0
        result["_processing"] = {
            "tier": tier,
            "elapsed_sec": round(elapsed, 2),
            "motion_analyzed": not cfg["skip_motion"],
            "timestamp": datetime.now().isoformat(),
        }

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        return {
            "name": name, "status": "ok",
            "elapsed": elapsed, "output": output_path,
        }

    except Exception as e:
        return {
            "name": name, "status": "error",
            "elapsed": time.time() - t0, "error": str(e),
        }


# ─── Tier Runner ─────────────────────────────────────────────────────────────

def run_tier(tier_name: str, files: list[dict], output_dir: str,
             resume: bool = False) -> list[dict]:
    """Process all files in a tier with the tier's parallelism setting."""
    cfg = TIERS[tier_name]
    workers = cfg["workers"]
    results = []

    if not files:
        return results

    # Filter for resume
    tasks = []
    skipped = 0
    for f in files:
        out_path = get_output_path(output_dir, f["path"], tier_name)
        if resume and is_already_analyzed(out_path):
            skipped += 1
            results.append({
                "name": f["name"], "status": "skipped",
                "elapsed": 0, "output": out_path,
            })
            continue
        tasks.append((f, out_path))

    # Header
    print()
    print(f"{'─' * 78}")
    tier_color = {"small": GREEN, "medium": CYAN, "large": YELLOW, "huge": MAGENTA}[tier_name]
    print(f"  {tier_color}{BOLD}TIER: {cfg['label']}{RESET}")
    total_size = sum(f["size"] for f in files)
    print(f"  {len(files)} files ({fmt_size(total_size)})"
          f" • {workers} worker{'s' if workers > 1 else ''}"
          f" • motion: {'on' if not cfg['skip_motion'] else 'adaptive'}")
    if skipped:
        print(f"  {DIM}{skipped} already analyzed (skipping){RESET}")
    print(f"{'─' * 78}")

    if not tasks:
        print(f"  {DIM}Nothing to process.{RESET}")
        return results

    completed = 0
    total = len(tasks)
    tier_start = time.time()

    if workers <= 1:
        # Sequential processing (for huge files)
        for f, out_path in tasks:
            completed += 1
            dur_str = fmt_dur(f["duration"]) if f["duration"] else ""
            skip_note = ""
            if tier_name == "huge" and f["duration"]:
                if f["duration"] > ULTRA_COARSE_DURATION:
                    skip_note = f" {DIM}(ultra-coarse, no motion){RESET}"
                elif f["duration"] > MOTION_SKIP_DURATION:
                    skip_note = f" {DIM}(no motion){RESET}"

            print(f"\n  ▸ [{completed}/{total}] {f['name'][:60]}")
            print(f"    {fmt_size(f['size'])} {dur_str}{skip_note}")

            r = analyze_one(f["path"], out_path, tier_name, f["duration"])
            results.append(r)

            icon = f"{GREEN}✓{RESET}" if r["status"] == "ok" else f"{RED}✗{RESET}"
            print(f"    {icon} {fmt_dur(r['elapsed'])}", end="")

            if r.get("error"):
                print(f" — {RED}{r['error'][:60]}{RESET}")
            else:
                tier_elapsed = time.time() - tier_start
                remaining_files = total - completed
                if completed > 0 and remaining_files > 0:
                    avg = tier_elapsed / completed
                    eta_sec = remaining_files * avg
                    print(f"  {DIM}(ETA for tier: ~{fmt_eta(eta_sec)}){RESET}")
                else:
                    print()
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {}
            for f, out_path in tasks:
                future = executor.submit(
                    analyze_one, f["path"], out_path, tier_name, f["duration"]
                )
                future_map[future] = f

            for future in as_completed(future_map):
                completed += 1
                f = future_map[future]
                try:
                    r = future.result()
                except Exception as e:
                    r = {"name": f["name"], "status": "error",
                         "elapsed": 0, "error": str(e)}
                results.append(r)

                icon = f"{GREEN}✓{RESET}" if r["status"] == "ok" else f"{RED}✗{RESET}"
                elapsed_str = fmt_dur(r["elapsed"])
                bar = progress_bar(completed, total, width=20)

                print(f"  {icon} {bar}  {elapsed_str:>8}  {r['name'][:45]}", end="")

                if r.get("error"):
                    print(f"  {RED}{r['error'][:40]}{RESET}")
                else:
                    tier_elapsed = time.time() - tier_start
                    remaining = total - completed
                    if completed > 0 and remaining > 0:
                        avg = tier_elapsed / completed
                        eta_sec = remaining * avg
                        print(f"  {DIM}ETA {fmt_eta(eta_sec)}{RESET}")
                    else:
                        print()

    tier_elapsed = time.time() - tier_start
    ok_count = sum(1 for r in results if r["status"] == "ok")
    err_count = sum(1 for r in results if r["status"] == "error")
    print(f"\n  Tier complete: {ok_count} ok, {err_count} errors, {skipped} skipped"
          f" — {fmt_dur(tier_elapsed)}")

    return results


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the visuals footage library (M1 Max optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", default=DEFAULT_SOURCE,
        help=f"Source directory (default: {DEFAULT_SOURCE})"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--tier", choices=["small", "medium", "large", "huge"],
        help="Process only one tier"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip files that already have valid analysis JSON"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show plan without processing"
    )
    parser.add_argument(
        "--force-motion", action="store_true",
        help="Force motion analysis even on very long files"
    )

    args = parser.parse_args()

    # ── Dependency check ──
    missing = []
    for mod, pkg, install in [
        ("cv2", "OpenCV", "opencv-python-headless"),
        ("librosa", "librosa", "librosa"),
        ("soundfile", "soundfile", "soundfile"),
        ("numpy", "NumPy", "numpy"),
        ("scipy", "SciPy", "scipy"),
    ]:
        try:
            __import__(mod)
        except ImportError:
            missing.append((pkg, install))

    if missing:
        print(f"\n  {RED}Missing dependencies:{RESET}")
        for pkg, install in missing:
            print(f"    ✗ {pkg}")
        install_cmd = "pip install " + " ".join(inst for _, inst in missing)
        print(f"\n  Fix with: {BOLD}{install_cmd}{RESET}")
        print(f"  Or re-run: {BOLD}./setup.sh{RESET}\n")
        sys.exit(1)

    if args.force_motion:
        global MOTION_SKIP_DURATION, ULTRA_COARSE_DURATION
        MOTION_SKIP_DURATION = float('inf')
        ULTRA_COARSE_DURATION = float('inf')

    # Banner
    print()
    print(f"{BOLD}{'═' * 78}{RESET}")
    print(f"{BOLD}  VISUALS LIBRARY ANALYZER{RESET}")
    print(f"  {DIM}M1 Max · 64 GB · Tiered parallel processing{RESET}")
    print(f"{BOLD}{'═' * 78}{RESET}")
    print()

    # Discover
    files = discover_files(args.source)
    if not files:
        print("No video files found.")
        sys.exit(0)

    tiered = assign_tiers(files)
    probe_durations(files)

    # Summary table
    total_size = sum(f["size"] for f in files)
    print(f"\n{BOLD}  {'Tier':<22} {'Files':>6} {'Size':>10} {'Workers':>9}{RESET}")
    print(f"  {'─' * 50}")
    tier_order = ["small", "medium", "large", "huge"]
    for tn in tier_order:
        tf = tiered[tn]
        cfg = TIERS[tn]
        ts = sum(f["size"] for f in tf)
        color = {"small": GREEN, "medium": CYAN, "large": YELLOW, "huge": MAGENTA}[tn]
        print(f"  {color}{cfg['label']:<22}{RESET} {len(tf):>6} {fmt_size(ts):>10} {cfg['workers']:>9}")
    print(f"  {'─' * 50}")
    print(f"  {'TOTAL':<22} {len(files):>6} {fmt_size(total_size):>10}")
    print(f"\n  Output → {args.output}")

    if args.dry_run:
        # Detailed file listing
        for tn in tier_order:
            tf = tiered[tn]
            if not tf:
                continue
            cfg = TIERS[tn]
            color = {"small": GREEN, "medium": CYAN, "large": YELLOW, "huge": MAGENTA}[tn]
            print(f"\n  {color}{BOLD}{cfg['label']}:{RESET}")
            for f in tf:
                dur_s = fmt_dur(f["duration"]) if f["duration"] else ""
                print(f"    {fmt_size(f['size']):>8}  {dur_s:>8}  {f['name'][:55]}")
        print(f"\n{DIM}  (Dry run — no files processed){RESET}\n")
        sys.exit(0)

    # Auto-proceed (use --dry-run to preview without processing)
    print()
    print(f"  {BOLD}Starting in 3 seconds...{RESET} (Ctrl+C to cancel)")
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(0)

    # Setup output
    os.makedirs(args.output, exist_ok=True)

    # Process tiers
    all_results = []
    global_start = time.time()
    interrupted = False

    tiers_to_run = [args.tier] if args.tier else tier_order

    try:
        for tn in tiers_to_run:
            tf = tiered.get(tn, [])
            tier_results = run_tier(tn, tf, args.output, resume=args.resume)
            all_results.extend(tier_results)
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n\n  {YELLOW}Interrupted — saving progress...{RESET}")
        print(f"  Use {BOLD}--resume{RESET} to continue where you left off.")

    # Write manifest
    global_elapsed = time.time() - global_start
    ok = sum(1 for r in all_results if r["status"] == "ok")
    errors = sum(1 for r in all_results if r["status"] == "error")
    skipped = sum(1 for r in all_results if r["status"] == "skipped")

    manifest = {
        "schema_version": "1.0.0",
        "analyzer": "visuals-library-analyzer",
        "source_directory": str(Path(args.source).resolve()),
        "output_directory": args.output,
        "machine": "M1 Max 64GB",
        "timestamp": datetime.now().isoformat(),
        "total_files": len(files),
        "analyzed": ok,
        "errors": errors,
        "skipped": skipped,
        "total_elapsed_sec": round(global_elapsed, 2),
        "files": all_results,
    }

    manifest_path = Path(args.output) / "_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # Final summary
    print()
    print(f"{BOLD}{'═' * 78}{RESET}")
    status_label = f"{YELLOW}INTERRUPTED{RESET}" if interrupted else f"{GREEN}COMPLETE{RESET}"
    print(f"  {BOLD}{status_label}")
    print(f"  {ok} analyzed · {errors} errors · {skipped} skipped")
    print(f"  Total time: {fmt_dur(global_elapsed)}")
    end_time = datetime.now().strftime("%-I:%M %p")
    print(f"  Finished at: {end_time}")
    print()
    print(f"  Output:   {args.output}")
    print(f"  Manifest: {manifest_path}")
    print(f"{BOLD}{'═' * 78}{RESET}")
    print()

    if errors > 0:
        print(f"  {YELLOW}Files with errors:{RESET}")
        for r in all_results:
            if r["status"] == "error":
                print(f"    {RED}✗{RESET} {r['name']}")
                print(f"      {r.get('error', 'unknown')[:70]}")
        print()


if __name__ == "__main__":
    main()
