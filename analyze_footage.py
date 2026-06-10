#!/usr/bin/env python3
"""
Batch footage analyzer with progress display and adaptive sampling.

Designed for large local media libraries — automatically scales analysis
resolution based on file size/duration to keep processing time reasonable.

Usage:
    python analyze_footage.py "/Volumes/archive/3000/3100/visuals/raw visuals footage"

Output goes to ~/Downloads/ghost-media/ by default.
"""

import json
import os
import sys
import time
import argparse
import subprocess
import signal
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# ─── Configuration ───────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wmv', '.flv', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.ogv', '.ts', '.mts',
}

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.aif',
    '.wma', '.opus', '.ape', '.alac',
}

# Adaptive sampling thresholds (duration in seconds → intervals in seconds)
# Longer files get coarser sampling to keep analysis time and JSON size sane.
SAMPLING_TIERS = [
    # (max_duration_sec, color_interval, motion_interval, brightness_interval, scene_threshold)
    (120,    0.5,  0.25, 0.25, 0.30),   # Under 2 min: fine-grained
    (600,    1.0,  0.5,  0.5,  0.30),    # 2-10 min: standard
    (1800,   2.0,  1.0,  1.0,  0.25),    # 10-30 min: moderate
    (3600,   4.0,  2.0,  2.0,  0.20),    # 30-60 min: coarse
    (7200,   6.0,  3.0,  3.0,  0.15),    # 1-2 hours: very coarse
    (float('inf'), 10.0, 5.0, 5.0, 0.12) # 2+ hours: ultra coarse
]

# Skip motion analysis for files longer than this (seconds)
MOTION_SKIP_THRESHOLD = 3600  # 1 hour


# ─── Utilities ───────────────────────────────────────────────────────────────

def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(int(seconds), 60)
        return f"{m}m {s}s"
    else:
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        return f"{h}h {m}m {s}s"


def get_duration_ffprobe(filepath: str) -> Optional[float]:
    """Fast duration check via ffprobe (no full decode)."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return None


def get_sampling_params(duration_sec: float) -> dict:
    """Select adaptive sampling parameters based on duration."""
    for max_dur, color_iv, motion_iv, bright_iv, scene_thresh in SAMPLING_TIERS:
        if duration_sec <= max_dur:
            return {
                "color_interval": color_iv,
                "motion_interval": motion_iv,
                "brightness_interval": bright_iv,
                "scene_threshold": scene_thresh * 100,  # Convert to 0-100 scale
                "skip_motion": duration_sec > MOTION_SKIP_THRESHOLD,
            }
    # Fallback (shouldn't reach here)
    return SAMPLING_TIERS[-1]


# ─── Progress Display ────────────────────────────────────────────────────────

class ProgressTracker:
    """Thread-safe progress display for batch processing."""

    def __init__(self, files: list[dict]):
        self.files = files
        self.total = len(files)
        self.completed = 0
        self.errors = 0
        self.start_time = time.time()
        self.results = []
        self.current_files = {}  # worker_id → filename

        # Calculate totals
        self.total_size = sum(f["size"] for f in files)

        # Print header
        print()
        print("=" * 78)
        print(f"  MEDIA ANALYZER — Batch Processing")
        print(f"  {self.total} files, {format_size(self.total_size)} total")
        print("=" * 78)
        print()

    def file_started(self, idx: int, filename: str):
        """Called when a file starts processing."""
        f = self.files[idx]
        dur_str = format_duration(f["duration"]) if f["duration"] else "unknown"
        size_str = format_size(f["size"])
        tier = "⚡" if f["duration"] and f["duration"] < 600 else "🔶" if f["duration"] and f["duration"] < 3600 else "🐢"

        print(f"  {tier} [{self.completed + 1}/{self.total}] Starting: {filename}")
        print(f"       Size: {size_str} | Duration: {dur_str}")

    def file_completed(self, idx: int, filename: str, success: bool,
                       elapsed: float, error: str = ""):
        """Called when a file finishes."""
        self.completed += 1
        if not success:
            self.errors += 1

        status = "✅" if success else "❌"
        elapsed_str = format_duration(elapsed)
        pct = (self.completed / self.total) * 100

        print(f"  {status} [{self.completed}/{self.total}] {filename} ({elapsed_str})")
        if error:
            print(f"       Error: {error}")

        # ETA calculation
        total_elapsed = time.time() - self.start_time
        if self.completed > 0:
            avg_per_file = total_elapsed / self.completed
            remaining = (self.total - self.completed) * avg_per_file
            print(f"       Progress: {pct:.0f}% | "
                  f"Elapsed: {format_duration(total_elapsed)} | "
                  f"ETA: ~{format_duration(remaining)}")
        print()

    def print_summary(self, output_dir: str):
        """Print final summary."""
        total_elapsed = time.time() - self.start_time
        success = self.completed - self.errors

        print()
        print("=" * 78)
        print(f"  COMPLETE")
        print(f"  {success}/{self.total} succeeded, {self.errors} errors")
        print(f"  Total time: {format_duration(total_elapsed)}")
        print(f"  Output: {output_dir}")
        print("=" * 78)
        print()


# ─── Analysis Worker ─────────────────────────────────────────────────────────

def analyze_file(filepath: str, output_dir: str, file_type: str,
                 duration: float, audio_sr: int = 22050) -> dict:
    """
    Analyze a single file with adaptive parameters.
    Runs in a subprocess for parallelism.
    """
    start = time.time()
    filename = Path(filepath).name

    try:
        if file_type == "audio":
            from media_analyzer.audio_analyzer import analyze_audio
            result = analyze_audio(filepath, sr=audio_sr)
        else:
            from media_analyzer.video_analyzer import analyze_video
            params = get_sampling_params(duration or 600)
            result = analyze_video(
                filepath,
                scene_threshold=params["scene_threshold"],
                color_interval=params["color_interval"],
                motion_interval=params["motion_interval"],
                brightness_interval=params["brightness_interval"],
                n_colors=3,
                skip_motion=params["skip_motion"],
            )

        # Write output
        safe_name = Path(filepath).stem
        # Sanitize filename for filesystem
        safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in safe_name).strip()[:120]
        out_file = Path(output_dir) / f"{safe_name}.{file_type}.analysis.json"

        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start
        return {
            "file": filepath,
            "name": filename,
            "type": file_type,
            "status": "success",
            "output": str(out_file),
            "elapsed_sec": round(elapsed, 2),
        }

    except Exception as e:
        elapsed = time.time() - start
        return {
            "file": filepath,
            "name": filename,
            "type": file_type,
            "status": "error",
            "error": str(e),
            "elapsed_sec": round(elapsed, 2),
        }


# ─── Main ────────────────────────────────────────────────────────────────────

def scan_directory(directory: str, recursive: bool = False) -> list[dict]:
    """Scan for media files and gather metadata."""
    dir_path = Path(directory)
    pattern = '**/*' if recursive else '*'

    files = []
    print(f"Scanning: {directory}")

    for p in sorted(dir_path.glob(pattern)):
        if not p.is_file():
            continue

        ext = p.suffix.lower()
        if ext in VIDEO_EXTENSIONS:
            file_type = "video"
        elif ext in AUDIO_EXTENSIONS:
            file_type = "audio"
        else:
            continue

        size = p.stat().st_size
        duration = get_duration_ffprobe(str(p))

        files.append({
            "path": str(p),
            "name": p.name,
            "type": file_type,
            "size": size,
            "duration": duration,
        })

    return files


def estimate_processing_time(files: list[dict]) -> str:
    """Rough estimate of total processing time."""
    total_sec = 0
    for f in files:
        dur = f["duration"] or 300  # Assume 5 min if unknown
        if f["type"] == "video":
            if dur > MOTION_SKIP_THRESHOLD:
                # Without motion: ~1 min per 10 min of footage
                total_sec += dur * 0.1
            elif dur > 1800:
                # Coarse sampling: ~1 min per 5 min
                total_sec += dur * 0.2
            else:
                # Standard: ~1 min per 2 min of footage
                total_sec += dur * 0.5
        else:
            # Audio is fast: ~10 sec per minute
            total_sec += dur * 0.17

    return format_duration(total_sec)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a folder of media with progress tracking"
    )
    parser.add_argument(
        "directory",
        help="Path to the media folder"
    )
    parser.add_argument(
        "-o", "--output",
        default=os.path.expanduser("~/Downloads/ghost-media"),
        help="Output directory (default: ~/Downloads/ghost-media)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int, default=2,
        help="Parallel workers (default: 2 — conservative for large files)"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Scan subdirectories"
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only analyze audio files"
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Only analyze video files"
    )
    parser.add_argument(
        "--force-motion",
        action="store_true",
        help="Force motion analysis even on long files (slow!)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and show plan without analyzing"
    )

    args = parser.parse_args()

    # Validate input
    if not Path(args.directory).is_dir():
        print(f"Error: '{args.directory}' is not a directory or is not mounted.")
        sys.exit(1)

    # Override motion skip if forced
    global MOTION_SKIP_THRESHOLD
    if args.force_motion:
        MOTION_SKIP_THRESHOLD = float('inf')

    # Scan
    files = scan_directory(args.directory, recursive=args.recursive)

    # Filter
    if args.audio_only:
        files = [f for f in files if f["type"] == "audio"]
    elif args.video_only:
        files = [f for f in files if f["type"] == "video"]

    if not files:
        print("No media files found.")
        sys.exit(0)

    # Sort: smallest first so we see early progress
    files.sort(key=lambda f: f["size"])

    # Show plan
    total_size = sum(f["size"] for f in files)
    total_duration = sum(f["duration"] or 0 for f in files)
    video_count = sum(1 for f in files if f["type"] == "video")
    audio_count = sum(1 for f in files if f["type"] == "audio")
    eta = estimate_processing_time(files)

    print(f"\nFound {len(files)} files ({video_count} video, {audio_count} audio)")
    print(f"Total size: {format_size(total_size)}")
    print(f"Total duration: {format_duration(total_duration)}")
    print(f"Estimated processing time: ~{eta} (with {args.workers} workers)")
    print(f"Output: {args.output}")
    print()

    # Show file list with analysis plan
    print("Files to analyze:")
    print("-" * 78)
    for i, f in enumerate(files, 1):
        dur = f["duration"]
        dur_str = format_duration(dur) if dur else "?"
        size_str = format_size(f["size"])
        params = get_sampling_params(dur or 600) if f["type"] == "video" else None

        motion_note = ""
        if f["type"] == "video" and params:
            if params["skip_motion"]:
                motion_note = " [no motion — too long]"
            else:
                motion_note = f" [motion @ {params['motion_interval']}s]"

        print(f"  {i:2d}. [{f['type'][:1].upper()}] {f['name']}")
        print(f"      {size_str} | {dur_str}{motion_note}")
    print("-" * 78)

    if args.dry_run:
        print("\n(Dry run — no analysis performed)")
        sys.exit(0)

    # Confirm
    print()
    response = input("Proceed? [Y/n] ").strip().lower()
    if response and response != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process
    tracker = ProgressTracker(files)

    if args.workers <= 1:
        # Sequential — simpler progress
        for i, f in enumerate(files):
            tracker.file_started(i, f["name"])
            result = analyze_file(
                f["path"], args.output, f["type"], f["duration"]
            )
            tracker.file_completed(
                i, f["name"],
                success=(result["status"] == "success"),
                elapsed=result["elapsed_sec"],
                error=result.get("error", "")
            )
            tracker.results.append(result)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {}
            for i, f in enumerate(files):
                future = executor.submit(
                    analyze_file, f["path"], args.output, f["type"], f["duration"]
                )
                future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                f = files[idx]
                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "file": f["path"], "name": f["name"],
                        "type": f["type"], "status": "error",
                        "error": str(e), "elapsed_sec": 0,
                    }

                tracker.file_completed(
                    idx, f["name"],
                    success=(result["status"] == "success"),
                    elapsed=result["elapsed_sec"],
                    error=result.get("error", "")
                )
                tracker.results.append(result)

    # Write manifest
    manifest = {
        "schema_version": "1.0.0",
        "analyzer": "media-analyzer-batch",
        "source_directory": str(Path(args.directory).resolve()),
        "output_directory": args.output,
        "total_files": tracker.total,
        "successful": tracker.completed - tracker.errors,
        "errors": tracker.errors,
        "total_elapsed_sec": round(time.time() - tracker.start_time, 2),
        "files": tracker.results,
    }

    manifest_path = Path(args.output) / "_batch_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    tracker.print_summary(args.output)


if __name__ == "__main__":
    main()
