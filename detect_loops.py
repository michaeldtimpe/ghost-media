#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  LOOP DETECTOR
  Detect loopable scenes by comparing first/last frames (SSIM)
  Updates enriched JSON files in-place with loopable flag per scene
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import re
import subprocess
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

# ─── Configuration ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
ENRICHED_DIR = BASE_DIR / "enriched"
SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")
TMP_DIR = Path("/tmp/loop_detect_frames")

SSIM_THRESHOLD = 0.85       # above this = loopable
MIN_BRIGHTNESS = 0.05       # skip near-black frames (false positive prevention)
FRAME_SIZE = (256, 256)     # resize for SSIM comparison
FRAME_OFFSET = 0.15         # seconds inward from start/end to avoid transitions


# ─── Source Resolution ────────────────────────────────────────────────────

def _sanitize_for_match(name):
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()


def find_source_video(file_info):
    """Find source video from enriched file_info."""
    original_path = Path(file_info.get("path", ""))
    if original_path.exists():
        return str(original_path)
    sanitized_target = _sanitize_for_match(original_path.stem)
    for candidate in SOURCE_DIR.iterdir():
        if original_path.stem[:25] in candidate.stem or candidate.stem[:25] in original_path.stem:
            return str(candidate)
    for candidate in SOURCE_DIR.iterdir():
        sanitized_candidate = _sanitize_for_match(candidate.stem)
        if len(sanitized_target) > 10 and len(sanitized_candidate) > 10:
            if sanitized_target[:25] in sanitized_candidate or sanitized_candidate[:25] in sanitized_target:
                return str(candidate)
    return None


# ─── Frame Extraction & SSIM ─────────────────────────────────────────────

def extract_frame(video_path, time_sec, output_path):
    """Extract a single frame via ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(max(0, time_sec)), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(output_path)],
            capture_output=True, timeout=30
        )
        return output_path.exists() and output_path.stat().st_size > 100
    except Exception:
        return False


def load_gray(path):
    """Load image as grayscale numpy array, resized."""
    img = Image.open(path).convert("L").resize(FRAME_SIZE, Image.LANCZOS)
    return np.array(img, dtype=np.float64) / 255.0


def compute_ssim(img1, img2):
    """Compute SSIM between two grayscale numpy arrays."""
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = img1.var()
    sigma2_sq = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)

    return float(num / den)


def check_scene_loopable(video_path, scene, tmp_dir):
    """Check if a scene is loopable by comparing first/last frames."""
    idx = scene["scene_index"]
    start = scene["start_sec"]
    end = scene["end_sec"]
    duration = scene["duration_sec"]

    # Skip very short scenes (< 1s can't meaningfully loop)
    if duration < 1.0:
        return False, 0.0

    frame_start = tmp_dir / f"s{idx}_start.jpg"
    frame_end = tmp_dir / f"s{idx}_end.jpg"

    t_start = start + FRAME_OFFSET
    t_end = max(start + 0.5, end - FRAME_OFFSET)

    ok1 = extract_frame(video_path, t_start, frame_start)
    ok2 = extract_frame(video_path, t_end, frame_end)

    if not (ok1 and ok2):
        return False, 0.0

    try:
        img1 = load_gray(frame_start)
        img2 = load_gray(frame_end)

        # Brightness check — skip near-black frames
        if img1.mean() < MIN_BRIGHTNESS and img2.mean() < MIN_BRIGHTNESS:
            return False, 0.0

        ssim = compute_ssim(img1, img2)
        return ssim >= SSIM_THRESHOLD, round(ssim, 4)
    finally:
        frame_start.unlink(missing_ok=True)
        frame_end.unlink(missing_ok=True)


# ─── Processing ──────────────────────────────────────────────────────────

def process_video(enriched_path, threshold, dry_run=False):
    """Process all scenes in one enriched file."""
    data = json.loads(enriched_path.read_text())
    scenes = data.get("scenes", {}).get("scenes", [])
    file_info = data.get("file", {})
    name = enriched_path.name.replace(".enriched.json", "")

    source_path = find_source_video(file_info)
    if not source_path or not Path(source_path).exists():
        return name, 0, 0, len(scenes), "source not found"

    if dry_run:
        return name, 0, 0, len(scenes), "dry run"

    tmp_dir = TMP_DIR / name[:50]
    tmp_dir.mkdir(parents=True, exist_ok=True)

    loopable_count = 0
    processed = 0

    for scene in scenes:
        is_loop, similarity = check_scene_loopable(source_path, scene, tmp_dir)
        scene["loopable"] = is_loop
        scene["loop_similarity"] = similarity
        if is_loop:
            loopable_count += 1
        processed += 1

    # Update enriched file
    data["loop_detection"] = {
        "ssim_threshold": threshold,
        "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "loopable_scenes": loopable_count,
        "total_scenes": len(scenes),
    }
    enriched_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    # Cleanup
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return name, loopable_count, processed, len(scenes), "ok"


def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect loopable scenes in enriched video files")
    parser.add_argument("--video", type=str, nargs="*", help="Filter by video name (substring)")
    parser.add_argument("--threshold", type=float, default=SSIM_THRESHOLD,
                        help=f"SSIM threshold for loopable (default: {SSIM_THRESHOLD})")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without processing")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos that already have loop detection data")
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"  ✗ Source directory not found: {SOURCE_DIR}")
        print(f"    Mount the archive drive and try again.")
        sys.exit(1)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    enriched_files = sorted(ENRICHED_DIR.glob("*.enriched.json"))

    if args.video:
        enriched_files = [f for f in enriched_files
                          if any(v.lower() in f.name.lower() for v in args.video)]

    if args.skip_existing:
        filtered = []
        for f in enriched_files:
            data = json.loads(f.read_text())
            if "loop_detection" not in data:
                filtered.append(f)
        enriched_files = filtered

    total_scenes = 0
    for f in enriched_files:
        data = json.loads(f.read_text())
        total_scenes += len(data.get("scenes", {}).get("scenes", []))

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  LOOP DETECTOR")
    print(f"  SSIM threshold: {args.threshold}")
    print(f"  Videos: {len(enriched_files)}  |  Scenes: {total_scenes}")
    print(f"  ════════════════════════════════════════════════════════════════════")

    if args.dry_run:
        for f in enriched_files:
            data = json.loads(f.read_text())
            n = len(data.get("scenes", {}).get("scenes", []))
            name = f.name.replace(".enriched.json", "")
            print(f"  ○ {name[:50]:<50}  {n:>5} scenes")
        print(f"\n  Would process {total_scenes} scenes across {len(enriched_files)} videos")
        return

    pipeline_start = time.time()
    total_loopable = 0
    total_processed = 0

    for idx, ef in enumerate(enriched_files, 1):
        name_short = ef.name.replace(".enriched.json", "")[:50]
        data = json.loads(ef.read_text())
        n_scenes = len(data.get("scenes", {}).get("scenes", []))

        print(f"\n  [{idx}/{len(enriched_files)}] {name_short}  ({n_scenes} scenes)")
        t0 = time.time()

        name, loopable, processed, total, status = process_video(
            ef, args.threshold, dry_run=False)

        elapsed = time.time() - t0
        total_loopable += loopable
        total_processed += processed

        if status == "ok":
            pct = (loopable / total * 100) if total > 0 else 0
            print(f"    ✓ {loopable}/{total} loopable ({pct:.0f}%)  "
                  f"in {fmt_time(elapsed)}", flush=True)
        else:
            print(f"    ✗ {status}")

    total_elapsed = time.time() - pipeline_start
    pct = (total_loopable / total_processed * 100) if total_processed > 0 else 0

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  LOOP DETECTION COMPLETE")
    print(f"  {total_processed} scenes processed in {fmt_time(total_elapsed)}")
    print(f"  {total_loopable} loopable ({pct:.0f}%)")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
