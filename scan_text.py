#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ENGLISH TEXT SCANNER
  Extracts frames at 1fps and flags which seconds contain English text
  Uses qwen2.5vl:7b via Ollama
═══════════════════════════════════════════════════════════════════════════════

Output: per-video JSON with a boolean flag for each second.
"""

import json
import os
import sys
import time
import base64
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# ─── Configuration ─────────────────────────────────────────────────────────

MODEL = "qwen2.5vl:7b"
OLLAMA_API = "http://localhost:11434"
import media_paths
SOURCE_DIR = media_paths.FOOTAGE_ROOT
ANALYSIS_DIR = Path(__file__).parent
OUTPUT_DIR = ANALYSIS_DIR / "text_flags"
FRAMES_DIR = OUTPUT_DIR / "frames"

PROMPT = """Look at this image carefully. Is there any English text visible in this frame?
English text means Latin-alphabet words, letters, numbers used as labels, titles, watermarks, credits, subtitles, logos with English words, etc.

Japanese, Chinese, Korean, or other non-Latin script does NOT count as English text.
Abstract shapes that happen to look like letters do NOT count.
Single isolated letters used as design elements do NOT count.

Answer with ONLY a JSON object:
{"has_english_text": true/false, "description": "brief description of what text if any"}"""

# Videos to scan (from user's list, excluding pass1 files)
TARGETS = [
    "モーションク_ラフィックス _Glancent_",
    "モーションク_ラフィックス _MO_6_",
    "モーションク_ラフィックス _nostalmic_-a73ba4",
    "モーションク_ラフィックス _Quiet or Upset_",
    "モーショングラフィックス _Glancent_",
    "_AviUtl_モーションク_ラフィックス _Graphic Ri_",
    "_AviUtl_モーションク_ラフィックス _Graphic XA_",
    "_AviUtl_モーションク_ラフィックス _Let_s start_",
    "_Motion Graphics_ __Created with After Effects_",
    "isshin b_w",
    "isshin REEL 2019",
    "isshin REEL 2020",
    "isshin REEL 2021",
    "isshin REEL 2022-d557b",
    "isshin REEL 2022",
    "isshin REEL 2023",
    "isshin REEL 2024-35e329",
    "isshin REEL 2024",
]


# ─── Helpers ───────────────────────────────────────────────────────────────

def check_ollama():
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            names = {m["name"] for m in data.get("models", [])}
            return MODEL in names or any(MODEL.split(":")[0] in n for n in names)
    except Exception:
        return False


def extract_frame(video_path, time_sec, output_path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return True
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(time_sec), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(output_path)],
            capture_output=True, timeout=30
        )
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def query_text_detection(image_path):
    """Ask the model if there's English text in the frame."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": MODEL,
        "prompt": PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": 256, "temperature": 0.1},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_API}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - start
            raw = data.get("response", "").strip()

            # Try to parse JSON
            parsed = try_parse_json(raw)
            if parsed and "has_english_text" in parsed:
                return parsed["has_english_text"], parsed.get("description", ""), elapsed, raw

            # Fallback: look for true/false in response
            raw_lower = raw.lower()
            if '"has_english_text": true' in raw_lower or '"has_english_text":true' in raw_lower:
                return True, raw[:200], elapsed, raw
            elif '"has_english_text": false' in raw_lower or '"has_english_text":false' in raw_lower:
                return False, "", elapsed, raw

            # Last resort
            return None, raw[:200], elapsed, raw

    except Exception as e:
        return None, str(e)[:200], time.time() - start, ""


def try_parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    bs = text.find("{")
    be = text.rfind("}") + 1
    if bs >= 0 and be > bs:
        try:
            return json.loads(text[bs:be])
        except json.JSONDecodeError:
            pass
    return None


_SOURCE_FILES_CACHE = None

def _get_source_files():
    global _SOURCE_FILES_CACHE
    if _SOURCE_FILES_CACHE is None:
        _SOURCE_FILES_CACHE = list(media_paths.iter_footage().values())
    return _SOURCE_FILES_CACHE


def find_source_video(analysis_name):
    """Find source video from analysis file."""
    af = ANALYSIS_DIR / f"{analysis_name}.analysis.json"
    if not af.exists():
        return None, 0

    data = json.loads(af.read_text())
    duration = data.get("metadata", {}).get("duration_sec", 0)
    file_info = data.get("file", {})
    original = Path(file_info.get("path", ""))

    if original.exists():
        return str(original), duration

    # Fuzzy search — handle Unicode normalization differences
    original_stem = original.stem
    for f in _get_source_files():
        if original_stem == f.stem:
            return str(f), duration
        if len(original_stem) > 5 and original_stem[:20] in f.stem:
            return str(f), duration
        if len(f.stem) > 5 and f.stem[:20] in original_stem:
            return str(f), duration

    return None, duration


def fmt_time(seconds):
    if seconds < 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bar(pct, width=15):
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    # Check Ollama
    print(f"\n  Checking {MODEL}...", end=" ", flush=True)
    if check_ollama():
        print("✓ ready")
    else:
        print("✗ not found")
        print(f"  Run: ollama pull {MODEL}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Plan work
    work = []
    total_frames = 0
    already_done = []

    for name in TARGETS:
        output_file = OUTPUT_DIR / f"{name}.text_flags.json"
        if output_file.exists():
            existing = json.loads(output_file.read_text())
            if existing.get("status") == "complete":
                already_done.append(name)
                continue

        source, duration = find_source_video(name)
        if not source:
            print(f"  ✗ {name[:45]} — source not found, skipping")
            continue

        n_frames = int(duration)  # 1 fps
        total_frames += n_frames
        work.append({"name": name, "source": source, "duration": duration, "frames": n_frames})

    print(f"\n  Found {len(TARGETS)} target videos")
    print(f"  Already complete: {len(already_done)} videos")
    print(f"  Remaining: {len(work)} videos, {total_frames} frames")
    print(f"  Estimated time: {fmt_time(total_frames * 65)}")

    if not work:
        print("\n  Nothing to do!")
        return

    # Banner
    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  ENGLISH TEXT SCANNER")
    print(f"  Model: {MODEL}")
    print(f"  {len(work)} videos, {total_frames} frames")
    print(f"  ════════════════════════════════════════════════════════════════════")

    # Run
    pipeline_start = time.time()
    global_frame = sum(int(json.loads((OUTPUT_DIR / f"{n}.text_flags.json").read_text()).get("total_frames", 0))
                       for n in already_done if (OUTPUT_DIR / f"{n}.text_flags.json").exists())
    global_total = global_frame + total_frames
    timings = []

    for vid_idx, w in enumerate(work, 1):
        name = w["name"]
        source = w["source"]
        duration = w["duration"]
        n_frames = w["frames"]

        # Overall progress
        overall_pct = global_frame / global_total if global_total else 0
        if timings:
            avg_t = sum(timings[-50:]) / len(timings[-50:])
            overall_eta = avg_t * (global_total - global_frame)
        else:
            overall_eta = 65 * (global_total - global_frame)

        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  [{vid_idx}/{len(work)}] {name[:50]}")
        print(f"  {fmt_time(duration)} duration, {n_frames} frames to scan")
        print(f"  Overall: {fmt_bar(overall_pct, 25)} {global_frame}/{global_total}  ETA: {fmt_time(overall_eta)}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        # Frame directory
        vid_frames = FRAMES_DIR / name.replace("/", "_")[:60]
        vid_frames.mkdir(parents=True, exist_ok=True)

        # Check for partial progress (resume support)
        output_file = OUTPUT_DIR / f"{name}.text_flags.json"
        results = {"video": name, "source": source, "duration_sec": duration, "flags": {}, "status": "in_progress"}
        start_from = 0

        if output_file.exists():
            existing = json.loads(output_file.read_text())
            if existing.get("status") == "in_progress" and existing.get("flags"):
                results = existing
                start_from = max(int(k) for k in results["flags"].keys()) + 1
                print(f"    Resuming from frame {start_from}")

        text_frames = sum(1 for v in results["flags"].values() if v.get("has_english_text"))

        for sec in range(start_from, n_frames):
            t = float(sec)
            frame_path = vid_frames / f"frame_{sec:05d}.jpg"

            # Extract
            if not extract_frame(source, t, frame_path):
                results["flags"][str(sec)] = {"time_sec": t, "has_english_text": None, "error": "extract_failed"}
                global_frame += 1
                continue

            # Query
            has_text, desc, elapsed, raw = query_text_detection(str(frame_path))
            timings.append(elapsed)
            global_frame += 1

            if has_text is True:
                text_frames += 1

            results["flags"][str(sec)] = {
                "time_sec": t,
                "has_english_text": has_text,
                "description": desc if has_text else "",
                "inference_sec": round(elapsed, 2),
            }

            # Clean up frame immediately
            frame_path.unlink(missing_ok=True)

            # Per-frame progress line
            vid_pct = (sec + 1) / n_frames
            overall_pct = global_frame / global_total if global_total else 0
            avg_t = sum(timings[-50:]) / len(timings[-50:])
            overall_eta = avg_t * (global_total - global_frame)

            status = "✓" if has_text is False else ("⚑ TEXT" if has_text else "~")

            print(f"    {fmt_bar(vid_pct)} {sec+1}/{n_frames}"
                  f"  @{t:>6.1f}s  {elapsed:>5.1f}s  {status:<6}"
                  f"  text:{text_frames}"
                  f"  total:{global_frame}/{global_total}  ETA:{fmt_time(overall_eta)}")

            # Save progress every 10 frames
            if sec % 10 == 0:
                output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        # Finalize this video
        results["status"] = "complete"
        results["text_frame_count"] = text_frames
        results["total_frames"] = n_frames
        results["text_ratio"] = round(text_frames / max(n_frames, 1), 4)
        output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        # Clean up frame dir
        for fp in vid_frames.glob("*.jpg"):
            fp.unlink()
        try:
            vid_frames.rmdir()
        except OSError:
            pass

        video_elapsed = time.time() - pipeline_start
        print(f"\n    Done: {text_frames} text frames / {n_frames} total ({results['text_ratio']:.1%})")
        print(f"    → {output_file.name}")

    # Summary
    total_elapsed = time.time() - pipeline_start
    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  SCAN COMPLETE")
    print(f"  {global_frame} frames in {fmt_time(total_elapsed)}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


def show_status():
    """Show progress of a running or completed scan."""
    if not OUTPUT_DIR.exists():
        print("  No scan data found.")
        return

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  TEXT SCAN STATUS")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    total_text = 0
    total_frames = 0
    total_done = 0
    total_remaining = 0

    for name in TARGETS:
        output_file = OUTPUT_DIR / f"{name}.text_flags.json"
        af = ANALYSIS_DIR / f"{name}.analysis.json"

        if not af.exists():
            continue

        adata = json.loads(af.read_text())
        expected_frames = int(adata.get("metadata", {}).get("duration_sec", 0))

        if output_file.exists():
            data = json.loads(output_file.read_text())
            done = len(data.get("flags", {}))
            text_count = sum(1 for v in data.get("flags", {}).values() if v.get("has_english_text"))
            status = data.get("status", "unknown")
            pct = done / max(expected_frames, 1) * 100

            icon = "✓" if status == "complete" else "…"
            print(f"  {icon} {name[:45]:45} {done:>5}/{expected_frames:>5} ({pct:>5.1f}%)"
                  f"  text:{text_count}")
            total_text += text_count
            total_frames += expected_frames
            total_done += done
            if status != "complete":
                total_remaining += expected_frames - done
        else:
            print(f"  ○ {name[:45]:45}     0/{expected_frames:>5} (  0.0%)  pending")
            total_frames += expected_frames
            total_remaining += expected_frames

    pct = total_done / max(total_frames, 1) * 100
    print(f"\n  Total: {total_done}/{total_frames} ({pct:.1f}%)  text frames: {total_text}")
    print(f"  Remaining: {total_remaining} frames (~{fmt_time(total_remaining * 65)})")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Show scan progress")
    args = parser.parse_args()

    if args.status:
        show_status()
    else:
        main()
