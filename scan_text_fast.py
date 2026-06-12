#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  FAST ENGLISH TEXT SCANNER — TWO-PASS
  Pass 1: EAST text detector (OpenCV) — sub-second, catches all text regions
  Pass 2: EasyOCR — confirms readable English text in flagged regions (~1s/frame)
═══════════════════════════════════════════════════════════════════════════════

Strategy:
  1. Coarse scan every COARSE_INTERVAL seconds using EAST detector (~0.1s/frame)
  2. Binary-search boundaries around EAST hits (still using EAST — fast)
  3. Confirm flagged regions with EasyOCR (only the hits, not all frames)
  4. Output .text_flags.json compatible with assemble_v2.py
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

import warnings
warnings.filterwarnings("ignore", message=".*pin_memory.*")


# ─── Configuration ─────────────────────────────────────────────────────────

MODEL = "qwen2.5vl:7b"
OLLAMA_API = "http://localhost:11434"
import media_paths
SOURCE_DIR = media_paths.FOOTAGE_ROOT
ANALYSIS_DIR = Path(__file__).parent
OUTPUT_DIR = ANALYSIS_DIR / "text_flags"
FRAMES_DIR = OUTPUT_DIR / "frames"

COARSE_INTERVAL = 60    # seconds between coarse samples
FINE_RESOLUTION = 2     # binary search stops at this granularity (seconds)
BOUNDARY_MARGIN = 5     # extra seconds to pad around detected text regions

EAST_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
EAST_MODEL_PATH = Path.home() / ".cache" / "east_text_detection.pb"
EAST_CONFIDENCE = 0.7   # minimum confidence for EAST text detection (0.5 too noisy on abstract visuals)
EAST_NMS_THRESHOLD = 0.4
EAST_INPUT_SIZE = (320, 320)  # EAST requires multiples of 32

CONFIRM_PROMPT = """Look at this image carefully. Is there any English text visible in this frame?
English text means Latin-alphabet words, letters, numbers used as labels, titles, watermarks, credits, subtitles, logos with English words, etc.

Japanese, Chinese, Korean, or other non-Latin script does NOT count as English text.
Abstract shapes that happen to look like letters do NOT count.
Single isolated letters used as design elements do NOT count.

Answer with ONLY a JSON object:
{"has_english_text": true/false, "description": "brief description of what text if any"}"""


# ─── Text detection (shared with scripts/check_render_text.py) ────────────
# EAST + EasyOCR machinery lives in text_detect.py; this module keeps the
# scanning strategy (coarse scan, binary-search refinement, flag building).
from text_detect import (  # noqa: E402
    EAST_CONFIDENCE,
    detect_text_east,
    extract_frame,
    get_east_net as _get_east_net,
    get_ocr_reader as _get_ocr_reader,
    verify_text_ocr,
)


def check_ollama():
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            names = {m["name"] for m in data.get("models", [])}
            return MODEL in names or any(MODEL.split(":")[0] in n for n in names)
    except Exception:
        return False


def query_llm_confirm(image_path):
    """Ask the vision LLM to confirm English text in a frame."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": MODEL,
        "prompt": CONFIRM_PROMPT,
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

            parsed = try_parse_json(raw)
            if parsed and "has_english_text" in parsed:
                return parsed["has_english_text"], parsed.get("description", ""), elapsed

            raw_lower = raw.lower()
            if '"has_english_text": true' in raw_lower or '"has_english_text":true' in raw_lower:
                return True, raw[:200], elapsed
            elif '"has_english_text": false' in raw_lower or '"has_english_text":false' in raw_lower:
                return False, "", elapsed

            return None, raw[:200], elapsed

    except Exception as e:
        return None, str(e)[:200], time.time() - start


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


def check_frame_east(video_path, time_sec, frames_dir):
    """Extract and check a single frame with EAST. Returns (has_text, n_regions, elapsed)."""
    frame_path = frames_dir / f"frame_{int(time_sec):06d}.jpg"
    if not extract_frame(video_path, time_sec, frame_path):
        return None, 0, 0
    has_text, n_regions, elapsed = detect_text_east(frame_path)
    frame_path.unlink(missing_ok=True)
    return has_text, n_regions, elapsed


_SOURCE_FILES_CACHE = None

def _get_source_files():
    global _SOURCE_FILES_CACHE
    if _SOURCE_FILES_CACHE is None:
        _SOURCE_FILES_CACHE = list(media_paths.iter_footage().values())
    return _SOURCE_FILES_CACHE


def find_source_video(analysis_name):
    af = ANALYSIS_DIR / f"{analysis_name}.analysis.json"
    if not af.exists():
        return None, 0
    data = json.loads(af.read_text())
    duration = data.get("metadata", {}).get("duration_sec", 0)
    file_info = data.get("file", {})
    original = Path(file_info.get("path", ""))
    if original.exists():
        return str(original), duration
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
    filled = int(width * min(pct, 1.0))
    return "█" * filled + "░" * (width - filled)


# ─── Two-Pass Scanner ───────────────────────────────────────────────────

def binary_search_boundary(video_path, frames_dir, lo, hi, direction="start"):
    """Binary search for text boundary using EAST detector."""
    queries = 0
    while hi - lo > FINE_RESOLUTION:
        mid = (lo + hi) / 2
        has_text, _, elapsed = check_frame_east(video_path, mid, frames_dir)
        queries += 1

        if has_text is None:
            if direction == "start":
                lo = mid
            else:
                hi = mid
            continue

        if direction == "start":
            if has_text:
                hi = mid
            else:
                lo = mid
        else:
            if has_text:
                lo = mid
            else:
                hi = mid

    boundary = lo if direction == "end" else hi
    return int(boundary), queries


def scan_video_two_pass(name, source_path, duration, frames_dir, skip_confirm=False):
    """Scan a video: EAST for speed, LLM to confirm.
    Returns: (flags_dict, stats_dict)
    """
    total_sec = int(duration)
    coarse_times = list(range(0, total_sec, COARSE_INTERVAL))
    if total_sec > 0 and coarse_times[-1] < total_sec - 5:
        coarse_times.append(total_sec - 1)

    total_east_queries = 0
    total_ocr_queries = 0

    # ── Pass 1: EAST coarse scan ──
    print(f"    Pass 1 — EAST coarse scan ({len(coarse_times)} samples at {COARSE_INTERVAL}s)")

    coarse_results = {}
    timings = []
    east_hits = []

    for i, t in enumerate(coarse_times):
        has_text, n_regions, elapsed = check_frame_east(source_path, t, frames_dir)
        total_east_queries += 1
        timings.append(elapsed)
        coarse_results[t] = (has_text, n_regions)

        status = f"⚑ {n_regions} regions" if has_text else ("✓" if has_text is False else "~")
        if has_text:
            east_hits.append(t)

        avg_t = sum(timings[-20:]) / len(timings[-20:])
        remaining = avg_t * (len(coarse_times) - i - 1)
        pct = (i + 1) / len(coarse_times)
        print(f"      {fmt_bar(pct)} {i+1}/{len(coarse_times)}"
              f"  @{t:>6.0f}s  {elapsed:>5.2f}s  {status}"
              f"  ETA:{fmt_time(remaining)}", flush=True)

    print(f"    Pass 1 done: {len(east_hits)} EAST hits in {total_east_queries} queries"
          f" ({sum(timings):.1f}s total)")

    if not east_hits:
        print(f"    No text detected — done")
        flags = {}
        for t, (has_text, _) in coarse_results.items():
            flags[str(t)] = {
                "time_sec": float(t),
                "has_english_text": False,
                "description": "",
                "sample_type": "east_coarse",
            }
        return flags, {
            "total_east_queries": total_east_queries,
            "total_ocr_queries": 0,
            "text_regions": [],
        }

    # ── EAST boundary refinement ──
    print(f"\n    Refining {len(east_hits)} EAST hits with binary search")

    search_regions = []
    for t in sorted(east_hits):
        lo = max(0, t - COARSE_INTERVAL)
        hi = min(total_sec, t + COARSE_INTERVAL)
        if search_regions and lo <= search_regions[-1][1]:
            search_regions[-1] = (search_regions[-1][0], hi)
        else:
            search_regions.append((lo, hi))

    east_regions = []
    for region_lo, region_hi in search_regions:
        region_hits = [t for t in east_hits if region_lo <= t <= region_hi]
        pre_text = max((t for t in coarse_results if t < min(region_hits)
                        and region_lo <= t and not coarse_results[t][0]),
                       default=region_lo)
        first_text = min(region_hits)

        start_sec, sq = binary_search_boundary(
            source_path, frames_dir, pre_text, first_text, "start")
        total_east_queries += sq

        last_text = max(region_hits)
        post_text = min((t for t in coarse_results if t > last_text
                         and not coarse_results[t][0]),
                        default=region_hi)
        end_sec, eq = binary_search_boundary(
            source_path, frames_dir, last_text, post_text, "end")
        total_east_queries += eq

        start_sec = max(0, start_sec - BOUNDARY_MARGIN)
        end_sec = min(total_sec, end_sec + BOUNDARY_MARGIN)
        east_regions.append((start_sec, end_sec))
        print(f"      EAST region: {fmt_time(start_sec)} – {fmt_time(end_sec)}"
              f"  ({end_sec - start_sec}s, +{sq + eq} queries)")

    # ── Pass 2: OCR confirmation on region midpoints ──
    if skip_confirm:
        print(f"\n    Skipping OCR confirmation (--no-confirm)")
        confirmed_regions = [(s, e, "") for s, e in east_regions]
    else:
        print(f"\n    Pass 2 — OCR confirmation of {len(east_regions)} region(s)")

        confirmed_regions = []
        ocr_timings = []
        for start, end in east_regions:
            region_len = end - start
            mid = start + region_len // 2

            frame_path = frames_dir / f"confirm_{int(mid):06d}.jpg"
            if not extract_frame(source_path, mid, frame_path):
                print(f"      Region {fmt_time(start)}–{fmt_time(end)}: "
                      f"could not extract frame")
                continue

            has_text, desc, elapsed = verify_text_ocr(str(frame_path))
            frame_path.unlink(missing_ok=True)
            total_ocr_queries += 1
            ocr_timings.append(elapsed)

            if has_text:
                confirmed_regions.append((start, end, desc))
                print(f"      Region {fmt_time(start)}–{fmt_time(end)}: "
                      f"{elapsed:.2f}s  ✓ \"{desc[:70]}\"", flush=True)
            else:
                print(f"      Region {fmt_time(start)}–{fmt_time(end)}: "
                      f"{elapsed:.2f}s  ✗ false positive", flush=True)

        if ocr_timings:
            print(f"    Pass 2 done: {len(confirmed_regions)} confirmed, "
                  f"{len(east_regions) - len(confirmed_regions)} rejected "
                  f"({sum(ocr_timings):.1f}s total)")

    # ── Build flags ──
    flags = {}
    text_seconds = set()
    for start, end, desc in confirmed_regions:
        for s in range(start, end + 1):
            text_seconds.add(s)

    for t, (has_text, _) in coarse_results.items():
        in_region = t in text_seconds
        flags[str(t)] = {
            "time_sec": float(t),
            "has_english_text": in_region,
            "description": "",
            "sample_type": "east_coarse",
        }

    for s in text_seconds:
        key = str(s)
        if key not in flags:
            desc = ""
            for start, end, d in confirmed_regions:
                if start <= s <= end:
                    desc = d
                    break
            flags[key] = {
                "time_sec": float(s),
                "has_english_text": True,
                "description": desc,
                "sample_type": "ocr_confirmed",
            }

    return flags, {
        "total_east_queries": total_east_queries,
        "total_ocr_queries": total_ocr_queries,
        "text_regions": confirmed_regions,
    }


# ─── Discovery ────────────────────────────────────────────────────────────

def discover_unscanned():
    analysis_files = sorted(ANALYSIS_DIR.glob("*.analysis.json"))
    existing_flags = set()
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.glob("*.text_flags.json"):
            stem = f.name.replace(".text_flags.json", "")
            existing_flags.add(stem)

    unscanned = []
    for af in analysis_files:
        name = af.name.replace(".analysis.json", "")
        if name in existing_flags:
            continue
        source, duration = find_source_video(name)
        if source:
            unscanned.append({"name": name, "source": source, "duration": duration})
        else:
            print(f"  ⚠ {name[:50]} — source not found, skipping")

    return unscanned


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    global COARSE_INTERVAL

    import argparse
    parser = argparse.ArgumentParser(description="Fast two-pass English text scanner (EAST + LLM)")
    parser.add_argument("--status", action="store_true", help="Show scan status")
    parser.add_argument("--interval", type=int, default=COARSE_INTERVAL,
                        help=f"Coarse sample interval in seconds (default: {COARSE_INTERVAL})")
    parser.add_argument("--only", type=str, nargs="*",
                        help="Only scan these video names (substring match)")
    parser.add_argument("--rescan", action="store_true",
                        help="Rescan videos that already have text flags")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip LLM confirmation (EAST only, faster but more false positives)")
    args = parser.parse_args()

    COARSE_INTERVAL = args.interval

    if args.status:
        show_status()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load EAST model upfront
    print(f"\n  Loading EAST text detector...", end=" ", flush=True)
    _get_east_net()
    print("✓ ready")

    if not args.no_confirm:
        print(f"  Loading EasyOCR...", end=" ", flush=True)
        _get_ocr_reader()
        print("✓ ready (will confirm EAST hits)")

    # Discover work
    if args.rescan:
        work = []
        for af in sorted(ANALYSIS_DIR.glob("*.analysis.json")):
            name = af.name.replace(".analysis.json", "")
            source, duration = find_source_video(name)
            if source:
                work.append({"name": name, "source": source, "duration": duration})
    else:
        work = discover_unscanned()

    if args.only:
        filtered = []
        for w in work:
            if any(term.lower() in w["name"].lower() for term in args.only):
                filtered.append(w)
        work = filtered

    if not work:
        print("\n  All videos already scanned. Use --rescan to redo.")
        return

    total_duration = sum(w["duration"] for w in work)
    est_queries = sum(int(w["duration"]) // COARSE_INTERVAL + 1 for w in work)

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  FAST TEXT SCANNER (EAST + OCR two-pass)")
    print(f"  Pass 1: EAST detector (~0.1s/frame)")
    print(f"  Pass 2: EasyOCR (confirms EAST hits, ~1s/frame)")
    print(f"  Coarse interval: {COARSE_INTERVAL}s")
    print(f"  Videos: {len(work)} ({fmt_time(total_duration)} total)")
    print(f"  Estimated EAST queries: ~{est_queries}")
    print(f"  ════════════════════════════════════════════════════════════════════")

    pipeline_start = time.time()
    total_east = 0
    total_ocr = 0
    total_text_regions = 0

    for vid_idx, w in enumerate(work, 1):
        name = w["name"]
        source = w["source"]
        duration = w["duration"]

        print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"  [{vid_idx}/{len(work)}] {name[:60]}")
        print(f"  {fmt_time(duration)} duration, source: {Path(source).name[:50]}")
        print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        vid_frames = FRAMES_DIR / name.replace("/", "_")[:60]
        vid_frames.mkdir(parents=True, exist_ok=True)

        flags, stats = scan_video_two_pass(
            name, source, duration, vid_frames, skip_confirm=args.no_confirm)

        total_east += stats["total_east_queries"]
        total_ocr += stats["total_ocr_queries"]
        total_text_regions += len(stats["text_regions"])

        # Save results
        text_count = sum(1 for v in flags.values() if v.get("has_english_text"))
        total_sec = int(duration)
        result = {
            "video": name,
            "source": source,
            "duration_sec": duration,
            "flags": flags,
            "status": "complete",
            "scan_method": "east_ocr" if not args.no_confirm else "east_only",
            "coarse_interval": COARSE_INTERVAL,
            "text_frame_count": text_count,
            "total_frames": total_sec,
            "text_ratio": round(text_count / max(total_sec, 1), 4),
            "east_queries": stats["total_east_queries"],
            "ocr_queries": stats["total_ocr_queries"],
            "text_regions": [
                {"start_sec": s, "end_sec": e, "description": d}
                for s, e, d in stats["text_regions"]
            ],
        }

        output_file = OUTPUT_DIR / f"{name}.text_flags.json"
        output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        print(f"\n    Result: {text_count} text seconds / {total_sec}s total"
              f" ({result['text_ratio']:.1%})")
        print(f"    EAST queries: {stats['total_east_queries']}"
              f"  |  OCR queries: {stats['total_ocr_queries']}")
        print(f"    → {output_file.name}")

        # Cleanup
        for fp in vid_frames.glob("*.jpg"):
            fp.unlink()
        try:
            vid_frames.rmdir()
        except OSError:
            pass

    total_elapsed = time.time() - pipeline_start
    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  SCAN COMPLETE")
    print(f"  {len(work)} videos in {fmt_time(total_elapsed)}")
    print(f"  EAST queries: {total_east}  |  OCR queries: {total_ocr}")
    print(f"  Text regions: {total_text_regions}")
    print(f"  Output: {OUTPUT_DIR}/")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


def show_status():
    print(f"\n  ═══════════════════════════════════════════════════════════════")
    print(f"  TEXT SCAN STATUS (all videos)")
    print(f"  ═══════════════════════════════════════════════════════════════\n")

    analysis_files = sorted(ANALYSIS_DIR.glob("*.analysis.json"))
    scanned = 0
    unscanned = 0
    total_text = 0

    for af in analysis_files:
        name = af.name.replace(".analysis.json", "")
        data = json.loads(af.read_text())
        duration = data.get("metadata", {}).get("duration_sec", 0)

        tf = OUTPUT_DIR / f"{name}.text_flags.json"
        if tf.exists():
            tdata = json.loads(tf.read_text())
            text_count = tdata.get("text_frame_count", 0)
            method = tdata.get("scan_method", "unknown")
            east_q = tdata.get("east_queries", tdata.get("queries_used", "?"))
            ocr_q = tdata.get("ocr_queries", tdata.get("llm_queries", "?"))
            status = tdata.get("status", "unknown")
            icon = "✓" if status == "complete" else "…"
            total_text += text_count
            scanned += 1
            print(f"  {icon} {name[:45]:45} {fmt_time(duration):>7}"
                  f"  text:{text_count:>4}  method:{method}"
                  f"  east:{east_q} llm:{ocr_q}")
        else:
            unscanned += 1
            print(f"  ○ {name[:45]:45} {fmt_time(duration):>7}  not scanned")

    print(f"\n  Scanned: {scanned}  |  Remaining: {unscanned}  |  Text frames: {total_text}")
    print()


if __name__ == "__main__":
    main()
