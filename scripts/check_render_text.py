"""Post-render English-text check: re-verify a finished render for lingering text.

The enrichment-time text filter (scan_text_fast.py) can miss text — coarse
60s sampling, detector misses, salvage margins. This closes the loop: scan the
RENDERED video, attribute any hit back to the source clip via the selection
plan, and optionally feed the offending source seconds back into text_flags/
so the next selection run excludes them.

Sampling:
  --plan selection_plan_v2.json   3 frames per planned clip (start+0.2s,
                                  midpoint, end−0.2s) — ~3×N frames total
  (no plan)                       1 frame per second of the render

Each sampled frame goes through the same two-pass detector as enrichment
(EAST ~0.1s/frame, EasyOCR confirm on EAST hits only). EAST-only hits are
reported separately from OCR-confirmed ones (stylized abstract visuals are
the known EAST false-positive mode).

Exit status: nonzero iff any OCR-CONFIRMED text is found.

--update-flags writes <source>.render_check.text_flags.json files covering
the offending clips' source ranges (same loader contract as the existing
*.runtime_bans.text_flags.json precedent).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from text_detect import (  # noqa: E402
    detect_text_east, extract_frame, get_east_net, get_ocr_reader, verify_text_ocr,
)

FPS = 30  # render frame rate; matches assemble_v2.FPS
CLIP_EDGE_OFFSET_SEC = 0.2
TEXT_FLAGS_DIR = Path(__file__).resolve().parent.parent / "text_flags"


def sample_times_from_plan(plan_rows):
    """Yield (render_time_sec, row) sample points: 3 per planned clip."""
    cursor_frames = 0
    for row in plan_rows:
        t0 = cursor_frames / FPS
        dur = row["n_frames"] / FPS
        cursor_frames += row["n_frames"]
        points = {min(t0 + CLIP_EDGE_OFFSET_SEC, t0 + dur / 2),
                  t0 + dur / 2,
                  max(t0 + dur - CLIP_EDGE_OFFSET_SEC, t0 + dur / 2)}
        for t in sorted(points):
            yield t, row


def sample_times_fallback(duration_sec, fps=1.0):
    t = 0.0
    while t < duration_sec:
        yield t, None
        t += 1.0 / fps


def probe_duration(video_path):
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format",
         str(video_path)], capture_output=True)
    if out.returncode != 0:
        raise SystemExit(f"ffprobe failed on {video_path}")
    return float(json.loads(out.stdout)["format"]["duration"])


def attribution(row):
    if row is None:
        return "(no plan — unattributed)"
    return (f"{row.get('clip_source_name', '?')} scene={row.get('scene_index', '?')} "
            f"clip_start={row.get('clip_start', '?')}s mode={row.get('render_mode', '?')}")


def update_flags(confirmed_hits):
    """Write render_check flag files covering the offending clips' source ranges.

    Conservative attribution: the text appeared somewhere inside the clip's
    used source range [clip_start, clip_start + src_duration], so flag the
    whole range (it's small — one phrase's worth of source footage).
    """
    by_source = {}
    for t, row, desc in confirmed_hits:
        if row is None:
            continue
        key = row["clip_source_name"]
        start = int(row["clip_start"])
        end = int(row["clip_start"] + row.get("src_duration", row["clip_duration"])) + 1
        entry = by_source.setdefault(key, {"source": row.get("clip_source", key),
                                           "seconds": {}})
        for s in range(start, end + 1):
            entry["seconds"][s] = desc

    written = []
    for name, entry in by_source.items():
        path = TEXT_FLAGS_DIR / f"{name}.render_check.text_flags.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {"video": name, "source": entry["source"], "status": "complete",
                    "scan_method": "render_check", "flags": {}}
        for s, desc in entry["seconds"].items():
            data["flags"][str(s)] = {
                "time_sec": float(s), "has_english_text": True,
                "description": desc[:120], "sample_type": "render_check",
            }
        data["text_frame_count"] = sum(
            1 for v in data["flags"].values() if v.get("has_english_text"))
        data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        TEXT_FLAGS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        written.append(path.name)
    return written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("render", type=Path, help="rendered .mp4 to check")
    p.add_argument("--plan", type=Path, default=None,
                   help="selection_plan_v2.json for clip-level sampling + attribution")
    p.add_argument("--fps", type=float, default=1.0,
                   help="fallback sampling rate when no plan is given (default 1/s)")
    p.add_argument("--update-flags", action="store_true",
                   help="feed confirmed hits back into text_flags/ (render_check files)")
    args = p.parse_args()

    if not args.render.exists():
        raise SystemExit(f"Render not found: {args.render}")

    if args.plan:
        rows = json.loads(args.plan.read_text())
        samples = list(sample_times_from_plan(rows))
        print(f"  Plan mode: {len(rows)} clips → {len(samples)} sample frames")
    else:
        dur = probe_duration(args.render)
        samples = list(sample_times_fallback(dur, args.fps))
        print(f"  Fallback mode: {dur:.0f}s render → {len(samples)} sample frames @ {args.fps}/s")

    print("  Loading EAST...", end=" ", flush=True)
    get_east_net()
    print("ready")

    east_hits = []      # (t, row, n_regions)
    confirmed = []      # (t, row, ocr_description)
    t_start = time.time()
    review_dir = args.render.parent / f"{args.render.stem}_text_check"

    with tempfile.TemporaryDirectory(prefix="render_text_check_") as tmp:
        tmp = Path(tmp)
        for i, (t, row) in enumerate(samples):
            frame = tmp / f"f_{i:06d}.jpg"
            if not extract_frame(args.render, t, frame):
                continue
            has_text, n_regions, _ = detect_text_east(frame)
            if not has_text:
                frame.unlink(missing_ok=True)
                continue
            east_hits.append((t, row, n_regions))
            ok, desc, _ = verify_text_ocr(frame)
            if ok:
                confirmed.append((t, row, desc))
                review_dir.mkdir(parents=True, exist_ok=True)
                # shutil.move, not Path.rename: the temp dir and the render's
                # review dir are often on different filesystems (/var/folders
                # vs the NAS-backed sets/), and rename() can't cross devices.
                shutil.move(str(frame), str(review_dir / f"confirmed_{t:08.1f}s.jpg"))
                print(f"  ✗ CONFIRMED @ {t:8.1f}s  \"{desc[:60]}\"  ← {attribution(row)}")
            frame.unlink(missing_ok=True)
            if (i + 1) % 200 == 0:
                print(f"    ... {i + 1}/{len(samples)} frames "
                      f"({time.time() - t_start:.0f}s)", flush=True)

    print(f"\n  Checked {len(samples)} frames in {time.time() - t_start:.0f}s")
    print(f"  EAST hits: {len(east_hits)}  |  OCR-confirmed: {len(confirmed)}")
    for t, row, n in east_hits:
        if not any(ct == t for ct, _, _ in confirmed):
            print(f"  ~ EAST-only @ {t:8.1f}s ({n} regions, OCR rejected)  ← {attribution(row)}")

    if confirmed and args.update_flags:
        written = update_flags(confirmed)
        print(f"  Flags updated: {', '.join(written)}")
    elif confirmed:
        print("  (re-run with --update-flags to feed these back into text_flags/)")

    if confirmed:
        print(f"  Flagged frames saved for review: {review_dir}/")
        print("\n  RESULT: TEXT FOUND — render needs attention")
        return 1
    print("\n  RESULT: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
