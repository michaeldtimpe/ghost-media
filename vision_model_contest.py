#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  VISION MODEL CONTEST
  Compare Ollama vision models for media analysis semantic enrichment
  Target:  M1 MacBook Pro Max · 64 GB RAM
═══════════════════════════════════════════════════════════════════════════════

Extracts frames from source videos at scene boundaries, sends each frame
to multiple Ollama vision models one at a time, and logs all responses
for side-by-side comparison.

Usage:
    python vision_model_contest.py                      # run full contest
    python vision_model_contest.py --status             # show progress
    python vision_model_contest.py --pause              # pause after current frame
    python vision_model_contest.py --resume             # resume paused contest
    python vision_model_contest.py --compare            # show side-by-side results
    python vision_model_contest.py --compare --video "Tame Impala"  # filter
    python vision_model_contest.py --source /path/to/videos

Requires: ollama, ffmpeg
"""

import json
import os
import sys
import time
import base64
import argparse
import subprocess
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timedelta

# ─── Configuration ──────────────────────────────────────────────────────────

MODELS = [
    "llama3.2-vision:11b",   # ~7GB  — Meta baseline
    "qwen2.5vl:7b",          # ~5GB  — Alibaba, video-native
    "gemma3:12b",             # ~8GB  — Google latest
    "minicpm-v:8b",           # ~5GB  — efficient, strong descriptions
    "qwen2.5vl:32b",         # ~18GB — big step up in quality
    "llava:34b",              # ~20GB — highest quality scene description
]

TEST_VIDEOS = [
    "Tame Impala - Nangs (Music Video).mp4",                    # music video, 103s
    "EXC3_CM3.mp4",                                              # motion graphics, 72s
    "PAPERS.mp4",                                                # abstract/artistic, 171s
    "BIRDMASK Visuals.mp4",                                      # VJ visuals, longer
    "Petit Biscuit - Sunset Lover (Official Music Video).mp4",   # cinematic music video
]

FRAMES_PER_VIDEO = 6
OLLAMA_API = "http://localhost:11434"

import media_paths
DEFAULT_SOURCE = str(media_paths.FOOTAGE_ROOT / "raw")
ANALYSIS_DIR = Path(__file__).parent
CONTEST_DIR = ANALYSIS_DIR / "vision_contest"
RESULTS_FILE = CONTEST_DIR / "results.json"
STATE_FILE = CONTEST_DIR / "state.json"
FRAMES_DIR = CONTEST_DIR / "frames"

VISION_PROMPT = """Analyze this video frame for use in an LLM-driven visual generation system. Provide a structured analysis:

1. **Visual Description**: Describe exactly what you see — objects, shapes, patterns, textures, people, environments. Be specific and concrete.

2. **Color Palette**: Describe the dominant colors, their relationships (complementary, analogous, monochromatic), and the overall color temperature (warm/cool/neutral).

3. **Composition & Movement**: Describe the spatial layout, depth, focal points. If motion is implied, describe the direction and type (pan, zoom, rotation, drift, pulse, etc.).

4. **Visual Style**: Classify the aesthetic — e.g., photorealistic, abstract, geometric, organic, psychedelic, minimalist, glitch, cinematic, etc.

5. **Mood & Energy**: What emotional tone does this frame convey? What energy level (calm, building, intense, chaotic)?

6. **Transition Potential**: How might this frame transition to/from adjacent content? What visual elements could be used as transition anchors?

Be concise but precise. Use terminology a visual artist or VJ would understand."""


# ─── Formatting Helpers ─────────────────────────────────────────────────────

def fmt_time(seconds):
    """Format seconds as H:MM:SS or M:SS."""
    if seconds < 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bar(pct, width=30):
    """Render a progress bar."""
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


# ─── State / Results ────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "status": "not_started",
        "started_at": None,
        "models_pulled": [],
        "completed": [],
        "paused": False,
        "errors": [],
        "model_timings": {},
    }


def save_state(state):
    CONTEST_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)


def load_results():
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return {"models": {}, "timing": {}}


def save_results(results):
    CONTEST_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    tmp.rename(RESULTS_FILE)


# ─── Frame Extraction ──────────────────────────────────────────────────────

def get_scene_times(video_name):
    """Get scene boundary timestamps from existing analysis data."""
    candidates = list(ANALYSIS_DIR.glob("*.analysis.json"))
    stem = Path(video_name).stem
    for c in candidates:
        if "pass1" in c.name:
            continue
        base = c.name.replace(".analysis.json", "")
        if stem[:20] in base or base[:20] in stem:
            try:
                data = json.loads(c.read_text())
                scenes = data.get("scenes", {})
                scene_list = scenes.get("scenes", []) if isinstance(scenes, dict) else scenes
                times = []
                for s in scene_list:
                    start = s.get("start_sec", 0)
                    end = s.get("end_sec", start)
                    mid = (start + end) / 2
                    times.append({"time": start, "label": f"scene_{s.get('scene_index', 0)}_start"})
                    if end - start > 3:
                        times.append({"time": mid, "label": f"scene_{s.get('scene_index', 0)}_mid"})
                return times, data.get("metadata", {})
            except Exception as e:
                print(f"  Warning: couldn't parse {c.name}: {e}")
    return [], {}


def extract_frames(video_path, video_name):
    """Extract frames at scene boundaries."""
    video_dir = FRAMES_DIR / hashlib.md5(video_name.encode()).hexdigest()[:12]
    video_dir.mkdir(parents=True, exist_ok=True)

    scene_times, metadata = get_scene_times(video_name)
    if not scene_times:
        duration = metadata.get("duration_sec", 60)
        scene_times = [
            {"time": i * duration / (FRAMES_PER_VIDEO + 1), "label": f"even_{i}"}
            for i in range(1, FRAMES_PER_VIDEO + 1)
        ]

    step = max(1, len(scene_times) // FRAMES_PER_VIDEO)
    selected = scene_times[::step][:FRAMES_PER_VIDEO]

    frames = []
    for st in selected:
        t, label = st["time"], st["label"]
        out_path = video_dir / f"{label}_{t:.2f}s.jpg"

        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append({"path": str(out_path), "time": t, "label": label})
            continue

        try:
            subprocess.run(
                ["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
                 "-frames:v", "1", "-q:v", "2", str(out_path)],
                capture_output=True, timeout=30
            )
            if out_path.exists() and out_path.stat().st_size > 0:
                frames.append({"path": str(out_path), "time": t, "label": label})
            else:
                print(f"    ⚠ ffmpeg produced no output @ {t:.1f}s")
        except subprocess.TimeoutExpired:
            print(f"    ⚠ ffmpeg timed out @ {t:.1f}s")
        except FileNotFoundError:
            print("ERROR: ffmpeg not found. Install with: brew install ffmpeg")
            sys.exit(1)

    return frames


# ─── Ollama Interface ──────────────────────────────────────────────────────

def check_ollama():
    """Verify ollama is running, return set of installed model names."""
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return {m["name"] for m in data.get("models", [])}
    except urllib.error.URLError:
        print("ERROR: ollama is not running. Start it with: ollama serve")
        sys.exit(1)


def pull_model(model_name):
    """Pull a model if not already available."""
    print(f"    Pulling {model_name}...", end=" ", flush=True)
    result = subprocess.run(
        ["ollama", "pull", model_name],
        capture_output=False, text=True, timeout=1800
    )
    if result.returncode == 0:
        print("✓")
        return True
    print("✗ FAILED")
    return False


def query_model(model_name, image_path, prompt):
    """Send image + prompt to Ollama via HTTP API. Returns (response, elapsed, error)."""
    start = time.time()
    try:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = json.dumps({
            "model": model_name,
            "prompt": prompt,
            "images": [img_b64],
            "stream": False,
            "options": {"num_predict": 1024},
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_API}/api/generate", data=payload,
            headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read())
            elapsed = time.time() - start
            return data.get("response", "").strip(), elapsed, None

    except Exception as e:
        return None, time.time() - start, str(e)[:200]


# ─── Live Progress ──────────────────────────────────────────────────────────

def print_live_header(model, model_idx, total_models, total_done, total_all):
    """Print a header when switching to a new model."""
    pct = total_done / total_all if total_all else 0
    print()
    print(f"  {'━' * 68}")
    print(f"  Model {model_idx}/{total_models}: {model}")
    print(f"  Overall: {fmt_bar(pct)} {total_done}/{total_all} ({pct:.0%})")
    print(f"  {'━' * 68}")


def print_frame_result(frame_idx, total_frames, video, label, elapsed, error,
                        model_elapsed, model_avg, total_done, total_all, total_elapsed):
    """Print one-line result for a completed frame."""
    pct = frame_idx / total_frames if total_frames else 0
    bar = fmt_bar(pct, width=20)

    # ETA based on overall average
    overall_avg = total_elapsed / total_done if total_done > 0 else 0
    remaining = overall_avg * (total_all - total_done)

    status = f"✗ {error[:40]}" if error else f"✓ {elapsed:.0f}s"
    video_short = Path(video).stem[:25]

    print(f"    {bar} {frame_idx}/{total_frames}  "
          f"{video_short:<25s} {label:<20s} {status:<15s} "
          f"avg:{model_avg:.0f}s  ETA:{fmt_time(remaining)}")


# ─── Status Display ─────────────────────────────────────────────────────────

def show_status():
    """Show contest progress."""
    state = load_state()
    results = load_results()

    completed = state.get("completed", [])
    total_done = len(completed)
    frames_per_model = len(TEST_VIDEOS) * FRAMES_PER_VIDEO
    total_all = len(MODELS) * frames_per_model

    pct = total_done / total_all if total_all else 0

    print()
    print(f"  ╔{'═' * 66}╗")
    print(f"  ║{'VISION MODEL CONTEST':^66}║")
    print(f"  ╠{'═' * 66}╣")
    print(f"  ║  Status:  {state['status']:<55}║")
    print(f"  ║  Paused:  {'YES' if state.get('paused') else 'NO':<55}║")
    print(f"  ║  Overall: {fmt_bar(pct)} {total_done}/{total_all} ({pct:.0%}){' ' * max(0, 20 - len(f'{total_done}/{total_all}'))}║")

    if state.get("started_at"):
        started = datetime.fromisoformat(state["started_at"])
        elapsed = (datetime.now() - started).total_seconds()
        print(f"  ║  Elapsed: {fmt_time(elapsed):<55}║")
        if total_done > 0:
            avg = elapsed / total_done
            remaining = avg * (total_all - total_done)
            print(f"  ║  ETA:     {fmt_time(remaining):<55}║")

    print(f"  ╠{'═' * 66}╣")

    for i, model in enumerate(MODELS, 1):
        model_done = sum(1 for c in completed if c["model"] == model)
        model_pct = model_done / frames_per_model if frames_per_model else 0
        timings = results.get("timing", {}).get(model, [])
        avg_t = f"{sum(timings)/len(timings):.0f}s" if timings else "—"
        icon = "✓" if model_done == frames_per_model else "▶" if model_done > 0 else "·"
        bar = fmt_bar(model_pct, width=15)

        print(f"  ║  {icon} {model:<28s} {bar} {model_done:>2}/{frames_per_model}"
              f"  avg:{avg_t:>5s}  ║")

    errs = state.get("errors", [])
    if errs:
        print(f"  ╠{'═' * 66}╣")
        print(f"  ║  Errors: {len(errs):<56}║")
        for e in errs[-3:]:
            msg = f"  {e['model']}: {e.get('error','')[:45]}"
            print(f"  ║  {msg:<64}║")

    print(f"  ╚{'═' * 66}╝")
    print()


# ─── Comparison Display ────────────────────────────────────────────────────

def show_comparison(video_filter=None):
    """Side-by-side comparison of model outputs."""
    results = load_results()
    models_data = results.get("models", {})

    if not any(models_data.values()):
        print("No results yet. Run the contest first.")
        return

    all_keys = set()
    for model, videos in models_data.items():
        for video, frames in videos.items():
            for frame_key in frames:
                all_keys.add((video, frame_key))

    for video, frame_key in sorted(all_keys):
        if video_filter and video_filter.lower() not in video.lower():
            continue

        print(f"\n{'═' * 90}")
        print(f"  VIDEO: {video}")
        print(f"  FRAME: {frame_key}")
        print(f"{'═' * 90}")

        for model in MODELS:
            entry = models_data.get(model, {}).get(video, {}).get(frame_key)
            if not entry:
                continue

            timing = entry.get("elapsed_sec", 0)
            err = entry.get("error")
            print(f"\n  ┌─ {model} ({timing:.1f}s) ─────────────────────────")
            if err:
                print(f"  │ ERROR: {err}")
            else:
                for line in (entry.get("response") or "").split("\n"):
                    print(f"  │ {line}")
            print(f"  └{'─' * 70}")

    # Summary
    print(f"\n{'═' * 90}")
    print(f"  {'Model':<35s} {'Avg':>7s} {'Min':>7s} {'Max':>7s} {'n':>4s}")
    print(f"  {'─'*35} {'─'*7} {'─'*7} {'─'*7} {'─'*4}")
    for model in MODELS:
        timings = results.get("timing", {}).get(model, [])
        if timings:
            avg = sum(timings) / len(timings)
            print(f"  {model:<35s} {avg:>6.0f}s {min(timings):>6.0f}s {max(timings):>6.0f}s {len(timings):>4}")
        else:
            print(f"  {model:<35s} {'—':>7s} {'—':>7s} {'—':>7s} {'—':>4s}")
    print(f"{'═' * 90}\n")


# ─── Main Contest ──────────────────────────────────────────────────────────

def run_contest(source_dir):
    """Run each model sequentially, one frame at a time."""
    source = Path(source_dir)
    if not source.exists():
        print(f"\n  ERROR: Source not found: {source}")
        print(f"  Mount the archive drive and try again.\n")
        sys.exit(1)

    CONTEST_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    results = load_results()

    if state.get("paused"):
        print("  Contest is paused. Use --resume to continue.")
        return

    # ── Ollama check ──
    print("\n  Checking ollama...", end=" ")
    installed = check_ollama()
    print(f"✓ ({len(installed)} models installed)")

    # ── Pull missing models ──
    print(f"\n  Preparing {len(MODELS)} models:")
    active_models = []
    for model in MODELS:
        if model in state.get("models_pulled", []):
            print(f"    {model} — ready")
            active_models.append(model)
            continue
        matched = any(model in m or model.split(":")[0] in m for m in installed)
        if matched:
            print(f"    {model} — already available")
            state.setdefault("models_pulled", []).append(model)
            save_state(state)
            active_models.append(model)
            continue
        if pull_model(model):
            state.setdefault("models_pulled", []).append(model)
            save_state(state)
            active_models.append(model)
        else:
            print(f"    ⚠ {model} — skipping (pull failed)")

    # ── Locate videos ──
    print(f"\n  Locating test videos:")
    available_files = {f.name: f for f in source.iterdir()
                       if f.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.webm'}}
    video_paths = {}
    for vname in TEST_VIDEOS:
        if vname in available_files:
            video_paths[vname] = available_files[vname]
            print(f"    ✓ {vname}")
        else:
            stem = Path(vname).stem[:25]
            matches = [f for f in available_files if stem[:15] in f]
            if matches:
                video_paths[vname] = available_files[matches[0]]
                print(f"    ✓ {vname} → {matches[0]}")
            else:
                print(f"    ✗ {vname} — not found")

    if not video_paths:
        print("\n  ERROR: No test videos found.\n")
        sys.exit(1)

    # ── Extract frames ──
    print(f"\n  Extracting frames:")
    video_frames = {}
    for vname, vpath in video_paths.items():
        frames = extract_frames(vpath, vname)
        if frames:
            video_frames[vname] = frames
            print(f"    {vname}: {len(frames)} frames")

    if not video_frames:
        print("\n  ERROR: No frames extracted.\n")
        sys.exit(1)

    # ── Build work queue ──
    all_frames = []
    for vname, frames in video_frames.items():
        for frame in frames:
            all_frames.append((vname, frame))

    completed_keys = {(c["model"], c["video"], c["frame"]) for c in state.get("completed", [])}
    frames_per_model = len(all_frames)
    total_all = len(active_models) * frames_per_model
    total_done = len(completed_keys)

    state["status"] = "running"
    state["started_at"] = state.get("started_at") or datetime.now().isoformat()
    save_state(state)

    contest_start = time.time()

    print(f"\n  {'═' * 68}")
    print(f"  VISION MODEL CONTEST")
    print(f"  {len(active_models)} models × {len(video_frames)} videos × {FRAMES_PER_VIDEO} frames"
          f" = {total_all} tasks ({total_done} already done)")
    print(f"  ~{fmt_time(68 * (total_all - total_done))} estimated"
          f" @ ~68s/frame sequential")
    print(f"  {'═' * 68}")

    # ── Process each model ──
    for model_idx, model in enumerate(active_models, 1):
        model_work = [(vname, frame) for vname, frame in all_frames
                      if (model, vname, frame["label"]) not in completed_keys]

        if not model_work:
            print(f"\n  ━━ Model {model_idx}/{len(active_models)}: {model} — already complete ━━")
            continue

        model_timings = []
        model_start = time.time()

        print_live_header(model, model_idx, len(active_models), total_done, total_all)

        for frame_idx, (vname, frame) in enumerate(model_work, 1):
            # ── Pause check ──
            state = load_state()
            if state.get("paused"):
                print(f"\n  ⏸  PAUSED at {total_done}/{total_all}. Use --resume to continue.\n")
                state["status"] = "paused"
                save_state(state)
                return

            # ── Query ──
            response, elapsed, error = query_model(model, frame["path"], VISION_PROMPT)
            model_timings.append(elapsed)
            total_done += 1

            # ── Save result ──
            results.setdefault("models", {}).setdefault(model, {}).setdefault(vname, {})
            results["models"][model][vname][frame["label"]] = {
                "response": response,
                "error": error,
                "elapsed_sec": elapsed,
                "timestamp": datetime.now().isoformat(),
                "frame_time_sec": frame["time"],
                "frame_path": frame["path"],
            }
            results.setdefault("timing", {}).setdefault(model, []).append(elapsed)
            save_results(results)

            # ── Save state ──
            state = load_state()
            state["completed"].append({"model": model, "video": vname, "frame": frame["label"]})
            if error:
                state["errors"].append({"model": model, "video": vname,
                                        "frame": frame["label"], "error": error})
            save_state(state)
            completed_keys.add((model, vname, frame["label"]))

            # ── Print progress ──
            model_avg = sum(model_timings) / len(model_timings)
            total_elapsed = time.time() - contest_start + (
                sum(t for ts in results.get("timing", {}).values() for t in ts)
                - sum(model_timings)  # don't double count current model
            ) if total_done == len(model_timings) else time.time() - (
                datetime.fromisoformat(state["started_at"]).timestamp()
            ) if state.get("started_at") else time.time() - contest_start

            print_frame_result(
                frame_idx, len(model_work), vname, frame["label"],
                elapsed, error, time.time() - model_start, model_avg,
                total_done, total_all, time.time() - contest_start
            )

        # ── Model summary ──
        model_total = time.time() - model_start
        model_avg = sum(model_timings) / len(model_timings) if model_timings else 0
        print(f"\n    Model complete: {len(model_timings)} frames in {fmt_time(model_total)}"
              f"  (avg {model_avg:.0f}s/frame)")

    # ── Done ──
    total_time = time.time() - contest_start
    print(f"\n  {'═' * 68}")
    print(f"  CONTEST COMPLETE")
    print(f"  {total_done} frames in {fmt_time(total_time)}")
    print(f"  Results:  {RESULTS_FILE}")
    print(f"  Compare:  python {Path(__file__).name} --compare")
    print(f"  {'═' * 68}\n")

    state = load_state()
    state["status"] = "complete"
    save_state(state)


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Vision Model Contest — compare Ollama vision models"
    )
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="Path to source video directory")
    parser.add_argument("--status", action="store_true",
                        help="Show current progress")
    parser.add_argument("--pause", action="store_true",
                        help="Pause after current frame")
    parser.add_argument("--resume", action="store_true",
                        help="Resume paused contest")
    parser.add_argument("--compare", action="store_true",
                        help="Show side-by-side results")
    parser.add_argument("--video", default=None,
                        help="Filter --compare to a video (substring)")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Override model list")
    parser.add_argument("--reset", action="store_true",
                        help="Clear all data and start fresh")

    args = parser.parse_args()

    if args.models:
        global MODELS
        MODELS = args.models

    if args.reset:
        if CONTEST_DIR.exists():
            import shutil
            shutil.rmtree(CONTEST_DIR)
            print("  Contest data cleared.")
        return

    if args.status:
        show_status()
        return

    if args.pause:
        state = load_state()
        state["paused"] = True
        save_state(state)
        print("  Pause signal sent. Will pause after current frame.")
        return

    if args.resume:
        state = load_state()
        state["paused"] = False
        state["status"] = "resuming"
        save_state(state)
        print("  Resuming...")
        run_contest(args.source)
        return

    if args.compare:
        show_comparison(args.video)
        return

    run_contest(args.source)


if __name__ == "__main__":
    main()
