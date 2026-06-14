#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SEMANTIC ENRICHMENT PIPELINE
  Add visual descriptions to existing media analysis using qwen2.5vl:7b
  Target:  M1 MacBook Pro Max · 64 GB RAM
═══════════════════════════════════════════════════════════════════════════════

Reads existing .analysis.json files, extracts frames at scene boundaries
from the source videos, and adds semantic descriptions via Ollama vision.

Sampling: ~1 frame per 30s of content (min 4, max 30 per video).
Enriched data is written to .enriched.json alongside the originals.

Usage:
    python enrich_analyses.py                     # run full pipeline
    python enrich_analyses.py --status            # show progress
    python enrich_analyses.py --pause             # pause after current frame
    python enrich_analyses.py --resume            # resume
    python enrich_analyses.py --source /path      # custom video source
    python enrich_analyses.py --dry-run           # show plan without running
    python enrich_analyses.py --video "Tame"      # only process matching videos

Requires: ollama (with qwen2.5vl:7b), ffmpeg
"""

import json
import os
import sys
import time
import argparse
import subprocess
import hashlib
from pathlib import Path
from datetime import datetime, timedelta

from vision_schema import SCENE_PROMPT, normalize_analysis
from vision_backends import get_backend

# ─── Configuration ──────────────────────────────────────────────────────────

MODEL = "qwen2.5vl:7b"
OLLAMA_API = "http://localhost:11434"

# Set in main() from --backend/--model (or $GHOST_VISION_BACKEND); defaults to ollama.
BACKEND = None

import media_paths
DEFAULT_SOURCE = str(media_paths.FOOTAGE_ROOT / "raw")
ANALYSIS_DIR = Path(__file__).parent
ENRICHMENT_DIR = ANALYSIS_DIR / "enriched"
STATE_FILE = ENRICHMENT_DIR / "state.json"
FRAMES_DIR = ENRICHMENT_DIR / "frames"

# Sampling: ~1 frame per SAMPLE_INTERVAL seconds, bounded by min/max
SAMPLE_INTERVAL_SEC = 30
MIN_FRAMES_PER_VIDEO = 4
MAX_FRAMES_PER_VIDEO = 30

# SCENE_PROMPT is imported from vision_schema (single source of truth).


# ─── Formatting ─────────────────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_bar(pct, width=25):
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


# ─── State ──────────────────────────────────────────────────────────────────

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "status": "not_started",
        "started_at": None,
        "completed_videos": [],
        "completed_frames": 0,
        "total_frames": 0,
        "paused": False,
        "errors": [],
        "timings": [],
    }


def save_state(state):
    ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.rename(STATE_FILE)


# ─── Analysis File Discovery ───────────────────────────────────────────────

def find_analysis_files(video_filter=None):
    """Find all .analysis.json files (excluding pass1 and meta files)."""
    files = []
    for f in sorted(ANALYSIS_DIR.glob("*.analysis.json")):
        if "pass1" in f.name or f.name.startswith("_"):
            continue
        if video_filter and video_filter.lower() not in f.name.lower():
            continue
        files.append(f)
    return files


def load_analysis(path):
    """Load an analysis JSON file."""
    return json.loads(path.read_text())


def find_source_video(analysis_data, analysis_path, source_dir):
    """Find the source video file for an analysis."""
    source = Path(source_dir)
    # Try the original filename from the analysis
    file_info = analysis_data.get("file", "")
    if isinstance(file_info, dict):
        original_file = file_info.get("path") or file_info.get("name", "")
    else:
        original_file = str(file_info)
    if original_file:
        # Try full path first
        full = Path(original_file)
        if full.exists():
            return full
        # Try just the filename in the source dir
        candidate = source / full.name
        if candidate.exists():
            return candidate

    # Fuzzy match from analysis filename
    base = analysis_path.stem.replace(".analysis", "")
    for ext in ['.mp4', '.mov', '.mkv', '.avi', '.webm']:
        for f in source.iterdir():
            if f.suffix.lower() == ext:
                if base[:20] in f.stem or f.stem[:20] in base:
                    return f
    return None


# ─── Frame Sampling ────────────────────────────────────────────────────────

def plan_frame_samples(analysis_data):
    """Decide which frames to sample based on scene boundaries and video length."""
    metadata = analysis_data.get("metadata", {})
    duration = metadata.get("duration_sec", 0)
    scenes = analysis_data.get("scenes", {})
    scene_list = scenes.get("scenes", []) if isinstance(scenes, dict) else scenes

    # Target frame count based on duration
    target = max(MIN_FRAMES_PER_VIDEO, min(MAX_FRAMES_PER_VIDEO, int(duration / SAMPLE_INTERVAL_SEC)))

    # If fewer scenes than target, use all scene starts + midpoints
    if len(scene_list) <= target:
        samples = []
        for s in scene_list:
            start = s.get("start_sec", 0)
            end = s.get("end_sec", start)
            mid = (start + end) / 2
            samples.append({
                "time": start,
                "label": f"scene_{s.get('scene_index', 0)}_start",
                "scene_index": s.get("scene_index", 0),
                "type": "scene_start"
            })
            if end - start > 5 and len(samples) < target:
                samples.append({
                    "time": mid,
                    "label": f"scene_{s.get('scene_index', 0)}_mid",
                    "scene_index": s.get("scene_index", 0),
                    "type": "scene_mid"
                })
        # Cap at target
        if len(samples) > target:
            step = max(1, len(samples) // target)
            samples = samples[::step][:target]
        return samples

    # More scenes than target: evenly distribute across scene boundaries
    step = max(1, len(scene_list) // target)
    selected_scenes = scene_list[::step][:target]

    samples = []
    for s in selected_scenes:
        start = s.get("start_sec", 0)
        samples.append({
            "time": start,
            "label": f"scene_{s.get('scene_index', 0)}_start",
            "scene_index": s.get("scene_index", 0),
            "type": "scene_start"
        })

    return samples[:target]


# ─── Frame Extraction ──────────────────────────────────────────────────────

def extract_frame(video_path, time_sec, output_path):
    """Extract a single frame from a video."""
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


# ─── Vision (backend-agnostic) ──────────────────────────────────────────────

def query_vision(image_path):
    """Send a frame to the active backend and return (parsed, raw, elapsed, error)."""
    raw, elapsed, error = BACKEND.analyze_frame(str(image_path), SCENE_PROMPT)
    if error:
        return None, raw, elapsed, error
    parsed = try_parse_json(raw)
    return parsed, raw, elapsed, None


def try_parse_json(text):
    """Try to extract JSON from a model response."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in markdown
    for marker in ["```json", "```"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.index("```", start) if "```" in text[start:] else len(text)
            try:
                return json.loads(text[start:end].strip())
            except json.JSONDecodeError:
                pass

    # Try to find { ... } block
    brace_start = text.find("{")
    brace_end = text.rfind("}") + 1
    if brace_start >= 0 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end])
        except json.JSONDecodeError:
            pass

    return None


# ─── Enrichment Output ─────────────────────────────────────────────────────

def build_enriched_output(analysis_data, frame_results):
    """Merge semantic descriptions into the analysis structure."""
    enriched = dict(analysis_data)
    enriched["schema_version"] = "2.0.0"
    enriched["enrichment"] = {
        "backend": BACKEND.name if BACKEND else "ollama",
        "model": BACKEND.model if BACKEND else MODEL,
        "timestamp": datetime.now().isoformat(),
        "frame_count": len(frame_results),
        "sample_interval_sec": SAMPLE_INTERVAL_SEC,
    }

    # Add per-scene enrichment
    scenes = enriched.get("scenes", {})
    scene_list = scenes.get("scenes", []) if isinstance(scenes, dict) else scenes

    # Index frame results by scene_index
    results_by_scene = {}
    for fr in frame_results:
        idx = fr.get("scene_index")
        if idx is not None:
            results_by_scene.setdefault(idx, []).append(fr)

    # Enrich each scene that has results
    for scene in scene_list:
        idx = scene.get("scene_index")
        if idx in results_by_scene:
            descs = results_by_scene[idx]
            # Use the first parsed result for this scene (validated against the schema)
            for d in descs:
                if d.get("parsed"):
                    clean, _ = normalize_analysis(d["parsed"])
                    scene["semantic"] = clean
                    break
            else:
                # Fallback: store raw text
                scene["semantic_raw"] = descs[0].get("raw", "")

    # Also store all frame analyses in a flat list for easy access
    enriched["frame_analyses"] = []
    for fr in frame_results:
        entry = {
            "time_sec": fr["time"],
            "label": fr["label"],
            "scene_index": fr.get("scene_index"),
        }
        if fr.get("parsed"):
            clean, _ = normalize_analysis(fr["parsed"])
            entry["analysis"] = clean
        elif fr.get("raw"):
            entry["analysis_raw"] = fr["raw"]
        if fr.get("error"):
            entry["error"] = fr["error"]
        if fr.get("provenance"):
            entry["_provenance"] = fr["provenance"]
        entry["inference_time_sec"] = fr.get("elapsed", 0)
        enriched["frame_analyses"].append(entry)

    return enriched


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def run_pipeline(source_dir, video_filter=None, dry_run=False, limit=None):
    source = Path(source_dir)
    if not source.exists():
        print(f"\n  ERROR: Source not found: {source}")
        print(f"  Mount the archive drive and try again.\n")
        sys.exit(1)

    ENRICHMENT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    state = load_state()
    if state.get("paused") and not dry_run:
        print("  Pipeline is paused. Use --resume to continue.")
        return

    # ── Backend check ──
    if not dry_run:
        print(f"\n  Checking backend {BACKEND.label}...", end=" ")
        ok, msg = BACKEND.health_check()
        if ok:
            print(f"✓ {msg}")
        else:
            print(f"✗ {msg}")
            sys.exit(1)

    # ── Discover analysis files ──
    analysis_files = find_analysis_files(video_filter)
    print(f"\n  Found {len(analysis_files)} analysis files")

    # ── Plan the work ──
    work_plan = []
    total_frames = 0
    already_done = set(state.get("completed_videos", []))

    for af in analysis_files:
        analysis = load_analysis(af)
        video_path = find_source_video(analysis, af, source)
        samples = plan_frame_samples(analysis)
        total_frames += len(samples)

        enriched_path = ENRICHMENT_DIR / af.name.replace(".analysis.json", ".enriched.json")
        done = af.name in already_done

        work_plan.append({
            "analysis_path": af,
            "analysis_data": analysis,
            "video_path": video_path,
            "samples": samples,
            "enriched_path": enriched_path,
            "done": done,
        })

    remaining = [w for w in work_plan if not w["done"]]
    if limit:
        remaining = remaining[:limit]
    remaining_frames = sum(len(w["samples"]) for w in remaining)

    print(f"  Total frames to analyze: {total_frames}")
    print(f"  Already complete: {len(already_done)} videos")
    print(f"  Remaining: {len(remaining)} videos, {remaining_frames} frames")
    print(f"  Estimated time: {fmt_time(remaining_frames * 55)}")

    if dry_run:
        print(f"\n  {'─' * 68}")
        print(f"  DRY RUN — planned work:")
        print(f"  {'─' * 68}")
        for w in work_plan:
            name = w["analysis_path"].name[:50]
            n = len(w["samples"])
            found = "✓" if w["video_path"] else "✗ VIDEO NOT FOUND"
            done = " (done)" if w["done"] else ""
            print(f"    {name:<52s} {n:>3} frames  {found}{done}")
        missing = [w for w in work_plan if not w["video_path"]]
        if missing:
            print(f"\n  ⚠ {len(missing)} videos missing from source directory")
        return

    # ── Run ──
    state["status"] = "running"
    state["started_at"] = state.get("started_at") or datetime.now().isoformat()
    state["total_frames"] = total_frames
    save_state(state)

    pipeline_start = time.time()
    global_frame = state.get("completed_frames", 0)

    print(f"\n  {'═' * 68}")
    print(f"  SEMANTIC ENRICHMENT PIPELINE")
    print(f"  Backend: {BACKEND.label}")
    print(f"  {len(remaining)} videos, {remaining_frames} frames")
    print(f"  {'═' * 68}")

    for vid_idx, work in enumerate(remaining, 1):
        af = work["analysis_path"]
        analysis = work["analysis_data"]
        video_path = work["video_path"]
        samples = work["samples"]
        enriched_path = work["enriched_path"]

        video_name = af.name.replace(".analysis.json", "")[:50]
        duration = analysis.get("metadata", {}).get("duration_sec", 0)

        # ── Pause check ──
        state = load_state()
        if state.get("paused"):
            print(f"\n  ⏸  PAUSED. Use --resume to continue.\n")
            state["status"] = "paused"
            save_state(state)
            return

        if not video_path:
            print(f"\n  ✗ {video_name} — source video not found, skipping")
            state = load_state()
            state["completed_videos"].append(af.name)
            state["errors"].append({"video": af.name, "error": "source video not found"})
            save_state(state)
            continue

        # ── Video header ──
        overall_pct = global_frame / total_frames if total_frames else 0
        overall_avg = (time.time() - pipeline_start) / max(1, global_frame) if global_frame > 0 else 55
        overall_eta = overall_avg * (total_frames - global_frame)

        print(f"\n  {'━' * 68}")
        print(f"  [{vid_idx}/{len(remaining)}] {video_name}")
        print(f"  {fmt_time(duration)} duration, {len(samples)} frames to analyze")
        print(f"  Overall: {fmt_bar(overall_pct)} {global_frame}/{total_frames}"
              f"  ETA: {fmt_time(overall_eta)}")
        print(f"  {'━' * 68}")

        # ── Extract & analyze frames ──
        frame_dir = FRAMES_DIR / hashlib.md5(af.name.encode()).hexdigest()[:12]
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_results = []
        video_start = time.time()

        for fi, sample in enumerate(samples, 1):
            # Pause check
            state = load_state()
            if state.get("paused"):
                print(f"\n  ⏸  PAUSED. Use --resume to continue.\n")
                state["status"] = "paused"
                save_state(state)
                return

            t = sample["time"]
            label = sample["label"]
            frame_path = frame_dir / f"{label}_{t:.2f}s.jpg"

            # Extract
            if not extract_frame(video_path, t, frame_path):
                print(f"    {fi:>3}/{len(samples)}  ✗ extract failed @ {t:.1f}s")
                frame_results.append({
                    "time": t, "label": label,
                    "scene_index": sample.get("scene_index"),
                    "error": "frame extraction failed"
                })
                global_frame += 1
                continue

            # Analyze
            parsed, raw, elapsed, error = query_vision(str(frame_path))
            global_frame += 1

            frame_results.append({
                "time": t, "label": label,
                "scene_index": sample.get("scene_index"),
                "parsed": parsed, "raw": raw,
                "elapsed": elapsed, "error": error,
                "provenance": {**BACKEND.provenance(), "elapsed": round(elapsed, 1)},
            })

            # Progress line
            vid_pct = fi / len(samples)
            overall_pct = global_frame / total_frames if total_frames else 0
            overall_avg = (time.time() - pipeline_start) / global_frame
            overall_eta = overall_avg * (total_frames - global_frame)
            status = f"✗ {error[:30]}" if error else ("✓" if parsed else "~")

            print(f"    {fmt_bar(vid_pct, 15)} {fi}/{len(samples)}"
                  f"  @{t:>7.1f}s  {elapsed:>5.1f}s  {status:<8s}"
                  f"  total:{global_frame}/{total_frames}  ETA:{fmt_time(overall_eta)}")

            # Update state timing
            state = load_state()
            state["completed_frames"] = global_frame
            if not error:
                state.setdefault("timings", []).append(elapsed)
            save_state(state)

        # ── Write enriched file ──
        enriched = build_enriched_output(analysis, frame_results)
        enriched_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))

        # ── Video summary ──
        video_elapsed = time.time() - video_start
        successes = sum(1 for fr in frame_results if fr.get("parsed"))
        failures = sum(1 for fr in frame_results if fr.get("error"))
        partial = len(frame_results) - successes - failures

        print(f"\n    Done: {successes} parsed, {partial} raw, {failures} failed"
              f"  ({fmt_time(video_elapsed)})")
        print(f"    → {enriched_path.name}")

        state = load_state()
        state["completed_videos"].append(af.name)
        save_state(state)

    # ── Complete ──
    total_elapsed = time.time() - pipeline_start
    print(f"\n  {'═' * 68}")
    print(f"  ENRICHMENT COMPLETE")
    print(f"  {global_frame} frames in {fmt_time(total_elapsed)}")
    print(f"  Output: {ENRICHMENT_DIR}/")
    print(f"  {'═' * 68}\n")

    state = load_state()
    state["status"] = "complete"
    save_state(state)


# ─── Re-enrich flagged (hard-case re-rate) ──────────────────────────────────

def _scene_is_flagged(scene, quality_by_index, quality_threshold, worklist_indices=None):
    """A scene is a re-enrichment candidate if its semantic failed validation/parse,
    its quality_score is below threshold, OR it appears in an audit worklist
    (e.g. scripts/audit_semantics.py mood/motion contradictions). OR-combined."""
    semantic = scene.get("semantic")
    # Parse failure: no semantic, or only raw text was stored.
    if not semantic or scene.get("semantic_raw"):
        return True, "no/raw semantic"
    # Validation failure: an enum field had to be snapped to "unclear".
    validation = semantic.get("_validation", {}) if isinstance(semantic, dict) else {}
    if any(v.get("normalized") == "unclear" for v in validation.values()):
        return True, "validation→unclear"
    # Low quality.
    q = quality_by_index.get(scene.get("scene_index"))
    if q is not None and q < quality_threshold:
        return True, f"quality {q:.2f}"
    # Audit worklist (semantic contradiction).
    if worklist_indices and scene.get("scene_index") in worklist_indices:
        return True, "audit contradiction"
    return False, ""


def run_reenrich_flagged(source_dir, quality_threshold=0.5, video_filter=None,
                         dry_run=False, worklist_path=None, limit=None):
    """Re-run only flagged scenes through the active backend (ideally claude-cli on
    the subscription) to fix up the cheap local model's hard cases.

    worklist_path: optional audit_semantics_worklist.json — its per-video
    contradictory scene indices are flagged in addition to parse/quality
    failures. limit: stop after this many scenes re-enriched (checkpointing)."""
    source = Path(source_dir)
    if not source.exists():
        print(f"\n  ERROR: Source not found: {source}\n  Mount the archive drive.\n")
        sys.exit(1)

    worklist = {}
    if worklist_path:
        wl = json.loads(Path(worklist_path).read_text())
        worklist = {k: set(v.get("contradictory_scenes", []))
                    for k, v in wl.get("reenrich_worklist", {}).items()}
        print(f"  Worklist: {sum(len(v) for v in worklist.values())} contradiction "
              f"scenes across {len(worklist)} videos")

    if not dry_run:
        ok, msg = BACKEND.health_check()
        print(f"\n  Backend {BACKEND.label}: {'✓ ' + msg if ok else '✗ ' + msg}")
        if not ok:
            sys.exit(1)

    files = sorted(ENRICHMENT_DIR.glob("*.enriched.json"))
    if video_filter:
        files = [f for f in files if video_filter.lower() in f.name.lower()]

    print(f"\n  {'═' * 68}")
    print(f"  RE-ENRICH FLAGGED  (threshold q<{quality_threshold})")
    print(f"  Backend: {BACKEND.label}  ·  {len(files)} enriched files")
    print(f"  {'═' * 68}")

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    total_flagged = total_fixed = 0

    for ef in files:
        data = json.loads(ef.read_text())
        scenes = data.get("scenes", {})
        scene_list = scenes.get("scenes", []) if isinstance(scenes, dict) else scenes

        # Quality sidecar (optional) → scene_index: quality_score
        quality_by_index = {}
        qf = ef.with_name(ef.name.replace(".enriched.json", ".quality.json"))
        if qf.exists():
            for s in json.loads(qf.read_text()).get("scenes", []):
                quality_by_index[s.get("scene_index")] = s.get("quality_score", 1.0)

        name = ef.name.replace(".enriched.json", "")
        wl_indices = worklist.get(name)
        flagged = []
        for scene in scene_list:
            hit, reason = _scene_is_flagged(scene, quality_by_index,
                                            quality_threshold, wl_indices)
            if hit:
                flagged.append((scene, reason))
        if not flagged:
            continue
        total_flagged += len(flagged)

        print(f"\n  {name[:50]} — {len(flagged)} flagged")
        if dry_run:
            for scene, reason in flagged[:8]:
                print(f"      scene {scene.get('scene_index')}: {reason}")
            continue

        video_path = find_source_video(data, ef, source)
        if not video_path:
            print(f"      ✗ source video not found, skipping")
            continue

        fa_by_scene = {fa.get("scene_index"): fa for fa in data.get("frame_analyses", [])}
        frame_dir = FRAMES_DIR / hashlib.md5(ef.name.encode()).hexdigest()[:12]
        frame_dir.mkdir(parents=True, exist_ok=True)

        fixed = 0
        for scene, reason in flagged:
            si = scene.get("scene_index")
            mid = (scene.get("start_sec", 0) + scene.get("end_sec", 0)) / 2
            frame_path = frame_dir / f"reenrich_scene_{si}_{mid:.2f}s.jpg"
            if not extract_frame(video_path, mid, frame_path):
                print(f"      scene {si}: ✗ extract failed")
                continue
            parsed, raw, elapsed, error = query_vision(str(frame_path))
            if error or not parsed:
                print(f"      scene {si}: ~ {error or 'unparsed'} ({elapsed:.1f}s)")
                continue
            clean, _ = normalize_analysis(parsed)
            scene["semantic"] = clean
            scene.pop("semantic_raw", None)
            if si in fa_by_scene:
                fa_by_scene[si]["analysis"] = clean
                fa_by_scene[si].pop("analysis_raw", None)
                fa_by_scene[si]["_provenance"] = {**BACKEND.provenance(),
                                                  "elapsed": round(elapsed, 1)}
            fixed += 1
            print(f"      scene {si}: ✓ re-enriched ({reason}, {elapsed:.1f}s)")

        if fixed:
            ef.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            total_fixed += fixed
            print(f"      → wrote {fixed} updates to {ef.name}")

        if limit and total_fixed >= limit:
            print(f"\n  --limit {limit} reached; stopping (re-run to continue — "
                  f"updated scenes won't re-flag).")
            break

    print(f"\n  {'═' * 68}")
    print(f"  RE-ENRICH COMPLETE — {total_fixed}/{total_flagged} flagged scenes updated")
    print(f"  {'═' * 68}\n")


# ─── Sampling-plan mode (pilot rescan) ──────────────────────────────────────

def run_sampling_plan(source_dir, video_filter=None, dry_run=False, only_new=False):
    """Re-enrich from bench sampling plans: one vision call per DISTINCT VISUAL
    STATE, with a `semantic` attached to EVERY represented scene (no inheritance),
    plus folded-in text detection. Writes a fresh .enriched.json and a derived
    .text_flags.json so the assembler omits text with no separate scan.

    Consumes enriched/<stem>.sampling_plan.json produced by bench/sampler.py. If the
    plan's representative frame already exists on disk (extracted by the bench), it
    is reused; otherwise it is extracted fresh.
    """
    source = Path(source_dir)
    plans = sorted(ENRICHMENT_DIR.glob("*.sampling_plan.json"))
    if video_filter:
        plans = [p for p in plans if video_filter.lower() in p.name.lower()]
    if only_new:
        # Skip videos already enriched via a sampling plan — protects an
        # import batch from silently re-running days of VLM over the
        # existing corpus (run_sampling_plan has no other resume logic).
        def _done(plan_path):
            stem = plan_path.name.replace(".sampling_plan.json", "")
            ef = ENRICHMENT_DIR / f"{stem}.enriched.json"
            if not ef.exists():
                return False
            try:
                mode = json.loads(ef.read_text()).get("enrichment", {}).get("mode")
            except (json.JSONDecodeError, OSError):
                return False
            return mode == "sampling_plan"
        skipped = [p for p in plans if _done(p)]
        plans = [p for p in plans if p not in set(skipped)]
        print(f"\n  --only-new: skipping {len(skipped)} already-enriched plan(s)")
    if not plans:
        print("\n  No sampling plans found (run `bench_run.py plan` first).")
        return

    if not dry_run:
        ok, msg = BACKEND.health_check()
        print(f"\n  Backend {BACKEND.label}: {'✓ ' + msg if ok else '✗ ' + msg}")
        if not ok:
            sys.exit(1)

    print(f"\n  {'═' * 68}")
    print(f"  SAMPLING-PLAN RESCAN  ·  backend {BACKEND.label}")
    print(f"  {len(plans)} plan(s)")
    print(f"  {'═' * 68}")
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for plan_path in plans:
        plan = json.loads(plan_path.read_text())
        stem = plan.get("display_stem", plan_path.name.replace(".sampling_plan.json", ""))
        n_states = plan.get("totals", {}).get("distinct_states", 0)
        print(f"\n  {stem[:55]} — {n_states} states")
        if dry_run:
            continue

        # Base structure: prefer the existing enriched file (has timelines), else analysis.
        ef = ENRICHMENT_DIR / f"{stem}.enriched.json"
        base_path = ef if ef.exists() else (ANALYSIS_DIR / f"{stem}.analysis.json")
        if not base_path.exists():
            print(f"      ✗ no base analysis/enriched file, skipping")
            continue
        data = json.loads(base_path.read_text())

        video_path = plan.get("source_path")
        if not video_path or not Path(video_path).exists():
            video_path = find_source_video(data, base_path, source)
        if not video_path:
            print(f"      ✗ source video not found, skipping")
            continue

        scene_data = data.get("scenes", {})
        scene_list = scene_data.get("scenes", []) if isinstance(scene_data, dict) else scene_data
        scenes_by_idx = {s.get("scene_index"): s for s in scene_list}

        frame_dir = FRAMES_DIR / hashlib.md5(stem.encode()).hexdigest()[:12]
        frame_dir.mkdir(parents=True, exist_ok=True)

        frame_analyses = []
        text_flag_seconds = {}     # int second -> description, for scenes with text
        described = errors = 0

        for sc in plan.get("scenes", []):
            si = sc.get("scene_index")
            scene_has_text = False
            text_desc = ""
            for rep in sc.get("representatives", []):
                t = rep.get("time_sec", 0.0)
                # reuse the bench-extracted frame if present, else extract fresh
                fp = ANALYSIS_DIR / rep.get("frame_path", "") if rep.get("frame_path") else None
                if not (fp and fp.exists()):
                    fp = frame_dir / f"s{si}_c{rep.get('cluster_id',0)}_{t:.2f}.jpg"
                    if not extract_frame(video_path, t, fp):
                        errors += 1
                        continue
                parsed, raw, elapsed, error = query_vision(str(fp))
                if error or not parsed:
                    errors += 1
                    continue
                clean, _ = normalize_analysis(parsed)
                described += 1
                frame_analyses.append({
                    "time_sec": t, "scene_index": si,
                    "cluster_id": rep.get("cluster_id"),
                    "analysis": clean,
                    "_provenance": {**BACKEND.provenance(), "elapsed": round(elapsed, 1)},
                    "inference_time_sec": round(elapsed, 2),
                })
                # primary semantic for the scene = first described state
                if si in scenes_by_idx and "semantic" not in scenes_by_idx[si]:
                    scenes_by_idx[si]["semantic"] = clean
                if clean.get("has_english_text"):
                    scene_has_text = True
                    text_desc = clean.get("text_content", "") or text_desc
            # fold text flag across the whole scene window (assembler omits whole scenes)
            if scene_has_text and si in scenes_by_idx:
                s = scenes_by_idx[si]
                for sec in range(int(s.get("start_sec", 0)), int(s.get("end_sec", 0)) + 1):
                    text_flag_seconds[sec] = text_desc

        # write enriched
        data["schema_version"] = "2.0.0"
        data["frame_analyses"] = frame_analyses
        data["enrichment"] = {
            "backend": BACKEND.name, "model": BACKEND.model,
            "timestamp": datetime.now().isoformat(),
            "frame_count": described, "mode": "sampling_plan",
            "plan_checksum_states": n_states,
        }
        ef.write_text(json.dumps(data, indent=2, ensure_ascii=False))

        # write derived text_flags so the assembler omits text with no separate scan
        TEXT_FLAGS_DIR = ANALYSIS_DIR / "text_flags"
        TEXT_FLAGS_DIR.mkdir(parents=True, exist_ok=True)
        flags = {str(sec): {"time_sec": float(sec), "has_english_text": True,
                            "description": desc, "sample_type": "vlm_enrich"}
                 for sec, desc in sorted(text_flag_seconds.items())}
        tf = {
            "video": stem, "source": str(video_path),
            "duration_sec": data.get("metadata", {}).get("duration_sec", 0),
            "flags": flags, "status": "complete", "scan_method": "vlm_enrich",
            "text_frame_count": len(flags),
        }
        (TEXT_FLAGS_DIR / f"{stem}.text_flags.json").write_text(
            json.dumps(tf, indent=2, ensure_ascii=False))

        print(f"      ✓ {described} states described, {errors} errors, "
              f"{len(text_flag_seconds)} text-seconds flagged")
        print(f"      → {ef.name} + text_flags/{stem}.text_flags.json")

    print(f"\n  {'═' * 68}\n  SAMPLING-PLAN RESCAN COMPLETE\n  {'═' * 68}\n")


# ─── Status ─────────────────────────────────────────────────────────────────

def show_status():
    state = load_state()
    done_v = len(state.get("completed_videos", []))
    done_f = state.get("completed_frames", 0)
    total_f = state.get("total_frames", 0)
    pct = done_f / total_f if total_f else 0
    timings = state.get("timings", [])
    avg = sum(timings) / len(timings) if timings else 0

    print(f"\n  ╔{'═' * 60}╗")
    print(f"  ║{'SEMANTIC ENRICHMENT PIPELINE':^60}║")
    print(f"  ╠{'═' * 60}╣")
    print(f"  ║  Model:    {MODEL:<48}║")
    print(f"  ║  Status:   {state['status']:<48}║")
    print(f"  ║  Videos:   {done_v} complete{' ' * 40}║")
    print(f"  ║  Frames:   {fmt_bar(pct)} {done_f}/{total_f} ({pct:.0%}){' ' * max(0, 15 - len(f'{done_f}/{total_f}'))}║")
    if avg:
        print(f"  ║  Avg/frame: {avg:.0f}s{' ' * 46}║")
    if state.get("started_at") and done_f > 0:
        elapsed = (datetime.now() - datetime.fromisoformat(state["started_at"])).total_seconds()
        remaining = avg * (total_f - done_f) if avg else 0
        print(f"  ║  Elapsed:  {fmt_time(elapsed):<48}║")
        print(f"  ║  ETA:      {fmt_time(remaining):<48}║")
    errs = state.get("errors", [])
    if errs:
        print(f"  ║  Errors:   {len(errs):<48}║")
    print(f"  ╚{'═' * 60}╝\n")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    global BACKEND
    parser = argparse.ArgumentParser(description="Semantic enrichment pipeline for media analysis")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source video directory")
    parser.add_argument("--status", action="store_true", help="Show progress")
    parser.add_argument("--pause", action="store_true", help="Pause after current frame")
    parser.add_argument("--resume", action="store_true", help="Resume paused pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without running")
    parser.add_argument("--video", default=None, help="Filter to matching videos (substring)")
    parser.add_argument("--reset", action="store_true", help="Clear enrichment data")
    parser.add_argument("--backend", default=None,
                        choices=["ollama", "claude-cli", "anthropic-api", "mlx"],
                        help="Vision backend (default: $GHOST_VISION_BACKEND or ollama)")
    parser.add_argument("--model", default=None, help="Override the backend's default model")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N videos (or, with --reenrich-flagged, "
                             "N scenes — checkpointing for long worklist runs)")
    parser.add_argument("--reenrich-flagged", action="store_true",
                        help="Re-run only flagged scenes (failed validation/parse or low quality)")
    parser.add_argument("--worklist", default=None,
                        help="With --reenrich-flagged: also re-run the contradiction "
                             "scenes in this audit_semantics worklist JSON")
    parser.add_argument("--only-new", action="store_true",
                        help="With --sampling-plan: skip videos whose enriched "
                             "file was already produced from a sampling plan")
    parser.add_argument("--sampling-plan", action="store_true",
                        help="Rescan from bench sampling plans (one call per distinct visual "
                             "state; attaches semantics to every represented scene + folds in "
                             "text detection). Pilot rescan path.")
    parser.add_argument("--quality-threshold", type=float, default=0.5,
                        help="quality_score below this flags a scene for re-enrichment")

    args = parser.parse_args()

    if args.reset:
        if ENRICHMENT_DIR.exists():
            import shutil
            shutil.rmtree(ENRICHMENT_DIR)
            print("  Enrichment data cleared.")
        return

    if args.status:
        show_status()
        return

    if args.pause:
        state = load_state()
        state["paused"] = True
        save_state(state)
        print("  Pause signal sent.")
        return

    BACKEND = get_backend(args.backend, args.model)

    if args.sampling_plan:
        run_sampling_plan(args.source, args.video, args.dry_run, args.only_new)
        return

    if args.reenrich_flagged:
        run_reenrich_flagged(args.source, args.quality_threshold, args.video,
                             args.dry_run, args.worklist, args.limit)
        return

    if args.resume:
        state = load_state()
        state["paused"] = False
        state["status"] = "resuming"
        save_state(state)
        print("  Resuming...")
        run_pipeline(args.source, args.video, limit=args.limit)
        return

    run_pipeline(args.source, args.video, args.dry_run, limit=args.limit)


if __name__ == "__main__":
    main()
