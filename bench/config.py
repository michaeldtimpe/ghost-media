#!/usr/bin/env python3
"""
Bake-off configuration: paths, the pilot video set, sampler/metric thresholds,
the engine matrix, the CLIP-anchored composite weights, and the metered-billing
guard. All tunables live here so a run is fully described by this file + the
frozen manifest (bench/manifest.py).
"""

import os
from pathlib import Path

from bench import keys

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent      # repo root
ANALYSIS_DIR = BASE_DIR                                 # *.analysis.json at root
ENRICHED_DIR = BASE_DIR / "enriched"
TEXT_FLAGS_DIR = BASE_DIR / "text_flags"

# Source footage. Prefer the local staged copies (fast, no archive-drive dependency);
# fall back to the external archive. resolve_source() searches these in order.
LOCAL_SOURCE_DIR = BASE_DIR / "raw_footage"
ARCHIVE_SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")
SOURCE_DIRS = [LOCAL_SOURCE_DIR, ARCHIVE_SOURCE_DIR]
SOURCE_DIR = ARCHIVE_SOURCE_DIR  # back-compat alias

BENCH_DIR = BASE_DIR / "bench"
RESULTS_DIR = BENCH_DIR / "results"
FRAMES_DIR = RESULTS_DIR / "frames"                    # extracted once, shared by engines
GROUNDTRUTH_DIR = BENCH_DIR / "groundtruth"

# ─── Pilot subset ────────────────────────────────────────────────────────────
# Substrings matched (case-insensitive) against *.analysis.json filenames. Chosen
# to span every visual regime + both filename encodings + scene-count extremes.

# The 6 files staged locally in raw_footage/ (user-selected). Substrings matched
# (case-insensitive) against *.analysis.json filenames.
PILOT_VIDEOS = [
    "EXC3_CM3",                          # motion graphics — abstract specificity
    "PAPERS",                            # abstract / low-res
    "Tame Impala",                       # real-world music video, fast cuts
    "show me lyrics",                    # on-screen English text (text precision)
    "isshin REEL 2024.analysis",         # Japanese+English text, 4K, non-ASCII (exact-ish, skips -35e329)
    "Disengaging",                       # clean VJ loop
    # Optional / heavier (not staged locally): "Mega 4K VJ Loop", "BIRDMASK",
    # "ROOM 5 THINK ABOUT YOU", "Some of my best Speed Drone"
]

# ─── Adaptive sampler thresholds ─────────────────────────────────────────────

BASE_CADENCE_SEC = 6.0          # n_candidates ~ duration / this, scaled by motion
MIN_CANDIDATES = 1
MAX_CANDIDATES = 12
DISTINCT_THRESHOLD = 0.90       # new cluster when cosine to all centroids < this
THRESHOLD_FALLBACK = [0.90, 0.85, 0.80]   # lower for low-variance/ambient/loop videos
MIN_STATES_PER_MIN = 1.0        # below this, step the threshold down (fallback)
MAX_STATES_PER_VIDEO_BENCH = 300  # cap distinct states scored per video (bench only)

# ─── Metric thresholds ───────────────────────────────────────────────────────

ADJACENT_COLLAPSE_SIM = 0.92    # adjacent-state descriptions with text-cosine >= this
                                # count as "collapsed" (semantic-collapse rate)

# ─── Engine matrix ───────────────────────────────────────────────────────────
# Real engines map to vision_backends.get_backend(backend, model). Synthetic
# engines are computed by the runner without a VLM call:
#   baseline      — descriptions already on disk (enriched/*.enriched.json)
#   clip-ceiling  — image->image retrieval (discriminability upper bound)

# Revised lineup (post-review). The Qwen2.5-VL 7B + 32B ladder runs on BOTH runtimes
# so the Ollama-vs-MLX comparison isolates runtime (note: Ollama now uses an MLX
# backend on Apple Silicon, so the delta is quant/caching/cold-vs-warm, not framework).
# MiniCPM (Ollama) is the high-throughput wildcard; InternVL (MLX) is the OCR/text
# specialist that may dominate the has_english_text submetric. claude-cli is the cloud
# quality reference (run explicitly; needs the subscription — see assert_billing_safe).
# Verify exact mlx-community repo ids + the Ollama minicpm-v tag before pulling.
ENGINES = {
    # Ollama (local)
    "ollama-qwen7b":  {"backend": "ollama", "model": "qwen2.5vl:7b"},      # baseline to beat
    "ollama-minicpm": {"backend": "ollama", "model": "minicpm-v:8b"},      # throughput wildcard (→4.5 if Ollama serves it)
    "ollama-qwen32b": {"backend": "ollama", "model": "qwen2.5vl:32b"},     # local quality/OCR ceiling
    # MLX (Apple-Silicon native)
    "mlx-qwen7b":     {"backend": "mlx", "model": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"},
    "mlx-qwen32b":    {"backend": "mlx", "model": "mlx-community/Qwen2.5-VL-32B-Instruct-4bit"},
    "mlx-internvl":   {"backend": "mlx", "model": "mlx-community/InternVL3-8B-MLX-4bit"},  # OCR specialist (verified id; plain -4bit 404s)
    "mlx-gemma3":     {"backend": "mlx", "model": "mlx-community/gemma-3-27b-it-4bit"}, # available in local archive (no download)
    # Cloud reference (optional; run explicitly)
    "claude-cli":     {"backend": "claude-cli", "model": "sonnet"},
    # Reference rows (no VLM call)
    "baseline":       {"synthetic": "baseline"},       # current system — the floor
    "clip-ceiling":   {"synthetic": "clip-ceiling"},   # image→image — discriminability ceiling
}

# Default set for `bench_run.py run` with no --engines: the local lineup + references.
# claude-cli is excluded by default (needs the subscription/key dance); add it explicitly.
DEFAULT_ENGINES = [
    "ollama-qwen7b", "ollama-minicpm", "ollama-qwen32b",
    "mlx-qwen7b", "mlx-qwen32b", "mlx-internvl",
    "baseline", "clip-ceiling",
]

JUDGE_BACKEND = "claude-cli"
JUDGE_MODEL = "sonnet"

# ─── CLIP-anchored composite weights ─────────────────────────────────────────
# Deliberate choice: the rank is computed ONLY from objective metrics. Within-video
# Recall@1 + coverage + text-F1 dominate because they map directly to the two
# original failure modes (discriminability/coverage and text omission); the
# adjacent-state term guards against winning with generic-but-CLIP-friendly prose.
# The LLM-judge (bench/judge.py) is NEVER included here — it is audit-only.

COMPOSITE_WEIGHTS = {
    "recall_at_1_within": 0.40,
    "coverage": 0.25,
    "text_f1": 0.20,
    "adjacent_discriminability": 0.15,
}

# ─── Throughput projection (cost/latency report) ─────────────────────────────
# Rough local-inference speedup factor of the M5 Max vs the current M1 Max, used
# only to print an M5-projected column. Refine once the M5 Max is back (~Jun 2026).
M5_SPEEDUP_VS_M1 = 3.0
FULL_CORPUS_SCENES = 19580      # measured; for $/full-corpus + wall-clock estimates


# ─── Billing guard ───────────────────────────────────────────────────────────

def assert_billing_safe(engine_names, allow_metered=False, will_judge=True):
    """Abort if a run would silently bill the metered Anthropic API.

    claude-cli routes to API billing (not the subscription) whenever
    ANTHROPIC_API_KEY is set (lessons.md footgun); anthropic-api is always
    metered. If either is in play and the key is set, refuse unless the user
    explicitly passed --allow-metered.
    """
    if allow_metered:
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return
    cloud = {"claude-cli", "anthropic-api"}
    backends = {ENGINES.get(n, {}).get("backend") for n in engine_names}
    touches_cloud = bool(backends & cloud) or (will_judge and JUDGE_BACKEND in cloud)
    if touches_cloud:
        raise SystemExit(
            "\n  ✗ ANTHROPIC_API_KEY is set — claude-cli/judge would bill the "
            "METERED API, not your subscription.\n"
            "    Fix one of:\n"
            "      • unset ANTHROPIC_API_KEY   (use the subscription, zero marginal cost)\n"
            "      • pass --allow-metered       (acknowledge metered billing)\n")


# ─── Source resolution + pilot registry ──────────────────────────────────────

def find_analysis_files(filters=None):
    """Return analysis files (excluding pass1/meta), optionally filtered by substring."""
    out = []
    for f in sorted(ANALYSIS_DIR.glob("*.analysis.json")):
        if "pass1" in f.name or f.name.startswith("_"):
            continue
        if filters and not any(s.lower() in f.name.lower() for s in filters):
            continue
        out.append(f)
    return out


_VID_EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def resolve_source(analysis_path, analysis_data, source_dirs=None):
    """Resolve an analysis file to its source video. Returns a Path or None.

    Search order: local raw_footage/ first (fast), then the archive drive. Within
    each: exact basename, then exact stem, then a last-resort fuzzy match. Callers
    log matched/total over a batch (the lessons.md prescription).
    """
    dirs = [Path(d) for d in (source_dirs or SOURCE_DIRS)]
    info = analysis_data.get("file", "")
    original = info.get("path") or info.get("name", "") if isinstance(info, dict) else str(info)
    base = keys.display_stem(analysis_path)
    name = Path(original).name if original else None

    # absolute path as recorded (only if it actually exists)
    if original and Path(original).exists():
        # prefer a local copy of the same basename if present
        for d in dirs:
            if (d / Path(original).name).exists():
                return d / Path(original).name
        return Path(original)
    # exact basename, then exact stem, in dir order
    for d in dirs:
        if not d.exists():
            continue
        if name and (d / name).exists():
            return d / name
        for f in d.iterdir():
            if f.suffix.lower() in _VID_EXT and f.stem == base:
                return f
    # last resort: fuzzy
    for d in dirs:
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix.lower() in _VID_EXT and base[:24] and (
                    base[:24] in f.stem or f.stem[:24] in base):
                return f
    return None


def pilot_registry(filters=None):
    """Resolve the pilot videos to a collision-checked identity registry."""
    import json
    afiles = find_analysis_files(filters or PILOT_VIDEOS)
    entries = []
    for af in afiles:
        data = json.loads(af.read_text())
        entries.append((af, resolve_source(af, data)))
    return keys.build_registry(entries)
