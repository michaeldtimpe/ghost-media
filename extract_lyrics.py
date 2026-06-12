#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  LYRICS EXTRACTOR
  Demucs vocal separation → Whisper transcription → timestamped lyrics JSON
═══════════════════════════════════════════════════════════════════════════════

Pipeline:
  1. Run Demucs (htdemucs) to separate vocals from the DJ set audio
  2. Run Whisper on the isolated vocals track
  3. Extract keywords from each segment
  4. Map segments onto the existing phrase/track timeline
  5. Output .lyrics.json compatible with assemble_v2.py

Requirements:
  pip install demucs openai-whisper

Usage:
  python3 extract_lyrics.py --set blue-sky-genesis-2025
  python3 extract_lyrics.py --all
  python3 extract_lyrics.py --input /path/to/audio.mp3
  python3 extract_lyrics.py --status
"""

import json
import os
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pin_memory.*")

# ─── Configuration ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
import media_paths
SETS_DIR = media_paths.SETS_ROOT
OUTPUT_DIR = BASE_DIR / "sets"
DEMUCS_DIR = BASE_DIR / "demucs_output"

DEMUCS_MODEL = "htdemucs"       # fast, good quality vocal separation
WHISPER_MODEL = "base"          # "tiny", "base", "small", "medium", "large"
                                # base is a good speed/quality tradeoff for lyrics

# Set definitions — same as assemble_v2.py
SET_CONFIGS = {
    "blue-sky-genesis-2025": {
        "audio": SETS_DIR / "blue-sky-genesis-2025" / "Mtimpe MIX1 02DEC MASTER MP3  V6  .mp3",
        "analysis": "Mtimpe MIX1 02DEC MASTER MP3 V6.deep-analysis.json",
    },
    "boxing-day-2025": {
        "audio": SETS_DIR / "boxing-day-2025" / "Mtimpe MIX 30OCT MASTER MP3 v5.mp3",
        "analysis": "Mtimpe MIX 30OCT MASTER WAV v5.deep-analysis.json",
    },
    "cheerleader-exodus-2025": {
        "audio": SETS_DIR / "cheerleader-exodus-2025" / "Mtimpe MIX2 10DEC MASTER MP3 v3.mp3",
        "analysis": "Mtimpe MIX2 10DEC MASTER WAV v3.deep-analysis.json",
    },
    "waiting-to-begin-2024": {
        "audio": SETS_DIR / "waiting-to-begin-2024" / "arche august dj set rev 4.mp3",
        "analysis": "arche august dj set rev 4.deep-analysis.json",
    },
    "will-call-2025": {
        "audio": SETS_DIR / "will-call-2025" / "Mtimpe MIX 09JUN MASTER MP3 v4.mp3",
        "analysis": "Mtimpe MIX 09JUN MASTER WAV v4.deep-analysis.json",
    },
}

# Keywords to ignore — common Whisper hallucinations on music/silence
STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "on", "to", "and", "of", "for",
    "that", "this", "with", "you", "i", "my", "me", "we", "he", "she",
    "was", "are", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "but", "or", "if", "so", "as", "at", "by", "up", "out",
    "all", "just", "like", "get", "got", "know", "think", "go", "going",
    "come", "want", "need", "let", "oh", "yeah", "uh", "ah", "um", "hmm",
    "na", "la", "da", "ooh", "whoa", "hey", "yo",
}

# Whisper often hallucinates these phrases on instrumental music
HALLUCINATION_PATTERNS = [
    r"thank you for watching",
    r"please subscribe",
    r"like and subscribe",
    r"see you next time",
    r"thanks for watching",
    r"please like",
    r"don't forget to subscribe",
    r"music\s*$",
    r"^\s*\[music\]\s*$",
    r"^\s*\[applause\]\s*$",
    r"^\s*\.\s*$",
    r"^\s*$",
]


# ─── Helpers ───────────────────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def extract_keywords(text):
    """Extract meaningful keywords from a text segment."""
    # Lowercase, strip punctuation
    words = re.findall(r"[a-z']+", text.lower())
    # Filter stop words and very short words
    keywords = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return list(dict.fromkeys(keywords))  # deduplicate, preserve order


def is_hallucination(text):
    """Check if text matches common Whisper hallucination patterns."""
    text_lower = text.strip().lower()
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def check_demucs():
    """Check if demucs is installed."""
    try:
        result = subprocess.run(["demucs", "--help"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_whisper():
    """Check if whisper is importable."""
    try:
        import whisper
        return True
    except ImportError:
        return False


# ─── Demucs Separation ───────────────────────────────────────────────────

def separate_vocals(audio_path, output_dir):
    """Run Demucs to isolate vocals. Returns path to vocals wav."""
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Demucs outputs to: output_dir/htdemucs/<stem>/vocals.wav
    stem = audio_path.stem
    vocals_path = output_dir / DEMUCS_MODEL / stem / "vocals.wav"

    if vocals_path.exists() and vocals_path.stat().st_size > 1000:
        print(f"    Vocals already separated: {vocals_path.name}")
        return vocals_path

    print(f"    Running Demucs ({DEMUCS_MODEL})...")
    print(f"    Input: {audio_path.name}")
    print(f"    This may take several minutes for long DJ sets.", flush=True)

    t0 = time.time()
    result = subprocess.run(
        [
            "demucs",
            "--name", DEMUCS_MODEL,
            "--out", str(output_dir),
            "--two-stems", "vocals",  # only separate vocals vs rest (faster)
            "-d", "mps",              # Apple Silicon GPU acceleration
            str(audio_path),
        ],
        capture_output=True,
        text=True,
        timeout=7200,  # 2 hour timeout for long sets
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"    Demucs error (exit {result.returncode}):")
        # Show last portion of stderr (progress bars can be very long)
        stderr_lines = result.stderr.strip().splitlines()
        # Skip progress bar lines, show actual errors
        error_lines = [l for l in stderr_lines if not l.strip().startswith(('%', '|', '\r'))]
        if error_lines:
            for line in error_lines[-20:]:
                print(f"    {line}")
        else:
            print(f"    {result.stderr[-2000:]}")
        return None

    if not vocals_path.exists():
        # Try alternate output path patterns
        possible = list((output_dir / DEMUCS_MODEL).rglob("vocals.wav"))
        if possible:
            vocals_path = possible[0]
        else:
            print(f"    Vocals file not found after separation")
            return None

    size_mb = vocals_path.stat().st_size / (1024 * 1024)
    print(f"    Separation complete: {size_mb:.0f} MB in {fmt_time(elapsed)}")
    return vocals_path


# ─── Whisper Transcription ───────────────────────────────────────────────

def transcribe_vocals(vocals_path):
    """Run Whisper on isolated vocals. Returns list of segments."""
    import whisper

    print(f"\n    Loading Whisper model ({WHISPER_MODEL})...", end=" ", flush=True)
    model = whisper.load_model(WHISPER_MODEL)
    print("ready")

    print(f"    Transcribing vocals...", flush=True)
    t0 = time.time()

    result = model.transcribe(
        str(vocals_path),
        language="en",
        word_timestamps=True,
        condition_on_previous_text=True,
        no_speech_threshold=0.6,      # higher = stricter silence detection
        logprob_threshold=-0.8,       # filter low-confidence segments
        compression_ratio_threshold=2.0,  # filter repetitive hallucinations
    )

    elapsed = time.time() - t0
    segments = result.get("segments", [])
    print(f"    Transcription complete: {len(segments)} raw segments in {fmt_time(elapsed)}")

    return segments


# ─── Processing ──────────────────────────────────────────────────────────

def process_segments(raw_segments):
    """Clean, filter, and extract keywords from Whisper segments."""
    processed = []
    total_filtered = 0

    for seg in raw_segments:
        text = seg.get("text", "").strip()

        # Skip empty or hallucinated segments
        if not text or is_hallucination(text):
            total_filtered += 1
            continue

        # Skip very low confidence segments
        avg_logprob = seg.get("avg_logprob", 0)
        no_speech_prob = seg.get("no_speech_prob", 0)
        if avg_logprob < -1.0 or no_speech_prob > 0.5:
            total_filtered += 1
            continue

        keywords = extract_keywords(text)

        # Skip if no meaningful keywords
        if not keywords:
            total_filtered += 1
            continue

        # Extract word-level timestamps if available
        words = []
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", "").strip(),
                "start": round(w.get("start", 0), 2),
                "end": round(w.get("end", 0), 2),
            })

        processed.append({
            "start_sec": round(seg["start"], 2),
            "end_sec": round(seg["end"], 2),
            "duration_sec": round(seg["end"] - seg["start"], 2),
            "text": text,
            "keywords": keywords,
            "confidence": round(-avg_logprob, 3),  # lower = more confident
            "no_speech_prob": round(no_speech_prob, 3),
            "words": words,
        })

    print(f"    Processed: {len(processed)} segments kept, {total_filtered} filtered")
    return processed


def map_to_tracks(segments, analysis_path):
    """Map lyric segments onto the track timeline from deep analysis."""
    if not analysis_path.exists():
        return segments, {}

    data = json.loads(analysis_path.read_text())
    tracks = data.get("tracks", [])

    if not tracks:
        return segments, {}

    # Build track lookup
    track_lyrics = {}  # track_title → list of segments
    for seg in segments:
        mid = (seg["start_sec"] + seg["end_sec"]) / 2
        for track in tracks:
            if track["start_sec"] <= mid < track["end_sec"]:
                seg["track_title"] = track["title"]
                seg["track_index"] = track["track_index"]
                title = track["title"]
                if title not in track_lyrics:
                    track_lyrics[title] = []
                track_lyrics[title].append(seg)
                break

    # Summary: keywords per track
    track_keyword_summary = {}
    for title, segs in track_lyrics.items():
        all_kw = []
        for s in segs:
            all_kw.extend(s["keywords"])
        # Count and rank keywords
        kw_counts = {}
        for kw in all_kw:
            kw_counts[kw] = kw_counts.get(kw, 0) + 1
        ranked = sorted(kw_counts.items(), key=lambda x: -x[1])
        track_keyword_summary[title] = {
            "segment_count": len(segs),
            "top_keywords": [kw for kw, _ in ranked[:15]],
            "all_keywords": [kw for kw, _ in ranked],
        }

    return segments, track_keyword_summary


def build_phrase_lyrics(segments, analysis_path, phrase_key="four_bar"):
    """Map lyrics onto phrases for direct use by the assembler."""
    if not analysis_path.exists():
        return {}

    data = json.loads(analysis_path.read_text())
    phrases = data.get("phrases", {}).get(phrase_key, [])

    if not phrases:
        return {}

    phrase_lyrics = {}  # phrase_index → keywords list

    for phrase in phrases:
        p_start = phrase["start_sec"]
        p_end = phrase["end_sec"]
        p_idx = phrase["phrase_index"]

        # Find segments overlapping this phrase
        kw_set = set()
        for seg in segments:
            # Check overlap
            if seg["end_sec"] > p_start and seg["start_sec"] < p_end:
                kw_set.update(seg["keywords"])

        if kw_set:
            phrase_lyrics[str(p_idx)] = sorted(kw_set)

    return phrase_lyrics


# ─── Main ─────────────────────────────────────────────────────────────────

def run_set(set_name, set_config):
    """Extract lyrics for a single DJ set."""
    audio_path = Path(set_config["audio"])
    analysis_name = set_config["analysis"]
    analysis_path = OUTPUT_DIR / analysis_name
    output_path = OUTPUT_DIR / f"{set_name}.lyrics.json"

    print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {set_name}")
    print(f"  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    if not audio_path.exists():
        print(f"    ERROR: Audio not found: {audio_path}")
        return False

    set_start = time.time()

    # Step 1: Separate vocals
    print(f"\n  [1/4] Vocal separation")
    vocals_path = separate_vocals(audio_path, DEMUCS_DIR)
    if not vocals_path:
        print(f"    FAILED: Could not separate vocals")
        return False

    # Step 2: Transcribe
    print(f"\n  [2/4] Transcription")
    raw_segments = transcribe_vocals(vocals_path)

    # Step 3: Process and filter
    print(f"\n  [3/4] Processing segments")
    segments = process_segments(raw_segments)

    # Step 4: Map to tracks and phrases
    print(f"\n  [4/4] Mapping to timeline")
    segments, track_summary = map_to_tracks(segments, analysis_path)

    # Build phrase-level keyword index for all phrase sizes
    phrase_lyrics_4 = build_phrase_lyrics(segments, analysis_path, "four_bar")
    phrase_lyrics_8 = build_phrase_lyrics(segments, analysis_path, "eight_bar")
    phrase_lyrics_16 = build_phrase_lyrics(segments, analysis_path, "sixteen_bar")

    # Collect all unique keywords
    all_keywords = set()
    for seg in segments:
        all_keywords.update(seg["keywords"])

    # Build output
    result = {
        "set_name": set_name,
        "audio_source": str(audio_path),
        "demucs_model": DEMUCS_MODEL,
        "whisper_model": WHISPER_MODEL,
        "extracted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {
            "total_segments": len(segments),
            "total_keywords": len(all_keywords),
            "vocal_coverage_sec": sum(s["duration_sec"] for s in segments),
            "phrases_with_lyrics": {
                "four_bar": len(phrase_lyrics_4),
                "eight_bar": len(phrase_lyrics_8),
                "sixteen_bar": len(phrase_lyrics_16),
            },
        },
        "all_keywords": sorted(all_keywords),
        "segments": segments,
        "track_summary": track_summary,
        "phrase_lyrics": {
            "four_bar": phrase_lyrics_4,
            "eight_bar": phrase_lyrics_8,
            "sixteen_bar": phrase_lyrics_16,
        },
    }

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    elapsed = time.time() - set_start

    print(f"\n  ╔═══════════════════════════════════════════════════════════════════╗")
    print(f"  ║  {set_name:<63}║")
    print(f"  ║  Segments: {len(segments):<8}  Keywords: {len(all_keywords):<8}"
          f"  Time: {fmt_time(elapsed):<14}║")
    print(f"  ║  Vocal coverage: {fmt_time(sum(s['duration_sec'] for s in segments)):<49}║")
    print(f"  ║  → {output_path.name:<63}║")
    print(f"  ╚═══════════════════════════════════════════════════════════════════╝")

    # Show track summary
    if track_summary:
        print(f"\n    Lyrics by track:")
        for title, info in sorted(track_summary.items(),
                                   key=lambda x: -x[1]["segment_count"]):
            kw_str = ", ".join(info["top_keywords"][:8])
            print(f"      {title[:40]:<42} {info['segment_count']:>3} segs"
                  f"  [{kw_str[:50]}]")

    return True


def remap_set(set_name, set_config):
    """Rebuild phrase_lyrics + track mapping from the existing segments in
    .lyrics.json against the current deep-analysis JSON — no Demucs/Whisper
    re-run. Needed whenever re-analysis changes phrase spans or indices
    (e.g. the 2.2.0 end_sec fix)."""
    output_path = OUTPUT_DIR / f"{set_name}.lyrics.json"
    analysis_path = OUTPUT_DIR / set_config["analysis"]
    if not output_path.exists():
        print(f"  ✗ {set_name}: no .lyrics.json to remap (run full extraction first)")
        return False
    if not analysis_path.exists():
        print(f"  ✗ {set_name}: analysis missing: {analysis_path.name}")
        return False

    data = json.loads(output_path.read_text())
    segments = data.get("segments", [])
    segments, track_summary = map_to_tracks(segments, analysis_path)

    phrase_lyrics = {
        "four_bar": build_phrase_lyrics(segments, analysis_path, "four_bar"),
        "eight_bar": build_phrase_lyrics(segments, analysis_path, "eight_bar"),
        "sixteen_bar": build_phrase_lyrics(segments, analysis_path, "sixteen_bar"),
    }
    data["segments"] = segments
    data["track_summary"] = track_summary
    data["phrase_lyrics"] = phrase_lyrics
    data.setdefault("stats", {})["phrases_with_lyrics"] = {
        k: len(v) for k, v in phrase_lyrics.items()}
    data["remapped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  ✓ {set_name}: phrase_lyrics rebuilt "
          f"({', '.join(f'{k}={len(v)}' for k, v in phrase_lyrics.items())})")
    return True


def show_status():
    """Show lyrics extraction status for all sets."""
    print(f"\n  ═══════════════════════════════════════════════════════════════")
    print(f"  LYRICS EXTRACTION STATUS")
    print(f"  ═══════════════════════════════════════════════════════════════\n")

    for name, cfg in SET_CONFIGS.items():
        lyrics_path = OUTPUT_DIR / f"{name}.lyrics.json"
        if lyrics_path.exists():
            data = json.loads(lyrics_path.read_text())
            stats = data.get("stats", {})
            print(f"  ✓ {name:<32}  {stats.get('total_segments', 0):>4} segments"
                  f"  {stats.get('total_keywords', 0):>4} keywords"
                  f"  model: {data.get('whisper_model', '?')}")
        else:
            audio = Path(cfg["audio"])
            exists = "ready" if audio.exists() else "MISSING"
            print(f"  ○ {name:<32}  not extracted  ({exists})")

    print()


def main():
    global WHISPER_MODEL

    import argparse
    parser = argparse.ArgumentParser(description="Extract lyrics from DJ sets via Demucs + Whisper")
    parser.add_argument("--set", type=str, choices=list(SET_CONFIGS.keys()),
                        help="DJ set to process")
    parser.add_argument("--all", action="store_true", help="Process all sets")
    parser.add_argument("--input", type=str, help="Process a specific audio file")
    parser.add_argument("--status", action="store_true", help="Show extraction status")
    parser.add_argument("--whisper-model", type=str, default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--reseparate", action="store_true",
                        help="Force re-run Demucs even if vocals exist")
    parser.add_argument("--remap-only", action="store_true",
                        help="Rebuild phrase_lyrics from existing segments against "
                             "the current analysis JSON (no Demucs/Whisper)")
    args = parser.parse_args()

    WHISPER_MODEL = args.whisper_model

    if args.status:
        show_status()
        return

    if args.remap_only:
        sets = {args.set: SET_CONFIGS[args.set]} if args.set else SET_CONFIGS
        print(f"\n  Remapping phrase_lyrics for {len(sets)} set(s)...")
        ok = all([remap_set(name, cfg) for name, cfg in sets.items()])
        sys.exit(0 if ok else 1)

    # Check dependencies
    print(f"\n  Checking dependencies...")

    if not check_demucs():
        print(f"  ✗ Demucs not found. Install: pip install demucs")
        sys.exit(1)
    print(f"  ✓ Demucs ({DEMUCS_MODEL})")

    if not check_whisper():
        print(f"  ✗ Whisper not found. Install: pip install openai-whisper")
        sys.exit(1)
    print(f"  ✓ Whisper ({WHISPER_MODEL})")

    DEMUCS_DIR.mkdir(parents=True, exist_ok=True)

    if args.input:
        # Process a single file
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"  ERROR: File not found: {input_path}")
            sys.exit(1)
        set_name = input_path.stem.replace(" ", "-").lower()
        run_set(set_name, {"audio": input_path, "analysis": f"{input_path.stem}.deep-analysis.json"})
        return

    # Determine which sets to process
    if args.set:
        sets = {args.set: SET_CONFIGS[args.set]}
    elif args.all:
        sets = SET_CONFIGS
    else:
        print(f"  Specify --set <name>, --all, or --input <path>")
        show_status()
        return

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  LYRICS EXTRACTOR")
    print(f"  Demucs: {DEMUCS_MODEL}  |  Whisper: {WHISPER_MODEL}")
    print(f"  Sets: {len(sets)}")
    print(f"  ════════════════════════════════════════════════════════════════════")

    batch_start = time.time()
    results = {}

    for name, cfg in sets.items():
        lyrics_path = OUTPUT_DIR / f"{name}.lyrics.json"
        if lyrics_path.exists() and not args.reseparate:
            print(f"\n  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(f"  {name} — already done, skipping")
            results[name] = True
            continue
        ok = run_set(name, cfg)
        results[name] = ok

    elapsed = time.time() - batch_start
    n_ok = sum(results.values())

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  COMPLETE: {n_ok}/{len(results)} sets in {fmt_time(elapsed)}")
    for name, ok in results.items():
        print(f"  {'✓' if ok else '✗'} {name}")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
