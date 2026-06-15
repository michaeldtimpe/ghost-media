#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  AUDIO SEMANTIC ENRICHMENT
  CLAP zero-shot tagging + LLM phrase composition → .semantic.json
═══════════════════════════════════════════════════════════════════════════════

The audio twin of enrich_analyses.py: where that script gives video scenes
semantic descriptions, this one gives DJ-set sections semantic descriptions.

Pipeline:
  1. Section the set — tracklist-aligned tracks when the deep analysis has
     them, else groups of 4-bar phrases targeting ~60 s per section.
  2. CLAP zero-shot tags per section (laion/larger_clap_music): genre, mood,
     instrumentation, vocals, texture — scored against curated vocabularies.
  3. Signal-feature digest per section from .deep-analysis.json (BPM, key,
     energy / brightness / percussiveness percentiles, energy arc, onsets).
  4. Lyrics keywords merged in from .lyrics.json when present.
  5. An LLM composes descriptive phrases + visual keywords per section via
     vision_backends.analyze_text (ollama / claude-cli / anthropic-api).
  6. Output: <set>.semantic.json in ./sets/, sibling to the lyrics JSON.

Requirements:
  torch + transformers (CLAP stage; in .venv via requirements.lock)
  a text backend for stage 5 — or --skip-llm for a tags-only pass

Usage:
  python3 enrich_audio.py --set will-call-2025
  python3 enrich_audio.py --all
  python3 enrich_audio.py --input /path/to/audio.wav
  python3 enrich_audio.py --status
  python3 enrich_audio.py --set X --backend claude-cli --llm-model haiku
  python3 enrich_audio.py --set X --skip-llm          # CLAP tags only
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import media_paths
from vision_backends import get_backend

# ─── Configuration ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
SETS_DIR = media_paths.SETS_ROOT   # canonical NAS sets root (env-overridable)
OUTPUT_DIR = BASE_DIR / "sets"     # repo-local sidecars (.deep-analysis/.lyrics/.semantic)

SCHEMA_VERSION = "1.0.0"

# NOTE: laion/larger_clap_music's HF conversion is broken (text tower collapses
# to cos≈0.999 for all inputs; audio-text cos ≈ 0 — verified 2026-06 on
# transformers 4.57 AND 5.9). clap-htsat-unfused discriminates correctly.
CLAP_MODEL = "laion/clap-htsat-unfused"
CLAP_SR = 48000                 # the checkpoint's training sample rate
WINDOW_SEC = 10.0               # CLAP's native input length
WINDOWS_PER_SECTION = 6         # evenly spaced windows, mean-pooled
SECTION_TARGET_SEC = 60.0       # phrase-group section length (no tracklist)
SOFTMAX_SCALE = 100.0           # CLIP-convention temperature for tag probs

# Set definitions — same as extract_lyrics.py / assemble_v2.py
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

# ─── Tag Vocabularies ──────────────────────────────────────────────────────
# Curated, electronic-leaning but broad. Scored zero-shot per axis; the axis
# template phrases the comparison the way CLAP's training captions read.

AXES = {
    "genre": [
        "house", "deep house", "tech house", "afro house", "acid house",
        "techno", "melodic techno", "minimal techno", "progressive house",
        "trance", "psytrance", "drum and bass", "breakbeat", "UK garage",
        "dubstep", "bass music", "trap", "hip hop", "ambient", "downtempo",
        "chillout", "IDM", "electro", "disco", "nu-disco", "funk", "soul",
        "pop", "indie dance", "synthwave", "hardstyle", "footwork", "latin",
    ],
    "mood": [
        "euphoric", "melancholic", "dark", "uplifting", "dreamy",
        "aggressive", "playful", "hypnotic", "tense", "serene", "nostalgic",
        "triumphant", "brooding", "sensual", "mysterious", "joyful",
        "anthemic", "introspective", "menacing", "bittersweet",
    ],
    "instrumentation": [
        "a four-on-the-floor kick drum", "a rolling bassline",
        "an acid bassline", "heavy sub bass", "arpeggiated synthesizers",
        "synth pads", "plucked synth melodies", "piano", "strings",
        "electric guitar", "brass", "saxophone", "hand percussion",
        "breakbeat drums", "808 drums", "claps and snares", "crisp hi-hats",
        "bells and chimes", "organ", "choir",
    ],
    "vocals": [
        "no vocals, purely instrumental", "female vocals", "male vocals",
        "chopped vocal samples", "rap vocals", "spoken word",
    ],
    "texture": [
        "lush", "sparse", "dense", "gritty", "clean", "warm", "cold",
        "distorted", "atmospheric", "punchy", "washed-out", "shimmering",
        "deep", "airy",
    ],
}

TEMPLATES = {
    "genre": "This is a {} track.",
    "mood": "This music feels {}.",
    "instrumentation": "This music features {}.",
    "vocals": "This track has {}.",
    "texture": "The production sounds {}.",
}

TOP_K = {"genre": 4, "mood": 4, "instrumentation": 5, "vocals": 2, "texture": 3}


# ─── Helpers ───────────────────────────────────────────────────────────────

def fmt_time(seconds):
    if seconds < 0:
        return "—"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def parse_llm_json(text):
    """Extract a JSON object from an LLM response (tolerates code fences)."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _str_list(value, max_items):
    """Coerce an LLM field to a clean list of strings."""
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()][:max_items]


# ─── Audio Reading ─────────────────────────────────────────────────────────

class AudioReader:
    """Windowed reads via soundfile when the codec seeks cleanly; falls back
    to a one-shot full decode (mp3 on older libsndfile builds)."""

    def __init__(self, path):
        self.path = Path(path)
        self._sf = None
        self._full = None
        try:
            import soundfile as sf
            self._sf = sf.SoundFile(str(self.path))
            self.duration = len(self._sf) / self._sf.samplerate
        except Exception:
            import librosa
            print("    (soundfile can't stream this file — full decode, one-off)")
            self._full, _ = librosa.load(str(self.path), sr=CLAP_SR, mono=True)
            self.duration = len(self._full) / CLAP_SR

    def window(self, start_sec, dur_sec):
        """Return float32 mono audio at CLAP_SR for [start, start+dur)."""
        if self._sf is not None:
            import librosa
            sr = self._sf.samplerate
            self._sf.seek(max(0, int(start_sec * sr)))
            data = self._sf.read(int(dur_sec * sr), dtype="float32",
                                 always_2d=True).mean(axis=1)
            if sr != CLAP_SR:
                data = librosa.resample(data, orig_sr=sr, target_sr=CLAP_SR)
            return data
        start = int(start_sec * CLAP_SR)
        return self._full[start:start + int(dur_sec * CLAP_SR)]

    def close(self):
        if self._sf is not None:
            self._sf.close()


# ─── CLAP Tagging ──────────────────────────────────────────────────────────

class ClapTagger:
    """laion/larger_clap_music zero-shot tagger. Loads once; text embeddings
    per axis are computed once and cached. Falls back MPS → CPU if an op is
    unsupported (HTSAT on MPS has had gaps across torch versions)."""

    def __init__(self, device=None):
        import torch
        from transformers import ClapModel, ClapProcessor
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"  Loading CLAP ({CLAP_MODEL}) on {self.device}...", flush=True)
        t0 = time.time()
        self.model = ClapModel.from_pretrained(CLAP_MODEL).eval()
        self.processor = ClapProcessor.from_pretrained(CLAP_MODEL)
        try:
            self.model.to(self.device)
        except Exception:
            self.device = "cpu"
        self._text_emb = {}
        print(f"  CLAP ready in {time.time() - t0:.1f}s")

    def _to_device(self, inputs):
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _process_audio(self, windows):
        # `audio=` (transformers ≥5) vs `audios=` (4.x) kwarg drift.
        try:
            return self.processor(audio=windows, sampling_rate=CLAP_SR,
                                  return_tensors="pt", padding=True)
        except (TypeError, ValueError):
            return self.processor(audios=windows, sampling_rate=CLAP_SR,
                                  return_tensors="pt", padding=True)

    def _forward(self, fn, inputs):
        """Run on self.device; on MPS failure, demote to CPU permanently."""
        try:
            with self.torch.no_grad():
                return fn(**self._to_device(inputs))
        except Exception:
            if self.device == "cpu":
                raise
            print("    (MPS op unsupported — falling back to CPU)")
            self.device = "cpu"
            self.model.to("cpu")
            self._text_emb.clear()
            with self.torch.no_grad():
                return fn(**self._to_device(inputs))

    @staticmethod
    def _unwrap(out):
        """get_*_features returns a bare tensor (transformers 4.x) or a
        BaseModelOutputWithPooling whose pooler_output is the projected
        512-dim joint embedding (5.x)."""
        return out if hasattr(out, "norm") else out.pooler_output

    def _normalize(self, emb):
        return emb / emb.norm(dim=-1, keepdim=True)

    def text_features(self, axis):
        if axis not in self._text_emb:
            prompts = [TEMPLATES[axis].format(t) for t in AXES[axis]]
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)
            emb = self._unwrap(self._forward(self.model.get_text_features, inputs))
            self._text_emb[axis] = self._normalize(emb).cpu()
        return self._text_emb[axis]

    def embed_windows(self, windows):
        """Mean-pooled, normalized audio embedding for a list of windows."""
        windows = [w for w in windows if len(w) > CLAP_SR // 2]  # ≥0.5 s
        if not windows:
            return None
        inputs = self._process_audio(windows)
        emb = self._unwrap(self._forward(self.model.get_audio_features, inputs))
        pooled = self._normalize(emb).mean(dim=0)
        return self._normalize(pooled.unsqueeze(0)).squeeze(0).cpu()

    def tag(self, audio_emb):
        """Score one audio embedding against every axis vocabulary.

        Returns {axis: [[term, prob, cos], ...]} — top-k per axis; prob is a
        within-axis softmax (relative ranking, not calibrated confidence)."""
        if audio_emb is None:
            return {}
        out = {}
        for axis in AXES:
            text_emb = self.text_features(axis)
            cos = (text_emb @ audio_emb).float()
            probs = self.torch.softmax(cos * SOFTMAX_SCALE, dim=0)
            order = self.torch.argsort(cos, descending=True)[:TOP_K[axis]]
            out[axis] = [
                [AXES[axis][i], round(float(probs[i]), 4), round(float(cos[i]), 4)]
                for i in order.tolist()
            ]
        return out


# ─── Sectioning ────────────────────────────────────────────────────────────

def build_sections(deep, target_sec):
    """Tracklist tracks when present; else 4-bar phrase groups of ~target_sec;
    else fixed windows over the file duration."""
    tracks = deep.get("tracks") or []
    if tracks:
        return [
            {"index": i, "start_sec": t["start_sec"], "end_sec": t["end_sec"],
             "title": t.get("title"), "source": "tracklist"}
            for i, t in enumerate(tracks)
        ]

    phrases = (deep.get("phrases") or {}).get("four_bar") or []
    bounds = []
    if phrases:
        cur_start = phrases[0]["start_sec"]
        for ph in phrases:
            if ph["end_sec"] - cur_start >= target_sec:
                bounds.append((cur_start, ph["end_sec"]))
                cur_start = ph["end_sec"]
        last_end = phrases[-1]["end_sec"]
        if last_end - cur_start > 1.0:
            if bounds and last_end - cur_start < target_sec / 2:
                bounds[-1] = (bounds[-1][0], last_end)  # merge short tail
            else:
                bounds.append((cur_start, last_end))
    else:
        duration = _set_duration(deep)
        edges = np.arange(0, duration, target_sec)
        bounds = [(float(a), float(min(a + target_sec, duration))) for a in edges]

    return [
        {"index": i, "start_sec": round(s, 3), "end_sec": round(e, 3),
         "title": None, "source": "phrase-group" if phrases else "fixed-window"}
        for i, (s, e) in enumerate(bounds)
    ]


def _set_duration(deep):
    dur = (deep.get("file") or {}).get("duration_sec")
    if dur:
        return float(dur)
    me = deep.get("multiband_energy") or []
    return float(me[-1]["time_sec"]) if me else 0.0


# ─── Signal-Feature Digest ─────────────────────────────────────────────────

class FeatureDigest:
    """Slices the deep-analysis timelines per section and expresses each
    section relative to the whole set (percentile ranks, not raw units)."""

    def __init__(self, deep):
        me = deep.get("multiband_energy") or []
        self.t = np.array([x["time_sec"] for x in me])
        self.rms = np.array([x.get("total_rms", 0.0) for x in me])
        self.bass = np.array(
            [x.get("sub_bass", 0.0) + x.get("bass", 0.0) for x in me])

        hp = deep.get("hpss_timeline") or []
        self.hp_t = np.array([x["time_sec"] for x in hp])
        self.harm = np.array([x.get("harmonic", 0.0) for x in hp])
        self.perc = np.array([x.get("percussive", 0.0) for x in hp])

        sp = deep.get("spectral_timeline") or []
        self.sp_t = np.array([x["time_sec"] for x in sp])
        self.centroid = np.array([x.get("centroid_hz", 0.0) for x in sp])

        self.onset_times = np.array(
            (deep.get("onsets") or {}).get("times_sec") or [])

        bpm = deep.get("bpm_timeline") or []
        self.bpm_t = np.array([x["time_sec"] for x in bpm])
        self.bpm = np.array([x.get("bpm", 0.0) for x in bpm])

        self.keys = deep.get("key_timeline") or []

    @staticmethod
    def _pct(ref, value):
        """Percentile rank of `value` within reference distribution `ref`."""
        return float((ref < value).mean() * 100) if len(ref) else 0.0

    @staticmethod
    def _arc(values):
        """Energy arc over a section: compare thirds of the RMS curve."""
        if len(values) < 6:
            return "steady"
        third = len(values) // 3
        first, mid, last = (values[:third].mean(), values[third:2 * third].mean(),
                            values[2 * third:].mean())
        if mid > first * 1.2 and mid > last * 1.2:
            return "peak-and-release"
        if last > first * 1.15:
            return "rising"
        if first > last * 1.15:
            return "falling"
        return "steady"

    def section(self, start, end):
        dur = max(end - start, 1e-6)
        sel = (self.t >= start) & (self.t < end)
        rms = self.rms[sel]
        out = {}

        rms_mean = float(rms.mean()) if rms.size else 0.0
        bass_mean = float(self.bass[sel].mean()) if rms.size else 0.0
        out["energy_pct"] = round(self._pct(self.rms, rms_mean), 1)
        # Band fields are raw power sums (not total_rms units) — express bass
        # as a set-relative percentile, same convention as brightness.
        out["bass_pct"] = round(self._pct(self.bass, bass_mean), 1)
        out["energy_arc"] = self._arc(rms)

        sp_sel = (self.sp_t >= start) & (self.sp_t < end)
        cen = self.centroid[sp_sel]
        out["brightness_pct"] = round(
            self._pct(self.centroid, float(cen.mean())), 1) if cen.size else 0.0

        hp_sel = (self.hp_t >= start) & (self.hp_t < end)
        h, p = self.harm[hp_sel], self.perc[hp_sel]
        total = float(h.mean() + p.mean()) if h.size else 0.0
        out["percussive_ratio"] = round(float(p.mean()) / total, 3) if total > 0 else 0.0

        n_onsets = int(((self.onset_times >= start) & (self.onset_times < end)).sum())
        out["onset_rate_per_sec"] = round(n_onsets / dur, 2)

        bpm_sel = (self.bpm_t >= start) & (self.bpm_t < end)
        bpms = self.bpm[bpm_sel]
        out["bpm"] = round(float(np.median(bpms)), 1) if bpms.size else None

        votes = {}
        for k in self.keys:
            if start <= k["time_sec"] < end:
                votes[k["label"]] = votes.get(k["label"], 0.0) + k.get("confidence", 1.0)
        out["key"] = max(votes, key=votes.get) if votes else None

        return out


# ─── Lyrics Merge ──────────────────────────────────────────────────────────

def load_lyrics_segments(set_name):
    path = OUTPUT_DIR / f"{set_name}.lyrics.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("segments") or []
    except (json.JSONDecodeError, OSError):
        return []


def section_lyrics(segments, start, end):
    """Keywords + a representative line for segments inside [start, end)."""
    hits = [s for s in segments
            if start <= (s["start_sec"] + s["end_sec"]) / 2 < end
            and s.get("no_speech_prob", 0) < 0.5]
    if not hits:
        return None
    counts = {}
    for s in hits:
        for kw in s.get("keywords", []):
            counts[kw] = counts.get(kw, 0) + 1
    keywords = sorted(counts, key=counts.get, reverse=True)[:12]
    rep = max(hits, key=lambda s: s.get("confidence", 0))
    return {"keywords": keywords, "line": rep.get("text", "").strip()}


# ─── LLM Composition ───────────────────────────────────────────────────────

SECTION_PROMPT = """\
You are writing semantic metadata for one section of a DJ set. The metadata \
drives selection of abstract visuals for a generated music video.

SECTION {idx} of {n} — {t0} to {t1} ({dur:.0f}s of a {total_min:.0f}-minute set)
{title_line}
Signal analysis (relative to this set):
- BPM: {bpm}
- Key: {key}
- Energy: {energy_pct:.0f}th percentile, arc: {energy_arc}
- Bass weight: {bass_pct:.0f}th percentile
- Brightness: {brightness_pct:.0f}th percentile
- Percussive vs harmonic balance: {percussive_ratio:.0%} percussive
- Onset density: {onset_rate_per_sec:.1f} events/sec

Zero-shot audio tags (CLAP similarity — hints, not ground truth; trust higher \
scores more):
{tag_block}
{lyrics_block}
Respond with ONLY a JSON object — no code fences, no commentary:
{{
  "phrases": ["4-6 short evocative phrases, 3-8 words each, describing how this section sounds and feels"],
  "summary": "one sentence",
  "visual_keywords": ["6-10 single or two-word tokens describing visuals suited to this music: motion, color temperature, texture, mood"],
  "energy": "low|medium|high",
  "mood": ["2-4 single-word moods"]
}}
Ground every claim in the data above. Never guess artist or track names.
"""

GLOBAL_PROMPT = """\
Below are one-line summaries of the {n} sections of a {total_min:.0f}-minute \
DJ set, in order. Write a JSON object describing the set's overall arc:

{section_lines}

Respond with ONLY a JSON object — no code fences, no commentary:
{{
  "summary": "2-3 sentences on the set's journey and overall character",
  "phrases": ["3-5 short phrases describing the set as a whole"]
}}
"""


def format_tag_block(tags):
    lines = []
    for axis, entries in tags.items():
        terms = ", ".join(f"{t} ({p:.2f})" for t, p, _ in entries)
        lines.append(f"- {axis}: {terms}")
    return "\n".join(lines)


def compose_section(backend, section, features, tags, lyrics, n_sections,
                    total_sec, retries=1):
    title_line = (f"Tracklist title: {section['title']}\n"
                  if section.get("title") else "")
    lyrics_block = ""
    if lyrics:
        lyrics_block = (
            f"\nLyrics detected (Whisper, noisy): keywords "
            f"{', '.join(lyrics['keywords'])}; sample line: \"{lyrics['line']}\"\n")

    prompt = SECTION_PROMPT.format(
        idx=section["index"] + 1, n=n_sections,
        t0=fmt_time(section["start_sec"]), t1=fmt_time(section["end_sec"]),
        dur=section["end_sec"] - section["start_sec"],
        total_min=total_sec / 60,
        title_line=title_line,
        bpm=features.get("bpm") or "unknown",
        key=features.get("key") or "unknown",
        tag_block=format_tag_block(tags),
        lyrics_block=lyrics_block,
        **{k: features[k] for k in
           ("energy_pct", "energy_arc", "bass_pct", "brightness_pct",
            "percussive_ratio", "onset_rate_per_sec")},
    )

    last_err = None
    for _ in range(retries + 1):
        text, elapsed, err = backend.analyze_text(prompt)
        if err:
            last_err = err
            continue
        obj = parse_llm_json(text)
        if obj is None:
            last_err = f"unparseable response: {text[:80]!r}"
            continue
        energy = str(obj.get("energy", "")).lower()
        return {
            "phrases": _str_list(obj.get("phrases"), 6),
            "summary": str(obj.get("summary", "")).strip(),
            "visual_keywords": _str_list(obj.get("visual_keywords"), 10),
            "energy": energy if energy in ("low", "medium", "high") else None,
            "mood": _str_list(obj.get("mood"), 4),
        }, elapsed, None
    return None, 0.0, last_err


def compose_global(backend, sections, total_sec):
    lines = []
    for s in sections:
        desc = s.get("description") or {}
        summary = desc.get("summary") or "(no summary)"
        lines.append(f"{s['index'] + 1}. [{fmt_time(s['start_sec'])}] {summary}")
    prompt = GLOBAL_PROMPT.format(n=len(sections), total_min=total_sec / 60,
                                  section_lines="\n".join(lines))
    text, _, err = backend.analyze_text(prompt)
    if err:
        return None, err
    obj = parse_llm_json(text)
    if obj is None:
        return None, "unparseable response"
    return {"summary": str(obj.get("summary", "")).strip(),
            "phrases": _str_list(obj.get("phrases"), 5)}, None


# ─── Per-Set Pipeline ──────────────────────────────────────────────────────

def resolve_audio(set_name, cfg):
    """Prefer a local WAV sibling of the analysis over the archive MP3."""
    local = OUTPUT_DIR / cfg["analysis"].replace(".deep-analysis.json", ".wav")
    if local.exists():
        return local
    return Path(cfg["audio"])


def enrich_set(set_name, analysis_path, audio_path, tagger, backend, args):
    print(f"\n{'─' * 70}")
    print(f"  SET: {set_name}")
    print(f"  Analysis: {analysis_path.name}")
    print(f"  Audio: {audio_path}")

    if not analysis_path.exists():
        print("  SKIP — deep analysis not found (run analyze_dj_set_deep.py first)")
        return False
    if not audio_path.exists():
        print("  SKIP — audio not found (archive drive mounted?)")
        return False

    out_path = OUTPUT_DIR / f"{set_name}.semantic.json"
    if out_path.exists() and not args.force:
        print(f"  SKIP — {out_path.name} exists (use --force to redo)")
        return True

    deep = json.loads(analysis_path.read_text())
    total_sec = _set_duration(deep)
    sections = build_sections(deep, args.section_sec)
    if args.limit_sections:
        sections = sections[:args.limit_sections]
    digest = FeatureDigest(deep)
    lyric_segments = load_lyrics_segments(set_name)
    reader = AudioReader(audio_path)

    print(f"  Duration: {fmt_time(total_sec)} | sections: {len(sections)} "
          f"({sections[0]['source']}) | lyrics: "
          f"{'yes' if lyric_segments else 'no'}")

    t_start = time.time()
    section_embs = []
    for sec in sections:
        s, e = sec["start_sec"], sec["end_sec"]
        dur = e - s

        # CLAP: evenly spaced windows, mean-pooled
        n_win = max(1, min(args.windows, int(dur // WINDOW_SEC) or 1))
        starts = np.linspace(s, max(s, e - WINDOW_SEC), n_win)
        windows = [reader.window(w, WINDOW_SEC) for w in starts]
        emb = tagger.embed_windows(windows)
        section_embs.append(emb)
        sec["tags"] = tagger.tag(emb)

        sec["features"] = digest.section(s, e)
        lyr = section_lyrics(lyric_segments, s, e)
        if lyr:
            sec["lyrics"] = lyr

        if not args.skip_llm:
            desc, elapsed, err = compose_section(
                backend, sec, sec["features"], sec["tags"], lyr,
                len(sections), total_sec)
            sec["description"] = desc
            if err:
                sec["description_error"] = err
            status = "ok" if desc else f"LLM FAILED: {err}"
        else:
            status = "tags only"

        top_genre = sec["tags"].get("genre", [["?"]])[0][0]
        print(f"    [{sec['index'] + 1:>3}/{len(sections)}] "
              f"{fmt_time(s)}–{fmt_time(e)}  {top_genre:<18} {status}",
              flush=True)

    reader.close()

    # Set-level: mean of section embeddings → global tags (+ LLM arc summary)
    valid = [e for e in section_embs if e is not None]
    global_block = {
        "bpm": (deep.get("global") or {}).get("bpm"),
        "key": (deep.get("global") or {}).get("key"),
        "duration_sec": round(total_sec, 1),
    }
    if valid:
        import torch
        mean_emb = torch.stack(valid).mean(dim=0)
        mean_emb = mean_emb / mean_emb.norm()
        global_block["tags"] = tagger.tag(mean_emb)
    if not args.skip_llm:
        gdesc, gerr = compose_global(backend, sections, total_sec)
        if gdesc:
            global_block["description"] = gdesc
        elif gerr:
            global_block["description_error"] = gerr

    output = {
        "schema_version": SCHEMA_VERSION,
        "set_name": set_name,
        "audio_source": str(audio_path),
        "analysis_source": analysis_path.name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "clap": CLAP_MODEL,
            "llm": None if args.skip_llm else backend.provenance(),
        },
        "params": {
            "section_target_sec": args.section_sec,
            "windows_per_section": args.windows,
            "window_sec": WINDOW_SEC,
        },
        "global": global_block,
        "sections": sections,
    }
    out_path.write_text(json.dumps(output, indent=1, ensure_ascii=False))

    n_described = sum(1 for s in sections if s.get("description"))
    print(f"  Wrote {out_path.name} — {len(sections)} sections, "
          f"{n_described} described, {fmt_time(time.time() - t_start)} elapsed")
    return True


# ─── Status ────────────────────────────────────────────────────────────────

def show_status():
    print(f"\n  {'set':<28} {'deep':>5} {'lyrics':>7} {'semantic':>9}")
    print(f"  {'─' * 52}")
    for set_name, cfg in SET_CONFIGS.items():
        deep_ok = (OUTPUT_DIR / cfg["analysis"]).exists()
        lyr_ok = (OUTPUT_DIR / f"{set_name}.lyrics.json").exists()
        sem_path = OUTPUT_DIR / f"{set_name}.semantic.json"
        if sem_path.exists():
            try:
                n = len(json.loads(sem_path.read_text()).get("sections", []))
                sem = f"✓ ({n})"
            except (json.JSONDecodeError, OSError):
                sem = "corrupt"
        else:
            sem = "—"
        print(f"  {set_name:<28} {'✓' if deep_ok else '—':>5} "
              f"{'✓' if lyr_ok else '—':>7} {sem:>9}")
    print()


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="CLAP tags + LLM descriptive phrases per DJ-set section")
    target = ap.add_mutually_exclusive_group()
    target.add_argument("--set", help="set name from SET_CONFIGS")
    target.add_argument("--all", action="store_true", help="all configured sets")
    target.add_argument("--input", help="arbitrary audio file (needs a "
                        "<stem>.deep-analysis.json in ./sets/)")
    target.add_argument("--status", action="store_true")
    ap.add_argument("--backend", default=None,
                    help="text backend: ollama | claude-cli | anthropic-api "
                         "(default: $GHOST_TEXT_BACKEND or ollama)")
    ap.add_argument("--llm-model", default=None, help="model for the backend")
    ap.add_argument("--skip-llm", action="store_true",
                    help="CLAP tags + features only, no phrase composition")
    ap.add_argument("--section-sec", type=float, default=SECTION_TARGET_SEC)
    ap.add_argument("--windows", type=int, default=WINDOWS_PER_SECTION)
    ap.add_argument("--device", default=None, help="torch device (default: mps)")
    ap.add_argument("--force", action="store_true", help="overwrite existing")
    ap.add_argument("--limit-sections", type=int, default=0,
                    help="process only the first N sections (testing)")
    args = ap.parse_args()

    if args.status:
        show_status()
        return

    # Resolve targets
    if args.set:
        if args.set not in SET_CONFIGS:
            sys.exit(f"unknown set '{args.set}' — options: "
                     f"{', '.join(SET_CONFIGS)}")
        targets = [args.set]
    elif args.all:
        targets = list(SET_CONFIGS)
    elif args.input:
        targets = None
    else:
        ap.print_help()
        return

    backend = None
    if not args.skip_llm:
        name = args.backend or os.environ.get("GHOST_TEXT_BACKEND") or "ollama"
        backend = get_backend(name, args.llm_model)
        ok, msg = backend.health_check()
        print(f"  Text backend {backend.label}: {msg}")
        if not ok:
            sys.exit("  Backend unhealthy — fix it or pass --skip-llm")

    tagger = ClapTagger(device=args.device)

    if targets is None:
        audio_path = Path(args.input)
        stem = audio_path.stem
        analysis_path = OUTPUT_DIR / f"{stem}.deep-analysis.json"
        enrich_set(stem, analysis_path, audio_path, tagger, backend, args)
    else:
        for set_name in targets:
            cfg = SET_CONFIGS[set_name]
            enrich_set(set_name, OUTPUT_DIR / cfg["analysis"],
                       resolve_audio(set_name, cfg), tagger, backend, args)

    print()


if __name__ == "__main__":
    main()
