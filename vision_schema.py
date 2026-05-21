#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  VISION ENRICHMENT SCHEMA
  Single source of truth for the frame-analysis prompt + enum vocabulary, plus a
  normalizer that snaps hallucinated/loose model output back onto the schema.
═══════════════════════════════════════════════════════════════════════════════

Why this exists
---------------
The enrichment prompt lists enum options, but the model can still return values
outside them (e.g. "trippy" for visual_style, "super" for mood.energy). Those
junk values flow straight into the assembler's scoring (visual_style drives the
style-hint bonus, mood.energy drives the semantic bonus), silently degrading
clip selection. `normalize_analysis` enforces the enum vocabulary and coerces the
list fields, recording every correction under a reserved `_validation` key so the
original model output stays inspectable.

Both `enrich_analyses.py` and `cleanup_enrichment.py` import `SCENE_PROMPT` and
`normalize_analysis` from here so the prompt never drifts between them.

Note: the enum vocabularies below are ghost-media's ACTUAL values (the ones the
53 existing enriched files and `assemble_v2.py` scoring already depend on). Only
`"unclear"` was added as an honest escape value — do not change the others.
"""

import copy
import difflib

# ─── Prompt (single source of truth) ─────────────────────────────────────────

SCENE_PROMPT = """Analyze this video frame for use in an LLM-driven visual generation system. Provide a structured JSON response with these fields:

{
  "visual_description": "Concrete description of what you see — objects, shapes, patterns, textures, people, environments. Be specific.",
  "color_palette": {
    "dominant_colors": ["list of 3-5 color names"],
    "color_relationship": "complementary | analogous | monochromatic | triadic | split-complementary | unclear",
    "temperature": "warm | cool | neutral | mixed | unclear"
  },
  "composition": {
    "layout": "description of spatial arrangement, depth, focal points",
    "implied_motion": "none | pan_left | pan_right | zoom_in | zoom_out | rotation_cw | rotation_ccw | drift | pulse | spiral | complex | unclear",
    "motion_description": "brief description of perceived movement or camera motion"
  },
  "visual_style": "photorealistic | abstract | geometric | organic | psychedelic | minimalist | glitch | cinematic | motion_graphics | particle | fractal | liquid | mixed | unclear",
  "mood": {
    "tone": "description of emotional quality",
    "energy": "calm | building | moderate | intense | chaotic | unclear"
  },
  "transition_anchors": ["list of visual elements that could serve as transition points"],
  "content_tags": ["list of descriptive tags for searchability"]
}

Pick exactly one option for each enum field. If a frame genuinely doesn't fit any
option, use "unclear" rather than inventing a new value.

Return ONLY valid JSON, no markdown or explanation."""


# ─── Schema vocabulary ───────────────────────────────────────────────────────
# Dotted paths into the analysis object. Each enum field MUST end up holding one
# of these values (or "unclear").

ENUMS = {
    "color_palette.color_relationship": [
        "complementary", "analogous", "monochromatic", "triadic",
        "split-complementary", "unclear",
    ],
    "color_palette.temperature": ["warm", "cool", "neutral", "mixed", "unclear"],
    "composition.implied_motion": [
        "none", "pan_left", "pan_right", "zoom_in", "zoom_out",
        "rotation_cw", "rotation_ccw", "drift", "pulse", "spiral",
        "complex", "unclear",
    ],
    "visual_style": [
        "photorealistic", "abstract", "geometric", "organic", "psychedelic",
        "minimalist", "glitch", "cinematic", "motion_graphics", "particle",
        "fractal", "liquid", "mixed", "unclear",
    ],
    "mood.energy": ["calm", "building", "moderate", "intense", "chaotic", "unclear"],
}

# Dotted paths that should always be flat lists of strings.
LIST_FIELDS = ("color_palette.dominant_colors", "transition_anchors", "content_tags")

# Reserved keys added by the pipeline; downstream consumers must ignore these
# (the assembler reads only named fields, so it already does).
RESERVED_KEYS = ("_validation", "_provenance")


# ─── Dotted-path helpers ─────────────────────────────────────────────────────

def _get_path(obj, path):
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def _set_path(obj, path, value):
    cur = obj
    parts = path.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _canon(s):
    """Loose canonical form for matching: lowercase, strip separators."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# ─── Normalizers ─────────────────────────────────────────────────────────────

def _snap_enum(raw, allowed):
    """Snap a single value onto the allowed set. Returns the chosen value."""
    if not isinstance(raw, str) or not raw.strip():
        return "unclear"
    s = raw.lower().strip()

    # 1. exact
    if s in allowed:
        return s

    # 2. separator-insensitive exact ("motion graphics" -> "motion_graphics")
    canon_map = {_canon(a): a for a in allowed}
    if _canon(s) in canon_map:
        return canon_map[_canon(s)]

    # 3. substring either direction (skip the trivial "unclear" unless explicit)
    cs = _canon(s)
    for a in allowed:
        if a == "unclear":
            continue
        ca = _canon(a)
        if ca and (ca in cs or cs in ca):
            return a

    # 4. fuzzy
    close = difflib.get_close_matches(s, [a for a in allowed if a != "unclear"],
                                      n=1, cutoff=0.6)
    if close:
        return close[0]

    # 5. give up honestly
    return "unclear"


def _coerce_list(raw):
    """Coerce a field into a flat list of non-empty strings.

    Handles: real lists, comma/semicolon-delimited strings, and nested dicts
    (e.g. a dominant-color object) — all of which models emit in practice.
    """
    if raw is None:
        return []

    if isinstance(raw, str):
        parts = [p.strip() for p in raw.replace(";", ",").split(",")]
        return [p for p in parts if p]

    if isinstance(raw, dict):
        raw = [raw]

    if not isinstance(raw, (list, tuple)):
        return [str(raw).strip()] if str(raw).strip() else []

    out = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append(s)
        elif isinstance(item, dict):
            # Pull the most string-like value out of a nested object.
            val = None
            for k in ("name", "color", "tag", "label", "value", "hex"):
                if isinstance(item.get(k), str) and item[k].strip():
                    val = item[k].strip()
                    break
            if val:
                out.append(val)
        else:
            s = str(item).strip()
            if s:
                out.append(s)
    return out


def normalize_analysis(parsed):
    """Validate + normalize one parsed frame analysis.

    Returns ``(clean, notes)``:
      - ``clean``: a corrected copy. If anything was changed, a ``_validation``
        key maps each touched field to ``{"raw": <original>, "normalized": <new>}``.
      - ``notes``: the same correction dict (empty if nothing changed), handy for
        logging without re-reading the object.

    ``parsed`` of ``None`` passes through unchanged.
    """
    if not isinstance(parsed, dict):
        return parsed, {}

    clean = copy.deepcopy(parsed)
    notes = {}

    # Enum fields
    for path, allowed in ENUMS.items():
        raw, present = _get_path(clean, path)
        if not present:
            continue
        snapped = _snap_enum(raw, allowed)
        if snapped != raw:
            notes[path] = {"raw": raw, "normalized": snapped}
            _set_path(clean, path, snapped)

    # List fields
    for path in LIST_FIELDS:
        raw, present = _get_path(clean, path)
        if not present:
            continue
        coerced = _coerce_list(raw)
        if coerced != raw:
            notes[path] = {"raw": raw, "normalized": coerced}
            _set_path(clean, path, coerced)

    if notes:
        clean["_validation"] = notes

    return clean, notes
