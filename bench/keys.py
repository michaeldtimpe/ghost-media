#!/usr/bin/env python3
"""
Canonical identity for the bake-off.

lessons.md documents a live join bug: the original filename (with `& × ' ( )`) and
the on-disk sanitized stem are *different identities*, and the repo's three
`find_source_video` helpers all join with fragile `[:20]/[:25]` prefix slices —
which collide on `isshin …2022` vs `…2022-d557b`. To make benchmark records
collision-proof we key everything on a SHA-256 of the **absolute source path**:

    canonical_key(abs_path) -> 16-hex-char id

The human-readable `display_stem` (the analysis-file stem, already sanitized and
matching the existing `enriched/*` sidecars) is carried alongside for display and
for naming sidecars, but is NEVER used as a lookup key. If two distinct source
paths ever map to the same display_stem we abort loudly rather than silently drop
one (the failure mode the prefix heuristics hide).
"""

import hashlib
import os
from pathlib import Path


class StemCollisionError(RuntimeError):
    """Two distinct source paths sanitized to the same display stem."""


def canonical_key(abs_source_path):
    """16-hex-char SHA-256 of the absolute, normalized source path."""
    norm = os.path.abspath(str(abs_source_path))
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def display_stem(analysis_path):
    """The sanitized stem used by existing enriched sidecars.

    `Foo (Bar).analysis.json` -> `Foo _Bar_` is NOT what happens here — the
    analysis filename is already the sanitized form on disk, so we just strip the
    `.analysis.json` suffix. This is the exact stem `enriched/<stem>.enriched.json`
    / `.clip_embeddings.json` / `.quality.json` use, so sidecars line up.
    """
    name = Path(analysis_path).name
    return name[:-len(".analysis.json")] if name.endswith(".analysis.json") else Path(name).stem


def build_registry(entries):
    """Build the canonical identity registry, aborting on stem collisions.

    `entries`: iterable of (analysis_path, source_path). `source_path` may be None
    when the source video can't be resolved — those are kept (key off the analysis
    path) but flagged via `source_resolved=False` so the runner can skip extraction.

    Returns: dict display_stem -> {
        key, display_stem, analysis_path, source_path, source_resolved
    }
    """
    registry = {}
    for analysis_path, source_path in entries:
        stem = display_stem(analysis_path)
        key_basis = source_path or analysis_path
        rec = {
            "key": canonical_key(key_basis),
            "display_stem": stem,
            "analysis_path": str(analysis_path),
            "source_path": str(source_path) if source_path else None,
            "source_resolved": source_path is not None,
        }
        if stem in registry:
            prev = registry[stem]
            if prev["source_path"] != rec["source_path"]:
                raise StemCollisionError(
                    f"display_stem {stem!r} maps to two sources:\n"
                    f"  {prev['source_path']}\n  {rec['source_path']}\n"
                    f"Disambiguate the analysis filenames before benchmarking.")
            continue  # identical re-entry, harmless
        registry[stem] = rec
    return registry
