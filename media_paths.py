#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  MEDIA PATHS
  Single source of truth for footage/sets locations + local NVMe cache.
═══════════════════════════════════════════════════════════════════════════════

Canonical layout (all source footage in ONE place on the NAS):

    /Volumes/archive/3000/3100/visuals/library/
        raw/             (formerly "raw visuals footage")
        arche-loops/     (formerly "New Arche Loops")
        synesthesia/     (formerly "synesthesia custom scenes")
        hungry-ghost/    (formerly "hungry ghost import")
        _staging/        (drop new collections here before import)

Everything is env-overridable, which is also how the local cache works: warm a
collection onto local NVMe and point the whole pipeline at it —

    python3 media_paths.py --warm hungry-ghost
    GHOST_FOOTAGE_ROOT=~/ghost-media-cache/library python3 analyze_visuals_library.py ...

(5 decode passes per video — scene detect, motion, embeddings, text scan,
loops — read at NVMe speed instead of SMB. After cached runs, stored file
paths point into the cache; run scripts/remap_media_paths.py to canonicalize
them — they're advisory either way, the fuzzy index rescues stale paths.)

Filename-keyed joins everywhere: filenames NEVER change when files move
between collections; only directories do.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

FOOTAGE_ROOT = Path(os.environ.get(
    "GHOST_FOOTAGE_ROOT", "/Volumes/archive/3000/3100/visuals/library"))
SETS_ROOT = Path(os.environ.get(
    "GHOST_SETS_ROOT", "/Volumes/archive/3000/3100/sets"))
CACHE_ROOT = Path(os.environ.get(
    "GHOST_MEDIA_CACHE", str(Path.home() / "ghost-media-cache")))
CACHE_MAX_GB = float(os.environ.get("GHOST_MEDIA_CACHE_MAX_GB", "100"))

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg"}

COLLECTIONS = ["raw", "arche-loops", "synesthesia", "hungry-ghost", "_staging"]

# Pre-migration location, used as a fallback so a half-migrated checkout
# still resolves footage.
LEGACY_RAW_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")

_index = None  # filename → Path (one-time recursive scan)


def _nfc(name):
    """NFC-normalize a filename: macOS/SMB filesystems return NFD-decomposed
    names while stored JSON paths are NFC — without this, files with
    non-ASCII names silently fail dict lookups."""
    return unicodedata.normalize("NFC", name)


def iter_footage(refresh=False):
    """All video files under FOOTAGE_ROOT (recursive, cached, NFC-keyed).
    `_staging/` is excluded — it's pre-import parking, not live corpus."""
    global _index
    if _index is None or refresh:
        _index = {}
        roots = [FOOTAGE_ROOT] if FOOTAGE_ROOT.exists() else []
        if not roots and LEGACY_RAW_DIR.exists():
            roots = [LEGACY_RAW_DIR]
        for root in roots:
            for p in sorted(root.rglob("*")):
                if "_staging" in p.parts:
                    continue
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                    # First hit wins on duplicate filenames; the remap
                    # utility's --verify reports collisions explicitly.
                    _index.setdefault(_nfc(p.name), p)
    return _index


def _sanitize_for_match(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_source(file_info):
    """Resolve an analysis JSON's file-info dict to an actual video path.

    Same contract as the old assemble_v2.find_source_video: exact stored
    path first, then prefix match, then sanitized fuzzy match over the
    filename index. Returns the stored path string when nothing matches
    (downstream fails gracefully).
    """
    original_path = Path(file_info.get("path", "")) if isinstance(file_info, dict) \
        else Path(str(file_info))
    if original_path.exists():
        return str(original_path)
    index = iter_footage()
    if _nfc(original_path.name) in index:
        return str(index[_nfc(original_path.name)])
    stem = original_path.stem
    sanitized_target = _sanitize_for_match(stem)
    for candidate in index.values():
        if stem[:25] in candidate.stem or candidate.stem[:25] in stem:
            return str(candidate)
    for candidate in index.values():
        sc = _sanitize_for_match(candidate.stem)
        if len(sanitized_target) > 10 and len(sc) > 10:
            if sanitized_target[:25] in sc or sc[:25] in sanitized_target:
                return str(candidate)
    return str(original_path)


# ─── Local cache ───────────────────────────────────────────────────────────

def _cache_size_bytes():
    return sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file())


def _evict_lru(needed_bytes):
    """Evict least-recently-used cache files until needed_bytes fit."""
    budget = CACHE_MAX_GB * 1e9
    files = sorted((p for p in CACHE_ROOT.rglob("*") if p.is_file()),
                   key=lambda p: p.stat().st_atime)
    total = sum(p.stat().st_size for p in files)
    while files and total + needed_bytes > budget:
        victim = files.pop(0)
        total -= victim.stat().st_size
        victim.unlink(missing_ok=True)


def cached_path(nas_path):
    """Copy-through cache: local copy of a NAS file, keyed by relative path
    + (size, mtime). Returns the local path, or the original on any failure."""
    src = Path(nas_path)
    try:
        if not src.exists():
            return str(src)
        try:
            rel = src.relative_to(FOOTAGE_ROOT.parent)
        except ValueError:
            rel = Path(src.name)
        dst = CACHE_ROOT / rel
        st = src.stat()
        if dst.exists():
            dst_st = dst.stat()
            if dst_st.st_size == st.st_size and int(dst_st.st_mtime) == int(st.st_mtime):
                return str(dst)
        _evict_lru(st.st_size)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return str(dst)
    except OSError:
        return str(src)


def warm(collection):
    """rsync a library collection into the cache (ahead of a long batch)."""
    src = FOOTAGE_ROOT / collection
    if not src.is_dir():
        raise SystemExit(f"No such collection: {src}")
    dst = CACHE_ROOT / FOOTAGE_ROOT.name / collection
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"warming {src} → {dst}")
    # --progress, not --info=progress2: macOS ships a pre-3.1 rsync.
    subprocess.run(["rsync", "-a", "--progress",
                    str(src) + "/", str(dst) + "/"], check=True)
    print(f"done. Point the pipeline at the cache with:\n"
          f"  GHOST_FOOTAGE_ROOT={CACHE_ROOT / FOOTAGE_ROOT.name}")


def status():
    print(f"FOOTAGE_ROOT: {FOOTAGE_ROOT}  (exists: {FOOTAGE_ROOT.exists()})")
    print(f"SETS_ROOT:    {SETS_ROOT}  (exists: {SETS_ROOT.exists()})")
    print(f"CACHE_ROOT:   {CACHE_ROOT}  (max {CACHE_MAX_GB:.0f} GB)")
    index = iter_footage(refresh=True)
    print(f"footage index: {len(index)} files")
    by_dir = {}
    for p in index.values():
        by_dir.setdefault(str(p.parent), []).append(p)
    for d, files in sorted(by_dir.items()):
        size = sum(f.stat().st_size for f in files) / 1e9
        print(f"  {len(files):4d} files  {size:7.1f} GB  {d}")
    if CACHE_ROOT.exists():
        print(f"cache: {_cache_size_bytes() / 1e9:.1f} GB used")


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--warm", metavar="COLLECTION",
                   help=f"rsync a collection into the cache ({', '.join(COLLECTIONS)})")
    p.add_argument("--status", action="store_true", help="show roots + index summary")
    args = p.parse_args()
    if args.warm:
        warm(args.warm)
    else:
        status()


if __name__ == "__main__":
    main()
