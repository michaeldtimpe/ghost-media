"""Remap stored media paths to the canonical library layout, and verify joins.

Stored paths live in three places:
  *.analysis.json            file.path  (repo root)
  enriched/*.enriched.json   file.path
  text_flags/*.text_flags.json  source

They're advisory (the filename index + fuzzy matcher rescue stale paths), but
rewriting them to the canonical location kills the ambiguity class for good
and makes --verify a meaningful health check after any move/import.

Default run = verify + report (no writes):
  - every analysis must resolve to exactly ONE file under FOOTAGE_ROOT
  - duplicate filenames across collections are reported (the hash-suffix
    poisoning class from lessons.md)
  - orphans (no resolvable source) are listed

--apply additionally rewrites the stored paths to the resolved canonical ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import media_paths as mp  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent


def collect_duplicate_filenames():
    """Filenames appearing more than once under FOOTAGE_ROOT."""
    seen = {}
    dups = {}
    if not mp.FOOTAGE_ROOT.exists():
        return dups
    for p in sorted(mp.FOOTAGE_ROOT.rglob("*")):
        if "_staging" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in mp.VIDEO_EXTS:
            name = mp._nfc(p.name)
            if name in seen:
                dups.setdefault(name, [seen[name]]).append(p)
            else:
                seen[name] = p
    return dups


def remap_file(path: Path, key_chain: list[str], apply: bool):
    """Resolve one stored path; return (status, old, new) and rewrite if apply."""
    data = json.loads(path.read_text())
    node = data
    for k in key_chain[:-1]:
        node = node.get(k, {})
    old = node.get(key_chain[-1], "")
    if not old:
        return "no-path", old, None
    resolved = mp.find_source({"path": old, "name": Path(old).name})
    if not Path(resolved).exists():
        return "orphan", old, None
    if resolved == old:
        return "ok", old, resolved
    if apply:
        node[key_chain[-1]] = resolved
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return "remapped", old, resolved


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apply", action="store_true",
                   help="rewrite stored paths (default: verify + report only)")
    args = p.parse_args()

    mp.iter_footage(refresh=True)
    print(f"FOOTAGE_ROOT: {mp.FOOTAGE_ROOT}")
    print(f"index: {len(mp.iter_footage())} video files\n")

    dups = collect_duplicate_filenames()
    if dups:
        print(f"⚠ DUPLICATE FILENAMES across collections ({len(dups)}):")
        for name, paths in dups.items():
            for q in paths:
                print(f"    {q}")
        print("  joins are filename-keyed — resolve these before importing!\n")
    else:
        print("✓ no duplicate filenames across collections\n")

    targets = (
        [(f, ["file", "path"]) for f in sorted(BASE_DIR.glob("*.analysis.json"))]
        + [(f, ["file", "path"]) for f in sorted((BASE_DIR / "enriched").glob("*.enriched.json"))]
        + [(f, ["source"]) for f in sorted((BASE_DIR / "text_flags").glob("*.text_flags.json"))]
    )
    counts = {"ok": 0, "remapped": 0, "orphan": 0, "no-path": 0}
    orphans = []
    for f, chain in targets:
        status, old, new = remap_file(f, chain, args.apply)
        counts[status] += 1
        if status == "orphan":
            orphans.append((f.name, old))

    print(f"checked {len(targets)} JSONs: "
          f"{counts['ok']} already canonical, "
          f"{counts['remapped']} {'remapped' if args.apply else 'remappable'}, "
          f"{counts['orphan']} orphans, {counts['no-path']} without a path")
    for name, old in orphans[:20]:
        print(f"  ORPHAN {name}: {old}")
    if len(orphans) > 20:
        print(f"  ... +{len(orphans) - 20} more")
    if not args.apply and counts["remapped"]:
        print("\nrun with --apply to rewrite")
    return 1 if (orphans or dups) else 0


if __name__ == "__main__":
    raise SystemExit(main())
