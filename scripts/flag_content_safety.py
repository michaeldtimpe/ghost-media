"""Flag scenes with disturbing/inappropriate content for hard exclusion.

We already have a rich VLM `semantic` per scene (visual_description, content_tags,
canonical_tags, mood.tone) from enrichment. Rather than pay for a second VLM
pass, this scans those descriptions for a curated unsafe-content lexicon and
writes `content_safety_flags.json` — a {source_name: [scene_index,...]} map the
assembler hard-excludes (conservative auto-exclude: a single match flags the
scene). It is intentionally broad (errs toward removing borderline content).

Matching is word-boundary regex over the lowercased description+tags+tone, so
"gun" doesn't fire on "begun". Mood adjectives that are ubiquitous on abstract
footage ("dark", "intense") are deliberately NOT in the lexicon — only concrete
violent/gore/sexual/weapon/disturbing content is.

Usage:
    python scripts/flag_content_safety.py            # scan + write
    python scripts/flag_content_safety.py --dry-run  # report only
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENRICHED_DIR = BASE_DIR / "enriched"
OUT_PATH = BASE_DIR / "content_safety_flags.json"

# HIGH-PRECISION unsafe content. The abstract/VJ corpus describes itself with
# words that collide with a naive violence lexicon — "graphic" (motion graphics),
# "shooting" (light), "hanging" (objects), "blade" (light blades), "blood-red"
# (a colour). Those are deliberately EXCLUDED; only terms that concretely denote
# gore/weapons/sexual content (and rarely fire on abstract footage) are kept.
UNSAFE_TERMS = [
    # gore / death (concrete)
    r"\bcorpse\b", r"dead bod(?:y|ies)", r"\bgore\b", r"\bgory\b", r"decapitat\w*",
    r"mutilat\w*", r"dismember\w*", r"\bmassacre\b", r"behead\w*", r"impaled",
    r"\bslaughter\w*", r"\btortur\w*", r"self.?harm", r"\bsuicide\b",
    r"\bbloody\b", r"blood[- ]?(?:soaked|spatter|splatter|stain\w*|drenched)",
    r"covered in blood", r"pool of blood",
    # weapons (concrete, in use)
    r"\bgun\b", r"\bguns\b", r"\bfirearm", r"\bpistol\b", r"\brifle\b", r"\bshotgun\b",
    r"\bmachete\b", r"\bknife\b", r"\bstab(?:bing|bed)\b", r"\bgunshot",
    # sexual / nudity
    r"\bnude\b", r"\bnaked\b", r"\bnudity\b", r"sexual\w*", r"\berotic\w*",
    r"genital\w*", r"\bporn\w*", r"\bfetish\b", r"\blewd\b",
    # overtly disturbing (concrete)
    r"\bgrotesque\b", r"\bgruesome\b",
]
PATTERNS = [re.compile(t, re.IGNORECASE) for t in UNSAFE_TERMS]


def _scene_text(semantic: dict) -> str:
    if not isinstance(semantic, dict):
        return ""
    parts = [semantic.get("visual_description", "") or ""]
    parts += semantic.get("content_tags", []) or []
    parts += semantic.get("canonical_tags", []) or []
    mood = semantic.get("mood", {})
    if isinstance(mood, dict):
        parts.append(mood.get("tone", "") or "")
    return " ".join(parts).lower()


def _matches(text: str):
    hits = []
    for pat in PATTERNS:
        m = pat.search(text)
        if m:
            hits.append(m.group(0))
    return hits


def scan():
    flagged = {}            # source_name → [scene_index]
    term_counts = Counter()
    samples = []
    n_scenes = n_flagged = 0
    for f in sorted(ENRICHED_DIR.glob("*.enriched.json")):
        data = json.loads(f.read_text())
        name = data.get("file", {}).get("name", f.stem)
        sl = data.get("scenes", {})
        sl = sl.get("scenes", []) if isinstance(sl, dict) else sl
        fa = {a.get("scene_index"): a.get("analysis", {}) for a in data.get("frame_analyses", [])}
        for sc in sl:
            si = sc.get("scene_index")
            sem = sc.get("semantic") or fa.get(si, {})
            n_scenes += 1
            hits = _matches(_scene_text(sem))
            if hits:
                flagged.setdefault(name, []).append(si)
                term_counts.update(hits)
                n_flagged += 1
                if len(samples) < 25:
                    desc = (sem.get("visual_description", "") if isinstance(sem, dict) else "")[:90]
                    samples.append((name[:32], si, hits[:3], desc))
    return flagged, term_counts, samples, n_scenes, n_flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    flagged, term_counts, samples, n_scenes, n_flagged = scan()
    print(f"  scanned {n_scenes} scenes across {len(list(ENRICHED_DIR.glob('*.enriched.json')))} files")
    print(f"  flagged {n_flagged} scenes ({100*n_flagged/max(n_scenes,1):.1f}%) "
          f"across {len(flagged)} sources")
    print(f"  top terms: {dict(term_counts.most_common(12))}")
    print("  samples:")
    for name, si, hits, desc in samples:
        print(f"    {name:<34} scene {si:>4}  {hits}  — {desc}")

    if args.dry_run:
        print("  (dry-run: not written)")
        return 0
    OUT_PATH.write_text(json.dumps({
        "flagged": {k: sorted(v) for k, v in flagged.items()},
        "n_flagged": n_flagged, "n_scenes": n_scenes,
        "lexicon": UNSAFE_TERMS,
    }, indent=2, ensure_ascii=False))
    print(f"  ✓ wrote {OUT_PATH.name} ({n_flagged} flagged scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
