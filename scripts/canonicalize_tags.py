"""Tag canonicalization: map free-form VLM tags onto a canonical vocabulary.

The corpus carries ~10k distinct free-form content_tags ("symmetry" vs
"symmetrical", "geometric shapes" vs "geometric pattern", "sea" vs "ocean"),
and the assembler's lyric↔tag dimension matches them lexically — synonyms
never match. This builds a canonical vocabulary and maps every tag to it via
CLIP text-embedding similarity. Zero VLM cost; hours → minutes of local MPS.

Pipeline:
  1. Collect tag frequencies from enriched/*.enriched.json (scene semantics
     + frame_analyses).
  2. Canonical vocabulary: walk tags by frequency, keeping a tag as canonical
     only if its CLIP cosine to every already-kept canonical is below
     CANON_DEDUP_SIM — the vocabulary self-dedupes — until VOCAB_SIZE terms.
  3. Map every distinct tag to its nearest canonical term (cosine ≥
     MAP_SIM_MIN; weaker matches stay unmapped).
  4. Write canonical_tags.json (terms + embeddings + mapping) and additively
     stamp `canonical_tags` next to every `content_tags` in the enriched
     JSONs (scene semantic + frame_analyses), preserving everything else.

The assembler loads canonical_tags.json to map lyric keywords into the same
space at runtime, so "ocean" (lyric) finally matches "sea" (tag).

Usage:
  python3 scripts/canonicalize_tags.py            # build + apply
  python3 scripts/canonicalize_tags.py --dry-run  # build + report, no writes
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent
ENRICHED_DIR = BASE_DIR / "enriched"
VOCAB_PATH = BASE_DIR / "canonical_tags.json"

VOCAB_SIZE = 200
CANON_DEDUP_SIM = 0.92   # candidate too close to an existing canonical → skip
MAP_SIM_MIN = 0.80       # tag→canonical mapping floor; below = unmapped
MIN_TAG_USES = 2         # ignore single-use tags when building the vocabulary


def iter_tag_holders(data):
    """Yield every dict in an enriched JSON that carries content_tags."""
    sd = data.get("scenes", {})
    scene_list = sd.get("scenes", []) if isinstance(sd, dict) else sd
    for s in scene_list:
        sem = s.get("semantic")
        if sem and sem.get("content_tags"):
            yield sem
    for fa in data.get("frame_analyses", []):
        an = fa.get("analysis")
        if an and an.get("content_tags"):
            yield an


def collect_tags():
    counts = Counter()
    files = sorted(ENRICHED_DIR.glob("*.enriched.json"))
    for f in files:
        data = json.loads(f.read_text())
        for holder in iter_tag_holders(data):
            for t in holder["content_tags"]:
                t = str(t).lower().strip()
                if t:
                    counts[t] += 1
    return counts, files


def build_vocabulary(counts, embed):
    """Greedy frequency-ordered vocabulary with CLIP-cosine dedup."""
    ranked = [t for t, c in counts.most_common() if c >= MIN_TAG_USES]
    kept_terms = []
    kept_embs = []
    for tag in ranked:
        if len(kept_terms) >= VOCAB_SIZE:
            break
        emb = embed[tag]
        if kept_embs and float(np.max(np.stack(kept_embs) @ emb)) >= CANON_DEDUP_SIM:
            continue
        kept_terms.append(tag)
        kept_embs.append(emb)
    return kept_terms, np.stack(kept_embs)


def main() -> int:
    global VOCAB_SIZE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="build the vocabulary + mapping and report, write nothing")
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = p.parse_args()
    VOCAB_SIZE = args.vocab_size

    print("Collecting tags...", flush=True)
    counts, files = collect_tags()
    distinct = list(counts)
    print(f"  {len(distinct)} distinct tags, {sum(counts.values())} uses, "
          f"{len(files)} enriched files")

    print("Encoding tags with CLIP (text encoder)...", flush=True)
    import torch  # noqa: F401  (ensures torch import error surfaces early)
    from clip_utils import load_clip, encode_text
    load_clip(verbose=True)
    # Batch-encode via encode_text one by one is slow; use the tokenizer batch.
    import clip_utils as cu
    embs = {}
    B = 256
    for i in range(0, len(distinct), B):
        batch = distinct[i:i + B]
        tokens = cu._clip_tokenizer(batch).to(cu._device)
        with cu.torch.no_grad():
            feats = cu._clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        for t, e in zip(batch, feats.cpu().numpy().astype(np.float32)):
            embs[t] = e
        if (i // B) % 10 == 0:
            print(f"  {min(i + B, len(distinct))}/{len(distinct)}", flush=True)

    print("Building canonical vocabulary...", flush=True)
    terms, term_embs = build_vocabulary(counts, embs)
    print(f"  {len(terms)} canonical terms (dedup sim {CANON_DEDUP_SIM})")

    print("Mapping all tags to canonical terms...", flush=True)
    mapping = {}
    sims_used = []
    for tag in distinct:
        sims = term_embs @ embs[tag]
        best = int(np.argmax(sims))
        if float(sims[best]) >= MAP_SIM_MIN:
            mapping[tag] = terms[best]
            sims_used.append(float(sims[best]))
    coverage_uses = sum(c for t, c in counts.items() if t in mapping)
    print(f"  mapped {len(mapping)}/{len(distinct)} tags "
          f"({coverage_uses / sum(counts.values()):.1%} of uses), "
          f"median map cosine {np.median(sims_used):.3f}")

    # Show a sample of non-identity mappings for eyeballing.
    samples = [(t, m) for t, m in mapping.items() if t != m][:0]
    interesting = sorted(((counts[t], t, m) for t, m in mapping.items() if t != m),
                         reverse=True)[:15]
    print("  sample mappings (by tag frequency):")
    for c, t, m in interesting:
        print(f"    {t!r} → {m!r}  ({c} uses)")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # No embeddings persisted: the assembler re-encodes the 200 terms at
    # runtime (it already loads CLIP for lyric matching) — keeps this file
    # small enough to commit alongside the perceptual baselines.
    VOCAB_PATH.write_text(json.dumps({
        "model": "ViT-B-32/openai",
        "vocab_size": len(terms),
        "dedup_sim": CANON_DEDUP_SIM,
        "map_sim_min": MAP_SIM_MIN,
        "terms": terms,
        "mapping": mapping,
    }, indent=1))
    print(f"\nwrote {VOCAB_PATH.name} ({VOCAB_PATH.stat().st_size / 1e6:.1f} MB)")

    print("Stamping canonical_tags into enriched files...", flush=True)
    n_holders = 0
    for f in files:
        data = json.loads(f.read_text())
        changed = False
        for holder in iter_tag_holders(data):
            canon = sorted({mapping[str(t).lower().strip()]
                            for t in holder["content_tags"]
                            if str(t).lower().strip() in mapping})
            if holder.get("canonical_tags") != canon:
                holder["canonical_tags"] = canon
                changed = True
            n_holders += 1
        if changed:
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"  stamped {n_holders} tag holders across {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
