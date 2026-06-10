#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SCENE QUERY
  Search the scene corpus in natural language using the existing CLIP embeddings.
═══════════════════════════════════════════════════════════════════════════════

The blog's point: natural-language queryability is the real prerequisite layer
for editing. ghost-media already generates per-scene CLIP embeddings (used inside
the assembler) but never exposed them. This turns that latent index into a tool:

    python query_scenes.py "golden liquid high energy"
    python query_scenes.py "dark glitchy cityscape" --top 15 --min-quality 0.5
    python query_scenes.py "warm organic flowing" --diversity 0.5 --extract

Reuses clip_utils.encode_text (same model the embeddings were built with), joins
per-scene metadata + quality scores, and ranks by cosine similarity. Optional
MMR-style diversity suppression keeps near-identical scenes from dominating.
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from clip_utils import encode_text

BASE_DIR = Path(__file__).parent
ENRICHED_DIR = BASE_DIR / "enriched"


def fmt_time(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_index():
    """Stack every scene's CLIP vector + metadata. Returns (matrix, meta_list)."""
    vectors, meta = [], []
    emb_files = sorted(ENRICHED_DIR.glob("*.clip_embeddings.json"))
    for ef in emb_files:
        src = ef.name.replace(".clip_embeddings.json", "")
        emb = json.loads(ef.read_text())

        # Per-scene metadata from the enriched file (loaded once).
        scene_meta, file_path = {}, None
        enr = ENRICHED_DIR / f"{src}.enriched.json"
        if enr.exists():
            ed = json.loads(enr.read_text())
            file_path = ed.get("file", {}).get("path")
            sl = ed.get("scenes", {}).get("scenes", [])
            for s in sl:
                sem = s.get("semantic", {}) or {}
                scene_meta[s.get("scene_index")] = {
                    "start": s.get("start_sec", 0),
                    "end": s.get("end_sec", 0),
                    "style": sem.get("visual_style", ""),
                    "tags": sem.get("content_tags", []) or [],
                }

        # Quality scores (optional).
        qual = {}
        qf = ENRICHED_DIR / f"{src}.quality.json"
        if qf.exists():
            for qs in json.loads(qf.read_text()).get("scenes", []):
                qual[qs["scene_index"]] = qs.get("quality_score", 1.0)

        for se in emb.get("scenes", []):
            si = se["scene_index"]
            m = scene_meta.get(si, {})
            vectors.append(np.asarray(se["embedding"], dtype=np.float32))
            meta.append({
                "source": src, "scene_index": si,
                "start": m.get("start", se.get("frame_time_sec", 0)),
                "end": m.get("end", 0),
                "style": m.get("style", ""),
                "tags": m.get("tags", []),
                "quality": qual.get(si, 1.0),
                "file_path": file_path,
            })

    M = np.stack(vectors) if vectors else np.zeros((0, 512), dtype=np.float32)
    return M, meta


def mmr_rank(adjusted, M, top, lam, pool_mult=10):
    """Greedy maximal-marginal-relevance: pick high-scoring but mutually-diverse
    scenes. score = adjusted_sim - lam * max cosine to already-selected."""
    order = list(np.argsort(-adjusted))
    pool = order[:max(top * pool_mult, 100)]
    selected = []
    while len(selected) < top and pool:
        best, best_val = None, -1e18
        sel_mat = M[selected] if selected else None
        for idx in pool:
            if idx in selected:
                continue
            div = float(np.max(sel_mat @ M[idx])) if sel_mat is not None else 0.0
            val = float(adjusted[idx]) - lam * div
            if val > best_val:
                best_val, best = val, idx
        selected.append(best)
        pool.remove(best)
    return selected


def extract_preview(meta_row, dur, out_dir):
    """ffmpeg-extract a short preview of one scene to out_dir. Needs the drive."""
    from generate_clip_embeddings import find_source_video
    video = find_source_video({"path": meta_row.get("file_path") or ""})
    if not video or not Path(video).exists():
        return None
    out = out_dir / f"{meta_row['source'][:30].replace('/', '_')}_scene{meta_row['scene_index']}.mp4"
    start = meta_row["start"]
    clip_dur = max(1.0, min(dur, (meta_row["end"] - start) or dur))
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(video),
         "-t", str(clip_dur), "-avoid_negative_ts", "make_zero",
         "-c:v", "libx264", "-an", str(out)],
        capture_output=True)
    return out if r.returncode == 0 and out.exists() else None


def main():
    ap = argparse.ArgumentParser(description="Natural-language scene search over CLIP embeddings")
    ap.add_argument("query", help="Natural-language description, e.g. 'golden liquid high energy'")
    ap.add_argument("--top", type=int, default=10, help="Number of results")
    ap.add_argument("--min-quality", type=float, default=None,
                    help="Hard floor: drop scenes with quality_score below this")
    ap.add_argument("--quality-weight", action="store_true",
                    help="Soft: multiply similarity by quality_score instead of a hard floor")
    ap.add_argument("--diversity", type=float, default=0.0,
                    help="MMR diversity strength 0..1 (0 = pure similarity)")
    ap.add_argument("--extract", action="store_true",
                    help="ffmpeg-extract previews of the top hits to /tmp (needs archive drive)")
    ap.add_argument("--extract-dur", type=float, default=4.0, help="Preview length (s)")
    args = ap.parse_args()

    M, meta = build_index()
    if len(meta) == 0:
        print("  No CLIP embeddings found. Run generate_clip_embeddings.py first.")
        return

    q = encode_text(args.query)
    sims = M @ q  # both sides L2-normalized → cosine similarity

    adjusted = sims.copy()
    keep_mask = np.ones(len(meta), dtype=bool)
    if args.min_quality is not None:
        keep_mask = np.array([m["quality"] >= args.min_quality for m in meta])
    if args.quality_weight:
        adjusted = adjusted * np.array([m["quality"] for m in meta])
    adjusted = np.where(keep_mask, adjusted, -1e18)

    n_avail = int(keep_mask.sum())
    if args.diversity > 0:
        ranked = mmr_rank(adjusted, M, min(args.top, n_avail), args.diversity)
    else:
        ranked = list(np.argsort(-adjusted))[:min(args.top, n_avail)]

    print(f"\n  Query: \"{args.query}\"  ·  {len(meta)} scenes indexed"
          + (f"  ·  {n_avail} pass quality≥{args.min_quality}" if args.min_quality is not None else ""))
    print(f"  {'─' * 78}")
    for rank, idx in enumerate(ranked, 1):
        m = meta[idx]
        tags = ", ".join(m["tags"][:5])
        print(f"  {rank:>2}. sim={sims[idx]:.3f} q={m['quality']:.2f}  "
              f"{m['source'][:34]:<34}  sc{m['scene_index']:<5} "
              f"{fmt_time(m['start'])}-{fmt_time(m['end'])}  "
              f"[{m['style'] or '?'}]  {tags}")

    if args.extract:
        out_dir = Path("/tmp/ghost_query")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  Extracting {len(ranked)} previews → {out_dir}")
        for idx in ranked:
            out = extract_preview(meta[idx], args.extract_dur, out_dir)
            print(f"    {'✓ ' + out.name if out else '✗ ' + meta[idx]['source'][:40] + ' (source not found)'}")
    print()


if __name__ == "__main__":
    main()
