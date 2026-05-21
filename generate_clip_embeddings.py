#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  CLIP EMBEDDING GENERATOR
  Generate CLIP embeddings for each scene → sidecar JSON files
  Enables semantic similarity search and lyrics-to-visual matching
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import re
import subprocess
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ─── Configuration ─────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
ENRICHED_DIR = BASE_DIR / "enriched"
SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")
TMP_DIR = Path("/tmp/clip_embed_frames")

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 32


# ─── Source Resolution ────────────────────────────────────────────────────

def _sanitize_for_match(name):
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()


def find_source_video(file_info):
    """Find source video from enriched file_info."""
    original_path = Path(file_info.get("path", ""))
    if original_path.exists():
        return str(original_path)
    sanitized_target = _sanitize_for_match(original_path.stem)
    for candidate in SOURCE_DIR.iterdir():
        if original_path.stem[:25] in candidate.stem or candidate.stem[:25] in original_path.stem:
            return str(candidate)
    for candidate in SOURCE_DIR.iterdir():
        sanitized_candidate = _sanitize_for_match(candidate.stem)
        if len(sanitized_target) > 10 and len(sanitized_candidate) > 10:
            if sanitized_target[:25] in sanitized_candidate or sanitized_candidate[:25] in sanitized_target:
                return str(candidate)
    return None


# ─── Frame Extraction ────────────────────────────────────────────────────

def extract_frame(video_path, time_sec, output_path):
    """Extract a single frame via ffmpeg."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(max(0, time_sec)), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "2", str(output_path)],
            capture_output=True, timeout=30
        )
        return output_path.exists() and output_path.stat().st_size > 100
    except Exception:
        return False


def extract_scene_frames(video_path, scenes, tmp_dir):
    """Extract midpoint frame for each scene. Returns list of (scene_index, frame_time, path)."""
    results = []
    for scene in scenes:
        idx = scene["scene_index"]
        mid = (scene["start_sec"] + scene["end_sec"]) / 2
        frame_path = tmp_dir / f"scene_{idx:05d}.jpg"

        if extract_frame(video_path, mid, frame_path):
            results.append((idx, round(mid, 2), frame_path))

    return results


# ─── CLIP Inference ──────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = None


def _load_clip():
    """Load CLIP model (cached singleton)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _device
    if _clip_model is not None:
        return

    import open_clip

    if torch.backends.mps.is_available():
        _device = torch.device("mps")
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=_device
    )
    _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    _clip_model.eval()
    print(f"  CLIP model: {CLIP_MODEL}/{CLIP_PRETRAINED} on {_device}")


def encode_images(frame_paths, batch_size=BATCH_SIZE):
    """Encode images with CLIP. Returns list of 512-dim numpy arrays."""
    _load_clip()
    embeddings = []

    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(_clip_preprocess(img))
            except Exception:
                # Placeholder zero vector for failed images
                images.append(torch.zeros(3, 224, 224))

        batch = torch.stack(images).to(_device)
        with torch.no_grad():
            features = _clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)

        embeddings.extend(features.cpu().numpy().tolist())

    return embeddings


def encode_text(text):
    """Encode text with CLIP. Returns 512-dim numpy array."""
    _load_clip()
    tokens = _clip_tokenizer([text]).to(_device)
    with torch.no_grad():
        features = _clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].tolist()


# ─── Processing ──────────────────────────────────────────────────────────

def process_video(enriched_path, batch_size):
    """Generate CLIP embeddings for all scenes in one enriched file."""
    data = json.loads(enriched_path.read_text())
    scenes = data.get("scenes", {}).get("scenes", [])
    file_info = data.get("file", {})
    name = enriched_path.name.replace(".enriched.json", "")

    source_path = find_source_video(file_info)
    if not source_path or not Path(source_path).exists():
        return name, 0, "source not found"

    tmp_dir = TMP_DIR / name[:50]
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Extract midpoint frames
    t0 = time.time()
    frame_results = extract_scene_frames(source_path, scenes, tmp_dir)
    extract_time = time.time() - t0

    if not frame_results:
        return name, 0, "no frames extracted"

    # Run CLIP inference
    t0 = time.time()
    frame_paths = [r[2] for r in frame_results]
    embeddings = encode_images(frame_paths, batch_size)
    infer_time = time.time() - t0

    # Build sidecar output
    scene_embeddings = []
    for (idx, frame_time, _), embedding in zip(frame_results, embeddings):
        scene_embeddings.append({
            "scene_index": idx,
            "frame_time_sec": frame_time,
            "embedding": [round(v, 6) for v in embedding],
        })

    output = {
        "model": f"{CLIP_MODEL}/{CLIP_PRETRAINED}",
        "embedding_dim": len(embeddings[0]) if embeddings else 512,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_file": enriched_path.name,
        "extract_time_sec": round(extract_time, 1),
        "inference_time_sec": round(infer_time, 1),
        "scenes": scene_embeddings,
    }

    output_path = ENRICHED_DIR / f"{name}.clip_embeddings.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False))

    # Cleanup frames
    for _, _, fp in frame_results:
        fp.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    return name, len(scene_embeddings), "ok"


def fmt_time(seconds):
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate CLIP embeddings for enriched video scenes")
    parser.add_argument("--video", type=str, nargs="*", help="Filter by video name (substring)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Inference batch size (default: {BATCH_SIZE})")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip videos that already have embeddings")
    args = parser.parse_args()

    if not SOURCE_DIR.exists():
        print(f"  ✗ Source directory not found: {SOURCE_DIR}")
        print(f"    Mount the archive drive and try again.")
        sys.exit(1)

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    enriched_files = sorted(ENRICHED_DIR.glob("*.enriched.json"))

    if args.video:
        enriched_files = [f for f in enriched_files
                          if any(v.lower() in f.name.lower() for v in args.video)]

    if args.skip_existing:
        existing = {f.name.replace(".clip_embeddings.json", "")
                    for f in ENRICHED_DIR.glob("*.clip_embeddings.json")}
        enriched_files = [f for f in enriched_files
                          if f.name.replace(".enriched.json", "") not in existing]

    total_scenes = 0
    for f in enriched_files:
        data = json.loads(f.read_text())
        total_scenes += len(data.get("scenes", {}).get("scenes", []))

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  CLIP EMBEDDING GENERATOR")
    print(f"  Model: {CLIP_MODEL}/{CLIP_PRETRAINED}")
    print(f"  Videos: {len(enriched_files)}  |  Scenes: {total_scenes}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  ════════════════════════════════════════════════════════════════════")

    # Load model upfront
    print(f"\n  Loading CLIP model...", end=" ", flush=True)
    _load_clip()

    pipeline_start = time.time()
    total_embedded = 0

    for idx, ef in enumerate(enriched_files, 1):
        name_short = ef.name.replace(".enriched.json", "")[:50]
        data = json.loads(ef.read_text())
        n_scenes = len(data.get("scenes", {}).get("scenes", []))

        print(f"\n  [{idx}/{len(enriched_files)}] {name_short}  ({n_scenes} scenes)")
        t0 = time.time()

        name, n_embedded, status = process_video(ef, args.batch_size)

        elapsed = time.time() - t0
        total_embedded += n_embedded

        if status == "ok":
            rate = n_embedded / elapsed if elapsed > 0 else 0
            print(f"    ✓ {n_embedded} embeddings  in {fmt_time(elapsed)}"
                  f"  ({rate:.1f} scenes/s)", flush=True)
        else:
            print(f"    ✗ {status}")

    total_elapsed = time.time() - pipeline_start

    print(f"\n  ════════════════════════════════════════════════════════════════════")
    print(f"  EMBEDDING GENERATION COMPLETE")
    print(f"  {total_embedded} scene embeddings in {fmt_time(total_elapsed)}")
    print(f"  Output: {ENRICHED_DIR}/*.clip_embeddings.json")
    print(f"  ════════════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
