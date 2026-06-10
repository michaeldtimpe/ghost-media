#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  CLIP UTILITIES
  Shared CLIP model loader + image/text encoders.
═══════════════════════════════════════════════════════════════════════════════

Extracted so the embedding generator and the natural-language query tool share
one encoder (same model, same device, same normalization) instead of one
importing the other's internals.
"""

import numpy as np
import torch

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "openai"
BATCH_SIZE = 32

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = None


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_clip(verbose=False):
    """Load the CLIP model once (cached singleton)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer, _device
    if _clip_model is not None:
        return
    import open_clip
    _device = get_device()
    _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=_device)
    _clip_tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
    _clip_model.eval()
    if verbose:
        print(f"  CLIP model: {CLIP_MODEL}/{CLIP_PRETRAINED} on {_device}")


# Backwards-compatible alias (generate_clip_embeddings.py used _load_clip).
_load_clip = load_clip


def encode_images(frame_paths, batch_size=BATCH_SIZE):
    """Encode images with CLIP. Returns a list of 512-dim python lists."""
    from PIL import Image
    load_clip()
    embeddings = []
    for i in range(0, len(frame_paths), batch_size):
        batch_paths = frame_paths[i:i + batch_size]
        images = []
        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                images.append(_clip_preprocess(img))
            except Exception:
                images.append(torch.zeros(3, 224, 224))
        batch = torch.stack(images).to(_device)
        with torch.no_grad():
            features = _clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        embeddings.extend(features.cpu().numpy().tolist())
    return embeddings


def encode_text(text):
    """Encode a text query with CLIP. Returns an L2-normalized float32 ndarray."""
    load_clip()
    tokens = _clip_tokenizer([text]).to(_device)
    with torch.no_grad():
        features = _clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()[0].astype(np.float32)
