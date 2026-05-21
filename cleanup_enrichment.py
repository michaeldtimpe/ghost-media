#!/usr/bin/env python3
"""
Post-enrichment cleanup:
1. Remove duplicate Glancent file (mangled filename encoding)
2. Re-run the 1 failed frame in Dom & Roland via Ollama
3. Fix "a a" stutter artifact across all enriched files
"""

import json
import os
import re
import sys
import time
import base64
import urllib.request
from pathlib import Path

from vision_schema import SCENE_PROMPT, normalize_analysis

ENRICHED_DIR = Path(__file__).parent / "enriched"
OLLAMA_API = "http://localhost:11434"
MODEL = "qwen2.5vl:7b"


def fix_stutter(obj):
    """Recursively fix 'a a' doubled-article stutter in all string values."""
    if isinstance(obj, str):
        # Fix "a a" but not inside words — match word boundary
        return re.sub(r'\ba a\b', 'a', obj)
    elif isinstance(obj, dict):
        return {k: fix_stutter(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_stutter(v) for v in obj]
    return obj


def query_vision(image_path):
    """Send image to vision model and return parsed JSON."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = json.dumps({
        "model": MODEL,
        "prompt": SCENE_PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {"num_predict": 1024},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_API}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
        elapsed = time.time() - start
        raw = data.get("response", "").strip()

    # Parse JSON
    try:
        return json.loads(raw), raw, elapsed
    except json.JSONDecodeError:
        pass
    # Try { ... } extraction
    bs = raw.find("{")
    be = raw.rfind("}") + 1
    if bs >= 0 and be > bs:
        try:
            return json.loads(raw[bs:be]), raw, elapsed
        except json.JSONDecodeError:
            pass
    return None, raw, elapsed


def main():
    print("\n  ═══════════════════════════════════════════════════════════")
    print("  ENRICHMENT CLEANUP")
    print("  ═══════════════════════════════════════════════════════════\n")

    # ── 1. Remove duplicate Glancent ──
    print("  [1/3] Removing duplicate Glancent file...")
    dup = ENRICHED_DIR / "モーションク_ラフィックス _Glancent_.enriched.json"
    if dup.exists():
        dup.unlink()
        print(f"    ✓ Removed: {dup.name}")
    else:
        print(f"    - Already removed")

    # ── 2. Re-run failed frame in Dom & Roland ──
    print("\n  [2/3] Re-running failed frame (Dom & Roland, scene 873)...")
    dr_file = ENRICHED_DIR / "A Twisting Tribute to Dom _ Roland  - Full Drum _ Bass VJ Mix.enriched.json"
    frame_path = ENRICHED_DIR / "frames" / "4317acc42017" / "scene_873_start_1908.03s.jpg"

    if not frame_path.exists():
        print(f"    ✗ Frame image not found: {frame_path}")
        sys.exit(1)

    data = json.loads(dr_file.read_text())

    print(f"    Querying {MODEL}...", end=" ", flush=True)
    parsed, raw, elapsed = query_vision(str(frame_path))

    if parsed:
        print(f"✓ parsed in {elapsed:.1f}s")
        clean, _ = normalize_analysis(parsed)
        # Update frame_analyses[9]
        fa = data["frame_analyses"][9]
        fa["analysis"] = clean
        fa.pop("analysis_raw", None)
        fa["inference_time_sec"] = elapsed

        # Also update the scene's semantic field
        scenes = data.get("scenes", {})
        scene_list = scenes.get("scenes", []) if isinstance(scenes, dict) else scenes
        for scene in scene_list:
            if scene.get("scene_index") == 873:
                scene["semantic"] = clean
                scene.pop("semantic_raw", None)
                break

        print(f"    ✓ Updated frame_analyses[9] and scene 873")
    else:
        print(f"~ raw again ({elapsed:.1f}s) — storing raw")
        print(f"    Preview: {raw[:150]}...")
        # Still an improvement attempt, keep raw
        data["frame_analyses"][9]["analysis_raw"] = raw

    # ── 3. Fix "a a" stutter across ALL files ──
    print("\n  [3/3] Fixing 'a a' stutter across all enriched files...")
    files = sorted(ENRICHED_DIR.glob("*.enriched.json"))
    fixed_count = 0
    total_fixes = 0

    for f in files:
        if f == dr_file:
            # We already have this in memory, fix it there
            before = json.dumps(data)
            data = fix_stutter(data)
            after = json.dumps(data)
            n = before.count('"a a ') - after.count('"a a ')
            if n > 0 or before != after:
                fixed_count += 1
                fixes = len(re.findall(r'\ba a\b', before)) - len(re.findall(r'\ba a\b', after))
                total_fixes += max(fixes, 0)
            # Write the Dom & Roland file (with re-run + stutter fix)
            f.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            continue

        text = f.read_text()
        original = text
        fdata = json.loads(text)
        fdata = fix_stutter(fdata)
        new_text = json.dumps(fdata, indent=2, ensure_ascii=False)
        if new_text != original:
            fixes = len(re.findall(r'\ba a\b', original)) - len(re.findall(r'\ba a\b', new_text))
            total_fixes += max(fixes, 0)
            fixed_count += 1
            f.write_text(new_text)

    print(f"    ✓ {fixed_count} files updated, {total_fixes} stutter instances fixed")

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  CLEANUP COMPLETE")
    print(f"  ═══════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
