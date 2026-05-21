#!/usr/bin/env python3
"""
Interactive ground-truth labeler for English-text presence on the bake-off's
representative frames. Resumable: already-labeled frames are skipped.

  python bench_label.py --n 250            # label up to 250 frames across the pilot
  python bench_label.py --videos "isshin"  # restrict to matching pilot videos
  python bench_label.py --stats            # show current label counts

Each frame is opened in the macOS previewer; answer:
  y = English text present   n = no English text   s = skip   q = save & quit

English text = readable Latin-alphabet words/letters/numbers (titles, captions,
watermarks, logos). Japanese/Chinese/Korean or other non-Latin script does NOT
count. Abstract shapes that merely look like letters do NOT count.
"""

import argparse
import subprocess

from bench import config, groundtruth as GT


def main():
    ap = argparse.ArgumentParser(description="Label English-text ground truth")
    ap.add_argument("--videos", nargs="+", default=None, help="restrict to matching pilot stems")
    ap.add_argument("--n", type=int, default=250, help="max frames to queue")
    ap.add_argument("--per-video", type=int, default=None, help="cap per video (balanced)")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    store = GT.load_labels()
    if args.stats:
        print("  text-label stats:", GT.stats(store))
        return

    registry = config.pilot_registry(args.videos)
    per_video = args.per_video or max(1, args.n // max(1, len(registry)))
    queue = GT.propose_text_frames(registry, target_per_video=per_video)
    done = GT.labeled_ids(store)
    queue = [r for r in queue if (r["key"], r["scene_index"], r["cluster_id"]) not in done][:args.n]

    print(f"\n  {len(queue)} frames to label (already have {GT.stats(store)['total']}).")
    print("  y=text  n=no-text  s=skip  q=save&quit\n")
    for i, rec in enumerate(queue, 1):
        fpath = config.BASE_DIR / rec["frame_path"]
        if not fpath.exists():
            continue
        subprocess.run(["open", str(fpath)], capture_output=True)
        hint = " (incumbent flags text here)" if rec["proposed_text"] else ""
        ans = input(f"  [{i}/{len(queue)}] {rec['display_stem'][:40]} "
                    f"@ {rec['time_sec']}s{hint}  y/n/s/q: ").strip().lower()
        if ans == "q":
            break
        if ans == "s":
            continue
        if ans in ("y", "n"):
            text = input("     text (optional): ").strip() if ans == "y" else ""
            GT.add_label(store, rec, ans == "y", text)
            if i % 10 == 0:
                GT.save_labels(store)
    GT.save_labels(store)
    print("\n  saved:", GT.stats(store))


if __name__ == "__main__":
    main()
