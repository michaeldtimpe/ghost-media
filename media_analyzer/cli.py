"""
Unified CLI for media-analyzer.

Usage:
    media-analyzer audio <file>          Analyze a single audio file
    media-analyzer video <file>          Analyze a single video file
    media-analyzer batch <directory>     Analyze all media in a directory
    media-analyzer batch <directory> --audio-only
    media-analyzer batch <directory> --video-only
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.aiff', '.aif',
    '.wma', '.opus', '.ape', '.alac',
}

VIDEO_EXTENSIONS = {
    '.mp4', '.mov', '.mkv', '.avi', '.webm', '.wmv', '.flv', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.ogv', '.ts', '.mts',
}


def analyze_single_audio(args):
    """Analyze a single audio file."""
    from media_analyzer.audio_analyzer import analyze_audio
    result = analyze_audio(args.input, sr=args.sr)
    output_path = args.output or (args.input + ".analysis.json")
    indent = None if args.compact else 2
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=indent)
    print(f"\nAudio analysis written to: {output_path}")


def analyze_single_video(args):
    """Analyze a single video file."""
    from media_analyzer.video_analyzer import analyze_video
    result = analyze_video(
        args.input,
        scene_threshold=args.scene_threshold,
        color_interval=args.color_interval,
        motion_interval=args.motion_interval,
        brightness_interval=args.brightness_interval,
        n_colors=args.n_colors,
        skip_motion=args.skip_motion,
    )
    output_path = args.output or (args.input + ".analysis.json")
    indent = None if args.compact else 2
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=indent)
    print(f"\nVideo analysis written to: {output_path}")


def _process_file(filepath: str, file_type: str, output_dir: str,
                  audio_sr: int, video_opts: dict) -> dict:
    """Worker function for batch processing (runs in subprocess)."""
    try:
        if file_type == "audio":
            from media_analyzer.audio_analyzer import analyze_audio
            result = analyze_audio(filepath, sr=audio_sr)
        else:
            from media_analyzer.video_analyzer import analyze_video
            result = analyze_video(filepath, **video_opts)

        # Write output
        name = Path(filepath).stem
        out_file = Path(output_dir) / f"{name}.{file_type}.analysis.json"
        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)

        return {"file": filepath, "type": file_type, "status": "success", "output": str(out_file)}
    except Exception as e:
        return {"file": filepath, "type": file_type, "status": "error", "error": str(e)}


def batch_analyze(args):
    """Analyze all media files in a directory."""
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    # Collect files
    audio_files = []
    video_files = []

    pattern = '**/*' if args.recursive else '*'

    for p in directory.glob(pattern):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in AUDIO_EXTENSIONS and not args.video_only:
            audio_files.append(str(p))
        elif ext in VIDEO_EXTENSIONS and not args.audio_only:
            video_files.append(str(p))

    total = len(audio_files) + len(video_files)
    if total == 0:
        print("No media files found.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files, {len(video_files)} video files ({total} total)")

    # Output directory
    output_dir = args.output_dir or str(directory / "analysis_output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    video_opts = {
        "scene_threshold": args.scene_threshold,
        "color_interval": args.color_interval,
        "motion_interval": args.motion_interval,
        "brightness_interval": args.brightness_interval,
        "n_colors": args.n_colors,
        "skip_motion": args.skip_motion,
    }

    # Process files
    workers = min(args.workers, total)
    results = []
    start_time = time.time()

    tasks = [(f, "audio") for f in audio_files] + [(f, "video") for f in video_files]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_process_file, fp, ft, output_dir, args.sr, video_opts): (fp, ft)
                for fp, ft in tasks
            }
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                results.append(r)
                status = "✓" if r["status"] == "success" else "✗"
                print(f"  [{i}/{total}] {status} {Path(r['file']).name}")
    else:
        for i, (fp, ft) in enumerate(tasks, 1):
            r = _process_file(fp, ft, output_dir, args.sr, video_opts)
            results.append(r)
            status = "✓" if r["status"] == "success" else "✗"
            print(f"  [{i}/{total}] {status} {Path(r['file']).name}")

    elapsed = time.time() - start_time
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")

    # Write batch manifest
    manifest = {
        "schema_version": "1.0.0",
        "analyzer": "media-analyzer-batch",
        "source_directory": str(directory.resolve()),
        "output_directory": output_dir,
        "total_files": total,
        "successful": success,
        "errors": errors,
        "elapsed_sec": round(elapsed, 2),
        "files": results,
    }

    manifest_path = Path(output_dir) / "_batch_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBatch complete: {success}/{total} succeeded in {elapsed:.1f}s")
    if errors > 0:
        print(f"  {errors} files had errors — see manifest for details")
    print(f"  Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="media-analyzer",
        description="Analyze audio and video files for ML-friendly JSON output",
    )
    subparsers = parser.add_subparsers(dest="command", help="Analysis mode")

    # --- Audio subcommand ---
    audio_parser = subparsers.add_parser("audio", help="Analyze a single audio file")
    audio_parser.add_argument("input", help="Path to audio file")
    audio_parser.add_argument("-o", "--output", help="Output JSON path")
    audio_parser.add_argument("--sr", type=int, default=22050, help="Sample rate (default: 22050)")
    audio_parser.add_argument("--compact", action="store_true", help="Compact JSON")
    audio_parser.set_defaults(func=analyze_single_audio)

    # --- Video subcommand ---
    video_parser = subparsers.add_parser("video", help="Analyze a single video file")
    video_parser.add_argument("input", help="Path to video file")
    video_parser.add_argument("-o", "--output", help="Output JSON path")
    video_parser.add_argument("--scene-threshold", type=float, default=30.0)
    video_parser.add_argument("--color-interval", type=float, default=1.0)
    video_parser.add_argument("--motion-interval", type=float, default=0.5)
    video_parser.add_argument("--brightness-interval", type=float, default=0.5)
    video_parser.add_argument("--n-colors", type=int, default=3)
    video_parser.add_argument("--skip-motion", action="store_true")
    video_parser.add_argument("--compact", action="store_true")
    video_parser.set_defaults(func=analyze_single_video)

    # --- Batch subcommand ---
    batch_parser = subparsers.add_parser("batch", help="Analyze all media in a directory")
    batch_parser.add_argument("directory", help="Directory to scan")
    batch_parser.add_argument("-o", "--output-dir", help="Output directory (default: <dir>/analysis_output)")
    batch_parser.add_argument("--audio-only", action="store_true", help="Only analyze audio files")
    batch_parser.add_argument("--video-only", action="store_true", help="Only analyze video files")
    batch_parser.add_argument("--recursive", "-r", action="store_true", help="Scan subdirectories")
    batch_parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers (default: 4)")
    batch_parser.add_argument("--sr", type=int, default=22050, help="Audio sample rate")
    batch_parser.add_argument("--scene-threshold", type=float, default=30.0)
    batch_parser.add_argument("--color-interval", type=float, default=1.0)
    batch_parser.add_argument("--motion-interval", type=float, default=0.5)
    batch_parser.add_argument("--brightness-interval", type=float, default=0.5)
    batch_parser.add_argument("--n-colors", type=int, default=3)
    batch_parser.add_argument("--skip-motion", action="store_true")
    batch_parser.set_defaults(func=batch_analyze)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
