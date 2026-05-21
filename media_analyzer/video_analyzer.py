"""
Video/footage analysis module.

Analyzes video files and produces structured JSON with:
- File metadata (duration, resolution, fps, codec)
- Scene change detection with timestamps
- Dominant color extraction per segment
- Motion intensity over time
- Brightness / contrast curves
- Per-scene summaries
"""

import json
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Lazy imports to keep startup fast
_cv2 = None
_colorsys = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_colorsys():
    global _colorsys
    if _colorsys is None:
        import colorsys
        _colorsys = colorsys
    return _colorsys


def get_video_metadata(filepath: str) -> dict[str, Any]:
    """Extract metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        filepath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    probe = json.loads(result.stdout)

    video_stream = None
    audio_stream = None
    for s in probe.get("streams", []):
        if s.get("codec_type") == "video" and video_stream is None:
            video_stream = s
        elif s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s

    fmt = probe.get("format", {})

    meta = {
        "duration_sec": round(float(fmt.get("duration", 0)), 3),
        "file_size_bytes": int(fmt.get("size", 0)),
        "format_name": fmt.get("format_name", "unknown"),
        "bit_rate": int(fmt.get("bit_rate", 0)),
    }

    if video_stream:
        # Calculate fps from r_frame_rate
        fps_str = video_stream.get("r_frame_rate", "0/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 0

        meta["video"] = {
            "codec": video_stream.get("codec_name", "unknown"),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": round(fps, 3),
            "total_frames": int(video_stream.get("nb_frames", 0)) or None,
            "pix_fmt": video_stream.get("pix_fmt", "unknown"),
        }

    if audio_stream:
        meta["audio"] = {
            "codec": audio_stream.get("codec_name", "unknown"),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
        }

    return meta


def detect_scenes(filepath: str, threshold: float = 30.0,
                  min_scene_sec: float = 0.5) -> dict[str, Any]:
    """
    Detect scene changes using frame difference analysis.
    Uses ffmpeg scene detection filter for speed, falls back to OpenCV.
    """
    # Try ffmpeg scene detection first (fast)
    try:
        return _detect_scenes_ffmpeg(filepath, threshold / 100.0, min_scene_sec)
    except Exception:
        pass

    # Fallback: OpenCV frame differencing
    return _detect_scenes_opencv(filepath, threshold, min_scene_sec)


def _detect_scenes_ffmpeg(filepath: str, threshold: float,
                          min_scene_sec: float) -> dict[str, Any]:
    """Scene detection via ffmpeg select filter."""
    cmd = [
        "ffmpeg", "-i", filepath,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    scene_times = [0.0]
    for line in result.stderr.split('\n'):
        if 'showinfo' in line and 'pts_time:' in line:
            try:
                pts_part = line.split('pts_time:')[1].split()[0]
                t = float(pts_part)
                if t - scene_times[-1] >= min_scene_sec:
                    scene_times.append(t)
            except (ValueError, IndexError):
                continue

    # Get total duration
    meta = get_video_metadata(filepath)
    duration = meta["duration_sec"]
    scene_times.append(duration)

    scenes = []
    for i in range(len(scene_times) - 1):
        scenes.append({
            "scene_index": i,
            "start_sec": round(scene_times[i], 3),
            "end_sec": round(scene_times[i + 1], 3),
            "duration_sec": round(scene_times[i + 1] - scene_times[i], 3),
        })

    return {
        "scene_count": len(scenes),
        "scene_change_times_sec": [round(t, 3) for t in scene_times[1:-1]],
        "scenes": scenes,
    }


def _detect_scenes_opencv(filepath: str, threshold: float,
                          min_scene_sec: float) -> dict[str, Any]:
    """Fallback scene detection using OpenCV frame differencing."""
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    min_frames = int(min_scene_sec * fps)

    # Sample every Nth frame for speed
    sample_step = max(1, int(fps / 4))  # ~4 samples per second

    scene_frames = [0]
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))  # Tiny for speed

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                mean_diff = float(np.mean(diff))
                if mean_diff > threshold and (frame_idx - scene_frames[-1]) >= min_frames:
                    scene_frames.append(frame_idx)

            prev_gray = gray

        frame_idx += 1

    cap.release()

    scene_times = [f / fps for f in scene_frames]
    duration = total_frames / fps
    scene_times.append(duration)

    scenes = []
    for i in range(len(scene_times) - 1):
        scenes.append({
            "scene_index": i,
            "start_sec": round(scene_times[i], 3),
            "end_sec": round(scene_times[i + 1], 3),
            "duration_sec": round(scene_times[i + 1] - scene_times[i], 3),
        })

    return {
        "scene_count": len(scenes),
        "scene_change_times_sec": [round(t, 3) for t in scene_times[1:-1]],
        "scenes": scenes,
    }


def analyze_colors(filepath: str, sample_interval_sec: float = 1.0,
                   n_colors: int = 3) -> dict[str, Any]:
    """Extract dominant colors at regular intervals using k-means."""
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    sample_step = max(1, int(fps * sample_interval_sec))

    color_timeline = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            t = frame_idx / fps
            small = cv2.resize(frame, (64, 36))
            pixels = small.reshape(-1, 3).astype(np.float32)

            # k-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            k = min(n_colors, len(pixels))
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

            # Count cluster sizes
            unique, counts = np.unique(labels, return_counts=True)
            order = np.argsort(-counts)

            colors = []
            for idx in order:
                bgr = centers[idx]
                rgb = [int(bgr[2]), int(bgr[1]), int(bgr[0])]
                hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
                pct = round(float(counts[idx]) / len(labels), 3)
                colors.append({
                    "rgb": rgb,
                    "hex": hex_color,
                    "percentage": pct,
                })

            color_timeline.append({
                "time_sec": round(t, 3),
                "dominant_colors": colors,
            })

        frame_idx += 1

    cap.release()

    # Overall dominant colors (average the top color at each sample)
    if color_timeline:
        all_top_colors = np.array([s["dominant_colors"][0]["rgb"] for s in color_timeline])
        avg_color = np.mean(all_top_colors, axis=0).astype(int).tolist()
    else:
        avg_color = [0, 0, 0]

    return {
        "sample_interval_sec": sample_interval_sec,
        "sample_count": len(color_timeline),
        "overall_average_color_rgb": avg_color,
        "timeline": color_timeline,
    }


def analyze_motion(filepath: str, sample_interval_sec: float = 0.5) -> dict[str, Any]:
    """Measure motion intensity using optical flow magnitude."""
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    sample_step = max(1, int(fps * sample_interval_sec))

    motion_timeline = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            t = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))

            if prev_gray is not None:
                # Farneback optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                mean_motion = float(np.mean(magnitude))
                max_motion = float(np.max(magnitude))

                motion_timeline.append({
                    "time_sec": round(t, 3),
                    "mean_motion": round(mean_motion, 4),
                    "max_motion": round(max_motion, 4),
                })

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if motion_timeline:
        all_mean = [m["mean_motion"] for m in motion_timeline]
        overall_motion = round(float(np.mean(all_mean)), 4)
        motion_std = round(float(np.std(all_mean)), 4)
    else:
        overall_motion = 0.0
        motion_std = 0.0

    return {
        "sample_interval_sec": sample_interval_sec,
        "sample_count": len(motion_timeline),
        "overall_mean_motion": overall_motion,
        "motion_variability": motion_std,
        "motion_level": _classify_motion(overall_motion),
        "timeline": motion_timeline,
    }


def _classify_motion(mean_motion: float) -> str:
    if mean_motion < 0.5:
        return "static"
    elif mean_motion < 2.0:
        return "low"
    elif mean_motion < 5.0:
        return "moderate"
    elif mean_motion < 10.0:
        return "high"
    else:
        return "very_high"


def analyze_brightness(filepath: str, sample_interval_sec: float = 0.5) -> dict[str, Any]:
    """Compute brightness and contrast over time."""
    cv2 = _get_cv2()
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    sample_step = max(1, int(fps * sample_interval_sec))

    brightness_timeline = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_step == 0:
            t = frame_idx / fps
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            mean_bright = float(np.mean(gray)) / 255.0
            contrast = float(np.std(gray)) / 255.0

            brightness_timeline.append({
                "time_sec": round(t, 3),
                "brightness": round(mean_bright, 4),
                "contrast": round(contrast, 4),
            })

        frame_idx += 1

    cap.release()

    if brightness_timeline:
        all_bright = [b["brightness"] for b in brightness_timeline]
        all_contrast = [b["contrast"] for b in brightness_timeline]
        avg_brightness = round(float(np.mean(all_bright)), 4)
        avg_contrast = round(float(np.mean(all_contrast)), 4)
    else:
        avg_brightness = 0.0
        avg_contrast = 0.0

    return {
        "sample_interval_sec": sample_interval_sec,
        "sample_count": len(brightness_timeline),
        "average_brightness": avg_brightness,
        "average_contrast": avg_contrast,
        "brightness_level": _classify_brightness_level(avg_brightness),
        "timeline": brightness_timeline,
    }


def _classify_brightness_level(brightness: float) -> str:
    if brightness < 0.2:
        return "very_dark"
    elif brightness < 0.35:
        return "dark"
    elif brightness < 0.55:
        return "moderate"
    elif brightness < 0.7:
        return "bright"
    else:
        return "very_bright"


def analyze_video(filepath: str, scene_threshold: float = 30.0,
                  color_interval: float = 1.0, motion_interval: float = 0.5,
                  brightness_interval: float = 0.5, n_colors: int = 3,
                  skip_motion: bool = False, quiet: bool = False) -> dict[str, Any]:
    """Full video analysis pipeline."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    def _log(msg: str):
        if not quiet:
            print(msg)

    _log(f"Loading: {filepath}")

    _log("  Extracting metadata...")
    metadata = get_video_metadata(filepath)

    _log("  Detecting scenes...")
    scenes = detect_scenes(filepath, threshold=scene_threshold)

    _log("  Analyzing colors...")
    colors = analyze_colors(filepath, sample_interval_sec=color_interval, n_colors=n_colors)

    motion = None
    if not skip_motion:
        _log("  Analyzing motion (this may take a moment)...")
        motion = analyze_motion(filepath, sample_interval_sec=motion_interval)
    else:
        _log("  Skipping motion analysis (--skip-motion)")

    _log("  Analyzing brightness/contrast...")
    brightness = analyze_brightness(filepath, sample_interval_sec=brightness_interval)

    result = {
        "schema_version": "1.0.0",
        "analyzer": "media-analyzer-video",
        "file": {
            "path": str(path.resolve()),
            "name": path.name,
            "format": path.suffix.lstrip('.').lower(),
        },
        "metadata": metadata,
        "scenes": scenes,
        "colors": colors,
        "brightness": brightness,
    }

    if motion is not None:
        result["motion"] = motion

    _log("  Done.")
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze video files for ML-friendly JSON output")
    parser.add_argument("input", help="Path to video file (mp4, mov, mkv, avi, webm, etc.)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: <input>.analysis.json)")
    parser.add_argument("--scene-threshold", type=float, default=30.0,
                        help="Scene change sensitivity (lower = more scenes, default: 30)")
    parser.add_argument("--color-interval", type=float, default=1.0,
                        help="Color sampling interval in seconds (default: 1.0)")
    parser.add_argument("--motion-interval", type=float, default=0.5,
                        help="Motion sampling interval in seconds (default: 0.5)")
    parser.add_argument("--brightness-interval", type=float, default=0.5,
                        help="Brightness sampling interval in seconds (default: 0.5)")
    parser.add_argument("--n-colors", type=int, default=3,
                        help="Number of dominant colors to extract (default: 3)")
    parser.add_argument("--skip-motion", action="store_true",
                        help="Skip motion analysis (faster)")
    parser.add_argument("--pretty", action="store_true", default=True,
                        help="Pretty-print JSON (default: true)")
    parser.add_argument("--compact", action="store_true", help="Compact JSON output")

    args = parser.parse_args()

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

    print(f"\nAnalysis written to: {output_path}")


if __name__ == "__main__":
    main()
