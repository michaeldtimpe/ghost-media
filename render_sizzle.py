#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  SIZZLE REEL RENDERER
  Generates reactive visuals from deep DJ set analysis
═══════════════════════════════════════════════════════════════════════════════

Renders highlight segments with:
  - Circular spectrum analyzer (7-band)
  - Particle system driven by beat energy + onsets
  - Waveform ribbon reacting to harmonic content
  - Color palette shifting with spectral centroid and chroma
  - Geometric overlays pulsing on beat
  - Track title overlays
  - Beat flash and drop impacts

Composites with audio via ffmpeg into final video.
"""

import json
import math
import os
import sys
import time
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ─── Configuration ─────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1920, 1080
FPS = 30
BG_COLOR = (8, 8, 16)

# Segments to render: (start_sec, duration_sec, label)
SEGMENTS = [
    (0, 25, "OPENING — Justice Genesis → fund$"),
    (118, 25, "BUILD — ZZafrika (Gesaffelstein Remix)"),
    (329, 25, "DROP — Ghost Voices → On My Mind"),
    (554, 25, "ENERGY — Calvin Harris I'm Not Alone"),
    (673, 20, "PEAK — Knock2 Gettin' Hott"),
    (1141, 25, "BASS — ZZafrika"),
    (2156, 25, "GROOVE — ISOxo Beam → TAKE IT OFF"),
    (2500, 25, "DRIVE — Stereo Love → Astronomia"),
    (3240, 25, "HEAT — Calvin Harris → Call On Me"),
    (3540, 25, "BASS DROP — EAT THE BASS"),
    (4050, 25, "FINALE — Mr. Brightside → Mr. Blue Sky"),
]

# Title card duration
TITLE_DURATION = 3.0  # seconds


# ─── Color Palettes ────────────────────────────────────────────────────────

def spectral_to_hue(centroid_hz):
    """Map spectral centroid to a hue value (0-360)."""
    # Low centroid = warm (reds/oranges), high = cool (blues/purples)
    normalized = np.clip((centroid_hz - 500) / 6000, 0, 1)
    # Wrap around color wheel: red → orange → yellow → green → cyan → blue → purple
    return (240 + normalized * 180) % 360


def hsl_to_rgb(h, s, l):
    """Convert HSL to RGB (all 0-1 except h is 0-360)."""
    h = h / 360.0
    if s == 0:
        r = g = b = l
    else:
        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue2rgb(p, q, h + 1/3)
        g = hue2rgb(p, q, h)
        b = hue2rgb(p, q, h - 1/3)
    return (int(r * 255), int(g * 255), int(b * 255))


def energy_palette(energy, centroid_hz, hp_ratio):
    """Generate a color palette based on audio features."""
    hue = spectral_to_hue(centroid_hz)
    sat = 0.6 + energy * 0.4
    # Harmonic = brighter, percussive = darker
    lightness = 0.15 + energy * 0.35 + min(hp_ratio * 0.05, 0.15)

    primary = hsl_to_rgb(hue, sat, lightness)
    accent = hsl_to_rgb((hue + 120) % 360, sat * 0.8, lightness * 1.2)
    highlight = hsl_to_rgb((hue + 60) % 360, sat, min(lightness * 1.5, 0.9))

    return primary, accent, highlight


# ─── Particle System ──────────────────────────────────────────────────────

class ParticleSystem:
    def __init__(self, max_particles=300):
        self.max = max_particles
        self.particles = []  # (x, y, vx, vy, life, max_life, size, color)

    def emit(self, count, cx, cy, energy, color, spread=200):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(1, 4 + energy * 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = np.random.uniform(0.5, 1.5 + energy)
            size = np.random.uniform(1, 3 + energy * 4)
            self.particles.append([
                cx + np.random.uniform(-spread * 0.1, spread * 0.1),
                cy + np.random.uniform(-spread * 0.1, spread * 0.1),
                vx, vy, life, life, size, color
            ])
        # Trim
        if len(self.particles) > self.max:
            self.particles = self.particles[-self.max:]

    def update(self, dt):
        alive = []
        for p in self.particles:
            p[0] += p[2] * dt * 60  # x += vx
            p[1] += p[3] * dt * 60  # y += vy
            p[3] += 0.02 * dt * 60  # gravity
            p[4] -= dt  # life
            if p[4] > 0:
                alive.append(p)
        self.particles = alive

    def draw(self, draw_obj):
        for p in self.particles:
            x, y, _, _, life, max_life, size, color = p
            alpha = life / max_life
            r, g, b = color
            c = (int(r * alpha), int(g * alpha), int(b * alpha))
            s = size * alpha
            if 0 <= x < WIDTH and 0 <= y < HEIGHT and s > 0.5:
                draw_obj.ellipse([x - s, y - s, x + s, y + s], fill=c)


# ─── Lookup Helpers ────────────────────────────────────────────────────────

def lookup_at_time(timeline, t, key=None):
    """Binary search for the nearest entry at time t."""
    if not timeline:
        return timeline[0] if timeline else {}
    lo, hi = 0, len(timeline) - 1
    time_key = "time_sec"
    while lo < hi:
        mid = (lo + hi) // 2
        if timeline[mid][time_key] < t:
            lo = mid + 1
        else:
            hi = mid
    entry = timeline[lo]
    if key:
        return entry.get(key, 0)
    return entry


def beats_in_range(beat_times, beat_features, t_start, t_end):
    """Get beats within a time range."""
    results = []
    for i, bt in enumerate(beat_times):
        if bt >= t_start and bt < t_end:
            results.append((bt, beat_features[i] if i < len(beat_features) else {}))
        elif bt >= t_end:
            break
    return results


def track_at_time(tracks, t):
    """Find which track is playing at time t."""
    for tr in tracks:
        if tr["start_sec"] <= t < tr["end_sec"]:
            return tr
    return tracks[-1] if tracks else {"title": ""}


# ─── Frame Renderer ───────────────────────────────────────────────────────

def render_frame(t, data, particles, prev_beat_t, frame_idx):
    """Render a single frame at time t."""
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Lookup current features
    mb = lookup_at_time(data["multiband_energy"], t)
    hp = lookup_at_time(data["hpss_timeline"], t)
    sp = lookup_at_time(data["spectral_timeline"], t)
    ch = lookup_at_time(data["chroma_timeline"], t)
    onset = lookup_at_time(data["onsets"]["strength_envelope"], t)

    energy = mb.get("total_rms", 0)
    centroid = sp.get("centroid_hz", 2000)
    hp_ratio = hp.get("hp_ratio", 1.0)
    percussive = hp.get("percussive", 0)
    onset_strength = onset.get("strength", 0)
    flux = sp.get("flux", 0)

    # Normalize energy for visuals (typical range 0-0.6)
    e_norm = min(energy / 0.5, 1.0)

    # Colors
    primary, accent, highlight = energy_palette(energy, centroid, hp_ratio)

    # ── Background glow ──
    glow_radius = int(200 + e_norm * 300)
    glow_intensity = int(e_norm * 40)
    if glow_intensity > 5:
        glow_layer = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
        glow_draw = ImageDraw.Draw(glow_layer)
        cx, cy = WIDTH // 2, HEIGHT // 2
        for r in range(glow_radius, 0, -4):
            alpha = glow_intensity * (r / glow_radius) ** 2
            c = (int(primary[0] * alpha / 255),
                 int(primary[1] * alpha / 255),
                 int(primary[2] * alpha / 255))
            glow_draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        img = Image.blend(img, glow_layer, 0.5)
        draw = ImageDraw.Draw(img)

    # ── Circular spectrum analyzer ──
    cx, cy = WIDTH // 2, HEIGHT // 2
    base_radius = 180
    bands_data = [
        mb.get("sub_bass", 0), mb.get("bass", 0), mb.get("low_mid", 0),
        mb.get("mid", 0), mb.get("high_mid", 0), mb.get("presence", 0),
        mb.get("brilliance", 0),
    ]
    band_names = list(BANDS.keys()) if 'BANDS' in dir() else [
        "sub_bass", "bass", "low_mid", "mid", "high_mid", "presence", "brilliance"
    ]
    n_bars = 64
    angle_step = 2 * math.pi / n_bars

    for i in range(n_bars):
        angle = i * angle_step - math.pi / 2
        # Interpolate between bands based on bar position
        band_pos = (i / n_bars) * len(bands_data)
        band_idx = int(band_pos)
        band_frac = band_pos - band_idx
        if band_idx >= len(bands_data) - 1:
            val = bands_data[-1]
        else:
            val = bands_data[band_idx] * (1 - band_frac) + bands_data[band_idx + 1] * band_frac

        bar_height = val / 0.3 * 200  # normalize
        bar_height = min(bar_height, 350)

        x1 = cx + math.cos(angle) * base_radius
        y1 = cy + math.sin(angle) * base_radius
        x2 = cx + math.cos(angle) * (base_radius + bar_height)
        y2 = cy + math.sin(angle) * (base_radius + bar_height)

        # Color gradient from primary to accent based on frequency
        ratio = i / n_bars
        r = int(primary[0] * (1 - ratio) + accent[0] * ratio)
        g = int(primary[1] * (1 - ratio) + accent[1] * ratio)
        b = int(primary[2] * (1 - ratio) + accent[2] * ratio)

        # Brighten on beat
        brightness_boost = 1.0 + onset_strength * 0.3
        r = min(255, int(r * brightness_boost))
        g = min(255, int(g * brightness_boost))
        b = min(255, int(b * brightness_boost))

        draw.line([(x1, y1), (x2, y2)], fill=(r, g, b), width=max(2, int(3 + e_norm * 2)))

    # ── Inner ring: chroma wheel ──
    chroma_radius = 160
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    for i, pc in enumerate(pitch_classes):
        angle = (i / 12) * 2 * math.pi - math.pi / 2
        val = ch.get(pc, 0)
        dot_size = 3 + val * 12
        x = cx + math.cos(angle) * chroma_radius
        y = cy + math.sin(angle) * chroma_radius

        # Chroma-mapped color
        chroma_hue = (i / 12) * 360
        cc = hsl_to_rgb(chroma_hue, 0.8, 0.3 + val * 0.4)
        draw.ellipse([x - dot_size, y - dot_size, x + dot_size, y + dot_size], fill=cc)

    # ── Geometric overlay: rotating polygon on beat ──
    beat_phase = (t * (global_bpm_at(data, t) / 60)) % 1.0
    rotation = t * 0.3 + beat_phase * math.pi * 0.25
    poly_sides = 6
    poly_radius = base_radius + 40 + e_norm * 30
    poly_alpha = 0.3 + beat_phase * 0.4

    points = []
    for i in range(poly_sides):
        angle = rotation + (i / poly_sides) * 2 * math.pi
        x = cx + math.cos(angle) * poly_radius
        y = cy + math.sin(angle) * poly_radius
        points.append((x, y))
    points.append(points[0])

    line_color = (
        int(highlight[0] * poly_alpha),
        int(highlight[1] * poly_alpha),
        int(highlight[2] * poly_alpha),
    )
    draw.line(points, fill=line_color, width=2)

    # Second polygon, counter-rotating
    points2 = []
    poly_radius2 = base_radius + 60 + e_norm * 50
    for i in range(poly_sides):
        angle = -rotation * 0.7 + (i / poly_sides) * 2 * math.pi + math.pi / poly_sides
        x = cx + math.cos(angle) * poly_radius2
        y = cy + math.sin(angle) * poly_radius2
        points2.append((x, y))
    points2.append(points2[0])
    draw.line(points2, fill=line_color, width=1)

    # ── Horizontal waveform ribbons (harmonic energy) ──
    harmonic = hp.get("harmonic", 0)
    ribbon_y_top = HEIGHT * 0.15
    ribbon_y_bot = HEIGHT * 0.85
    ribbon_width = WIDTH * 0.8
    ribbon_x_start = WIDTH * 0.1
    n_ribbon_pts = 80

    for ribbon_y, ribbon_amp, ribbon_color in [
        (ribbon_y_top, harmonic * 300, (*primary, )),
        (ribbon_y_bot, percussive * 300, (*accent, )),
    ]:
        pts = []
        for i in range(n_ribbon_pts):
            x = ribbon_x_start + (i / n_ribbon_pts) * ribbon_width
            phase = t * 3 + i * 0.15
            amp = ribbon_amp * math.sin(phase) * (0.5 + 0.5 * math.sin(i * 0.3 + t * 2))
            y = ribbon_y + amp
            pts.append((x, y))
        if len(pts) > 1:
            c = tuple(min(255, int(v * (0.3 + e_norm * 0.5))) for v in ribbon_color)
            draw.line(pts, fill=c, width=2)

    # ── Particles ──
    # Emit on beats / onsets
    if onset_strength > 3:
        count = int(onset_strength * 3)
        particles.emit(count, cx, cy, e_norm, highlight, spread=base_radius)

    # Continuous ambient particles
    if frame_idx % 3 == 0:
        particles.emit(1, cx + np.random.uniform(-200, 200),
                       cy + np.random.uniform(-200, 200),
                       e_norm * 0.3, primary, spread=50)

    particles.update(1.0 / FPS)
    particles.draw(draw)

    # ── Beat flash ──
    if onset_strength > 5:
        flash_alpha = min(onset_strength / 15, 0.4)
        flash_layer = Image.new('RGB', (WIDTH, HEIGHT),
                                (int(255 * flash_alpha), int(255 * flash_alpha), int(255 * flash_alpha)))
        img = Image.blend(img, flash_layer, flash_alpha * 0.3)
        draw = ImageDraw.Draw(img)

    # ── Corner info ──
    try:
        font_small = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 16)
        font_track = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_bpm = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 20)
    except Exception:
        font_small = ImageFont.load_default()
        font_track = font_small
        font_bpm = font_small

    # Track name
    track = track_at_time(data.get("tracks", []), t)
    if track:
        title = track.get("title", "")[:60]
        draw.text((40, HEIGHT - 60), title, fill=(200, 200, 200), font=font_track)

    # BPM
    bpm_entry = lookup_at_time(data["bpm_timeline"], t)
    bpm = bpm_entry.get("bpm", 0)
    draw.text((WIDTH - 150, 30), f"{bpm:.0f} BPM", fill=(180, 180, 180), font=font_bpm)

    # Time
    mins, secs = divmod(int(t), 60)
    draw.text((40, 30), f"{mins}:{secs:02d}", fill=(120, 120, 120), font=font_small)

    # Energy meter (bottom left)
    meter_x, meter_y = 40, HEIGHT - 90
    meter_w, meter_h = 200, 6
    draw.rectangle([meter_x, meter_y, meter_x + meter_w, meter_y + meter_h],
                    fill=(30, 30, 40))
    fill_w = int(meter_w * e_norm)
    draw.rectangle([meter_x, meter_y, meter_x + fill_w, meter_y + meter_h],
                    fill=primary)

    return img


def global_bpm_at(data, t):
    entry = lookup_at_time(data["bpm_timeline"], t)
    return entry.get("bpm", 128)


# ─── Title Card ────────────────────────────────────────────────────────────

def render_title_card(text, subtitle=""):
    img = Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = font_title

    # Center text
    bbox = draw.textbbox((0, 0), text, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((WIDTH - tw) // 2, HEIGHT // 2 - 40), text, fill=(220, 220, 230), font=font_title)

    if subtitle:
        bbox2 = draw.textbbox((0, 0), subtitle, font=font_sub)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((WIDTH - tw2) // 2, HEIGHT // 2 + 30), subtitle, fill=(120, 120, 140), font=font_sub)

    return img


# ─── Segment Renderer ─────────────────────────────────────────────────────

def render_segment(data, start_sec, duration_sec, label, output_dir, seg_index):
    """Render all frames for one segment."""
    n_frames = int(duration_sec * FPS)
    particles = ParticleSystem(max_particles=400)

    # Title card frames
    title_frames = int(TITLE_DURATION * FPS)

    frame_paths = []
    seg_dir = output_dir / f"seg_{seg_index:02d}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Render title card
    title_img = render_title_card(label.split(" — ")[0] if " — " in label else label,
                                  label.split(" — ")[1] if " — " in label else "")

    for i in range(title_frames):
        # Fade in/out
        alpha = 1.0
        if i < FPS * 0.5:
            alpha = i / (FPS * 0.5)
        elif i > title_frames - FPS * 0.5:
            alpha = (title_frames - i) / (FPS * 0.5)

        frame = Image.blend(Image.new('RGB', (WIDTH, HEIGHT), BG_COLOR), title_img, alpha)
        path = seg_dir / f"frame_{i:05d}.png"
        frame.save(path)
        frame_paths.append(path)

    # Render visual frames
    prev_beat_t = 0
    for i in range(n_frames):
        t = start_sec + i / FPS
        frame = render_frame(t, data, particles, prev_beat_t, i)

        # Crossfade from title
        if i < FPS:
            title_alpha = 1.0 - i / FPS
            frame = Image.blend(frame, title_img, title_alpha * 0.3)

        path = seg_dir / f"frame_{title_frames + i:05d}.png"
        frame.save(path)
        frame_paths.append(path)

    return frame_paths, seg_dir


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    analysis_path = Path("sets/Mtimpe MIX1 02DEC MASTER MP3 V6.deep-analysis.json")
    import media_paths
    audio_path = media_paths.SETS_ROOT / "blue-sky-genesis-2025" / "Mtimpe MIX1 02DEC MASTER MP3  V6  .mp3"
    output_dir = Path("sets/sizzle_render")
    final_output = Path("sets/sizzle_reel.mp4")

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  SIZZLE REEL RENDERER")
    print(f"  ═══════════════════════════════════════════════════════════\n")

    # Load analysis
    print(f"  Loading analysis...", end=" ", flush=True)
    data = json.loads(analysis_path.read_text())
    print(f"done")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = sum(int((d + TITLE_DURATION) * FPS) for _, d, _ in SEGMENTS)
    print(f"  Segments: {len(SEGMENTS)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Estimated duration: {total_frames / FPS:.0f}s")
    print()

    # Render each segment
    all_seg_dirs = []
    rendered = 0
    t_start = time.time()

    for seg_idx, (start, duration, label) in enumerate(SEGMENTS):
        seg_frames = int((duration + TITLE_DURATION) * FPS)
        print(f"  [{seg_idx + 1}/{len(SEGMENTS)}] {label[:50]}")
        print(f"    {start / 60:.1f}m, {duration}s, {seg_frames} frames...", end=" ", flush=True)

        t0 = time.time()
        _, seg_dir = render_segment(data, start, duration, label, output_dir, seg_idx)
        elapsed = time.time() - t0
        rendered += seg_frames

        fps_rate = seg_frames / elapsed
        remaining = (total_frames - rendered) / fps_rate if fps_rate > 0 else 0
        print(f"{elapsed:.1f}s ({fps_rate:.1f} fps, ETA: {remaining:.0f}s)")
        all_seg_dirs.append(seg_dir)

    # ── Compile with ffmpeg ──
    print(f"\n  Compiling video segments...")

    # Create a concat file listing all segment videos
    segment_videos = []
    for seg_idx, seg_dir in enumerate(all_seg_dirs):
        start_sec = SEGMENTS[seg_idx][0]
        duration_sec = SEGMENTS[seg_idx][1]
        total_seg_duration = duration_sec + TITLE_DURATION
        seg_video = output_dir / f"seg_{seg_idx:02d}.mp4"

        # Render segment to video
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(seg_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-crf", "18", "-preset", "fast",
            str(seg_video)
        ], capture_output=True)

        # Add audio to segment
        seg_with_audio = output_dir / f"seg_{seg_idx:02d}_audio.mp4"
        # Audio starts at start_sec, but we have TITLE_DURATION of silence first
        # Use amerge: silence for title, then audio from start_sec
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(seg_video),
            "-ss", str(max(0, start_sec - TITLE_DURATION)),
            "-i", str(audio_path),
            "-t", str(total_seg_duration),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v", "-map", "1:a",
            "-shortest",
            str(seg_with_audio)
        ], capture_output=True)

        segment_videos.append(seg_with_audio)

    # Concat all segments
    concat_file = output_dir / "concat.txt"
    with open(concat_file, 'w') as f:
        for sv in segment_videos:
            f.write(f"file '{sv.resolve()}'\n")

    print(f"  Concatenating {len(segment_videos)} segments...")
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        str(final_output)
    ], capture_output=True)

    # Cleanup frame images
    print(f"  Cleaning up frames...")
    for seg_dir in all_seg_dirs:
        for png in seg_dir.glob("*.png"):
            png.unlink()
        seg_dir.rmdir()
    for sv in segment_videos:
        sv.unlink()
    for sv in output_dir.glob("seg_*.mp4"):
        sv.unlink()
    concat_file.unlink()

    total_time = time.time() - t_start
    file_size = final_output.stat().st_size / (1024 * 1024)

    print(f"\n  ═══════════════════════════════════════════════════════════")
    print(f"  SIZZLE REEL COMPLETE")
    print(f"  Output: {final_output}")
    print(f"  Size: {file_size:.1f} MB")
    print(f"  Total render time: {total_time:.0f}s")
    print(f"  ═══════════════════════════════════════════════════════════\n")


# Re-export BANDS for the renderer
BANDS = {
    "sub_bass": (20, 60), "bass": (60, 250), "low_mid": (250, 500),
    "mid": (500, 2000), "high_mid": (2000, 4000), "presence": (4000, 8000),
    "brilliance": (8000, 11025),
}

if __name__ == "__main__":
    main()
