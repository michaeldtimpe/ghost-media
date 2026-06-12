# Repository audit
**Findings: 81 (consolidated across chunks)**

## Area: . (chunk 1)

## Bugs & security

**Medium — `media_analyzer/cli.py:73–74` — Filename collision in recursive batch mode**

```python
name = Path(filepath).stem
out_file = Path(output_dir) / f"{name}.{file_type}.analysis.json"
```

When `batch_analyze` runs with `--recursive` (`**/*` glob at line 94), any two files sharing the same stem across subdirectories (e.g., `dir_a/intro.mp4` and `dir_b/intro.mp4`) write to the identical output path `output_dir/intro.video.analysis.json`. The second file silently overwrites the first, with no warning in the manifest. The batch manifest at line 168 records each file's output path, so the collision is visible post-hoc but the data loss is already done.

**Impact:** Silent data loss in recursive batch analysis when source directories contain same-stem files.

**Suggested fix:** Derive the output name from the relative path rather than the bare stem, e.g. `out_file = Path(output_dir) / (str(Path(filepath).relative_to(directory)).replace(os.sep, "_") + f".{file_type}.analysis.json")`, or include a hash of the full path.

## Structural improvements

1. **Extract hardcoded archive path from `bench/config.py:24`** — `ARCHIVE_SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")` is an absolute, machine-specific path. The file already has a fallback pattern (`SOURCE_DIRS` list with `LOCAL_SOURCE_DIR` first), but the back-compat alias `SOURCE_DIR = ARCHIVE_SOURCE_DIR` (line 26) and the standalone constant still bake in the path. Move to an environment variable with a documented default, or a `.env` / config file. **Risk:** Low — only affects benchmark runs when the local `raw_footage/` mirror is absent. **Verify:** Confirm `resolve_source()` callers use `SOURCE_DIRS` (they do) and that no other code imports `SOURCE_DIR` directly.

2. **`bench/config.py:158–163` — Docstring/code mismatch in `find_analysis_files`** — The docstring says "excluding pass1/meta" but the filter at line 159 only excludes `"pass1"` and names starting with `"_"`. Files containing `"meta"` are not excluded. **Risk:** Low — cosmetic, but could mislead future maintainers. **Verify:** Grep for any `*meta*.analysis.json` files in the repo root; if none exist, the mismatch is harmless but should still be corrected.

3. **`bench/config.py:142` — Unknown engine names silently treated as safe** — `backends = {ENGINES.get(n, {}).get("backend") for n in engine_names}` yields `None` for unrecognized engine names, which never matches the `cloud` set. A typo in an engine name would bypass the billing guard. **Risk:** Low — `assert_billing_safe` is a safety net, not a security boundary; typos would also cause runtime failures downstream. **Verify:** Add an explicit check `if n not in ENGINES: raise ValueError(f"Unknown engine: {n}")` at the top of the function.

## Area: . (chunk 2)

- 1. **`sys.exit(1)` in library functions** — Both `analyze_dj_set` and `analyze_deep` call `sys.exit(1)` on missing file. This is a code smell but not a practical bug since these functions are only cal
- 2. **Duplicate code** — `parse_tracklist`, `build_track_segments`, and `_classify_brightness` are duplicated between the two files. This is a structural issue.
- 4. **`analyze_deep` computes STFT twice** — `compute_multiband_energy` (line 117) returns `S` and `freqs`, which are used by `compute_beat_features` (line 747). But `compute_spectral_timeline` (line 7
- 6. **`analyze_tracks_deep` uses `zip(beat_times, beat_features)`** — This is correct since they're the same length.
- 1. **`sys.exit(1)` in library functions** — Both `analyze_dj_set` (line 282) and `analyze_deep` (line 679) call `sys.exit(1)` on missing file. This is a code smell because it terminates the entire pro
- 1. **`sys.exit(1)` in library functions** — This is a code smell but not a practical bug. I'll rate it as Low.
- 2. **Duplicate code** — `parse_tracklist`, `build_track_segments`, and `_classify_brightness` are duplicated. This is a structural issue.
- 3. **Redundant STFT computation** — `compute_spectral_timeline` computes its own STFT. This is a performance issue.
- 5. **`compute_spectral_timeline` flux length mismatch** — I verified this is correct (prepends 0 to match length).
- 6. **`analyze_deep` drops per-beat energy from output** — This is intentional (Phase 4 pruning).
- 1. **Medium** — Unguarded `min()`/`max()` on potentially empty BPM timeline in both files. If audio is shorter than the BPM window (8s default), `bpm_timeline` is empty, and `min(bpms)`/`max(bpms)` ra
- 1. **Duplicate code** — `parse_tracklist`, `build_track_segments`, and `_classify_brightness` are duplicated between the two files.
- 2. **Redundant STFT computation** — `compute_spectral_timeline` computes its own STFT even though `compute_multiband_energy` already computed it.
- 3. **`sys.exit(1)` in library functions** — Both `analyze_dj_set` and `analyze_deep` call `sys.exit(1)` on missing file.
- Actually, let me reconsider. The task says to report SERIOUS, code-grounded bugs/security issues AND high-leverage STRUCTURAL improvements. The `sys.exit(1)` issue is not serious (no imports), and the
- 1. **Medium** — Unguarded `min()`/`max()` on potentially empty BPM timeline in both files.
- 1. **Medium** — Unguarded `min()`/`max()` on potentially empty BPM timeline in both files. `analyze_dj_set.py:316-317` and `analyze_dj_set_deep.py:711-712` (and `analyze_dj_set_deep.py:829`).
- 1. **Medium** — Unguarded `min()`/`max()` on potentially empty BPM timeline.
- Actually, the task says "report SERIOUS, code-grounded bugs/security issues AND high-leverage STRUCTURAL improvements". The `min()`/`max()` bug is code-grounded and real, even if unlikely to trigger.

## Area: . (chunk 3)

## Bugs & security

**High — `assemble_music_video.py:494–500` — Cleanup without `try/finally` or existence checks**

```python
    # Cleanup
    print(f"  Cleaning up...")
    for path, _ in clip_paths:
        path.unlink()
    concat_video.unlink()
    concat_file.unlink()
    clip_dir.rmdir()
```

If any `subprocess.run` call earlier in `assemble_video` raises (e.g. `TimeoutExpired` at line 418 or 472), this cleanup block is never reached, leaving orphaned clip files, the concat list, and the raw concat video on disk. Furthermore, if the concat step silently fails (returns non-zero but doesn't raise), `concat_video.unlink()` raises `FileNotFoundError` because the file was never created. `clip_dir.rmdir()` raises `OSError` if any clips remain.

**Impact:** Disk-space leak on every failed assembly run; unhandled `FileNotFoundError`/`OSError` crashes the program during cleanup.

**Suggested fix:** Wrap the entire assembly body in `try/finally` and guard each removal: `if concat_video.exists(): concat_video.unlink()`, use `shutil.rmtree(clip_dir, ignore_errors=True)` instead of `rmdir()`.

---

**Medium — `assemble_music_video.py:418` — `extract_clip` timeout propagates uncaught**

```python
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return output_path.exists()
```

`subprocess.run` raises `subprocess.TimeoutExpired` when the 120-second deadline is exceeded. The caller at line 438–443 expects a boolean return value and has no `try/except`. A single slow extraction (e.g. seeking into a multi-gigabyte file on a slow volume) crashes the entire assembly pipeline.

**Impact:** Complete assembly failure on a single slow clip extraction; no partial result saved.

**Suggested fix:** Catch `subprocess.TimeoutExpired` inside `extract_clip` and return `False`, or increase the timeout and log a warning.

---

**Medium — `assemble_music_video.py:371–372` — Clip duration can become negative when clip starts near end of source video**

```python
        max_available = best_clip.duration_sec - best_clip.time_sec
        clip_duration = min(clip_duration, max_available - 0.5)
        clip_duration = max(clip_duration, 1.0)
```

When `best_clip.time_sec` is within 0.5 s of the source video's end, `max_available - 0.5` is negative. `min(phrase.duration_sec, negative)` yields a negative value, then `max(negative, 1.0)` clamps to 1.0 s. The clip is then extracted starting at `best_clip.time_sec` (near EOF) for 1.0 s — ffmpeg either fails or produces a truncated clip, and the audio-video timing drifts.

**Impact:** Silent clip extraction failure or timing drift in the final video.

**Suggested fix:** Guard with `if max_available < 1.5: continue` (skip this clip) or cap `clip_duration = max(0.5, max_available - 0.5)` and log a warning.

---

**Medium — `assemble_music_video.py:365` — `select_clips` crashes on empty clip database**

```python
        best_score, best_clip = scored[0]
```

If `build_clip_database()` returns an empty list (no `.enriched.json` files in `ENRICHED_DIR`), `scored` is empty and `scored[0]` raises `IndexError`. The crash occurs with no diagnostic message.

**Impact:** Unhelpful `IndexError` instead of a clear "no clips available" message.

**Suggested fix:** Add `if not scored: print("No clips available for phrase …"); continue` before line 365.

---

**Medium — `assemble_music_video.py:195–199` — Unguarded dict access in `extract_phrase_features`**

```python
    multiband = data["multiband_energy"]
    hpss = data["hpss_timeline"]
    spectral = data["spectral_timeline"]
    chroma = data["chroma_timeline"]
    bpm_tl = data["bpm_timeline"]
```

Five direct `data[key]` lookups with no `.get()` fallback. If the deep-analysis JSON is from an older schema version or is partially written, a `KeyError` crashes the assembler with no context about which key is missing.

**Impact:** Hard crash on malformed or version-mismatched deep-analysis JSON.

**Suggested fix:** Use `data.get("multiband_energy", [])` etc., and emit a warning if any timeline is empty.

---

**Medium — `analyze_footage.py:215–218` — Filename collision in recursive batch mode**

```python
        safe_name = Path(filepath).stem
        safe_name = "".join(c if c.isalnum() or c in ' -_' else '_' for c in safe_name).strip()[:120]
        out_file = Path(output_dir) / f"{safe_name}.{file_type}.analysis.json"
```

When `--recursive` is used (`scan_directory` at line 250 uses `**/*` glob), any two files sharing the same stem across subdirectories (e.g. `dir_a/intro.mp4` and `dir_b/intro.mp4`) write to the identical output path. The second silently overwrites the first.

**Impact:** Silent data loss in recursive batch analysis.

**Suggested fix:** Derive the output name from the relative path, e.g. `safe_name = str(Path(filepath).relative_to(directory)).replace(os.sep, "_").replace(".", "_")`.

## Structural improvements

1. **Extract hardcoded paths from `assemble_music_video.py:36–41`** — Six module-level constants (`ENRICHED_DIR`, `ANALYSIS_PATH`, `AUDIO_PATH`, `SOURCE_DIR`, `OUTPUT_DIR`, `FINAL_OUTPUT`) are absolute or repo-relative paths baked into the source. Move to a config file (TOML/YAML) or environment variables with documented defaults. **Risk:** Low — only affects portability. **Verify:** Confirm no other scripts import these symbols (grep for `from assemble_music_video import`).

2. **Deduplicate formatting helpers across `analyze_footage

## Area: . (chunk 4)

- 1. **`embed_dim` computation at line 1005-1007** - Only checks `pool[0]` for embedding dimension, but the first clip might not have an embedding while others do. This could cause a shape mismatch in `
- 2. **`extract_clip` timeout at line 1139 and 1151** - `subprocess.TimeoutExpired` is not caught, causing the entire assembly to crash.
- 4. **`pool[0]` access at line 1006** - This is inside the `if MMR_LAMBDA > 0 and pool:` block, so pool is non-empty. But `pool[0][1].clip_embedding` could be None, which is handled by the ternary.
- 1. **`extract_clip` timeout at line 1139, 1151, 1165** - `subprocess.TimeoutExpired` is not caught, causing the entire assembly to crash.
- 3. **`embed_dim` computation at line 1005-1007** - Only checks `pool[0]` for embedding dimension, but the first clip might not have an embedding while others do. This could cause a shape mismatch in `
- - "Medium — `assemble_music_video.py:195–199` — Unguarded dict access in `extract_phrase_features`"
- 4. **Unguarded dict access in `extract_phrase_features`** - Direct `data[key]` access at lines 411-415 could raise `KeyError` if the key is missing.

## Area: . (chunk 5)

## Bugs & security

**Medium — `detect_loops.py:47–49` — Empty `file_info.path` causes `find_source_video` to return first file in SOURCE_DIR**

```python
    sanitized_target = _sanitize_for_match(original_path.stem)
    for candidate in SOURCE_DIR.iterdir():
        if original_path.stem[:25] in candidate.stem or candidate.stem[:25] in original_path.stem:
            return str(candidate)
```

When `file_info.get("path", "")` yields `""` (or a trailing-slash path like `"/some/dir/"`), `Path("").stem` is `""`. The expression `"" in candidate.stem` is always `True`, so the first entry from `SOURCE_DIR.iterdir()` is returned unconditionally. Loop detection then runs against the wrong source video, writing incorrect `loopable`/`loop_similarity` values back into the enriched JSON.

**Impact:** Data corruption — enriched files receive loop-detection results from an unrelated video.

**Suggested fix:** Guard against empty stems before the fuzzy-match loops: `if not original_path.stem: return None` after line 44.

---

**Medium — `detect_loops.py:114–118` — Orphaned frame file when one of two extractions fails**

```python
    ok1 = extract_frame(video_path, t_start, frame_start)
    ok2 = extract_frame(video_path, t_end, frame_end)

    if not (ok1 and ok2):
        return False, 0.0
```

When `ok1` is `True` but `ok2` is `False`, `frame_start` exists on disk but the early return at line 118 bypasses the `try/finally` block (lines 120–132) that would clean it up. The `finally` is never reached because the `return` precedes the `try`. Each such failure leaves one orphaned `.jpg` in `/tmp/loop_detect_frames/`.

**Impact:** Accumulating disk waste in `/tmp/`; over a large enriched corpus this can leave thousands of stray frames.

**Suggested fix:** Clean up before the early return:
```python
    if not (ok1 and ok2):
        frame_start.unlink(missing_ok=True)
        frame_end.unlink(missing_ok=True)
        return False, 0.0
```

## Structural improvements

1. **Extract hardcoded `SOURCE_DIR` from `detect_loops.py:27`** — `SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")` is an absolute, machine-specific path. The same path appears in `assemble_v2.py` and `enrich_analyses.py` (per survey notes). Move to an environment variable with a documented default or a shared config module. **Risk:** Low — only affects portability. **Verify:** Confirm no other code imports this symbol directly; grep shows it's only used within `detect_loops.py`.

2. **Eliminate double file read in `detect_loops.py:139` and `detect_loops.py:248`** — `main()` reads each enriched file at line 248 to count scenes for the progress display, then `process_video()` reads the same file again at line 139 to process it. For large enriched JSON files (some >10 MB), this doubles I/O. **Risk:** Low — purely performance. **Verify:** Pass the already-loaded `data` dict from `main()` into `process_video()` instead of re-reading.

3. **Replace `rmdir()` with `shutil.rmtree` in `detect_loops.py:176`** — `tmp_dir.rmdir()` only removes empty directories. If any frames leaked (see finding above), the `OSError` is silently swallowed at line 177–178, leaving stale temp directories. **Risk:** Low — `/tmp` is cleaned on reboot, but repeated runs accumulate cruft. **Verify:** Use `shutil.rmtree(tmp_dir, ignore_errors=True)` to guarantee cleanup.

4. **Atomic write for enriched files in `detect_loops.py:172`** — `enriched_path.write_text(json.dumps(...))` writes the entire file in one call. If the process is killed mid-write (OOM, SIGKILL), the enriched JSON is truncated and corrupted. **Risk:** Medium — data loss on a single enriched file. **Verify:** Write to a `.tmp` sibling path then `os.replace()` for atomicity.

5. **Silent zero-tensor fallback in `clip_utils.py:65–66`** — When `Image.open(p)` fails, `encode_images` substitutes `torch.zeros(3, 224, 224)`. The resulting CLIP embedding is meaningless but indistinguishable from a valid embedding downstream. Callers have no way to know which entries are garbage. **Risk:** Low — only affects frames that fail to load (corrupt files, wrong paths). **Verify:** Return `None` for failed images and let callers decide, or emit a warning list alongside the embeddings.

## Area: . (chunk 6)

## Bugs & security

**Medium — `generate_clip_embeddings.py:38–44` — Empty stem causes `find_source_video` to return first file in `SOURCE_DIR`**

```python
    original_path = Path(file_info.get("path", ""))
    if original_path.exists():
        return str(original_path)
    sanitized_target = _sanitize_for_match(original_path.stem)
    for candidate in SOURCE_DIR.iterdir():
        if original_path.stem[:25] in candidate.stem or candidate.stem[:25] in original_path.stem:
            return str(candidate)
```

When `file_info.get("path", "")` yields `""` (e.g. enriched file missing the `"file"` key or `"path"` field), `Path("").stem` is `""`. The expression `"" in candidate.stem` is always `True`, so the first entry from `SOURCE_DIR.iterdir()` is returned unconditionally. CLIP embeddings are then generated from the wrong source video, writing incorrect embeddings into the sidecar.

**Impact:** Data corruption — wrong embeddings silently written for the affected video, degrading semantic similarity search and scoring in the assembler.

**Suggested fix:** Guard against empty stems before the fuzzy-match loops: `if not original_path.stem: return None` after line 41.

---

**Medium — Non-atomic JSON writes across 5 locations in 3 files**

```python
# enrich_analyses.py:520
enriched_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False))

# enrich_analyses.py:657
ef.write_text(json.dumps(data, indent=2, ensure_ascii=False))

# enrich_analyses.py:779
ef.write_text(json.dumps(data, indent=2, ensure_ascii=False))

# flag_quality.py:206
out_path.write_text(json.dumps(output, ensure_ascii=False))

# generate_clip_embeddings.py:136
output_path.write_text(json.dumps(output, ensure_ascii=False))
```

All five write the entire JSON payload in a single `write_text()` call. If the process is killed mid-write (OOM, SIGKILL, power loss), the output file is truncated and corrupted. The state file in `enrich_analyses.py:96–98` already uses the correct atomic pattern (`tmp.write_text` → `tmp.rename`), but the data files do not. Enriched files can exceed 10 MB, making mid-write corruption more likely.

**Impact:** Corrupted sidecar JSON on process kill; downstream consumers (assembler, quality pass) crash or produce incorrect results on the corrupted file.

**Suggested fix:** Write to a `.tmp` sibling path then `os.replace()` (or `Path.rename()`) for atomicity at all five locations.

---

**Low — `generate_clip_embeddings.py:142` — `rmdir()` silently fails when temp dir is non-empty**

```python
    try:
        tmp_dir.rmdir()
    except OSError:
        pass
```

`rmdir()` only removes empty directories. If any frame files leaked (e.g. from a failed extraction that still wrote a partial file), the `OSError` is silently swallowed, leaving stale temp directories in `/tmp/clip_embed_frames/`. Over repeated runs this accumulates disk waste.

**Impact:** Accumulating orphaned temp directories in `/tmp/`.

**Suggested fix:** Use `shutil.rmtree(tmp_dir, ignore_errors=True)` to guarantee cleanup.

## Structural improvements

1. **Consolidate the two `find_source_video` implementations** — `enrich_analyses.py:120–146` and `generate_clip_embeddings.py:36–50` implement the same source-video resolution logic with different signatures and different robustness. The `enrich_analyses.py` version is superior (falls back to analysis filename fuzzy match; accepts `source_dir` as parameter). Extract a shared function into a common module (e.g. `path_utils.py`) and have both scripts import it. **Risk:** Low — the enriched version is strictly more robust. **Verify:** After consolidation, confirm both callers pass the correct arguments and that the empty-stem guard is present.

2. **Extract hardcoded `SOURCE_DIR` from `generate_clip_embeddings.py:26`** — `SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")` is an absolute, machine-specific path with no environment-variable override. The same path appears in `enrich_analyses.py:48` (`DEFAULT_SOURCE`) and `detect_loops.py:27`. Move to a shared config module or environment variable with a documented default. **Risk:** Low — only affects portability. **Verify:** Grep confirms no other code imports this symbol directly.

3. **Eliminate double file read in `generate_clip_embeddings.py:187–188` and `206–207`** — `main()` reads each enriched file at line 187 to count total scenes for the header display, then `process_video()` reads the same file again at line 90. For large enriched JSON files (some >10 MB), this doubles I/O. **Risk:** Low — purely performance. **Verify:** Pass the already-loaded `data` dict from `main()` into `process_video()` instead of re-reading.

4. **Atomic writes for all JSON output** — As described in the Medium finding above, convert all five direct `write_text(json.dumps(...))` calls to the tmp+rename pattern already used for the state file. **Risk:** Low — the pattern is proven in the same codebase. **Verify:** Confirm no other process reads the file while it's being written (these are batch scripts, so this is safe).

## Area: . (chunk 7)

## Bugs & security

**Medium — `vision_backends.py:250` — Hardcoded `media_type: "image/jpeg"` in `AnthropicAPIBackend.analyze_frame`**

```python
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": _read_image_b64(image_path),
                    },
```

The Anthropic API requires `media_type` to match the actual image format (supports `image/jpeg`, `image/png`, `image/gif`, `image/webp`). `_read_image_b64` reads raw bytes without inspecting format. If a non-JPEG frame (e.g. PNG from a different extraction path) is passed, the API may reject the request or misdecode the image, producing garbage analysis output. The `OllamaBackend` at line 115 has no this issue because Ollama's `/api/generate` infers format from the base64 payload.

**Impact:** Silent analysis failure or incorrect semantic output when non-JPEG images reach the Anthropic API backend.

**Suggested fix:** Detect format from the file extension or magic bytes before the API call:
```python
import mimetypes
_media_type, _ = mimetypes.guess_type(image_path) or ("image/jpeg", None)
```
Then use `_media_type` in the `source` dict.

---

**Medium — `query_scenes.py:119–123` — Unguarded `subprocess.run` for ffmpeg in `extract_preview`**

```python
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(start), "-i", str(video),
         "-t", str(clip_dur), "-avoid_negative_ts", "make_zero",
         "-c:v", "libx264", "-an", str(out)],
        capture_output=True)
```

No `timeout=` argument. If ffmpeg hangs (e.g. seeking into a multi-gigabyte file on a slow or unresponsive archive drive, or encountering a corrupt container), the entire `query_scenes.py` process blocks indefinitely with no recovery path. All other subprocess calls in the codebase (`vision_backends.py:159`, `:182`, `:127`) carry explicit timeouts.

**Impact:** Complete hang of the scene-query tool on a single slow or corrupt source video; no partial results returned.

**Suggested fix:** Add `timeout=120` and wrap in `try/except subprocess.TimeoutExpired` returning `None` (consistent with the function's existing `return None` on failure).

## Structural improvements

1. **Centralise image-format detection in `vision_backends.py`** — Replace the hardcoded `"image/jpeg"` at line 250 with a helper (e.g. `_guess_media_type(image_path)`) that checks magic bytes or `mimetypes.guess_type`. This also future-proofs the function if the pipeline ever produces PNG/WebP frames. **Risk:** Low — `mimetypes` is stdlib; magic-byte fallback is trivial. **Verify:** Unit-test `_guess_media_type` against known JPEG/PNG/WebP samples; confirm Anthropic API accepts all four types.

2. **Add timeout to `extract_preview` in `query_scenes.py:119`** — Mirror the timeout discipline already present in `vision_backends.py` (300 s for model calls, 30 s for CLI version checks). A 120-second timeout is appropriate for a short clip extraction. **Risk:** Low — only affects the optional `--extract` path. **Verify:** Confirm `subprocess.TimeoutExpired` is caught and `None` is returned, matching the existing failure contract.

3. **`query_scenes.py:74` — Direct key access on embedding data** — `vectors.append(np.asarray(se["embedding"], dtype=np.float32))` uses bare `se["embedding"]` with no `.get()` fallback. If a `.clip_embeddings.json` sidecar is partially written or corrupted (e.g. from a mid-write kill, a known risk per the non-atomic-write findings in chunks 4–5), this crashes `build_index` with an uninformative `KeyError`. **Risk:** Low — only affects corrupted sidecars. **Verify:** Switch to `se.get("embedding")` and skip scenes with `None` embeddings, emitting a warning count at the end of `build_index`.

## Area: . (chunk 10)

## Bugs & security

**Medium — `render_sizzle.py:544,557,578` — Three `subprocess.run` calls for ffmpeg lack `timeout=`**

```python
# Line 544
subprocess.run([
    "ffmpeg", "-y",
    "-framerate", str(FPS),
    "-i", str(seg_dir / "frame_%05d.png"),
    ...
], capture_output=True)

# Line 557
subprocess.run([
    "ffmpeg", "-y",
    "-i", str(seg_video),
    "-ss", str(max(0, start_sec - TITLE_DURATION)),
    ...
], capture_output=True)

# Line 578
subprocess.run([
    "ffmpeg", "-y",
    "-f", "concat", "-safe", "0",
    ...
], capture_output=True)
```

All three ffmpeg invocations omit `timeout=`. If ffmpeg hangs (e.g., seeking into a multi-gigabyte file on a slow archive drive, or encountering a corrupt container), the entire sizzle-reel render blocks indefinitely with no recovery path. Every other subprocess call in the codebase (`vision_backends.py:159,182,127`, `extract_lyrics.py:131,167`, `query_scenes.py:119`) carries an explicit timeout.

**Impact:** Complete hang of the sizzle-reel renderer on a single slow or corrupt source video; no partial results returned.

**Suggested fix:** Add `timeout=120` to each call and wrap in `try/except subprocess.TimeoutExpired` returning a failure indicator (consistent with the rest of the codebase).

---

**Medium — `render_sizzle.py:588–597` — Cleanup without `try/finally` or existence checks**

```python
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
```

If any `subprocess.run` call earlier in `main()` raises (e.g., `TimeoutExpired` once timeouts are added, or `FileNotFoundError` if ffmpeg is missing), this cleanup block is never reached, leaving orphaned frame PNGs, segment MP4s, and the concat list on disk. Furthermore, `seg_dir.rmdir()` raises `OSError` if any PNGs leaked (e.g., from a failed `frame.save()`), aborting the rest of the cleanup. `concat_file.unlink()` raises `FileNotFoundError` if the concat step was never reached.

**Impact:** Disk-space leak on every failed render run; unhandled `OSError`/`FileNotFoundError` crashes the program during cleanup.

**Suggested fix:** Wrap the render body in `try/finally`; use `shutil.rmtree(seg_dir, ignore_errors=True)` instead of `rmdir()`; guard each removal with `.exists()` checks.

---

**Medium — `extract_lyrics.py:444` — Non-atomic JSON write**

```python
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
```

Writes the entire lyrics JSON in a single `write_text()` call. If the process is killed mid-write (OOM, SIGKILL, power loss), the output file is truncated and corrupted. The state file in `enrich_analyses.py:96–98` already uses the correct atomic pattern (`tmp.write_text` → `tmp.rename`), but this data file does not. Lyrics files can be large (many segments with word-level timestamps).

**Impact:** Corrupted `.lyrics.json` on process kill; downstream assembler crashes or produces incorrect results on the corrupted file.

**Suggested fix:** Write to a `.tmp` sibling path then `os.replace()` (or `Path.rename()`) for atomicity.

---

**Medium — `reprocess_pass2.py:390–391` — Non-atomic JSON write**

```python
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
```

Same non-atomic write pattern. The backup at line 386–388 (`os.rename` to `.pass1.analysis.json`) is atomic, but the new write is not. If the process is killed mid-dump, the analysis JSON is truncated and corrupted, and the backup rename already happened, so the original is lost too.

**Impact:** Corrupted `.analysis.json` on process kill; the original pass-1 backup is already renamed, so data is unrecoverable.

**Suggested fix:** Write to a `.tmp` sibling path then `os.replace()` for atomicity.

## Structural improvements

1. **Extract hardcoded archive paths from three files** — `extract_lyrics.py:39` (`SETS_DIR`), `render_sizzle.py:490–491` (`analysis_path`, `audio_path`), and `reprocess_pass2.py:42` (`SOURCE_DIR`) all contain absolute, machine-specific paths (`/Volumes/archive/3000/3100/…`). Move to environment variables with documented defaults or a shared config module. **Risk:** Low — only affects portability. **Verify:** Grep confirms no other code imports these symbols directly.

2. **Remove dead code in `render_sizzle.py:238` and `render_sizzle.py:611–615`** — `band_names` is assigned at line 238 but never used after that. The `BANDS` dict at line 611 is defined after `render_frame`, and `'BANDS' in dir()` inside `render_frame` always returns `False` (because `dir()` checks the local namespace, not the module namespace), so the hardcoded fallback list is always used. **Risk:** Low — purely cosmetic. **Verify:** Delete `band_names` assignment and the `BANDS` dict; confirm `render_frame` still runs correctly.

3. **Replace `rmdir()` with `shutil.rmtree` in `render_sizzle.py:592`** — `seg_dir.rmdir()` only removes empty directories. If any PNG files leaked (e.g., from a failed `frame.save()`), the `OSError` aborts the rest of the cleanup. **Risk:** Low — `/tmp`-style directories are cleaned on reboot, but repeated runs accumulate cruft. **Verify:** Use `shutil.rmtree(seg_dir, ignore_errors=True)` to guarantee cleanup.

## Area: . (chunk 11)

## Bugs & security

**Medium — Non-atomic JSON writes in `scan_text.py:340`, `scan_text.py:347`, `scan_text_fast.py:667`**

```python
# scan_text.py:340
output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

# scan_text.py:347
output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False))

# scan_text_fast.py:667
output_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
```

All three write the entire JSON payload in a single `write_text()` call. If the process is killed mid-write (OOM, SIGKILL, power loss), the output file is truncated and corrupted. Downstream consumers — the assembler, `show_status()` at `scan_text.py:392`/`scan_text_fast.py:705`, and the resume logic at `scan_text.py:254` and `scan_text.py:289` — crash on corrupted files with an uninformative `json.JSONDecodeError`. The state file in `enrich_analyses.py:96–98` already uses the correct atomic pattern (`tmp.write_text` → `tmp.rename`), but these data files do not.

**Impact:** Corrupted `.text_flags.json` on process kill; scanner resume logic and status display crash on corrupted files.

**Suggested fix:** Write to a `.tmp` sibling path then `os.replace()` (or `Path.rename()`) for atomicity at all three locations.

---

**Low — `rmdir()` silently fails when temp dir is non-empty in `scan_text.py:353` and `scan_text_fast.py:679`**

```python
# scan_text.py:352–355
        try:
            vid_frames.rmdir()
        except OSError:
            pass

# scan_text_fast.py:678–681
        try:
            vid_frames.rmdir()
        except OSError:
            pass
```

`rmdir()` only removes empty directories. If any frame files leaked (e.g., from a failed `extract_frame` that still wrote a partial file, or from a crash between frame creation and deletion), the `OSError` is silently swallowed, leaving stale temp directories in `text_flags/frames/`. Over repeated runs this accumulates disk waste.

**Impact:** Accumulating orphaned temp directories in `text_flags/frames/`.

**Suggested fix:** Use `shutil.rmtree(vid_frames, ignore_errors=True)` to guarantee cleanup.

## Structural improvements

1. **Deduplicate shared functions between `scan_text.py` and `scan_text_fast.py`** — Both files implement identical `extract_frame` (`scan_text.py:77–88` / `scan_text_fast.py:190–201`), `check_ollama` (`scan_text.py:66–74` / `scan_text_fast.py:204–212`), `try_parse_json` (`scan_text.py:135–147` / `scan_text_fast.py:256–268`), `find_source_video` (`scan_text.py:159–183` / `scan_text_fast.py:290–308`), `fmt_time` (`scan_text.py:186–191` / `scan_text_fast.py:311–316`), and `fmt_bar` (`scan_text.py:194–196` / `scan_text_fast.py:319–321`). Extract these into a shared module (e.g., `text_scan_utils.py`). **Risk:** Low — the functions are textually identical. **Verify:** After extraction, confirm both scripts import and use the shared functions correctly; run a quick smoke test of each scanner.

2. **Extract hardcoded `SOURCE_DIR` from both files** — `scan_text.py:26` and `scan_text_fast.py:37` both contain `SOURCE_DIR = Path("/Volumes/archive/3000/3100/visuals/raw visuals footage")`. This absolute, machine-specific path appears in at least five other files across the repo. Move to an environment variable with a documented default or a shared config module. **Risk:** Low — only affects portability. **Verify:** Grep confirms no other code imports these symbols directly; the change is purely local to each file.

3. **Atomic writes for all JSON output** — As described in the Medium finding, convert all three direct `write_text(json.dumps(...))` calls to the tmp+rename pattern already proven in `enrich_analyses.py:96–98`. **Risk:** Low — the pattern is established in the same codebase. **Verify:** Confirm no other process reads the file while it's being written (these are batch scripts, so this is safe).

## Area: media_analyzer (chunk 12)

## Bugs & security

**Medium — `media_analyzer/video_analyzer.py:52` — `get_video_metadata` invokes ffprobe without `timeout=`**

```python
    result = subprocess.run(cmd, capture_output=True, text=True)
```

`subprocess.run` has no `timeout=` argument. If ffprobe hangs (e.g., on a corrupt container, a file on an unresponsive network/archive drive, or a deeply nested directory structure), the entire video analysis pipeline blocks indefinitely with no recovery path. The sibling function `_detect_scenes_ffmpeg` at line 125 correctly uses `timeout=300`; this one does not.

**Impact:** Complete hang of `analyze_video` on a single problematic source file; no partial results saved.

**Suggested fix:** Add `timeout=60` (ffprobe is fast on well-formed files) and wrap in `try/except subprocess.TimeoutExpired` raising a clear `RuntimeError`.

---

**Medium — Non-atomic JSON writes in `media_analyzer/audio_analyzer.py:345–346` and `media_analyzer/video_analyzer.py:511–512`**

```python
# audio_analyzer.py:345–346
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=indent)

# video_analyzer.py:511–512
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=indent)
```

Both `main()` functions open the output file for writing (truncating it) then stream JSON into it. If the process is killed mid-dump (OOM, SIGKILL, power loss), the output file is left empty or truncated. Downstream consumers crash with `json.JSONDecodeError`. The atomic tmp+rename pattern is already proven in `vision_model_contest.py:119–121` and `enrich_analyses.py:96–98` within this repo.

**Impact:** Corrupted `.analysis.json` on process kill; downstream assembler or enrichment pipeline crashes on the corrupted file.

**Suggested fix:** Write to a `.tmp` sibling path then `os.replace()` (or `Path.rename()`) for atomicity at both locations.

---

**Low — `media_analyzer/audio_analyzer.py:128–129` — `np.corrcoef` returns NaN for constant chroma, producing misleading key detection**

```python
        corr_maj = float(np.corrcoef(rolled, major_profile)[0, 1])
        corr_min = float(np.corrcoef(rolled, minor_profile)[0, 1])
```

When `chroma_mean` is constant (e.g., all zeros for silent audio), `np.roll` produces a constant vector. `np.corrcoef` on a zero-variance input returns `[[1, nan], [nan, 1]]`. `float(nan)` is `nan`, and `nan > -1` is `False`, so the loop never updates `best_corr`. The function returns `{"key": "C", "mode": "major", "confidence": -1}` — a fabricated key with a confidence value that semantically means "perfect negative correlation" rather than "no data."

**Impact:** Misleading key/mode output for silent or near-silent audio segments; confidence of `-1` is semantically wrong.

**Suggested fix:** Guard with `np.isnan(corr_maj)` / `np.isnan(corr_min)` and skip NaN correlations, or return `confidence: 0.0` with a `"key": "unknown"` sentinel when all correlations are NaN.

## Structural improvements

1. **Extract hardcoded `DEFAULT_SOURCE` from `vision_model_contest.py:60`** — `DEFAULT_SOURCE = "/Volumes/archive/3000/3100/visuals/raw visuals footage"` is an absolute, machine-specific path used as the argparse default at line 605. The same path appears in at least five other files across the repo. Move to an environment variable with a documented default or a shared config module. **Risk:** Low — only affects portability; `--source` override already works. **Verify:** Grep confirms no other code imports this symbol directly.

2. **Remove dead `total_elapsed` computation in `vision_model_contest.py:566–571`** — A complex multi-branch ternary expression computes `total_elapsed` but the variable is never used; `time.time() - contest_start` is passed to `print_frame_result` at line 576 instead. This dead code is confusing and obscures the actual ETA logic. **Risk:** Low — purely cosmetic. **Verify:** Delete lines 566–571; confirm `print_frame_result` still receives correct arguments.

3. **Deduplicate `fmt_time` and `fmt_bar` from `vision_model_contest.py:86–98`** — These two formatting helpers are textually identical to copies in `scan_text.py`, `scan_text_fast.py`, `assemble_v2.py`, and other files. Extract them into a shared module (e.g., `utils.py` or `media_analyzer/__init__.py`). **Risk:** Low — the functions are pure and trivially testable. **Verify:** After extraction, confirm all callers import from the shared location; run a smoke test of each script.

4. **Eliminate double `get_video_metadata` call in `media_analyzer/video_analyzer.py`** — `analyze_video` calls `get_video_metadata` at line 437, then `detect_scenes` → `_detect_scenes_ffmpeg` calls it again at line 139. This runs ffprobe twice per file. **Risk:** Low — purely performance. **Verify:** Pass the already-loaded `metadata` dict from `analyze_video` into `detect_scenes` (add an optional `metadata` parameter), and skip the redundant call in `_detect_scenes_ffmpeg` when provided.

## Coverage gaps

These areas could not be analyzed (verbose or empty model output) and may still contain issues:
- chunk 8 (bench): vision_schema.py, bench/TESTPLAN.md, bench/__init__.py, bench/groundtruth.py …
- chunk 9 (bench): bench/mlx_backend.py, bench/report.py, bench/runner.py, bench/sampler.py …
