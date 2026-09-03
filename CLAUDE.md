# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repo.

## What this is
`ghost-media` is an algorithmic music-video generator. Three layers:
- **Toolkit** — the installable `media_analyzer/` package (CLI: `media-analyzer`); turns audio/video into JSON.
- **Pipeline** — the root `*.py` scripts that consume that JSON and render videos; `assemble_v2.py` is the heart of it.
- **Bench** — the `bench/` package (CLI: `bench_run.py`): a vision-engine bake-off harness that scores VLMs (Ollama / MLX / Claude) on the footage to pick the production enrichment engine. The winner feeds `enrich_analyses.py --sampling-plan`. See **[bench/TESTPLAN.md](bench/TESTPLAN.md)**.

## Start here
1. **[AGENTS.md](AGENTS.md)** — contributor guide: key files, common tasks, gotchas.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — data flow, JSON schemas, scoring + diversity params.
3. **[README.md](README.md)** — overview, quickstart, repo layout.
4. **[lessons.md](lessons.md)** — engineering lessons + gotchas (Vision, Audio, and Selection sections — read these before changing scoring or selection mechanics).
5. **[bench/TESTPLAN.md](bench/TESTPLAN.md)** — vision-engine bake-off plan + status; bake-off + corpus rescan + perceptual-diversity uplift all **complete**.
6. **[audio_field_audit.md](audio_field_audit.md)** — per-field contract for `.deep-analysis.json` (schema 2.2.0): role, cost, recommendation per field. Reference when modifying audio output.
7. **[scripts/](scripts/)** — `audit_repeats.py` for diagnosing visible-repeat complaints; `capture_perceptual_baseline.py` for tuning selection constants without paying for full renders.
8. **[MIGRATION.md](MIGRATION.md)** — moving the working setup (repo + gitignored data + archive media) to a new host: copy lists, hardcoded paths, post-move verification.

## Working conventions
- **Generated data is gitignored** (`*.analysis.json`, `enriched/`, `text_flags/`, `sets/`, `demucs_output/`, `bench/results/`, `raw_footage/`, media). Never commit it. Code, docs, and `examples/` only.
- The pipeline scripts run from the repo root and read/write data **relative to the root** (`./enriched/`, `./*.analysis.json`, `./sets/`). The toolkit batch drivers default their output to `~/Downloads/ghost-media`.
- Source footage and DJ-set audio live on the NAS (`/Volumes/archive/...`) — it must be mounted for analysis/assembly. All media paths come from `media_paths.py` (canonical footage root `visuals/library/<collection>/`, env-overridable via `GHOST_FOOTAGE_ROOT` etc.); see `MIGRATION.md` for the layout, the remap/verify utility, and the local-cache workflow.
- All analysis timelines are sampled at 8 Hz; scenes (not frames) are the atomic unit for clip selection.
- Vision enrichment + lyrics extraction need Ollama / Demucs+Whisper respectively; both are optional for the assembler, which only needs numpy + FFmpeg.

## Environment
macOS on Apple Silicon (64 GB). Prefer `python3 -m venv` / the provided `setup.sh`; FFmpeg via Homebrew.

## Quality gates (cleat)

`python3 quality/bin/gate.py` runs every quality gate; it also runs when you
stop, and a failing gate is handed back to you as the next thing to fix. A
failure names the file, the line and what fixes it — split the function, give
the value its real type, make the test pass, handle the error.

Do not edit `quality.json`, anything under `quality/`, or the hooks to make a
gate pass, and do not run `--write-baseline`: the baselines record debt a
person accepted, and only a person loosens them, in a reviewed commit. The
gates only ever tighten; that is the point.
