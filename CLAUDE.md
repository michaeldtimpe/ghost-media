# CLAUDE.md

Guidance for Claude Code (and other agents) working in this repo.

## What this is
`ghost-media` is an algorithmic music-video generator. Two layers:
- **Toolkit** — the installable `media_analyzer/` package (CLI: `media-analyzer`); turns audio/video into JSON.
- **Pipeline** — the root `*.py` scripts that consume that JSON and render videos; `assemble_v2.py` is the heart of it.

## Start here
1. **[AGENTS.md](AGENTS.md)** — contributor guide: key files, common tasks, gotchas.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — data flow, JSON schemas, scoring + diversity params.
3. **[README.md](README.md)** — overview, quickstart, repo layout.

## Working conventions
- **Generated data is gitignored** (`*.analysis.json`, `enriched/`, `text_flags/`, `sets/`, `demucs_output/`, media). Never commit it. Code, docs, and `examples/` only.
- The pipeline scripts run from the repo root and read/write data **relative to the root** (`./enriched/`, `./*.analysis.json`, `./sets/`). The toolkit batch drivers default their output to `~/Downloads/ghost-media`.
- Source footage and DJ-set audio live on an external archive drive (`/Volumes/archive/...`) — it must be mounted for analysis/assembly. Paths are documented in `ARCHITECTURE.md` / `AGENTS.md`.
- All analysis timelines are sampled at 8 Hz; scenes (not frames) are the atomic unit for clip selection.
- Vision enrichment + lyrics extraction need Ollama / Demucs+Whisper respectively; both are optional for the assembler, which only needs numpy + FFmpeg.

## Environment
macOS on Apple Silicon (64 GB). Prefer `python3 -m venv` / the provided `setup.sh`; FFmpeg via Homebrew.
