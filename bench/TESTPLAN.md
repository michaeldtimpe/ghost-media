# Vision-Engine Bake-Off — Test Plan

Revised after reviewer feedback. The harness lives in `bench/`; this doc is the
execution plan: what we test, on what, how we judge, and in what order.

## Status / how to resume (updated 2026-05-25 — migrated to M5 Max)

**Migration complete.** The project moved from the M1 Max (64 GB) to an **M5 Max
(128 GB)**. The migration carried the *data* (repo, footage, sampling plans, extracted
frames, enriched corpus) but **not the compute env**, which was rebuilt from scratch.
Because we're now on the M5, the old two-phase hardware plan collapses: the **full
`DEFAULT_ENGINES` lineup incl. the 32B engines runs in one pass**.

**Environment (rebuilt):** dedicated `.venv` on **Python 3.12** (`PYTHON=python3.12
./setup.sh` then `pip install torch open_clip_torch mlx-vlm`); `requirements.lock`
committed; `pyproject` pinned `>=3.12,<3.14`. Functional probe + `smoke_test.py` pass
(torch/sympy + mlx + open_clip import; CLIP runs on `mps`). Ollama 0.24.0.

**Models staged + fingerprinted** (`bench/results/model_fingerprints.json`):
- ollama: `qwen2.5vl:7b` (Q4_K_M), `qwen2.5vl:32b` (Q4_K_M, 21 GB), `minicpm-v:8b` (Q4_0).
- mlx (HF cache): `Qwen2.5-VL-7B-Instruct-4bit`, `Qwen2.5-VL-32B-Instruct-4bit`,
  `InternVL3-8B-MLX-4bit` — the config's old `InternVL3-8B-4bit` 404s; **fixed** to the
  real `-MLX-4bit` id in `config.py`.
- kappa archive (`/Volumes/home/hub`) inventoried: **none of the VL engines are there**
  (only text MLX models + the dropped gemma-3-27b) — all weights downloaded fresh.
- `python3 bench_run.py health` = all 8 engines ✓.

**Embedding consistency verified:** the migrated plan embeddings reproduce M5-encoded
CLIP at **cosine 1.000000** (open_clip's new QuickGELU *warning* is cosmetic — both
machines use standard GELU). No re-plan needed; **281 distinct states** across the 6
clips. `baseline`/`clip-ceiling` re-scored on M5.

**Smoke PASSED (Phase 3 gate):** `ollama-qwen7b` on EXC3_CM3 — composite **0.529 vs
baseline 0.217**, coverage **0.98 vs 0.12**, JSON adherence 0.98, enum-snaps 0,
non-compliance 0, 40/41 parsed, descriptions specific. Decisively beats the floor.

**Throughput finding + fix:** Ollama loaded qwen2.5vl:7b at its **full 128k context →
52 GB / ~16.5 s/frame**. Capped `num_ctx=8192` in `vision_backends.py` → **14 GB**, same
quality. A *memory* fix, not latency (per-frame ~16.5 s is inherent). See lessons.md
("A model server's default context window is a hidden memory bomb").

**Migration key note:** `keys.canonical_key` is a SHA of the **absolute source path**,
so M1 keys ≠ M5 keys. Fresh runs are self-consistent (`score_all` rebuilds the index
under the live key); only the migrated synthetic raw had to be re-run on M5 (done).

**NOT yet done — resume here:**
1. **Full bake-off (Phase 4)** — awaiting go-ahead. Run `python3 bench_run.py run`
   (bare = full `DEFAULT_ENGINES`). Budget ≈ 16.5 s/frame × 281 states/engine ≈ 75
   min/engine (overnight for the lineup). Then `compare`.
2. **Text ground-truth (Phase 2)** — interactive: `python3 bench_label.py --n 250`
   (opens each frame; y/n). Needed for the text-F1 metric (currently 0). Oversample
   hard cases; include negative controls. NB: qwen7b flagged text on 25/41 EXC3_CM3
   frames — GT will show whether that's accurate or over-flagging.
3. **Optional pre-sweep hardening** — 3-tier health (runtime/semantic/OCR sanity),
   per-frame RAM/output-len telemetry, calibration audit.
4. **claude-cli + judge** stay deferred (they touch `ANTHROPIC_API_KEY`; use
   `env -u ANTHROPIC_API_KEY …` to bill the subscription — the bench hard-blocks otherwise).

## Why (the trap to avoid)
"Video understanding for tagging" is **four partly-competing capabilities**, not one:
1. **OCR / on-screen text** (drives `has_english_text` + `text_content`)
2. **Dense, discriminative scene description**
3. **Temporal consistency** across adjacent frames/states
4. **Throughput** per frame at an acceptable hallucination rate

Most bake-offs accidentally optimize only for #2 ("describe this vividly"), which
**overstates real pipeline quality**. We explicitly optimize for production
usefulness: schema stability, low hallucination, OCR fidelity, consistency, and
**downstream retrieval** — not caption flair. A model that says *"DJ under blue
lights"* correctly and consistently beats one that invents *"an energetic
underground techno artist in a cyberpunk rave"*.

## Engine lineup (revised)

The Qwen2.5-VL **7B + 32B** ladder runs on **both** runtimes so the comparison
isolates runtime. MiniCPM (Ollama) is the throughput wildcard; InternVL (MLX) is
the OCR specialist that may dominate the text submetric. claude-cli is the cloud
quality reference. `baseline` = current system (floor); `clip-ceiling` = image→image
upper bound.

| Engine (config key) | Backend / model | Role |
|---|---|---|
| `ollama-qwen7b` | ollama · qwen2.5vl:7b | **Baseline to beat** — OCR + consistency + schema stability sweet spot |
| `ollama-minicpm` | ollama · minicpm-v:8b *(→4.5 if served)* | High-throughput wildcard enricher; rich but more variance — not a sole truth source |
| `ollama-qwen32b` | ollama · qwen2.5vl:32b | Local quality/OCR ceiling |
| `mlx-qwen7b` | mlx · Qwen2.5-VL-7B-Instruct-4bit | Cross-runtime vs `ollama-qwen7b` |
| `mlx-qwen32b` | mlx · Qwen2.5-VL-32B-Instruct-4bit | Flagship local ceiling (largest MLX advantage at scale) |
| `mlx-internvl` | mlx · InternVL3-…-4bit *(verify id)* | OCR/text specialist — submetric dominance |
| `claude-cli` | claude-cli · sonnet | Cloud quality reference (run explicitly; subscription) |
| `baseline` / `clip-ceiling` | — | Floor / ceiling references |

Dropped from the earlier shortlist: Gemma-3 (reviewers expect Qwen to edge it on
OCR/video; keep only as an optional diversity add). Verify exact `mlx-community`
repo ids and the Ollama `minicpm-v` version tag before pulling.

### Runtime note (important)
Ollama now uses an **MLX backend on Apple Silicon** (~Mar 2026; ~7× decode speedup
on M1 Max). So Ollama-vs-MLX is no longer framework-vs-framework — it's **quant
suffix, caching, and cold-vs-warm start**. Expect Ollama warm-cache to win repeat
inference; raw `mlx-vlm` to win cold start and to offer control over frame-stepping
/ kv-cache. Still worth measuring; just interpret it that way.

### Frame ingestion (deliberate)
We **extract representative frames upfront with ffmpeg and send single images** per
call — we do **not** use the models' native video wrappers (which drop frames
unpredictably depending on quant/layout). The adaptive sampler (`bench/sampler.py`)
already produces the medoid frame per distinct visual state.

### Temporal coherence
Describing **one medoid per distinct state** avoids intra-state frame flicker *by
construction*. The `adjacent-state discriminability` metric rewards neighbouring
**distinct** states being described differently (anti-generic); it is not in tension
with flicker-avoidance (that's within-state). Optional future probe: describe ≥2
candidates in a multi-frame cluster and measure agreement (consensus voting) — the
plan schema already allows >1 representative per state.

## Footage (staged in `raw_footage/`)
Six short, high-signal clips spanning the regimes (durations are the cost driver):

| Clip | Dur | Regime / stress |
|---|---|---|
| EXC3_CM3 | 1:12 | motion graphics — abstract specificity |
| PAPERS | 2:50 | abstract + low-res / compression |
| Tame Impala – Nangs | 1:43 | real-world music video, fast cuts, low-light faces |
| show me lyrics | 3:47 | on-screen **English** text — precision |
| isshin REEL 2024 | 1:32 (4K) | Japanese **+** English text, angled/tiny captions, non-ASCII filename |
| Disengaging | 2:13 | clean VJ loop |

Optional heavier (on the archive drive): Mega 4K VJ Loop (2hr, near-dup/scale,
capped at 300 scenes), a drone clip (photorealistic + watermarks; shortest is ~10m),
ROOM 5 (human subjects + overlay text). The **text ground-truth** should oversample
hard cases (motion blur, strobes/smoke, LED screens, angled/tiny captions).

## Metrics & what "winning" means
Composite rank is **CLIP-anchored objective metrics only** (within-video Recall@1 +
coverage + text-F1 + adjacent discriminability). The LLM-judge is **audit-only**.
But the decision is **submetric-aware** — a model can win the production slot by
dominating one axis:

- **Discriminability / "did we really see it"**: within-video Recall@1, MRR (round-trip). *Downstream-retrieval truth, not vividness.*
- **OCR / text gate**: text precision/recall/F1 vs human GT, + EAST/OCR incumbent (baseline).
- **Coverage**: distinct-states-described / present (baseline ≈ 7%; target ≈ 100%).
- **Production reliability**: `json_adherence`, `enum_snap_rate`, text `noncompliance_rate`. An engine with low adherence is disqualified for tagging regardless of prose.
- **Throughput**: sec/frame, est. corpus hours (M1-now / M5-projected).
- **Audit (judge)**: accuracy / specificity / hallucination / under-description. If it disagrees with the composite, investigate — never average.

**Caption-style lock-in warning:** the chosen engine's caption style defines the
corpus's embedding/retrieval space; switching engines later can silently invalidate
search/clustering. So evaluate **downstream retrieval usefulness**, and once chosen,
re-embed consistently.

## Hardware sequencing
- **Now (M1 Max):** run the 7B/8B engines + small/cheap runs + all references; gather the OCR/description/reliability signal.
- **M5 Max (~early Jun 2026):** run the 32B engines and the **full-corpus rescan** with the winner (and any escalation engine for hard OCR/edge frames).

## Procedure
```
# 0. choose billing path for the cloud reference (optional)
env -u ANTHROPIC_API_KEY ...            # claude-cli on the subscription

# 1. health + pulls
python3 bench_run.py health
ollama pull qwen2.5vl:7b && ollama pull minicpm-v:8b   # (+ qwen2.5vl:32b on M5)
pip install mlx-vlm                       # MLX engines (best on M5)

# 2. sampling plans (already built for the 6) + histogram
python3 bench_run.py plan

# 3. text ground-truth (~250 frames, ≥60-80 positives, hard cases oversampled)
python3 bench_label.py --n 250

# 4. run the bake-off (defaults = local lineup + references; add claude-cli explicitly)
python3 bench_run.py run --engines ollama-qwen7b,ollama-minicpm,mlx-qwen7b,mlx-internvl,baseline,clip-ceiling --judge
#   on M5, add: ollama-qwen32b,mlx-qwen32b   (and env -u ... claude-cli)

# 5. read results
python3 bench_run.py compare
python3 bench_run.py compare --video "show me lyrics"   # side-by-side descriptions

# 6. pilot rescan with the winner, then eyeball in the assembler
env -u ANTHROPIC_API_KEY python3 enrich_analyses.py --backend <winner> --sampling-plan --video <clip>
```

## Acceptance criteria
A winning engine must beat `baseline` on within-video Recall@1, coverage (≫7%, near
100% of states), tag entropy, and text-F1 ≥ EAST/OCR — **with** `json_adherence`
high (≥~0.95) and low non-compliance, and the judge audit corroborating (or an
investigated disagreement). Pick a **production engine** (likely a 7B/8B for the
cost curve) and optionally an **escalation engine** (32B and/or InternVL) for hard
OCR / long-tail frames.
