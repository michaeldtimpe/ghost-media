# Vision-Engine Bake-Off — Test Plan

Revised after reviewer feedback. The harness lives in `bench/`; this doc is the
execution plan: what we test, on what, how we judge, and in what order.

## Status / how to resume (updated 2026-05-21)

**Harness:** built + verified end-to-end. `python3 smoke_test.py` passes (incl. 15
bench checks). Synthetic floor/ceiling render via `bench_run.py`. On the 6-clip pilot
the **current system (`baseline`) covers only 7% of distinct states** (R@1-within
0.33); the adaptive sampler yields **281 distinct states across 6 clips** (vs ~513 for
the *entire* old corpus). Plans are built (`enriched/*.sampling_plan.json`).

**Footage:** 6 clips staged locally in `raw_footage/` (gitignored): EXC3_CM3, PAPERS,
Tame Impala, show me lyrics, isshin REEL 2024, Disengaging. The bench prefers these
over the archive drive.

**Models staged (no fresh download needed for these two):**
- `ollama-qwen7b` → **pulled** (`qwen2.5vl:7b`, 6 GB).
- `mlx-gemma3` → **copied** from the local archive `/Volumes/home/hub`
  (`mlx-community/gemma-3-27b-it-4bit`, 16 GB) into `~/.cache/huggingface/hub`.
  `mlx-vlm 0.5.0` installed. (Archive note: `/Volumes/home/hub` is a HF cache full of
  *text* MLX models; **gemma-3-27b is the only vision model there** — Qwen2.5-VL,
  InternVL, MiniCPM-V are NOT in it and must be downloaded.)

**NOT yet done — resume here:**
1. **No real engine has been run yet** (the box had other models loaded; we deliberately
   held off to avoid overloading it). Resume with ONE clip on ONE engine:
   `python3 bench_run.py run --engines ollama-qwen7b --videos EXC3_CM3` → eyeball
   descriptions + `has_english_text` via `bench_run.py compare --video EXC3_CM3`.
2. Then the full pilot with the staged engines (`ollama-qwen7b,mlx-gemma3,baseline,clip-ceiling`),
   then `compare`.
3. Build text ground-truth (`bench_label.py --n 250`) for the text P/R metric.
4. Acquire remaining engines (not in archive): `ollama pull minicpm-v` (verify 2.6 vs
   4.5 tag); MLX `Qwen2.5-VL-7B/32B-4bit` + `InternVL3-…-4bit` (verify exact HF ids).
   Defer the 32B engines to the M5 Max (~early Jun 2026 — see memory `m5-max-return`).

**Gotchas:** `ANTHROPIC_API_KEY` was inadvertently printed to a transcript — **rotate
it**. For the `claude-cli` engine/judge use `env -u ANTHROPIC_API_KEY …` to bill the
subscription (the bench hard-blocks otherwise).

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
