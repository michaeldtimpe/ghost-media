# Vision-Engine Bake-Off — Test Plan

Revised after reviewer feedback. The harness lives in `bench/`; this doc is the
execution plan: what we test, on what, how we judge, and in what order.

## Status / how to resume (updated 2026-05-28 — bake-off arc landed on main)

**TL;DR.** The vision bake-off, text-GT labeling, the full production rescan, **and**
the merge to `main` are all **done**. Landed as **PR #1** (merge commit `28ef789`,
6 commits preserving the bake-off → hybrid → MLX → rescan → lessons arc). Production
engine = **`mlx-qwen7b`** + **`mlx-internvl` on parse-failure** (V1 cascade). The
54-video corpus is fully enriched at **19,592/19,594 scenes with semantic (99.99%)** —
the 2 holdouts are end-of-file tail scenes ffmpeg can't seek to. The bench harness,
hybrids, text-F1 scoring, and CLIP embedding sidecars are all in sync with the
production dataset. Pre-MLX backups archived offsite at
`~/backups/ghost-media/pre-mlx-corpus-2026-05-28.tar.gz`.

A follow-up audio rigor pass landed shortly after as **PR #2** (`e430b60`) and
**PR #3** (`75ce526`); see `audio_field_audit.md` + `lessons.md` "# Audio side".
Notably, `assemble_v2.py` now reads `bpm_timeline.confidence` and `onsets.times_sec`
from the deep-analysis schema 2.1.0 — both fields were already produced by the
analyzer, the uplift just wired them into scoring.

### Environment (still current)
Dedicated `.venv` on **Python 3.12** at the repo root — invoke as `./.venv/bin/python`.
torch 2.12, mlx-vlm 0.5, open_clip 3.3; `requirements.lock` committed; `pyproject`
pinned `>=3.12,<3.14`. Ollama 0.24.0 with `num_ctx=8192` (the 128k default ballooned
qwen2.5vl:7b to 52 GB — see lessons.md). All 8 base engines + 9 hybrids healthy.

### Models staged + fingerprinted (`bench/results/model_fingerprints.json`)
- ollama: `qwen2.5vl:7b` (Q4_K_M), `qwen2.5vl:32b`, `minicpm-v:8b` (Q4_0)
- mlx (HF cache): `Qwen2.5-VL-7B-Instruct-4bit`, `Qwen2.5-VL-32B-Instruct-4bit`,
  `InternVL3-8B-MLX-4bit` (config's old `InternVL3-8B-4bit` 404s — **fixed**)
- kappa `/Volumes/home/hub` had **no VL engines** (only text MLX + dropped gemma-3)
- Embedding consistency: M1 plans reproduce on M5 at **cosine 1.000000** (281 states)

### Phase 4 bake-off results (M5 sequential + M1 InternVL parallel, merged)
```
engine            comp   R@1in  cover  jAdh  s/frm
clip-ceiling     0.650   1.000  1.00   1.00    —    (ceiling)
mlx-qwen7b       0.474   0.457  0.99   0.99    7.6  ← production winner
mlx-internvl     0.466   0.423  1.00   1.00    4.3* ← escalation winner (*M5 prompt-probe)
ollama-qwen7b    0.465   0.432  1.00   1.00   14.7
ollama-qwen32b   0.453   0.416  0.98   0.98   39.9
mlx-qwen32b      0.452   0.406  0.98   0.98   26.4
ollama-minicpm   0.436   0.387  0.95   0.95    5.3  ← DROPPED (snap 1.26/frame)
baseline         0.217   0.394  0.12   0.12    —    (floor)
```
- **The 32B engines do NOT beat the 7Bs** on any per-video R@1 subset. They produce
  ~30% longer descriptions for the same composite, parse-fail on 3 frames the 7Bs
  handle, and over-flag `has_english_text` systematically.
- **MLX is ~1.9× faster than Ollama** for the same Qwen 7B; same composite to within
  0.01. *Runtime is throughput, not quality* — at the aggregate; see lesson.
- **InternVL is the OCR/specialist** — highest adjacent-discriminability (0.31),
  perfect coverage + jAdh, zero hard errors. Different style vocabulary
  (`geometric`-heavy) than the Qwen variants.

### Phase 5 hybrid bake-off (`bench/hybrid.py` + 9 spec variants)
Best hybrid: **V2 (error-fallback to qwen32b) 0.479** edges base by **+0.005**. V1
(error-fallback to InternVL) and V5 (multi-trigger cascade) tie at 0.478. The user
hypothesis "7B + 32B for descriptions" (V7) **lost decisively**: composite 0.446,
**−0.028 vs base**. Verbose descriptions are anti-discriminative. See lessons.md.

The +0.005 win came entirely from rescuing 3 parse-fail frames (coverage 0.99→1.00 +
their R@1 contribution). Multi-trigger cascades (snap≥2, style=unclear, desc<150)
added no composite value beyond the error trigger — pick the simpler rule.

### Decision (recorded)
- **Production: `mlx-qwen7b`** — best calibration, conservative on text (under-flags
  vs the 32B faction), 93% snap-clean, 7.6 s/frm.
- **Escalation: `mlx-internvl` on parse-failure only** — V1 spec
  (`bench/hybrids/V1-err-internvl.spec.json`). Recommended over V2's +0.001 because
  InternVL has 100% jAdh + 0 errors while qwen32b has its own class-specific parse
  failures.
- **Drop**: `ollama-minicpm` (3/4 of frames need schema snapping), both **32B** engines
  (no composite gain, slower, over-flag text, model-class parse failures).

### Done (2026-05-26 → 2026-05-28)
1. ~~Wire MLX into `enrich_analyses.py`~~ — already in place from prior work;
   `bench/mlx_backend.py` is a thin re-export.
2. ~~Pilot rescan~~ — EXC3_CM3 ran end-to-end: 39/39 scenes, 0 errors, V1 cascade
   rescued 3 parse-fails. Confirmed cascade behaves as bake-off predicted.
3. ~~Text ground-truth~~ — 162 labels in `bench/groundtruth/text_labels.json`
   (38 pos / 124 neg). Auto-generated by Claude vision in lieu of interactive labeling;
   user reviewed 12 borderline cases and accepted. Spot-check: descriptions of cards
   and stylized typography are conservatively labelled (e.g. "scattered Latin letters"
   borderline calls). text-F1 across engines after re-score:
   ```
   engine          composite  text-F1
   mlx-qwen7b      0.621      0.73   ← production
   ollama-qwen7b   0.621      0.78
   mlx-internvl    0.602      0.68   ← under-flags has_english_text vs Qwens
   baseline (EAST) 0.292      0.37
   ```
   InternVL's "OCR specialist" reputation is about character-level recognition, not
   the binary `has_english_text` flag — it actually *under-flags* on stylized text.
4. ~~Full-corpus rescan~~ — **54 videos, 18,397 plan states, 19,594 scenes,
   19,591 with semantic (100%)**. Backend split: 16,956 Qwen 7B + 1,384 InternVL
   rescues (7.5% cascade ratio). Wall time **~37 h on M5 alone** (M1 attempted but
   net-negative; neo can't run the model; see lessons.md). The original 14h estimate
   assumed the bench's default 300-state-per-video sampling cap; uncapping to capture
   the full richness of long videos (e.g., 7 Hours Visual: 4,969 plan states vs the
   ~300 it would have been capped at) pushes runtime ~3× but yields a **38× richer
   dataset** than the prior pipeline (~513 scenes total).
5. **Optional: judge audit** (deferred — touches `ANTHROPIC_API_KEY`):
   `env -u ANTHROPIC_API_KEY ./.venv/bin/python bench_run.py run --engines
   mlx-qwen7b,mlx-internvl,mlx-qwen32b --judge`. The one experiment that could
   vindicate the 32B on caption-richness grounds CLIP can't see.
6. **Optional: composite re-weighting** post-GT — adjacent-discriminability 0.15→0.20
   / R@1in 0.40→0.35 promotes InternVL. Cheap via `score` (no engines re-run).

### Hybrid harness reference (for future variants)
- Specs: `bench/hybrids/*.spec.json` (declarative — author once, run anywhere).
- Build: `./.venv/bin/python bench_run.py hybrid build` synthesizes raw from existing engine outputs.
- Score: `./.venv/bin/python bench_run.py score` picks them up unchanged.
- Real cost probe: `./.venv/bin/python bench_run.py hybrid run-cost --spec NAME --n 30` (pre/post-flight ollama unload baked in).

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
