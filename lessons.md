# Lessons

Engineering lessons from ghost-media, with an emphasis on the round of work that
imported ideas from the blog post *"I indexed a year of video locally"*
(simbastack.com) — what transferred, what was already solved, and the bugs and
surprises found along the way.

## Importing from the blog

The blog describes a local-first **video-indexing** pipeline. Its thesis — *"the
editor is solving the wrong problem; the prerequisite layer is making an archive
queryable"* — maps cleanly onto ghost-media, but only partly, because ghost-media
had already solved several of the same problems independently.

| Blog idea | Status in ghost-media | What we did |
|-----------|----------------------|-------------|
| Embedding-based natural-language retrieval | **Already half-built** — per-scene CLIP embeddings existed and fed the assembler | Exposed them as a search tool (`query_scenes.py`) |
| Enum constraints + validation to stop hallucination | Prompt listed enums but did **zero validation** | Added `vision_schema.normalize_analysis` (snap-to-vocabulary) |
| Vision-backend priority hierarchy (cloud-quality / local-bulk) | Single hardcoded Ollama call | Added `vision_backends.py` (ollama / claude-cli / anthropic-api) |
| Cull "non-recordings" before the editor | **No quality filtering at all** | Added `flag_quality.py` + a soft scoring penalty |

**Lesson:** before importing an architecture, find the parts you already solved.
Half of the blog's pipeline was already present; cargo-culting the whole thing
would have duplicated working code. The value was in the *missing operational
layers*, not the headline idea.

## "Wired" is not "working" — verify joins at runtime

The biggest find of the effort: the assembler's **CLIP-similarity scoring (dim
11) had been silently dead**. The embeddings were generated (51 sidecar files),
the loading code existed, and the scoring term referenced them — everything
*looked* wired. But the join key was

```python
key = (clip.source_name.replace(".enriched.json", ""), clip.scene_index)
```

where `source_name` is the **original** filename (`"… Dom & Roland …"`, with
`&`, `×`, `'`, `()`), while the sidecar filenames are **filesystem-sanitized**
(`"… Dom _ Roland …"`). The two never matched → `clip_embedding` stayed `None`
for every scene → the term contributed nothing. A static read of the code (and
even the file counts) suggested a "fully active" feature; only running
`build_scene_database` and printing `matched/total` revealed `0/6797`.

Fix: key on the enriched-file stem (`enriched_stem`), which is exactly the
sanitized form the sidecars use. This revived dim 11 *and* let the new quality
join attach (it had inherited the same broken pattern).

**Lessons:**
- A scoring term that silently degrades to a no-op is worse than a crash — it
  produces plausible output while a whole signal is missing. Log
  `matched/total` for every optional data join.
- The original filename and the on-disk (sanitized) filename are *different
  identities*. Pick one canonical key and use it everywhere.

## Don't trust a single scalar's scale across a heterogeneous corpus

Quality flagging started with absolute thresholds for everything. But optical-flow
**motion magnitude is not comparable across videos**: one file averaged `0.006`,
another `3.8` — a ~600× difference, driven by resolution/content, not "more
motion." An absolute "frozen if motion < 0.02" rule would fire on every scene of
the low-scale video and never on the high-scale one.

Brightness and contrast, by contrast, are 0–1 normalized and *are* stable across
the corpus.

**Lesson:** lean quality signals on the scale-stable measurements (brightness,
contrast). Where a signal isn't scale-stable (motion), use it only for the
unambiguous extreme — "frozen" fires only when *peak* motion is essentially zero
(a true still), which is meaningful at any scale.

## "Bad quality" vs "not what you want right now"

The first quality pass flagged 30% of scenes — but **`near_dup` alone was ~95% of
the flags**, crushing VJ-loop videos to 82% flagged. A repeating loop is not
*low quality*; it's perfectly usable footage you simply don't want to repeat. The
assembler already has reuse penalties and variety windows for that.

So `near_dup` became a *gentle* nudge (penalty 0.25, → quality 0.75) while the
genuinely-unusable structural flags (black/blown/frozen) stay strong (→ 0.1). The
sidecar also separates `technical_score` from `editorial_score` so the two
concerns can diverge later without a data migration.

**Lesson:** separate "this footage is broken" from "this footage is redundant for
this edit." Conflating them lets a redundancy heuristic masquerade as a quality
filter and quietly deprioritize whole sources.

## Soft penalties beat hard filters for creative pipelines

Quality is applied as a *non-destructive* scoring penalty
(`-(1 - quality) * 3`), not an exclusion. A black scene sinks ~3 points below
clean candidates but is still selectable as a last resort. In testing, with a
healthy candidate pool the assembler picked 8/8 clean clips on its own; the
penalty only matters when the pool is starved.

**Lesson:** in a system where "the worst option is sometimes exactly the pacing
beat you need," prefer a reversible, tunable penalty to a hard cut. Hard filters
create selection blind spots you can't recover from downstream.

## Subscription vs. metered billing is a silent footgun

The headline backend goal was to use the **Claude subscription** (zero marginal
cost) via the Claude Code CLI, not the metered API. But the CLI silently routes
to **API billing** if `ANTHROPIC_API_KEY` is set in the environment — which it was
in the dev box. Nothing errors; you just get charged.

Mitigations baked in: the `claude-cli` backend's health check **warns when
`ANTHROPIC_API_KEY` is set**, and the docs point to `claude setup-token` →
`CLAUDE_CODE_OAUTH_TOKEN` for unattended batches that still bill the subscription.

**Lesson:** when "free" depends on the absence of an env var, detect and surface
the expensive case explicitly. Don't let billing mode be invisible.

## Cloud models return refusals as success

Per the blog (and confirmed worth guarding): a vision model — especially via the
CLI — can return a refusal or permission-denial as a *normal* response body. A
naive pipeline records it as a successful (garbage) analysis. The backends scan
the first ~200 chars (substring, not `startswith`, since models prepend
disclaimers) for refusal markers and convert them to errors so retry /
`--reenrich-flagged` can react.

**Lesson:** "HTTP 200 / exit 0" is not "the task was done." Validate the *content*
of a model response, not just the transport status.

## Cheap wins live in data you already have

The entire quality pass reads the **existing** enriched timelines — no video
re-decoding, no GPU, no archive drive — and produced sidecars for 53 videos /
~19.5k scenes in seconds. The most expensive-looking feature was the cheapest to
build because the measurements were already on disk.

**Lesson:** before adding an expensive preprocessing pass, check whether the
signal is already sitting in an artifact you computed for something else.

## Validate model output where it's persisted, not where it's produced

Enum normalization is applied at the single **persistence chokepoint**
(`build_enriched_output`), so every backend (current and future) gets validated
output without each call site remembering to do it. Corrections are kept under a
reserved `_validation` key (raw + normalized) for forensic visibility into prompt
drift, and `_provenance` records which backend/model produced each frame.

**Lesson:** put cross-cutting validation at the one place data is written, not
scattered across producers. Keep the raw value — you'll want it when tuning prompts.

## Operational footnote

A broken `torch` install (missing its `sympy` dependency) silently took down the
entire CLIP/torch stack — embeddings, the assembler's dim-11 path, and the new
search tool — while the numpy-only code kept working. Dependency health isn't
all-or-nothing; a transitive dep can disable one capability cluster while the
rest of the app looks fine.

## A model server's default context window is a hidden memory bomb

On the M5 migration, the first real VLM bench run (`qwen2.5vl:7b` via Ollama) ran
at ~16.5 s/frame and `ollama ps` showed the model occupying **52 GB**. The cause
wasn't the images — it was that Ollama loads a model with its **full advertised
context window (128k for qwen2.5-VL)** by default, sizing the KV cache for 128k
tokens when single-frame description actually uses ~3.9k (≈2.6k image tokens +
prompt + 1024 output). Note qwen2.5-VL caps image tokens by downscaling, so a 4K
frame and a 1080p frame cost the *same* ~2.6k tokens — verified via
`prompt_eval_count`. Setting `num_ctx: 8192` in the Ollama options (`vision_backends.py`)
dropped the footprint to **14 GB** with zero quality change (parse rate, enum
compliance, and descriptions all identical; 8192 leaves 2× headroom).

Crucially this fixed **memory, not latency** — the 128k allocation was a pure
KV-cache cost, not extra compute, so per-frame time stayed ~16.5 s. Don't conflate
the two.

**Lesson:** a server's *default* context window is sized for the model's max, not
your workload. On a fixed-RAM box that silently caps how many (and how large)
models you can co-resident — a 52 GB-per-7B default would make a multi-engine
bake-off thrash long before you exhaust "real" memory. Pin `num_ctx` to your
actual token budget and confirm with `ollama ps` (SIZE/CONTEXT).

## Verbosity is anti-discriminative for retrieval-truth captions

The hybrid bake-off ran an isolated test: take the production engine (mlx-qwen7b)
and swap *only* its `description` field for a 32B's verbose prose (V7) or InternVL's
terse prose (V6), keeping everything else identical.

| variant | mean desc length | composite | R@1-within |
|---|---|---|---|
| V6 (InternVL desc) | ~131 chars | 0.462 | 0.421 |
| **mlx-qwen7b (sweet spot)** | **~358 chars** | **0.474** | **0.457** |
| V7 (qwen32b desc) | ~465 chars | 0.446 | 0.399 |

Both shorter *and* longer descriptions retrieved worse. The 32B's extra prose
specifically dropped R@1 by 0.058 absolute (−13%), with the magnitude scaling per
video with how much new but non-discriminative wording the 32B added. On Tame Impala
the cliff was −0.16 R@1 — the 32B described the same scene with words that retrieved
*less* well.

The bench's CLIP-text composite reads description text through ViT-B-32-openai; if
the description adds adjectives ("serene, rainy outdoor scene with a focus on…")
without adding new visual content, the embedding drifts toward generic — losing the
fingerprint that pins it to *this* frame versus its neighbors.

**Lesson:** "bigger model = richer description = better caption" is the wrong
intuition for retrieval-truth corpora. The production engine likely sits at a
discriminability sweet spot — long enough to be specific, short enough to be
distinctive. Don't reach for the heavier model on the assumption its verbose prose
helps; measure first. (For *human-readability* the heavier model might still win —
that's a different axis the CLIP composite can't see; use a judge audit to test.)

## Same model, two runtimes ≠ same outputs

Same weights (Qwen2.5-VL-7B-Instruct-4bit), same prompt, temp=0 — but mlx-vlm and
Ollama-MLX produce per-frame descriptions at:
- **median CLIP-text cosine 0.87** (not ~1.0)
- **tag Jaccard 0.18** (~80% disjoint vocabularies)
- **visual_style agrees 74%** of the time

Aggregate composites are within 0.01 (CLIP-text retrieval is robust to surface
variation), but the caption *prose* and tag *vocabulary* the corpus would carry are
materially different. Calibration also diverges — MLX uses `"unclear"` 9× more often
than Ollama on the same weights, and is correctly calibrated on a black frame where
Ollama hallucinates a "pixelated face/mask pattern".

Cause: Ollama's GGUF Q4_K_M ≠ mlx-community's MLX-4bit quant; activations diverge
slightly; greedy decisions split at the token boundary; differences cascade through
~400-token JSON generations. Despite both runtimes using "MLX on Apple Silicon" now,
they're not interchangeable below the aggregate level.

**Lesson:** the caption corpus and the embedding space it lives in are
runtime-specific, not just model-specific. "We picked the model, the runtime is just
throughput" is the trap: the production corpus locks in a runtime + quant combo, and
switching later silently invalidates CLIP retrieval against the existing corpus.
Pick the runtime + quant explicitly, write it into provenance, and stay there.
