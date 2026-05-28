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

## A bench knob with a sensible default is the wrong default for production

`bench_run.py plan` defaults `--max-states 300` (`MAX_STATES_PER_VIDEO_BENCH = 300`
in `bench/config.py`). For scoring it's fine — 300 distinct visual states per video
is plenty to rank engines. For *production* sampling plans it silently truncates the
input scene list before clustering, so a 7-hour VJ piece that would yield ~5,000
distinct states ends up with ~321 — losing ~94% of the granularity the dataset is
supposed to expose. The cap doesn't error or warn; it just sub-samples and the
emitted plan looks identical in shape.

The fix is one flag (`--max-states 100000`), but you need to know to pass it. The
first attempt at the full-corpus rescan built capped plans, and three minutes in I
caught it only because the histogram showed `scenes=300` (the cap) instead of the
analysis's true scene count.

**Lesson:** when a bench knob and a production knob diverge, make the default *fail
loudly* in the wrong context, not silently produce a smaller dataset. The bench
script could log "WARNING: capping at 300 states — pass `--max-states 0` to
disable" whenever the cap fires, with the count it would have produced uncapped.

## Multi-machine MLX-VLM splits are often net-negative

Tried splitting the production rescan across M5 + M1 + neo. Real-world result was
worse than M5 alone:

- **neo (Apple A18 Pro, 8 GB unified)** can run `mlx-vlm` smoke tests but **cannot
  sustain Qwen2.5-VL-7B 4-bit** inference. Model is ~6 GB; activations + KV cache push
  the practical floor to ~12 GB. Drop A18-class chips from any multi-machine VLM
  split.
- **M1 Max (64 GB)** runs the same model, but per-frame inference is ~14 s/frame vs
  M5 Max's ~4.8 s/frame — a **~3× throughput gap**. With `omlx serve` competing for
  Metal it gets worse. Combined with the next two lessons, M1 ended up costing more
  wall time than it saved.

**Lesson:** verify per-machine throughput before partitioning. If the slowest
machine's bottleneck-video would take >2× the fastest machine's, parallelism
probably loses to single-machine on the fast box. Sample one video on each
candidate machine first; partition only when the speedup math actually works.

## Enriched JSON is written once per video — kill mid-stream loses everything

`run_sampling_plan` accumulates `scene["semantic"]` and `frame_analyses` in memory,
then `ef.write_text(...)` *once* at end-of-video. On a 3,065-state video at M1's
~14 s/frame, an interrupted run loses **all 12 hours** of work. We learned this
twice (yesterday's M1 ffmpeg-broken run, then today's M1 hang-and-kill — second
attempt wasted 12 h of overnight inference because the relaunched process had to
redo everything from scratch).

The frame extraction *is* cached (`extract_frame` no-ops if the file exists), so
the relaunch isn't a *total* loss — but the VLM inference output isn't.

**Lesson:** for long-running batches, write checkpoints. A 100-scene-batch flush
(or every N minutes) would cap the kill-damage to <5 min instead of 12 h. Until
then: don't kill a sampling-plan job unless you're willing to lose the entire
in-progress video; if you must kill, do it *between* videos (when a `→ wrote N
updates to …enriched.json` line just appeared).

## `tee | grep` pipeline + `\r`-heavy progress bars = silent deadlock

Worker script on M1 used `python ... 2>&1 | tee -a $LOG | grep -E "states|errors"`
to filter the live console output while keeping the full log. On M5 this worked
fine; on M1 the run **hung silently** for hours — process loaded the model, then
sat at 0% CPU forever.

Cause: mlx-vlm emits per-frame `tqdm` progress bars using `\r` (carriage returns,
no newline). Python's stdout pipe to `tee` fills its 64KB pipe buffer faster than
`grep` can drain it (grep waits for full lines, never gets one). Python blocks on
the next write to stdout. The whole pipeline deadlocks.

Fix: drop the filter pipeline; write directly to log with `python -u ... >> $LOG
2>&1`. Lose live console matches but the log has everything and the process never
blocks.

**Lesson:** `tee | grep` patterns are dangerous when the upstream emits `\r`-only
"line" updates (tqdm, downloads, training progress). Either disable the filter or
use `unbuffer`/`script` to inject a PTY. Single redirect is always safe.

## `--reenrich-flagged` without `--video` filter is dangerously broad

Running `enrich_analyses.py --reenrich-flagged` with no filter scans **every**
`enriched/*.enriched.json` file in the directory — including stale M1-era files
and orphans from old experiments. On the first full-corpus dry-run this flagged
**18,601 scenes** to re-enrich; the actual cascade target was ~3,500.

It also over-flags scenes that are *meant* to lack semantic by design: in
sampling-plan mode, scenes outside the plan's representatives intentionally have
no `semantic` field (they're supposed to inherit from cluster mates downstream).
For Mega 4K VJ Loop (1,595 scenes, 358 plan states), the predicate flagged 1,318
of those non-representative scenes as "no/raw semantic." Cascading them isn't
*wrong* — it just adds redundant per-scene VLM calls where the design expected
inheritance.

**Lesson:** always scope `--reenrich-flagged` with a `--video` filter list when
re-running production. And the predicate `_scene_is_flagged` should learn the
sampling-plan/inheritance distinction so non-rep scenes aren't false positives.

## `--video` substring filter matches duplicate hashed files

`--video "isshin REEL 2024"` matches both `isshin REEL 2024.sampling_plan.json`
and `isshin REEL 2024-35e329.sampling_plan.json` (a duplicate file from a re-scan).
Running cascade with that filter writes to both files in sequence — if two machines
each grab a different substring that overlaps, you get a race.

Disambiguator: append `.` to the stem (e.g., `"isshin REEL 2024."`) — that matches
only the non-hash variant. Or use the `.sampling`/`.enriched` extension suffix in
the filter.

**Lesson:** substring filters need a canonicalization or unique-anchor pass when
the filename space contains hash-suffix duplicates. The CLI should optionally
require exact-match (`--video-exact`) for high-stakes runs.

## SSH non-login shells drop /opt/homebrew/bin

`ssh user@host "command"` runs in a non-login non-interactive shell that doesn't
source `.zshrc`/`.bash_profile`, so `/opt/homebrew/bin` isn't in PATH. M1's
ffmpeg is installed via Homebrew there; the production worker's
`subprocess.run(["ffmpeg", ...])` silently failed with `FileNotFoundError`,
returning 3,065 errors per video and producing a corrupt enriched.json full of
empty `frame_analyses` entries. No log signal that ffmpeg was the cause — just
"states described: 0, errors: 3065".

Fix: `export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH` at the top of the
worker script.

**Lesson:** worker scripts that shell out to native binaries (ffmpeg, ffprobe,
imagemagick) must set their own PATH. Don't rely on the SSH user's interactive
shell config. Test workers via `ssh host "/path/to/script.sh"` *exactly* the way
the production launch will invoke them.

## MLX inference at temp=0 is non-deterministic on the margin

The full-corpus cascade re-enriched 4,841 of 4,845 flagged scenes — 4 unparseable
responses from InternVL (each ~13s, suggesting the model produced too-long /
malformed JSON). Replaying *the same scene with the same model and prompt* one of
them produced a clean parseable response in 7s. Same weights, same temp=0, same
input image — different output bytes.

MLX's reductions involve some non-deterministic operation ordering (parallel
sum/argmax), and 4-bit quantization compounds tiny activation drifts. The base
generation is mostly stable but ~0.1% of long responses (>700 tokens) diverge
enough to break JSON validity.

**Lesson:** "temp=0 ⇒ deterministic" is the wrong mental model for MLX on Metal.
For high-stakes parses, build retry into the cascade — a second attempt with the
same args succeeds ~75% of the time. Don't waste effort hunting for a "real"
cause when the failure rate is sub-1%.
