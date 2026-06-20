# Lessons

## Offsite rollback artefacts (not in repo)

Two arcs produced sizeable comparative datasets we kept as research assets
rather than rollback insurance:

- `~/backups/ghost-media/pre-mlx-corpus-2026-05-28.tar.gz` (5.0 MB compressed,
  53 files) — pre-MLX `.enriched.pre-mlx.json` from before the full-corpus
  vision rescan. Pair (old-captions × new-captions × same-scenes) is useful
  for retrieval-drift studies, caption-style A/B, future judge benchmarks.
  Reversal cost if this tarball is also lost: **~37 h on M5** to re-run the
  rescan from scratch (full-corpus, 18,397 plan states, 7.5% InternVL
  cascade ratio).

- `~/backups/ghost-media/{set}.pre-audio-uplift.deep-analysis.json` × 5 sets
  (~37 MB each) — pre-2.1.0 deep-analysis JSONs from before the rigor uplift.
  Has the full `chroma_timeline`, `multiband` mids, spectral extras, per-beat
  `beats.features`, and `onsets.strength_envelope` that the 2.1.0 schema
  drops. Reversal cost: ~5 min/set to re-run `analyze_dj_set_deep.py` on the
  original audio (audio still on archive drive).


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

*Recurrence (2026-06, hungry-ghost import):* the same class bit again through a
different knob — `bench_run.py plan` without `--videos` silently scopes to the
**6-video PILOT set** (`pilot_registry` falls back to `PILOT_VIDEOS`). The import
chain "built sampling plans" in 3 seconds, enrichment found nothing new to do, and
the chain marched on green. Caught because the stage timestamps were implausible,
not because anything failed. `plan` now prints a loud warning when run without
`--videos` (pass `--videos .` for the full corpus). Two instances make the
pattern: audit every bench CLI default before reusing it in a production runbook,
and treat suspiciously fast stages as failures until proven otherwise.

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

# Audio side

The audio rigor uplift (Phases 0–6) closed the obvious gaps between what
`analyze_dj_set_deep.py` computes and what `assemble_v2.py` consumes. Field-level
contract is documented in [`audio_field_audit.md`](audio_field_audit.md); the
lessons below are the operational tax we paid getting there.

## Validate a zero-shot model with a discrimination probe before trusting tags

`laion/larger_clap_music` — the music-specialized CLAP checkpoint, the obvious
pick for `enrich_audio.py` — is broken in its HuggingFace conversion: every
text embeds to cos ≈ 0.999 of every other text, and audio–text cosines sit at
≈ 0 for all pairs. Weights load with zero missing/unexpected keys, so nothing
warns; the tag output *looks* plausible (top-k always returns something) while
being uniform noise. Verified on transformers 4.57 **and** 5.9 — it's the
conversion, not the library version. `laion/clap-htsat-unfused` discriminates
correctly (a sine tone scores "a pure sine tone beep" at 0.67 vs 0.04 for
"techno music") and is what `enrich_audio.py` ships with.

**Lesson:** a zero-shot tagger that ranks a fixed vocabulary always produces
confident-looking output, even fed garbage embeddings. Before trusting any new
checkpoint, run a 30-second synthetic probe — white noise vs a sine tone
against matching text labels — and check the text–text cosine matrix isn't
degenerate. Top-k order alone tells you nothing.

## Persisted signals should justify their storage cost even when unwired

Pre-uplift, `.deep-analysis.json` carried 14 top-level keys, several with
35,831-sample timelines × multiple fields per sample. Of those, `assemble_v2.py`
read four (`total_rms`, `sub_bass`+`bass`, `harmonic`+`percussive`, `centroid_hz`,
`bpm`). The rest sat in the file producing no scoring or downstream behavior.

The fix wasn't "wire everything" — most weren't worth wiring. The fix was to
split each field three ways: actively consumed, strategic latent (kept because
key/transition-aware sequencing is a credible future), or unjustified
accumulation (dropped from JSON; compute retained in-process for any
internal aggregate).

The audit produced a 65% file-size reduction (37 MB → 13 MB on a 70-min set)
without touching any compute path — pure persistence pruning.

**Lesson:** unwired output is not free. A field being computed but unread costs
storage and parse time, but more importantly it lies about the system's
intended behavior. Decide per-field: read it, keep it for a named future, or
drop it.

## Dead JSON fields create false documentation

The strongest argument for the audit doc as a permanent artifact (not throwaway
scaffolding): any future developer who reads `compute_chromagram` and sees its
output assigned into `result["chroma_timeline"]` will reasonably assume some
downstream tool reads it. The 37,000-sample chroma timeline was producing
exactly zero scoring decisions, but the code shape said otherwise.

This is the inverse of "Wired is not working" (line 27): there, a feature
appears active but isn't; here, a feature appears intentional but isn't. The
mitigation is the same — write down the contract somewhere it can be read
without running the code.

**Lesson:** the contract of a JSON output is what code reads from it, not what
the producer writes into it. Document the read side in a discoverable place
(here: `audio_field_audit.md`) so writes don't masquerade as semantics.

## Librosa beat tracking ships unvalidated

`librosa.beat.beat_track` returns beat times. It does not return any indication
of confidence, tracking stability, or whether it locked octave. Octave-doublings
(2× or 0.5× the true tempo), mid-set tracker failures, and grid drift all ship
silently to downstream consumers unless validated externally.

The new `beat_quality` block emits three observability metrics: IOI outlier
rate (±15% fractional tolerance, robust against the outliers themselves
inflating np.std), octave-doubling rate plus max consecutive-window run length
(a 3-window run = 24s of locked-wrong tempo is qualitatively worse than 3
scattered windows), and metronomic deviation. Non-blocking warnings; the
analyzer still completes.

**Lesson:** when a third-party library produces a primary signal with no
internal quality readout, build the readout yourself. Especially when the
signal feeds production downstream.

## Librosa's tempogram locks at half/double tempo during DJ transitions

The 5-set Phase 5b backfill caught this clearly: 3 of 5 sets had runs of 7–15
consecutive 8-second windows (≈56–120s of wall time) where the windowed BPM
ratio fell in `[0.45, 0.55]` or `[1.9, 2.1]` relative to `global_bpm`. Looking
at the source audio, these correspond to track transition windows where
`librosa.feature.tempo` briefly locks onto the incoming track's beat division
before settling. The `octave_doubling_run_max` warning fires at 3+ windows by
design — these are real lock events, not threshold noise.

The downstream impact is subtle: `bpm_timeline.bpm` values during those
windows are off by 2×, which Phase 2's `bpm_confidence` (computed from the
tempogram autocorrelation, not from BPM-stability) can't fix on its own.
Future work could either (a) post-process windowed BPM to median-filter
octave jumps, or (b) wire the locked-window mask into Phase 2 confidence so
clip selection inside transition zones inherits low trust.

**Lesson:** librosa beat/tempo APIs make per-window decisions independently;
they don't enforce continuity across windows. In long DJ-set audio with
multiple tempo regimes, expect this to surface at every transition. Detect
it post-hoc, don't trust raw `bpm_timeline` inside the runs.

*Update (2.2.0):* option (a)+(b) landed as `repair_bpm_timeline` — folded
windows are corrected against a 31-entry rolling median and keep
`confidence × 0.25`, so the assembler's existing dur-match weighting
distrusts them with no assembler change. Validation still runs on the raw
timeline first so `octave_doubling_pct` keeps measuring librosa's behavior.

## Downbeat estimation from energy heuristics doesn't converge on this corpus

The cut-timing uplift wanted phrase grids anchored at musical downbeats
(today's grid starts wherever librosa's first beat fell — an arbitrary 0–15
beat offset from the real phrase structure). Three candidate signals were
prototyped against all 5 sets at 16-beat phase granularity: |ΔRMS| arrival,
bass arrival (positive sub_bass+bass delta), and brilliance arrival. They
disagree with each other on most sets (e.g. boxing-day: best phase 14 vs 5
vs 9) and margins over the median phase are thin (0.04–0.33). Four-on-the-
floor material puts kick energy on *every* beat, starving simple per-beat
energy features of bar-phase information. Tracklist timestamps can't
arbitrate either — 1-second resolution vs a ~0.47s beat period.

So 2.2.0 persists `beats.downbeats_sec` + margins for observability, but
phrase anchoring is opt-in (`--anchor-phrases`) until a listen test
validates an anchor. A wrong anchor is worse than none: it shifts *every*
cut to a consistently wrong beat, which is exactly the artifact the uplift
was meant to remove.

**Lesson:** when a heuristic decides something global (one offset applied to
~550 cuts), require corroboration between independent signals before wiring
it in. Persist the estimate; gate the behavior.

## `metronomic_deviation_max_sec` is NOT "tracker drift"

The third beat-quality check measures the maximum absolute deviation between
observed beats and an extrapolated constant-tempo grid built from `global_bpm`.
On the smoke set this lit up at 44s — alarming if you read the name as "drift"
but accurate to what the math computes: this set has windowed BPM ranging
83–199 across 70 minutes, so a constant-tempo extrapolation built on the global
129 BPM is going to diverge by tens of seconds. That's healthy DJ-set tempo
variation, not librosa failing to track.

The original name in the plan was "phase drift"; reviewer rightly flagged that
the metric *conflates* local tracker quality with global metronomic adherence.
Renamed before merge.

**Lesson:** when a metric name implies a verdict ("drift", "error", "failure"),
the math has to actually mean that. If it doesn't, rename to match the math —
even at the cost of a less catchy label. Useful as one signal among several is
worth more than misleading as a verdict.

## `PhraseFeatures` scoring weights co-evolve with `select_clips`

The score-weight constants in `score_scene` (motion_match × 1.0, bright_match
× 0.8, dur_match × 0.9, sat × 0.5, contrast × 0.5, …) are not a balanced design
arrived at in one sitting — they reflect years of tuning against the
diversity windows in `select_clips` and the `MAX_SOURCE_USAGE_MULT` cap. New
scoring components that show up at weight 1.0+ will either swamp the rest of
the function or destabilize the diversity invariants the cap relies on.

The Phase 2 additions landed conservatively: `onset_match` at weight 0.4 (about
half of `dur_match`); `bpm_confidence` as a *multiplier* on `dur_match` with a
floor of 0.6 (so even at zero confidence the duration term contributes 60% of
its existing weight). Both are dataset-relative — percentile-rank within set
for onset density, per-set 95th-pct calibration for the confidence divisor —
because hardcoded scales silently break when the input distribution shifts
(see "Don't trust a single scalar's scale across a heterogeneous corpus",
line 56).

The post-uplift smoke run showed mean score 1.30 vs baseline 1.25 (+4%) and
source diversity broadened from a single-leader to a 3-way tie at top — both
healthy directions. A weight-1.0 onset_match would have produced different
results, almost certainly worse.

**Lesson:** when adding a new scoring component to a tuned pipeline, land at
<0.5 weight or as a multiplier on an existing term. Verify against a smoke set
with frozen seed; ratchet up only after observing the distribution shift.

# Selection side

## Raw CLIP cosines don't discriminate on a same-genre corpus — calibrate them

Wiring `.semantic.json` visual keywords into scoring (dim 13) as a plain
`max(0, cos) * 0.4` bonus churned 98% of selections (pure MMR path-cascade
noise from tiny uniform deltas) while moving actual semantic alignment by
+0.0001 — the dim was structurally present and functionally absent. Cause: on
a corpus that is *all* abstract electronic visuals, text→image cos for any
mood/texture phrase spans a band of a few hundredths, so a sub-1.0 weight on
the raw value can't outvote even the smallest motion/brightness differences.

The fix follows the codebase's own convention (bpm-confidence divisor,
onset/motion percentile ranks): per section, compute cos against the entire
clip-embedding corpus once, take p5/p95, and min-max the per-clip cos into
[0,1] before weighting. Same frozen-seed A/B: alignment +3.0%, distinct
sources unchanged, mean score flat. Conservative landing per the
`PhraseFeatures`-weights rule, now with an effect to ratchet.

Two corollaries. (1) Selection churn is not evidence of effect — MMR +
variety windows amplify any perturbation into widespread reshuffling, so
measure the dimension's *target* quantity (here: mean cos of chosen clips),
not how much the output changed. (2) Dim 11 (lyric→CLIP, weight 0.6,
uncalibrated) almost certainly shares this narrow-band weakness — candidate
for the same p5/p95 treatment behind its own A/B.

## Metadata diversity is not perceptual diversity

The assembler's `select_clips` enforced three diversity controls
(`VARIETY_WINDOW=15`, `SCENE_VARIETY_WINDOW=30`, `MAX_SOURCE_USAGE_MULT=2.0`)
all keyed off filename strings (`source_name`) and structural indexes
(`scene_index`). The console output bragged "40/40 sources used" while the
viewer perceived constant repeats. The audit (Phase 0 of the
perceptual-diversity uplift) made the gap concrete: 6 of the user-visible
"repeats" in the smoke video were the same source file appearing under two
filenames (`isshin REEL 2022.webm` ↔ `isshin REEL 2022-d557b.webm`, cosine
1.000) — different `source_name` strings, identical embeddings, so the
metadata-keyed diversity windows had no idea they were the same.

The CLIP embeddings sat in `SceneClip.clip_embedding` from sidecars and were
used exactly once in scoring — for matching scenes against the phrase's
lyric-text embedding. They were never used for scene-vs-scene similarity at
selection time.

**Lesson:** when "looks the same" is the property you care about, measure it
in the embedding space, not the filesystem. Filename uniqueness, scene-index
uniqueness, and source-rotation windows protect against indexing-level
repeats but not perceptual ones. The 12-component scoring function already
loaded the embeddings; selection was the missing consumer.

## Hash-suffix duplicate files are silent corpus poisoning

The audit found `isshin REEL 2022.webm` and `isshin REEL 2022-d557b.webm` as
two enriched files (each with its own sampling plan, text flags, quality
sidecar, CLIP embeddings) referencing visually-identical content. They
existed because re-runs of the analyzer renamed the output with a hash
suffix when the original was already present; both got committed to the
enrichment pipeline. Same for the 2024 reel. Every downstream consumer
treated them as independent sources.

The cost wasn't disk — it was the assembler picking the same scene from both
"sources" in close succession because each appeared eligible under
`VARIETY_WINDOW=15` (different `source_name`). Two corpus files producing six
back-to-back identical clips in one 41-minute render.

A scan for the pattern `*-XXXXXX.enriched.json` paired with a no-suffix
variant catches this. The dedup is mechanical: keep the non-hash filename,
delete the hash-suffix sidecars (enriched, embeddings, quality, sampling
plan, text flags). Offsite tarball as insurance.

**Lesson:** filename diversity is not source diversity. When the same
analyzer can write the same content under two filenames, downstream code
treats them as independent sources whether they are or not. Audit the corpus
periodically for hash-suffix-vs-base duplicate pairs; the lessons.md note
"`--video` substring filter matches duplicate hashed files" was the earlier
half of this same problem.

## MMR with windowed history fixes selection, but the candidate pool ceiling matters more than λ

Phase C ported the MMR re-ranker from `query_scenes.py:89–107` into
`select_clips`. First attempt: `MMR_LAMBDA=0.4`, pool of 30 (top-30 scored
candidates re-ranked by `score - λ × max_cosine_to_last_10_selected`). Result
on three test sets: `waiting-to-begin-2024` and `boxing-day-2025` improved,
but **`cheerleader-exodus-2025` regressed** on the primary metric (pairs
≥0.85 within window 5 went from 42 → 51).

The diagnostic log (`bench/mmr_diagnostics.log`, first 10 phrases per run)
showed why: by phrase 3, **29 of 30 candidates already had max_recent_cosine
> 0.7**. The pool was uniformly perceptually similar to recent picks —
because all the top-scoring candidates for any given phrase tend to *look
alike* (they all match the same energy + brightness + contrast targets). λ
can rearrange a homogeneous pool but can't conjure variety from it.

Widening the pool to 80 with λ=0.5 fixed cheerleader (51 → 36) without
regressing the other two sets. Pushing λ to 0.7 swung cheerleader back to
47 — too aggressive. The per-set tuning was stable enough that one global λ
worked, but only after the pool gave MMR room to find genuinely different
clips.

**Lesson:** with MMR over a scored candidate pool, the *width* of the pool
matters more than the diversity weight. A narrow pool ensures all
candidates are score-equivalents that may also be perceptual-equivalents. Widen
first, then tune λ. The diagnostic log of (raw_score, normalized_score,
max_recent_cosine, mmr_score) per candidate is essential for understanding
which lever to pull when one set responds differently from the others.

Per-phrase score normalization (min-max within the pool) is also mandatory:
without it, the raw-score range varies wildly per phrase and λ becomes
brittle. A phrase where candidates score 0.6–0.8 vs one where they score
1.5–4.0 produces a different effective penalty for the same λ. Normalize to
[0, 1] before applying the cosine penalty.

## "Sources used 40/40" is not the same as "40 distinct visual experiences"

The Phase 0 audit script computed close-pair-count metrics at multiple
thresholds (cosine ≥ 0.95, 0.90, 0.85, 0.80, 0.75) across multiple windows
(1, 3, 5, 10, 30 phrases). The point: the human eye registers "this looks
like the previous one" at far lower cosine than 1.0. Mean consecutive
cosine of 0.69 (baseline on `waiting-to-begin-2024`) means almost half the
selection has consecutive pairs at ≥ 0.70 — a visually obvious resemblance,
even though no scene is literally repeated.

Phase D's perceptual-diversity logger emits the same metrics during every
run (`compute_perceptual_diversity` in `assemble_v2.py`), so future tuning
has a concrete observable rather than relying on the "40/40 sources" tally
that bears no relation to what the eye sees.

**Lesson:** when the user complaint is perceptual, instrument the perceptual
dimension. Source-distribution stats answer a different question
("operational diversity") and can read healthy while the experience is
monotonous. Both numbers want to be in the assembler's standard log block,
not one substituting for the other.

# Render side

## The render layer must enforce the sync contract, or the scoring layer is decorative

v2.0 scored scenes against phrases with eleven weighted dimensions, then the
render layer quietly broke the only assumption that made any of it matter:
that clip i occupies exactly phrase i's span on the audio timeline. Two bugs
compounded: clips were truncated to scene length when the scene was shorter
than the phrase (median scene 3.8s vs median 4-bar phrase 7.0s — so *most*
clips), and every extraction padded +0.5s of unaccounted video. The concat
just butt-joined whatever lengths arrived. Cuts drifted off phrase
boundaries cumulatively; minutes in, the clip "matched" to a phrase was
playing against entirely different audio.

The v2.1 fix plans every clip in *frames anchored to the audio timeline*
(`end_frame = round((phrase.end_sec - timeline_start) * FPS)`), so per-clip
rounding cannot accumulate, and fills short scenes (slow-mo speedfit /
ping-pong / loop) instead of truncating. A failed extraction renders black
filler of the planned frame count rather than being dropped — a dropped
clip would shift every later cut. A post-extraction ffprobe pass verifies
`rendered frames == planned frames` per clip and prints net drift.

**Lesson:** in any "plan then render" pipeline, the renderer needs an
explicit, *verified* contract with the planner. The selection layer had four
diversity mechanisms and a diagnostic log; the render layer had zero
assertions — and that's exactly where the product-breaking bug lived for
two months. Instrument the boring layer.

## ffmpeg `-t` placement decides whether filters can change duration

`-t` after `-i` is an *output* option: it caps the encoded result. With a
`setpts=PTS/0.7` slow-mo (or a `reverse`+`concat` ping-pong) in the filter
chain, an output-side `-t {scene_duration}` silently truncates the stretched
result back to the original length — the filter runs, then its entire
purpose is cut off. The failure is invisible in exit codes; only frame
counting caught it. Input-side `-t` (before `-i`) bounds how much source is
*read*, which is what scene extraction wants.

Two adjacent gotchas from the same debugging session: (1) naive ping-pong
(`split → reverse → concat`) duplicates the seam frame — forward ends on
frame X, reverse begins on frame X — which reads as a stutter and gets
dropped (with a timestamp gap) by a later `concat -c copy`; trim the first
frame of the reversed half. (2) `-frames:v N` is the only duration control
that survives all of this — frame counts, never wall-clock seconds, are the
unit the sync contract is written in.

# Per-song re-architecture + electric-dreams integration (2026-06)

## DJ sets are songs, not one stream — make the song the unit of selection
The biggest coherence win wasn't a scoring tweak: it was changing the planner's
unit of organization from "whole 68-min set, one global clip pool" to
"per-song mini-video." `scripts/segment_songs.py` infers song boundaries from
the existing deep-analysis (key/BPM/energy/spectral novelty, snapped to 16-bar
phrases — no audio re-analysis) and writes `tracks[]`. `select_clips` then
resets variety/usage/MMR state at each song boundary and gives each song a
mostly-exclusive source pool. Result on waiting-to-begin: median uninterrupted
run 3.7s→7.5s, single-play 30%→74%, looping 70%→26%. Bias segmentation toward
FEWER/LONGER songs (`MIN_SONG_SEC=120`): over-segmentation chops identity,
under-segmentation degrades gracefully.

## `frames_emitted` must stay GLOBAL across songs
The frame-exact contract spans the whole render. The per-song refactor resets
everything EXCEPT `frames_emitted`/`timeline_start` — resetting those re-anchors
each song to frame 0 and desyncs the back half. The regression gate
(`GHOST_IGNORE_TRACKS=1` → byte-identical plan vs pre-refactor) is what proved
the wrap was clean before per-song logic was layered on.

## Phrase merges silently drop new PhraseFeatures fields
`merge_phrases_adaptive` and `adjust_cuts_for_vocals` reconstruct
`PhraseFeatures`, listing fields explicitly — so a newly-added field
(`track_index`) defaulted to -1 on every merged phrase, fragmenting 65 phrases
into a spurious "song." Fix: re-derive `track_index` from the preserved
`track_title` at the top of `select_clips` rather than trusting it survives the
merge passes. Lesson: new identity fields need re-derivation after any pass that
rebuilds the dataclass.

## Source-sequence runs preserve the original videos' long-form context
"Show clips in their original order" is a real continuity lever: once a
long-form source (high mean scene length) is chosen, continue with ITS next
scenes IN ORDER for a bounded run (`_next_in_sequence`, bypasses the variety
window). It's a selection-only change — each clip is still planned frame-exact.

## Cohesion vs MMR don't fight if they reference different sets
Within-song cohesion rewards similarity to a slow-moving SONG anchor; MMR
penalizes similarity to the last-N selected (fast, local). Different reference
sets → they coexist. The anchor must be mostly the STABLE pool centroid
(0.7·stable + 0.3·running), never a pure running centroid — a pure running
centroid self-reinforces and collapses a song to one subtype.

## A naive content-safety lexicon drowns in VJ vocabulary
Flagging unsafe scenes by scanning VLM descriptions works, but "graphic"
(motion graphics), "shooting" (light), "hanging" (objects), "blade" (light
blades), "blood-red" (a colour) collide with abstract footage — a naive lexicon
flagged 1,149 scenes (4%, mostly false). A high-precision lexicon (concrete
gore/weapon/sexual terms only) flagged 159 (0.5%) genuine ones.
`scripts/flag_content_safety.py` → `content_safety_flags.json` → assembler hard
excludes. No new VLM pass — mines the existing enrichment.

## Per-scene text feedback converges over renders, not in one
`check_render_text --update-flags` flags the CURRENT render's OCR hits, but the
next render (different seed/corpus) selects DIFFERENT text-bearing scenes, so
the total hit count doesn't drop in one pass. Caption-heavy sources needed a
hard exclude (`CAPTION_HEAVY_SOURCES` in `build_scene_database`), not just the
soft avoid_sources penalty, which leaked burned-in text.

## Concat-copy seam flashes: closed-GOP + scene-end guard
A "single frame of the next scene" at cuts has two causes: (1) B-frame
reordering across `concat -c copy` segment seams — fix with `-bf 0` +
`open-gop=0` so segments are independently decodable; (2) speedfit/loop reads to
a mis-detected scene end pulling the source's NEXT-scene frame — fix with a
~2-frame `SCENE_END_GUARD` on the read (tpad clones to refill).

## electric-dreams headless render — Electron gotchas (Part B)
Filtering footage through the ED VJ engine offline (`--render` in
electric-dreams) hit three Electron traps worth remembering:
(1) `allow-scripts` blocks Electron's postinstall → `dist/` is a stub; extract
the cached zip manually (`ditto -x -k ~/Library/Caches/electron/*/electron-*.zip
node_modules/electron/dist/`) and write `path.txt`.
(2) Electron's bundled Chromium has NO H.264/AAC decoder → the HTML5 video
underlay can only play VP8/VP9 WebM; transcode inputs to VP9 first, but mux the
ORIGINAL file's audio into the output so quality is untouched.
(3) Electron GUI apps detach stdout on macOS → `console.log` is lost; log to a
file. Also: `electron <path>` is relative to CWD — the built main is at
`apps/electron/out/main/index.js`, not `out/main/index.js`.
(4) Cost: `webContents.capturePage()` at retina (2560×1440) runs ~8× real-time.
The full 68-min render needs a smaller window + JPEG frames to be practical
(~9h → ~2h). Audio must stay real-time (Web Audio can't go fully offline), so
the capture loop is paced to the audio clock.
