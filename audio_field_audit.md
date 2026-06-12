# Audio field-usage audit (`.deep-analysis.json`)

Authoritative source of truth for what each field in `.deep-analysis.json` is for, who reads it, and whether it stays. This doc is referenced by `lessons.md`'s Audio section and gates Phase 4 of the audio rigor uplift.

## Schema 2.2.0 changes (cut-timing refinement, 2026-06)

- **`phrases[].end_sec` semantics changed**: now the start of the NEXT phrase (the beat after the chunk's last beat), so `[start_sec, end_sec)` spans the full musical phrase. Pre-2.2.0 it was the last beat's *timestamp*, undershooting by one beat — every assembler cut landed one beat early (cut-alignment audit: median cut→next-phrase gap = exactly one beat period on all 5 sets). The assembler's one-beat extension workaround is now gated on `schema_version < 2.2.0`.
- **`bpm_timeline` is repaired before persisting**: half/double-tempo locked windows (librosa octave locks at DJ transitions) are folded onto the rolling local median; folded windows keep `confidence × 0.25` so the assembler's dur-match weighting distrusts them. See `repair_bpm_timeline`.
- **`beat_quality.octave_corrected_windows` / `.octave_corrected_times_sec`**: count + locations of folded windows (validation still runs on the raw timeline first, so `octave_doubling_pct` keeps measuring librosa's behavior).
- **`beats.downbeats_sec` + estimator fields** (`downbeat_estimator`, `downbeat_bar_offset`, `downbeat_bar_margin`, `phrase_anchor_offset`, `phrase_anchor_candidate_16`, `phrase_anchor_margin`): HEURISTIC bass-arrival downbeat estimate, persisted for observability and the cut-alignment audit only. Candidate signals disagree on this corpus (lessons.md), so phrase anchoring stays opt-in (`--anchor-phrases`); `phrase_anchor_offset` records what was actually applied (0 unless opted in).

Reference set: `sets/arche august dj set rev 4.deep-analysis.json` (37.1 MB, 4159s duration, 35,831 timeline samples at 8 fps, 546 4-bar phrases). Field sizes are bytes for this set; ratios are stable across sets.

**Role legend:**
- `persisted+consumed` — written to JSON, read by an external consumer
- `persisted+orphan` — written to JSON, no current reader
- `persisted+aggregate` — written to JSON, only read internally by `analyze_tracks_deep` for derived per-track stats
- `intermediate` — used inside `analyze_dj_set_deep` but never written

**Recommendation legend:**
- `wire` — Phase 2 will add a reader
- `keep` — consumer exists or strategic latent value
- `drop-from-json` — Phase 4 target: remove from output dict, keep the compute path
- `drop-entirely` — Phase 4: no consumer, no aggregate; remove compute + output

---

## Top-level structure

| Field | Bytes | % of file | Role | Recommendation |
|---|---:|---:|---|---|
| `schema_version` | 7 | 0.0% | persisted+consumed | keep (bump to "2.1.0") |
| `analyzer` | 22 | 0.0% | persisted+consumed | keep |
| `file` | 200 | 0.0% | persisted+consumed | keep |
| `global` | 132 | 0.0% | mixed (see below) | keep parent |
| `tracks` | 38,643 | 0.1% | mixed (see below) | keep parent |
| `transitions` | 18,323 | 0.1% | persisted+orphan | keep (Phase 5b will validate; flagged as candidate for future wiring) |
| `beats` | 1,236,757 | **4.7%** | mixed (see below) | reshape |
| `bpm_timeline` | 118,686 | 0.4% | persisted+consumed | keep + wire `confidence` (Phase 2) |
| `multiband_energy` | 6,993,577 | **26.5%** | mixed (see below) | reshape |
| `hpss_timeline` | 3,234,869 | **12.2%** | persisted+consumed | keep |
| `onsets` | 1,769,185 | **6.7%** | mixed (see below) | reshape + wire `times_sec` (Phase 2) |
| `chroma_timeline` | 7,122,635 | **27.0%** | persisted+aggregate | **drop-from-json** (compute `chroma_dominant` per track inline) |
| `spectral_timeline` | 5,679,296 | **21.5%** | mixed (see below) | reshape |
| `key_timeline` | 38,595 | 0.1% | persisted+orphan | keep (deferred latent signal for future key-aware sequencing) |
| `phrases` | 166,361 | 0.6% | persisted+consumed | keep |
| `beat_quality` | — | new | persisted+consumed (will be) | **add** (Phase 3) |

## `global`

| Field | Role | Read by | Recommendation |
|---|---|---|---|
| `bpm` | persisted+consumed | `assemble_v2.py:454` (phrase-level), various | keep |
| `bpm_range` | persisted+orphan | — | keep |
| `beat_count` | persisted+orphan | — | keep |
| `key` | persisted+orphan | — | keep (will pair with key-aware sequencing) |
| `onset_count` | persisted+orphan | — | keep |
| `onset_rate_per_sec` | persisted+orphan | — | keep |

## `tracks[]` per-track

| Field | Role | Read by | Recommendation |
|---|---|---|---|
| `track_index`, `title`, `start_sec`, `end_sec`, `duration_sec` | persisted+consumed | `assemble_v2.py:457` (phrase track lookup) | keep |
| `beat_count`, `bpm` (dict), `energy` (dict) | persisted+orphan | — | keep (per-track stats reference) |
| `bands` (dict per band) | persisted+orphan | — | keep |
| `harmonic_percussive` (dict) | persisted+orphan | — | keep |
| `spectral` (dict — centroid/flatness/flux/brightness/texture) | persisted+orphan | — | keep |
| `chroma_dominant` | persisted+orphan | — | keep (derived; cheap to recompute inline) |

## `transitions[]` per-transition

| Field | Role | Recommendation |
|---|---|---|
| All ~17 fields per transition | persisted+orphan | keep (small footprint; latent for transition-aware cuts) |

## `beats`

| Field | Bytes | Role | Recommendation |
|---|---:|---|---|
| `beats.times_sec` | ~140 K | persisted+consumed | keep (already used by `validate_beat_grid` Phase 3) |
| `beats.features` (list of 9682 items, 7 fields each) | ~1.1 MB | persisted+aggregate (used only by `analyze_tracks_deep` line 363–366 for per-track aggregates) | **drop-from-json**, compute aggregates inline |

## `bpm_timeline[]`

| Field | Role | Recommendation |
|---|---|---|
| `time_sec`, `bpm` | persisted+consumed | keep |
| `confidence` | persisted+orphan → persisted+consumed (Phase 2) | **wire** in Phase 2 |

## `multiband_energy[]` — 35,831 items × 9 fields

| Field | Role | Recommendation |
|---|---|---|
| `time_sec`, `total_rms`, `sub_bass`, `bass` | persisted+consumed (`assemble_v2.py:433–435`) | keep |
| `low_mid`, `mid`, `high_mid` | persisted+orphan | **drop-from-json** |
| `presence`, `brilliance` | persisted+orphan | keep (latent shimmer signal candidate per reviewer 2) |

Estimated reduction: dropping 3 of 9 fields → ~33% smaller per item → ~8.8% of total file.

## `hpss_timeline[]` — 35,831 items × 4 fields

| Field | Role | Recommendation |
|---|---|---|
| `time_sec`, `harmonic`, `percussive` | persisted+consumed (`assemble_v2.py:448–449`) | keep |
| `hp_ratio` | persisted+aggregate (used by `analyze_transitions` line 469–470 only — windowed averages) | **drop-from-json**, recompute inline at transition time |

Reduction: ~25% of this 12.2% slice ≈ 3.0% of total.

## `onsets`

| Field | Bytes | Role | Recommendation |
|---|---:|---|---|
| `onsets.count` | 4 | persisted+orphan | keep |
| `onsets.times_sec` (9668 floats) | 103 K | persisted+orphan → persisted+consumed (Phase 2) | **wire** in Phase 2 (`onset_density`) |
| `onsets.strength_envelope` (35,831 × 2 fields) | 1.67 MB | persisted+orphan | **drop-from-json** |

Reduction: ~94% of onsets sub-dict ≈ 6.3% of total.

## `chroma_timeline[]` — 35,831 items × 14 fields

All fields persisted+aggregate (only consumed by `analyze_tracks_deep` line 435–438 for per-track dominant chroma). Strategic latent signal for future key-aware sequencing.

**Recommendation:** **drop-from-json**, but refactor `analyze_tracks_deep` to receive the chromagram array directly (or compute per-track dominant inline from segments of the raw chroma matrix). Compute path stays; the persisted artifact goes.

If future work needs the full per-sample timeline, it can be recomputed from audio in ~10 s/set (line 631 `librosa.feature.chroma_cqt` call).

Reduction: 100% of this 27.0% slice ≈ 27.0% of total. **Single biggest win.**

## `spectral_timeline[]` — 35,831 items × 7 fields

| Field | Role | Recommendation |
|---|---|---|
| `time_sec`, `centroid_hz` | persisted+consumed (`assemble_v2.py:439`) | keep |
| `flux` | persisted+aggregate (used by `analyze_transitions` line 473–474 for windowed averages) | **drop-from-json** (recompute inline at transition time) |
| `bandwidth_hz`, `flatness`, `rolloff_hz`, `contrast_mean` | persisted+orphan | **drop-from-json** |

Reduction: ~71% of this 21.5% slice ≈ 15.3% of total.

## `key_timeline[]`

Persisted+orphan. Small footprint (38 K). Strategic latent for future key-aware sequencing. **Keep persisted.**

## `phrases.{four_bar|eight_bar|sixteen_bar}[]`

Per-phrase: `phrase_index`, `start_sec`, `end_sec`, `beat_count`, `bars`, `energy_mean`, `energy_peak`, `energy_shape`. All persisted+consumed by `assemble_v2.py:413` (the assembler builds its phrase pipeline from `four_bar`). **Keep all.**

---

## Phase 4 drop summary (file-size impact)

| Drop target | % of file |
|---|---:|
| `chroma_timeline` (full) | 27.0% |
| `spectral_timeline.{bandwidth_hz, flatness, rolloff_hz, contrast_mean, flux}` | 15.3% |
| `multiband_energy.{low_mid, mid, high_mid}` | 8.8% |
| `onsets.strength_envelope` | 6.3% |
| `beats.features` per-beat list | 4.7% |
| `hpss_timeline.hp_ratio` | 3.0% |
| **Total estimated reduction** | **~65%** |

Resulting file size estimate for the reference set: 37.1 MB → ~13.0 MB.

## Phase 2 wiring summary

- **Read** `bpm_timeline.confidence` (already persisted, divisor calibrated at 8.28 per Phase 0).
- **Read** `onsets.times_sec` (already persisted; percentile-rank within set per phrase).
- **Add** `motion_std` to `SceneClip` dataclass in `assemble_v2.py` (computed from existing `motion_vals` at line 338). Percentile-rank across full scene database during build.

## Intermediates (for the record, not written to JSON)

- `S`, `freqs` — STFT magnitude + frequencies; feed `compute_beat_features`. Pure intermediate; ~50 MB in-memory.
- `y_harm`, `y_perc` — harmonic/percussive decomposed audio; feed downstream features. Pure intermediate; ~150 MB in-memory.
- `onset_env` (full envelope) — used internally for beat tracking; not the same as the persisted `strength_envelope` (which is the sampled version).
