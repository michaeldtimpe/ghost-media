"""
ghost-media vision-engine bake-off harness.

Compares vision engines (claude-cli / ollama / mlx) on a representative footage
subset, anchored on objective CLIP metrics, to fix the prior enrichment's 2.6%
first-hand coverage and generic tags. See bench/config.py for knobs and the
approved plan for the design rationale.

Modules:
  config      paths, pilot set, thresholds, engine matrix, billing guard
  keys        sha256 canonical identity + collision detection
  util        atomic JSON writes, git SHA, small formatters
  manifest    frozen per-run manifest for reproducibility
  sampler     adaptive CLIP-cluster frame sampler -> sampling_plan.json
  metrics     reconstruction round-trip / discriminability / coverage / text P-R
  judge       claude-cli faithfulness audit (diagnostic only, never scored)
  runner      drives the bake-off (extract once, loop engines, resume/pause)
  report      CLIP-anchored scoreboard renderer
  groundtruth text-presence + description spot-check label store
"""

__all__ = ["config", "keys", "util", "manifest"]
