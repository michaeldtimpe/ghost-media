#!/usr/bin/env python3
"""
Vision-engine bake-off entry point.

  python bench_run.py health                          # backend availability (mlx may be absent)
  python bench_run.py plan   [--videos S...] [--force] # build sampling plans + histogram
  python bench_run.py run    [--engines a,b] [--judge] [--allow-metered]
  python bench_run.py compare [--video SUBSTR]         # CLIP-anchored scoreboard
  python bench_run.py score                            # re-score from existing raw outputs

Engines: claude-cli, ollama, mlx, baseline (current system, floor), clip-ceiling
(image→image upper bound). Default engine set = all five. The composite rank uses
objective CLIP metrics only; the LLM-judge is an audit layer.
"""

import argparse

from bench import config, runner, report, sampler
from bench.hybrid import register_hybrids
register_hybrids()  # adds bench/hybrids/*.spec.json to config.ENGINES so they're scorable


def _engines(arg):
    if not arg:
        return list(config.DEFAULT_ENGINES)
    names = [e.strip() for e in arg.split(",") if e.strip()]
    bad = [e for e in names if e not in config.ENGINES]
    if bad:
        raise SystemExit(f"unknown engine(s): {bad}; choose from {list(config.ENGINES)}")
    return names


def main():
    ap = argparse.ArgumentParser(description="ghost-media vision-engine bake-off")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_h = sub.add_parser("health"); p_h.add_argument("--engines")
    p_p = sub.add_parser("plan")
    p_p.add_argument("--videos", nargs="+"); p_p.add_argument("--force", action="store_true")
    p_p.add_argument("--max-states", type=int, default=None)
    p_r = sub.add_parser("run")
    p_r.add_argument("--engines"); p_r.add_argument("--videos", nargs="+")
    p_r.add_argument("--judge", action="store_true")
    p_r.add_argument("--allow-metered", action="store_true")
    p_r.add_argument("--judge-n", type=int, default=40)
    p_s = sub.add_parser("score"); p_s.add_argument("--engines"); p_s.add_argument("--videos", nargs="+")
    p_c = sub.add_parser("compare")
    p_c.add_argument("--video", default=None); p_c.add_argument("--engines")
    p_c.add_argument("--videos", nargs="+")

    p_hyb = sub.add_parser("hybrid", help="2-model hybrid synthesizer + cost probe")
    p_hyb.add_argument("action", choices=["build", "run-cost", "list"])
    p_hyb.add_argument("--spec", help="spec name (for run-cost)")
    p_hyb.add_argument("--n", type=int, default=30, help="frames for run-cost probe")

    args = ap.parse_args()

    if args.cmd == "health":
        runner.health(_engines(args.engines))
        return

    if args.cmd == "plan":
        if not args.videos:
            # Same failure class as the --max-states cap (lessons.md): a bench
            # default silently scoping a production run. The 2026-06 import
            # built plans for 6 pilots instead of 99 new videos because of this.
            print("\n  ⚠ no --videos given: planning the PILOT set only "
                  f"({len(config.PILOT_VIDEOS)} videos).")
            print("    For the full corpus pass `--videos .` "
                  "(substring-matches every analysis file).\n")
        reg = config.pilot_registry(args.videos)
        stats = sampler.build_plans(reg, max_states=args.max_states, force=args.force)
        sampler.print_histogram(stats)
        return

    if args.cmd == "run":
        reg = config.pilot_registry(args.videos)
        engines = _engines(args.engines)
        print(f"\n  engines: {engines}\n  videos: {list(reg)}")
        # ensure plans exist
        sampler.build_plans(reg)
        boards = runner.run(engines, reg, do_judge=args.judge,
                            allow_metered=args.allow_metered, judge_n=args.judge_n)
        report.render()
        return

    if args.cmd == "score":
        reg = config.pilot_registry(args.videos)
        runner.score_all(reg, _engines(args.engines))
        report.render()
        return

    if args.cmd == "compare":
        reg = config.pilot_registry(args.videos)
        if args.video:
            report.dump_video(reg, args.video, _engines(args.engines))
        else:
            report.render()
        return

    if args.cmd == "hybrid":
        from bench import hybrid as H
        if args.action == "list":
            for s in H.load_specs():
                bits = [f"base={s.base}"]
                if s.escalation: bits.append(f"esc={s.escalation}")
                if s.overrides:  bits.append(f"override={list(s.overrides)}")
                print(f"  {s.name:32s} mode={s.mode:18s} {' '.join(bits)}")
                if s.description: print(f"      {s.description}")
        elif args.action == "build":
            H.build_all()
        elif args.action == "run-cost":
            if not args.spec:
                raise SystemExit("hybrid run-cost requires --spec NAME")
            H.cascade_run(args.spec, n_sample=args.n)
        return


if __name__ == "__main__":
    main()
