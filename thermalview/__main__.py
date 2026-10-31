"""Command line.

  python -m thermalview probe              inspect the extracted M100 data
  python -m thermalview prepare [--dev]    build the cleaned 1-minute table
  python -m thermalview forecast           4-model forecasting experiment
  python -m thermalview thermal            fit heat models + simulator check
  python -m thermalview placement          policy comparison in the simulator
  python -m thermalview figures            paper figures from results/
  python -m thermalview sensitivity        placement robustness to the limit and the safety margin
  python -m thermalview all [--dev]        every step above, in order
  python -m thermalview serve              start the operator console
  python -m thermalview apidoc             write docs/API.md from the API description
  python -m thermalview paper              fill the paper draft with numbers from results/
  python -m thermalview transfer --test-config config_jan.toml
                                           frozen evaluation-month models scored on another month
  python -m thermalview jobs [--test-config config_jan.toml]
                                           real job starts (Slurm job table): event study
"""
from __future__ import annotations

import argparse
import sys
import time

from thermalview import pipeline
from thermalview.config import load_config

QUICK = {"forecast": {"max_train_rows": 40_000, "bootstrap_iters": 200},
         "evaluation": {"sim_days": 1, "seeds": [1, 2, 3]},
         "workload": {"load_factors": [1.5]},
         "sensitivity": {"limit_percentiles": [95]}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="thermalview", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["probe", "prepare", "forecast", "thermal", "placement", "figures", "all", "serve", "sensitivity",
                                       "apidoc", "paper", "transfer", "jobs"])
    ap.add_argument("--dev", action="store_true", help="use synthetic development data instead of M100")
    ap.add_argument("--quick", action="store_true",
                    help="small, fast settings to check everything runs (not for paper results)")
    ap.add_argument("--config", default=None, help="path to config.toml")
    ap.add_argument("--lan", action="store_true",
                    help="serve: accept connections from other computers on the network (binds 0.0.0.0)")
    ap.add_argument("--test-config", default=None, help="transfer: config of the month to test on")
    ap.add_argument("--public", action="store_true",
                    help="serve: also give a temporary https link that works from any network (needs cloudflared)")
    ap.add_argument("--port", type=int, default=None, help="serve: port (default from config)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, QUICK if args.quick else None)
    t = time.time()
    if args.step == "apidoc":
        from thermalview.web import apispec
        out = cfg.root / "docs" / "API.md"
        out.write_text(apispec.to_markdown(), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    if args.step == "paper":
        from thermalview import paper
        paper.build(cfg)
        return 0
    if args.step == "jobs":
        from thermalview import jobs
        jobs.run(cfg, load_config(args.test_config, QUICK if args.quick else None) if args.test_config else None)
        return 0
    if args.step == "transfer":
        from thermalview import transfer
        if not args.test_config:
            ap.error("transfer needs --test-config (for example config_jan.toml)")
        transfer.run(cfg, load_config(args.test_config, QUICK if args.quick else None))
        return 0
    if args.step == "serve":
        from thermalview.web.server import serve
        if args.lan:
            cfg["app"]["host"] = "0.0.0.0"
        if args.port:
            cfg["app"]["port"] = args.port
        serve(cfg, public=args.public)
        return 0
    if args.step == "prepare":
        pipeline.step_prepare(cfg, dev=args.dev)
    elif args.step == "all":
        pipeline.step_all(cfg, dev=args.dev)
    else:
        getattr(pipeline, f"step_{args.step}")(cfg)
    print(f"\n{args.step} finished in {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
