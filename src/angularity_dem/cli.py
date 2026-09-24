"""CLI: run / campaign / analyze / verify."""

from __future__ import annotations

import argparse
import os
import time

from .analysis import analyze_directory
from .protocol import Params, run_simulation


def _cmd_run(args):
    preset = Params.paper if args.preset == "paper" else Params.demo
    P = preset(
        corners=args.corners,
        width=args.width,
        seed=args.seed,
        eps_end=args.eps_end,
        out=args.out or f"results/run_n{args.corners}_s{args.seed}.npz",
        label=f"n{args.corners}",
    )
    os.makedirs(os.path.dirname(P.out), exist_ok=True)
    t0 = time.time()
    run_simulation(P)
    print(f"done in {time.time() - t0:.1f} s")


def _cmd_campaign(args):
    preset = Params.paper if args.preset == "paper" else Params.demo
    corners = [int(c) for c in args.corners.split(",")]
    os.makedirs(args.outdir, exist_ok=True)
    for N in corners:
        for s in range(args.samples):
            P = preset(
                corners=N,
                width=args.width,
                seed=args.seed + s,
                eps_end=args.eps_end,
                out=os.path.join(args.outdir, f"run_n{N}_s{args.seed + s}.npz"),
                label=f"n{N}",
            )
            t0 = time.time()
            print(f"=== N={N} sample {s + 1}/{args.samples} ===")
            run_simulation(P)
            print(f"=== done in {time.time() - t0:.1f} s ===")


def _cmd_analyze(args):
    print(analyze_directory(args.results, args.figdir, args.report))


def _cmd_verify(args):
    import numpy as np

    from .geometry import area_equivalent_diameter, circularity, fig1_table

    tab = fig1_table()
    print("Fig. 1 check (unit circumradius):")
    for N, v in tab.items():
        print(f"  N={N:>2}: d_area/r={v['d_area_over_r']:.4f}  circ={v['circularity']:.4f}")
    assert abs(area_equivalent_diameter(5, 1.0) - 1.739916) < 1e-5
    assert abs(circularity(5, 1.0) - 0.929950) < 1e-5
    assert abs(circularity(64, 1.0) - 0.999598) < 1e-5
    assert all(tab[N]["circularity"] < tab[M]["circularity"]
               for N, M in zip((5, 6, 7, 10, 13, 17, 32), (6, 7, 10, 13, 17, 32, 64)))

    from .analysis import eta_from_phi, friction_param

    for eta in (1.5, 2.0, 2.755, 4.0):
        assert abs(eta_from_phi(friction_param(eta * 2e4, 2e4)) - eta) < 1e-12
    a, b, c = 3.425, -1.215, 0.573
    print(f"\nEq. (7) check: phi/mu(N=5) = {a * 5**b + c:.3f} (paper: > 1)")
    assert a * 5**b + c > 1.0
    print(f"Eq. (7) check: phi/mu(N=64) = {a * 64**b + c:.3f} (paper: ~0.6)")
    assert 0.55 < a * 64**b + c < 0.65
    print("\nall analytic checks passed")


def main(argv=None):
    p = argparse.ArgumentParser(prog="angularity-dem")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="single biaxial simulation")
    pr.add_argument("--corners", type=int, default=5)
    pr.add_argument("--preset", choices=["demo", "paper"], default="demo")
    pr.add_argument("--width", type=float, default=0.07)
    pr.add_argument("--eps-end", type=float, default=0.15)
    pr.add_argument("--seed", type=int, default=1)
    pr.add_argument("--out", default=None)
    pr.set_defaults(func=_cmd_run)

    pc = sub.add_parser("campaign", help="run all corner numbers")
    pc.add_argument("--corners", default="5,6,13,64")
    pc.add_argument("--preset", choices=["demo", "paper"], default="demo")
    pc.add_argument("--width", type=float, default=0.07)
    pc.add_argument("--eps-end", type=float, default=0.15)
    pc.add_argument("--samples", type=int, default=1)
    pc.add_argument("--seed", type=int, default=1)
    pc.add_argument("--outdir", default="results")
    pc.set_defaults(func=_cmd_campaign)

    pa = sub.add_parser("analyze", help="figures + verification report")
    pa.add_argument("--results", default="results")
    pa.add_argument("--figdir", default=None)
    pa.add_argument("--report", default=None)
    pa.set_defaults(func=_cmd_analyze)

    pv = sub.add_parser("verify", help="fast analytic self-checks")
    pv.set_defaults(func=_cmd_verify)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
