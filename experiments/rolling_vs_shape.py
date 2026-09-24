"""Can discs emulate angularity? Rolling friction vs particle shape.

Krengel et al. (2023) argue that disc-based models cannot substitute for
polygonal particles, and specifically criticise "rolling friction" because it
only opposes rotation and cannot create geometric interlocking. This experiment
tests that claim by trying to make quasi-circular particles (N=64) reproduce
the pentagon (N=5) fingerprint using the disc-model "tricks":

  * rolling friction        mu_r in {0.02, 0.05, 0.1, 0.2, 0.4} at mu = 0.5
  * sliding friction        mu in {0.2, 0.35, 0.7, 1.0} at mu_r = 0
  * both                    mu = 1.0, mu_r = 0.2

and comparing to the shape series N = 5..64 (results_v2). The angular-particle
fingerprint is not just "higher strength": it is the combination

    strength up, critical porosity up (looser yet stronger),
    rotation down, coordination number Z down (fewer contacts),
    normal force per contact up, contact anisotropy up.

Usage:
    uv run python experiments/rolling_vs_shape.py run --jobs 7
    uv run python experiments/rolling_vs_shape.py analyze
"""

from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from gonzo.analysis import critical_state, friction_param, load_run

SHAPE_DIR = "results_v2"
DISC_DIR = "results_rolling"
FIGDIR = "experiments/figures"
REPORT = "experiments/rolling_vs_shape_report.txt"

ROLLING = [0.02, 0.05, 0.1, 0.2, 0.4]
FRICTION = [0.2, 0.35, 0.7, 1.0]
SEEDS = (1, 2, 3)
ANCHOR = (0.5, 0.0)  # disc baseline, available in the shape series


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def _job(spec):
    from gonzo.protocol import Params, run_simulation

    N, mu, mu_r, seed, outdir = spec
    P = Params.demo(corners=N, width=0.08, seed=seed, eps_end=0.15)
    P.mu = mu
    P.mu_r = mu_r
    P.out = os.path.join(outdir, f"run_mu{mu:g}_mur{mu_r:g}_s{seed}.npz")
    t0 = time.time()
    run_simulation(P, verbose=False)
    return spec, time.time() - t0


def run_jobs(jobs, ncores):
    os.makedirs(DISC_DIR, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=ncores) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for fut in as_completed(futs):
            spec, dt = fut.result()
            done += 1
            print(
                f"  {done}/{len(jobs)}  N={spec[0]} mu={spec[1]:g} "
                f"mu_r={spec[2]:g} seed={spec[3]}  {dt:.0f}s",
                flush=True,
            )


def cmd_run(args):
    jobs = [(64, 0.5, mr, s, DISC_DIR) for mr in ROLLING for s in SEEDS]
    jobs += [(64, mu, 0.0, s, DISC_DIR) for mu in FRICTION for s in SEEDS]
    jobs += [(64, 1.0, 0.2, s, DISC_DIR) for s in SEEDS]
    print(f"{len(jobs)} disc runs")
    run_jobs(jobs, args.jobs)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def force_heterogeneity(fn: np.ndarray) -> float:
    """Share of total normal force carried by the strongest 20% contacts."""
    if len(fn) == 0:
        return np.nan
    f = np.sort(fn)[::-1]
    k = max(1, int(round(0.2 * len(f))))
    return float(f[:k].sum() / f.sum())


def _points_shape():
    runs = {}
    for fn in sorted(os.listdir(SHAPE_DIR)):
        m = re.match(r"run_n(\d+)_s(\d+)\.npz$", fn)
        if m:
            runs.setdefault(int(m.group(1)), []).append(
                load_run(os.path.join(SHAPE_DIR, fn))
            )
    pts = {}
    for N, rs in runs.items():
        cs = [critical_state(r) for r in rs]
        pt = {k: float(np.mean([c[k] for c in cs])) for k in
              ("phi_mu", "phi", "porosity", "Z", "medFN", "medrot_end",
               "aniso", "eta", "servo_ok_fraction")}
        pt["het"] = float(np.mean([force_heterogeneity(r["end_fn"]) for r in rs]))
        pt["label"] = f"N={N}"
        pt["N"] = N
        pts[N] = pt
    return pts


def _points_disc():
    runs = {}
    for fn in sorted(os.listdir(DISC_DIR)):
        m = re.match(r"run_mu([\d.]+)_mur([\d.]+)_s(\d+)\.npz$", fn)
        if m:
            key = (float(m.group(1)), float(m.group(2)))
            runs.setdefault(key, []).append(load_run(os.path.join(DISC_DIR, fn)))
    pts = {}
    for (mu, mu_r), rs in runs.items():
        cs = [critical_state(r) for r in rs]
        pt = {k: float(np.mean([c[k] for c in cs])) for k in
              ("phi_mu", "phi", "porosity", "Z", "medFN", "medrot_end",
               "aniso", "eta", "servo_ok_fraction")}
        pt["het"] = float(np.mean([force_heterogeneity(r["end_fn"]) for r in rs]))
        pt["label"] = f"mu={mu:g}" + (f",mu_r={mu_r:g}" if mu_r else "")
        pt["mu"], pt["mu_r"] = mu, mu_r
        pts[(mu, mu_r)] = pt
    return pts


OBS = ("phi_mu", "porosity", "medrot_end", "Z", "medFN", "aniso")


def cmd_analyze(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shape = _points_shape()
    disc = _points_disc()
    if not disc:
        raise SystemExit("no disc runs found - run first")

    # reference: pentagon and disc anchors from the shape series
    penta = shape[5]
    disc0 = shape[64]
    disc_pts = {k: v for k, v in disc.items()}

    # best disc match to the pentagon across all observables
    def dist(pt):
        tot = 0.0
        for k in OBS:
            lo, hi = sorted((penta[k], disc0[k]))
            span = abs(hi - lo)
            if span < 1e-12:
                continue
            tot += ((pt[k] - penta[k]) / span) ** 2
        return float(np.sqrt(tot))

    best_key = min(
        (k for k in disc_pts if disc_pts[k]["servo_ok_fraction"] >= 0.5),
        key=lambda k: dist(disc_pts[k]),
    )
    best = disc_pts[best_key]

    lines = []
    add = lines.append
    add("CAN DISCS EMULATE ANGULARITY?  rolling friction vs particle shape")
    add("=" * 70)
    add("")
    add("Demo preset (w=0.08 m, ~250 particles), identical protocol for all runs.")
    add("Pentagon (N=5) = angular reference; N=64 = quasi-circular reference.")
    add("'servo' = fraction of critical-state rows where the lateral servo holds")
    add("sigma_c; low values flag stick-slip jams (excluded from the best match).")
    add("")
    cols = ("phi", "phi_mu", "porosity", "medrot_end", "Z", "medFN", "aniso",
            "servo_ok_fraction")
    hdr = f"{'series':>22} " + " ".join(f"{k:>8}" for k in cols)
    add(hdr)
    add("-" * len(hdr))
    row = f"{'N=5 pentagon':>22} " + " ".join(
        f"{penta[k]:>8.3f}" if k != "medFN" else f"{penta[k]:>8.0f}" for k in cols)
    add(row)
    row = f"{'N=64 disc (mu_r=0)':>22} " + " ".join(
        f"{disc0[k]:>8.3f}" if k != "medFN" else f"{disc0[k]:>8.0f}" for k in cols)
    add(row)
    for key in sorted(disc_pts, key=lambda k: (k[1], k[0])):
        if key == ANCHOR:
            continue
        pt = disc_pts[key]
        row = f"{'disc ' + pt['label']:>22} " + " ".join(
            f"{pt[k]:>8.3f}" if k != "medFN" else f"{pt[k]:>8.0f}" for k in cols)
        if pt["servo_ok_fraction"] < 0.5:
            row += "  <-- jam (unreliable)"
        add(row)
    add("")
    add(f"best disc match to pentagon: {best['label']}  (distance {dist(best):.2f})")
    add("")
    add("mismatch of the best disc model vs the pentagon, in units of the")
    add("pentagon-to-disc span for each observable:")
    for k in OBS:
        span = abs(penta[k] - disc0[k])
        gap = (best[k] - penta[k]) / span if span > 1e-12 else 0.0
        add(f"  {k:>10}: pentagon {penta[k]:>9.3f}  best disc {best[k]:>9.3f}  "
            f"residual {gap*100:+7.1f}% of span")
    add("")
    add("force heterogeneity (share of F_N on strongest 20% contacts):")
    add(f"  N=5 pentagon {penta['het']:.3f}   N=64 disc {disc0['het']:.3f}   "
        f"best disc {best['het']:.3f}")

    os.makedirs(FIGDIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    Ns = sorted(shape)
    shape_c = plt.cm.viridis(np.linspace(0, 0.85, len(Ns)))

    def base(ax, xkey, ykey, xlabel, ylabel):
        ax.plot([shape[N][xkey] for N in Ns], [shape[N][ykey] for N in Ns],
                "-", color="k", lw=1.0, zorder=1)
        for N, c in zip(Ns, shape_c):
            ax.plot(shape[N][xkey], shape[N][ykey], "o", color=c, ms=7,
                    zorder=3)
            ax.annotate(str(N), (shape[N][xkey], shape[N][ykey]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        rr = sorted([k for k in disc_pts if k[1] > 0 and k[0] == 0.5])
        if rr:
            ax.plot([disc_pts[k][xkey] for k in rr], [disc_pts[k][ykey] for k in rr],
                    "--", color="crimson", lw=1.0, zorder=1)
            for k in rr:
                ax.plot(disc_pts[k][xkey], disc_pts[k][ykey], "s", color="crimson",
                        ms=6, zorder=3)
                ax.annotate(f"{k[1]:g}", (disc_pts[k][xkey], disc_pts[k][ykey]),
                            textcoords="offset points", xytext=(4, -10),
                            fontsize=8, color="crimson")
        ff = sorted([k for k in disc_pts if k[1] == 0 and k[0] != 0.5])
        if ff:
            pts = ff
            ax.plot([disc_pts[k][xkey] for k in pts], [disc_pts[k][ykey] for k in pts],
                    ":", color="darkorange", lw=1.0, zorder=1)
            for k in ff:
                ax.plot(disc_pts[k][xkey], disc_pts[k][ykey], "^",
                        color="darkorange", ms=6, zorder=3)
                ax.annotate(f"mu={k[0]:g}", (disc_pts[k][xkey], disc_pts[k][ykey]),
                            textcoords="offset points", xytext=(4, -10),
                            fontsize=8, color="darkorange")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    base(axes[0, 0], "porosity", "phi_mu", "critical-state porosity",
         r"$\phi/\mu$")
    axes[0, 0].set_title("shape effect vs disc tricks (porosity)")
    base(axes[0, 1], "Z", "phi_mu", "coordination number Z", r"$\phi/\mu$")
    axes[0, 1].set_title("few contacts vs many weak contacts")
    base(axes[1, 0], "medrot_end", "phi_mu", "median rotation [rad]", r"$\phi/\mu$")
    axes[1, 0].set_title("rotation")
    base(axes[1, 1], "het", "medFN", "force heterogeneity (top 20% share)",
         r"median $F_N$ [N]")
    axes[1, 1].set_title("force network fingerprint")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([], [], color="k", marker="o", ls="-", label="polygons (shape)"),
        Line2D([], [], color="crimson", marker="s", ls="--",
               label="discs + rolling friction"),
        Line2D([], [], color="darkorange", marker="^", ls=":",
               label="discs + sliding friction"),
    ]
    axes[0, 0].legend(handles=handles, fontsize=8, loc="best")
    fig.suptitle("Can discs emulate angularity? (labels: shape = corner number,"
                 " rolling = mu_r, friction = mu)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "rolling_vs_shape.png"), dpi=150)
    plt.close(fig)

    report = "\n".join(lines)
    with open(REPORT, "w") as f:
        f.write(report + "\n")
    print(report)
    print(f"\nfigure: {FIGDIR}/rolling_vs_shape.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "analyze"])
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "run":
        cmd_run(args)
    else:
        cmd_analyze(args)


if __name__ == "__main__":
    main()
