"""Benchmark + regression harness for optimization work.

Design: chaotic dynamics make end-to-end trajectory comparison meaningless
after ~10^4 steps (fp-rounding amplifies). Instead:

1. `baseline` runs setup, saves the frozen state, records 300 probe steps of
   wall forces from that state (deterministic per code version up to fp
   rounding), plus 20k-step window statistics of its own trajectory.
2. `check` loads the FROZEN state, re-runs the 300-step probe (must match the
   baseline probe to ~1e-7 relative: same physics, only fp reordering), and
   runs its own 20k probe (window statistics must match within 5%).

Usage: uv run python bench/bench.py baseline|check
"""

import sys
import time

import numpy as np

from gonzo.protocol import Params, build_system, _Runner

SETUP = {6: (60_000, 30_000), 64: (60_000, 30_000)}


def make_state(P):
    rng = np.random.default_rng(P.seed)
    sysd = build_system(P, rng)
    R = _Runner(P, sysd)
    n_settle, n_cons = SETUP[P.corners]
    for _ in range(n_settle):
        R.step_once(0.0, 9.81)
    R.yt = float(np.max(sysd["pos"][:, 1] + sysd["rad"]))
    R.s_l = R.s_r = R.s_lid = 0.0
    for _ in range(n_cons):
        R.step_once(P.mu, 0.0)
        R.servo_walls()
    # clear tangential-spring history so probes from this state (in-memory or
    # reloaded) start deterministically; note dem_step step_now restarts at 0
    # and its staleness test (now - last <= 3) would otherwise resurrect
    # setup-phase springs
    R.s["hist_part"][:] = -1
    R.s["hist_stp"][:] = -(10**9)
    return R


def shear_block(R, n_steps, collect_forces):
    P = R.P
    forces = np.empty((n_steps, 3)) if collect_forces else None
    for k in range(n_steps):
        fl, fr, flid = R.step_once(P.mu, 0.0)
        R.servo_walls(do_lid=False)
        R.yt -= P.v_y * P.dt
        if collect_forces:
            forces[k] = (fl, fr, flid)
    return forces


def load_runner(path, P):
    s = dict(np.load(path))
    R = _Runner(P, s)
    R.xl = float(s["xl"]); R.xr = float(s["xr"]); R.yt = float(s["yt"])
    R.s_l = float(s["s_l"]); R.s_r = float(s["s_r"]); R.s_lid = float(s["s_lid"])
    return R


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    out = {}
    bench = {}
    for N in (6, 64):
        P = Params.demo(corners=N, seed=3)
        # timing on a fresh consolidated, shearing system
        R = make_state(P)
        t0 = time.time()
        shear_block(R, 2000, False)
        bench[f"us_per_step_N{N}"] = (time.time() - t0) / 2000 * 1e6
        # pristine frozen state for stats; saved copy for the force probe
        Rf = make_state(P)
        spath = f"bench/state_N{N}.npz"
        np.savez(
            spath, **Rf.s,
            xl=Rf.xl, xr=Rf.xr, yt=Rf.yt,
            s_l=Rf.s_l, s_r=Rf.s_r, s_lid=Rf.s_lid,
        )
        if mode == "baseline":
            Rp = load_runner(spath, P)
            forces = shear_block(Rp, 300, True)
            np.savez(f"bench/probe_N{N}.npz", forces=forces)
        # statistical window from the pristine trajectory (identical in both modes)
        forces20k = shear_block(Rf, 20_000, True)
        out[f"forces_stats_N{N}"] = np.array(
            [forces20k[10_000:].mean(axis=0), forces20k[10_000:].std(axis=0)]
        )
    out["bench"] = np.array([bench[f"us_per_step_N6"], bench[f"us_per_step_N64"]])
    print("bench:", {k: round(v, 1) for k, v in bench.items()})

    if mode == "baseline":
        np.savez("bench/baseline.npz", **out)
        print("baseline saved (states + probes + stats)")
        return

    base = np.load("bench/baseline.npz")
    ok = True
    for N in (6, 64):
        P = Params.demo(corners=N, seed=3)
        forces = shear_block(load_runner(f"bench/state_N{N}.npz", P), 300, True)
        ref = np.load(f"bench/probe_N{N}.npz")["forces"]
        rel_head = np.abs(forces - ref).max() / max(np.abs(ref).max(), 1e-12)
        s0, s1 = out[f"forces_stats_N{N}"], base[f"forces_stats_N{N}"]
        rel_stat = np.abs(s0 - s1).max() / max(np.abs(s1).max(), 1e-12)
        print(f"N={N}: frozen-state force rel.diff = {rel_head:.2e} "
              f"(<= 1e-7), window-stat rel.diff = {rel_stat:.2e} (<= 0.05)")
        ok &= rel_head < 1e-7 and rel_stat < 0.05
    print("REGRESSION:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
