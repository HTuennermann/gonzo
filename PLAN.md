# Speedup Plan (executed)

Status: Phase 1 + 2 complete. Measured results (this machine, 8 cores):

| stage | N=6 us/step | N=64 us/step |
|---|---|---|
| original | 330 | 3622 |
| O(log K) ordered clip | 235* | 3047* |
| lens-window clip | ~110* | ~160* |
| preallocated scratch | 109 | 157 |
| compiled run_block (protocol path) | ~105 engine + ~0 python | ~150 |
| campaign --jobs 7 (end to end) | 40 runs: ~105 min serial -> 8.2 min | |

*on the interim (broken-probe) harness; final numbers from bench/bench.py.

The decisive change was the lens-window clip: for circumcircle-intersecting
pairs only Pb edges whose lines can cut the circle lens are visited (window
from the circle-intersection angle; exact for regular polygons, verified on
3000 thin-contact cases against shapely). Thin quasi-static contacts have
windows of 1-3 edges out of 64.

Regression protocol (bench/bench.py): frozen-state A/B (bitwise per code
version, forces must match to 1e-7) + 20k-step window statistics (<= 5%),
plus pytest incl. shapely cross-validation. All pass; the 40-run campaign
on the new engine reproduces the paper verification report 14/14.

## Phase 1 - algorithmic fixes inside numba (biggest win, low risk)

1. **O(log K) edge ordering** instead of O(K^2) sort. Regular-polygon edge
   normals are angularly sorted, so the "far-to-near along d" order is a walk
   from the edge whose normal is nearest to +d, outward in both directions.
   Binary search for the start edge, then emit. Expected: 5-15x on N=64
   narrow phase.
2. **O(1) separating-axis rejection** between circumcircle test and clip:
   for regular polygons the vertex support along any direction is analytic
   (index = round(theta/dtheta)), so a 2-3 axis SAT costs ~20 flops.
   Expected: 2-4x fewer clips (false candidates rejected).
3. **Collinear-vertex pruning** of clip output (round-polygon slivers
   accumulate near-degenerate vertices; keeps m ~ 4-6).
4. **Preallocated scratch** (head/nxt/buf1/bu2 created once in `_Runner`,
   passed in) - removes 4 allocations per step.
5. Micro: cache per-particle cos/sin; maintain running bbox instead of
   O(n) min/max scan each step.

Target after Phase 1: N=64 ~500-800 us/step, N=5 ~110-130 us/step.

## Phase 2 - driver overhead and throughput

6. **Compiled phase driver**: move the per-step Python loop (stress
   smoothing, servo update, shear bookkeeping) into a numba `run_steps(k)`
   that returns every k-th step. Python overhead is ~30-40 us/step = ~25%
   for N<=13.
7. **Parallel samples**: `campaign --jobs J` - run seeds as J processes
   (simulations are independent). Campaign wall-clock / min(J, cores),
   zero physics risk. (40-run campaign: ~1.7 h -> ~20 min on 6-8 cores.)

Combined Phase 1+2 target: 40-run demo campaign ~25-40 min (from ~1.7 h);
paper-scale single run from ~2-4 days to ~10-20 h single process.

## Phase 3 - threading within one run (only if paper-scale needed)

8. `prange` pair loop, per-thread force/torque buffers + reduction; history
   slots partitioned by owner particle (no cross-thread writes). Expected
   2-4x on 4-8 cores. Medium risk (races); needs a dedicated test that
   single- and multi-threaded runs agree to fp tolerance.

## Phase 4 - GPU port (out of scope for now)

Timestep loop is inherently sequential; per-step work is tiny (~10-50k
contact evaluations). Overlaps ~1e-10 m on 0.1 m coordinates force float64,
which is 1/32-1/64 rate on consumer GPUs - not viable. Only worth it on
A100/H100-class hardware via a numba.cuda port of the pair loop; estimated
~3-10 h for one paper-scale run there.

## Verification protocol for every change

- `uv run pytest` (incl. shapely cross-checks).
- Regression: fixed-seed N=6 and N=64 runs, 20k steps; forces/positions must
  match the current implementation to ~1e-12 relative per step (clip-order
  changes only reshuffle fp rounding). Beyond ~50k steps trajectories
  diverge chaotically - compare window-averaged stresses instead.
- Benchmark table (us/step for N in {5, 32, 64}) re-run after each item.

Suggested order: 1 -> 2 -> 4 -> 7 -> 5 -> 6 -> (3), with benchmarks and
regression checks after each; commit per item.
