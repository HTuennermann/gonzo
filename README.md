# gonzo

*[gon](https://en.wikipedia.org/wiki/Polygon) + DEM — a 2D polygon discrete-element
study of particle angularity in granular shear.*

`gonzo` simulates biaxial compression of regular polygonal particles (5–64
corners) with the discrete-element method and reproduces the results of:

> Krengel, D., Chen, J., Kikumoto, M. (2023).
> *Effects of particle angularity on the bulk-characteristics of granular
> assemblies under plane strain condition.*
> Computers and Geotechnics 164, 105812.
> [doi:10.1016/j.compgeo.2023.105812](https://doi.org/10.1016/j.compgeo.2023.105812)

The paper is **not** redistributed here; use the link above.

## What it does

2D polygonal DEM implementing the Matuttis-Chen contact model used in the paper:

- overlap area `A` of convex polygons (Sutherland-Hodgman clipping with an
  analytic lens-window optimisation),
- elastic normal force `F_N = Y A / l` (Eq. 1),
- characteristic length `l = 4|ra||rb|/(|ra|+|rb|)` (Eq. 2),
- viscous damping `F_N^dis = γ sqrt(m̄Y) Ȧ / l` (Eq. 3),
- Cundall-Strack tangential spring with Coulomb cap `|F_T| <= μ F_N` (Eq. 5),
- servo-controlled rigid box: isotropic consolidation to σ_c = 20 kPa, then
  biaxial shear at constant lid velocity and constant lateral pressure.

Material parameters follow the paper's Table 1 (bidisperse 80/20 mix of
2.5/5.0 mm circumradii, ρ = 2830 kg/m², Y = 10⁹ N/m, μ = 0.5, Δt = 2 µs), and
the verification report compares against the paper's quantitative findings:

- internal friction law φ/μ = 3.425 N^(−1.215) + 0.573 (Eq. 7),
- critical-state coordination number Z → 3.24,
- median critical-state normal force F_N ≈ 203.7 ± 7.1 N,
- porosity, rotation and contact-anisotropy trends vs corner number.

## Install / run (uv)

```bash
uv sync                       # runtime + dev dependencies
uv run pytest                 # unit tests (incl. shapely cross-checks)

# fast analytic self-checks: Fig. 1 shape metrics, Eq. 7 limits
uv run gonzo verify

# full pipeline: 8 shapes x 5 samples, ~8 min on 8 cores
uv run gonzo campaign --corners 5,6,7,10,13,17,32,64 --samples 5 --jobs 7 --outdir results
uv run gonzo analyze --results results
```

Outputs: per-run `.npz` time series in `results/`, paper-analogue figures
(Figs. 4–17) in `figures/`, and `verification_report.txt` (14 checks).

Single run or full paper scale:

```bash
uv run gonzo run --corners 5 --preset demo     # ~1 min, ~260 particles
uv run gonzo run --corners 5 --preset paper    # w=0.25 m, ~2500 particles, ~2 h
```

## Assumptions (unspecified in the paper)

- **Damping constant γ** (Eq. 3) is not given in the paper; default `γ = 1.0`
  (near-critical contact damping). Weakly influential in the quasi-static
  regime; configurable.
- **Tangential stiffness k_T = (2/7)·k_N** per contact (standard DEM choice);
  the paper names the Cundall-Strack spring but gives no k_T.
- The **demo preset** (w = 0.08 m, ~260 particles, v_y = 30 mm/s) trades sample
  size and strain rate for speed (inertial number ~10⁻⁴, still quasi-static).
  Small angular samples can exhibit global stick-slip jams, so critical-state
  statistics are conditioned on the lateral servo tracking σ_c; the tracked
  fraction is reported.

## Layout

- `src/gonzo/geometry.py` - polygon construction, paper Fig. 1 shape metrics
- `src/gonzo/contact.py`  - numba convex clipping + contact geometry
- `src/gonzo/engine.py`   - force loop, Cundall-Strack history, walls, integrator
- `src/gonzo/protocol.py` - assembly, deposition, consolidation, shear
- `src/gonzo/analysis.py` - critical-state metrics, Eq. (6)/(7), figures
- `src/gonzo/cli.py`      - `run` / `campaign` / `analyze` / `verify`
- `tests/`                - pytest suite (shapely cross-validation)
- `bench/`                - chaos-safe timing + regression harness
- `experiments/`          - discs-vs-angularity study (rolling friction, see below)
- `PLAN.md`               - performance work log (23x narrow-phase speedup)

## Does rolling friction on discs substitute for angularity?

The paper argues disc-based models (rolling friction, clusters of discs) cannot
replace polygonal particles, in particular criticising rolling friction because
it only opposes rotation and cannot create geometric interlocking.
`experiments/rolling_vs_shape.py` tests this by trying to make quasi-circular
particles (N=64) reproduce the pentagon fingerprint with rolling friction
(`mu_r`) or sliding friction (`mu`):

    uv run python experiments/rolling_vs_shape.py run --jobs 7
    uv run python experiments/rolling_vs_shape.py analyze

Findings (demo preset, 3 samples per point): sliding friction alone raises
strength only weakly and leaves porosity, rotation and coordination unchanged,
i.e. it cannot emulate angularity. **Rolling friction does**: as `mu_r` grows,
strength, critical-state porosity, force per contact and contact anisotropy all
increase while median rotation and coordination number fall - the same
fingerprint as the N=5 -> N=64 shape series, and it can be tuned to match the
pentagon closely. Residual differences remain in the force-network
heterogeneity and the strength-coordination relation. Caveat: at this sample
size strong interlocking (high `mu_r`, and pentagons themselves) produces
stick-slip jams; the report flags runs where the lateral servo cannot hold the
confining pressure.
