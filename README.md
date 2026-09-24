# angularity-dem

Reproduction and verification of:

> Krengel, D., Chen, J., Kikumoto, M. (2023).
> *Effects of particle angularity on the bulk-characteristics of granular
> assemblies under plane strain condition.*
> Computers and Geotechnics 164, 105812.
> https://doi.org/10.1016/j.compgeo.2023.105812

2D polygonal DEM biaxial compression of regular polygons
(N ∈ {5, 6, 7, 10, 13, 17, 32, 64} corners), implementing the Matuttis-Chen
contact model used in the paper:

- overlap area `A` of convex polygons (Sutherland-Hodgman clipping),
- elastic normal force `F_N = Y A / l` (Eq. 1),
- characteristic length `l = 4|ra||rb|/(|ra|+|rb|)` (Eq. 2),
- viscous damping `F_N^dis = γ sqrt(m̄Y) Ȧ / l` (Eq. 3),
- Cundall-Strack tangential spring with Coulomb cap `|F_T| <= μ F_N` (Eq. 5),
- servo-controlled rigid box walls: isotropic consolidation to
  σ_c = 20 kPa, then biaxial shear at constant lid velocity with constant
  lateral confining pressure (paper Sec. 2.3).

Material parameters follow paper Table 1: bidisperse 80/20 mix of 2.5/5.0 mm
circumradii, ρ = 2830 kg/m², Y = 10⁹ N/m, μ = 0.5, Δt = 2 µs.

## Usage (uv)

```bash
uv sync                       # install runtime + dev deps
uv run pytest                 # unit tests (incl. shapely cross-check of clipping)
uv run angularity-dem verify  # fast analytic self-checks (Fig. 1, Eq. 7)

# full pipeline (demo preset, ~260 particles; ~8 min for all 8 shapes x 5 samples on 8 cores):
uv run angularity-dem campaign --corners 5,6,7,10,13,17,32,64 --samples 5 --jobs 7 --outdir results
uv run angularity-dem analyze --results results

# single run / paper-scale (WARNING: days of CPU time):
uv run angularity-dem run --corners 5 --preset paper
```

Outputs: per-run `.npz` time series in `results/`, paper-analogue figures
(Figs. 4-17) in `figures/`, and `verification_report.txt` comparing against
the paper's quantitative findings:

- Eq. (7): φ/μ = 3.425 N^(-1.215) + 0.573,
- critical-state coordination number Z → 3.24,
- median critical-state normal force F_N ≈ 203.7 ± 7.1 N,
- porosity/rotation/anisotropy trends vs corner number.

## Assumptions (unspecified in the paper)

- **Damping constant γ = 0.5** (Eq. 3) - not given in the paper; chosen for a
  near-elastic, well-damped contact (ζ ≈ 0.25). Weak influence in quasi-static
  regime; configurable via `Params.gamma`.
- **Tangential stiffness k_T = (2/7)·k_N** per contact (`Params.xi_t`),
  a standard DEM choice; the paper names the Cundall-Strack spring but gives
  no k_T value.
- **Demo preset** uses w = 0.08 m (~260 particles) and v_y = 100 mm/s
  (inertial number ~10⁻⁴, still quasi-static; paper: 0.25 m, ~2500 particles,
  10 mm/s) so the campaign finishes in minutes instead of days.
- Deposition is RSA placement + gravitational settling (frictionless, as in
  the paper), trimmed to h = 1.2·w.

## Layout

- `src/angularity_dem/geometry.py` - polygon construction, Fig. 1 metrics
- `src/angularity_dem/contact.py`  - numba convex clipping + contact geometry
- `src/angularity_dem/engine.py`   - force loop, history springs, walls, integrator
- `src/angularity_dem/protocol.py` - assembly, deposition, consolidation, shear
- `src/angularity_dem/analysis.py` - critical-state metrics, Eq. (6)/(7), figures
- `tests/`                        - pytest suite (shapely cross-validation)
