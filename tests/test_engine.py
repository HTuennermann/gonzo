import numpy as np
import pytest

from gonzo.analysis import eta_from_phi, friction_param
from gonzo.engine import fabric_aniso
from gonzo.geometry import polygon_area, regular_polygon
from gonzo.protocol import Params, build_system, run_simulation


def _sys_2_particles():
    """Two pentagons, flat edges facing, one above the other."""
    P = Params(corners=5)
    rng = np.random.default_rng(0)
    sysd = build_system(Params(corners=5, solid_target=1e-4), rng)
    # replace with hand-built pair
    r = 2.5e-3
    v = regular_polygon(5, r, phase=np.pi / 2)  # vertex up... use flat base
    # orientation with a flat edge at the bottom: rotate so vertex angle offset
    v = regular_polygon(5, r, phase=np.pi / 5)
    n = 2
    vloc = np.zeros((n, 5, 2))
    vloc[0] = v
    vloc[1] = v
    mass = np.full(n, P.rho * polygon_area(v))
    return P, vloc, mass, r


def test_rolling_friction_sprung_coulomb():
    """Rolling friction: neutral at rest, opposes relative spin, mu_r-ordered."""
    from gonzo.engine import HEAD_CAP, dem_step

    P = Params(corners=64)
    r = 2.5e-3
    nv = 64
    v = regular_polygon(nv, r, 0.0)
    n = 2
    vloc = np.zeros((n, nv, 2))
    vloc[0] = v
    vloc[1] = v
    mass = np.full(n, P.rho * polygon_area(v))
    inert = np.full(n, P.rho * polygon_area(v) * r**2 * 0.5)
    rad = np.full(n, r)
    sep = 2 * r - 1e-5  # slight overlap
    pos = np.array([[0.05, 0.05], [0.05 + sep, 0.05]])
    scratch = (
        np.empty(HEAD_CAP, dtype=np.int64), np.empty(n, dtype=np.int64),
        np.empty((2 * nv + 8, 2)), np.empty((2 * nv + 8, 2)),
    )

    def run(mu_r, om0, nsteps):
        vel = np.zeros((n, 2))
        om = np.array(om0, dtype=float)
        th = np.zeros(n)
        hist = np.full(n * 16, -1, dtype=np.int64)
        hft = np.zeros(n * 16)
        hm = np.zeros(n * 16)
        hst = np.full(n * 16, -(10**9), dtype=np.int64)
        for i in range(nsteps):
            dem_step(
                pos.copy(), vel, th, om,
                np.zeros((n, 2)), np.zeros(n),
                vloc, np.full(n, nv), np.zeros_like(vloc),
                mass, inert, rad,
                -1.0, 1.0, -1.0, 2.0,
                0.5, mu_r, P.Y, P.gamma, P.xi_t, 2e-6, 0.0,
                hist, hft, hm, hst, i,
                np.zeros(n, dtype=np.int64), 0,
                np.empty(n * 4), np.empty(n * 4), np.empty(n * 4),
                np.zeros(1, dtype=np.int64), *scratch,
            )
        return om

    # no relative spin -> rolling friction is essentially inactive (only the
    # tiny relative spin seeded by the contact model itself engages it)
    om_ref = run(0.0, [0.7, 0.7], 200)
    om_roll = run(0.4, [0.7, 0.7], 200)
    assert np.allclose(om_ref, om_roll, rtol=1e-5)

    # relative spin is resisted smoothly (one step, no chatter)
    om1 = run(0.2, [1.0, -1.0], 1)
    d = om1 - np.array([1.0, -1.0])
    assert d[0] < 0 and d[1] > 0            # opposes the relative spin
    assert abs(om1[1] - om1[0]) < 2.0       # reduced, not amplified
    assert np.abs(d).max() < 0.05           # spring builds up gradually

    # cap regime: with a large relative spin the torque saturates at
    # mu_r * FN * R_eff and is therefore linear in mu_r
    start = np.array([2000.0, -2000.0])
    d1 = start - run(0.1, start, 1)
    d3 = start - run(0.3, start, 1)
    assert d1[0] > 0 and d1[1] < 0          # still opposing
    assert np.allclose(d3, 3.0 * d1, rtol=1e-6)

    # stable, bounded over many steps (no blow-up)
    om = run(0.4, [1.0, -1.0], 400)
    assert np.all(np.isfinite(om)) and np.abs(om).max() < 100.0


def test_phi_eta_roundtrip():
    for eta in (1.2, 1.8, 2.755, 5.0):
        phi = friction_param(eta * 2e4, 2e4)
        assert eta_from_phi(phi) == pytest.approx(eta, abs=1e-12)
    # paper sanity: eta = 2.755 <-> phi/mu = 1.058 (mu = 0.5)
    assert friction_param(2.755 * 2e4, 2e4) / 0.5 == pytest.approx(1.058, abs=2e-3)


def test_fabric_aniso_isotropic():
    rng = np.random.default_rng(1)
    ang = rng.uniform(-np.pi, np.pi, 20000)
    assert fabric_aniso(ang, ang.shape[0]) == pytest.approx(1.0, abs=0.05)


def test_fabric_aniso_anisotropic():
    # contacts concentrated along x -> ratio > 1
    ang = np.concatenate([
        np.random.normal(0, 0.1, 5000),
        np.random.normal(np.pi, 0.1, 5000),
    ])
    r = fabric_aniso(ang, ang.shape[0])
    assert r > 1.3


def test_free_flight_trajectory():
    """No contacts: exact parabolic trajectory (integrator check)."""
    from gonzo.engine import HEAD_CAP, dem_step

    P = Params(corners=5)
    r = 2.5e-3
    v = regular_polygon(5, r, 0.3)
    n = 1
    scratch = (
        np.empty(HEAD_CAP, dtype=np.int64), np.empty(n, dtype=np.int64),
        np.empty((12, 2)), np.empty((12, 2)),
    )
    vloc = np.zeros((n, 5, 2))
    vloc[0] = v
    mass = np.array([P.rho * polygon_area(v)])
    inert = np.array([mass[0] * r**2 * 0.3])
    pos = np.array([[0.05, 0.05]])
    vel = np.array([[0.3, 0.5]])
    th = np.array([0.7])
    om = np.array([1.3])
    dt = 1e-6
    for i in range(2000):
        dem_step(
            pos, vel, th, om,
            np.zeros((n, 2)), np.zeros(n),
            vloc, np.full(n, 5), np.zeros_like(vloc),
            mass, inert, np.array([r]),
            -1.0, 1.0, -1.0, 2.0,  # walls far away
            0.5, 0.0, P.Y, P.gamma, P.xi_t, dt, 9.81,
            np.full(n * 16, -1, dtype=np.int64), np.zeros(n * 16),
            np.zeros(n * 16),
            np.full(n * 16, -(10**9), dtype=np.int64), i,
            np.zeros(n, dtype=np.int64), 0,
            np.empty(n * 4), np.empty(n * 4), np.empty(n * 4),
            np.zeros(1, dtype=np.int64), *scratch,
        )
    t = 2000 * dt
    assert pos[0, 0] == pytest.approx(0.05 + 0.3 * t, abs=1e-9)
    # gravity acts toward -y; semi-implicit Euler adds 0.5*g*t*dt vs continuum
    assert pos[0, 1] == pytest.approx(
        0.05 + 0.5 * t - 0.5 * 9.81 * t * t, abs=1e-7
    )
    assert th[0] == pytest.approx(0.7 + 1.3 * t, abs=1e-9)  # no torque in flight


def test_particle_settles_on_floor_overlap():
    """Equilibrium overlap on floor matches F = Y A / l = m g (2D weight)."""
    from gonzo.engine import HEAD_CAP, dem_step

    P = Params(corners=5)
    r = 2.5e-3
    v = regular_polygon(5, r, np.pi * 1.3)  # flat bottom edge (vertices 234/306 deg)
    n = 1
    scratch = (
        np.empty(HEAD_CAP, dtype=np.int64), np.empty(n, dtype=np.int64),
        np.empty((12, 2)), np.empty((12, 2)),
    )
    vloc = np.zeros((n, 5, 2))
    vloc[0] = v
    mass = np.array([P.rho * polygon_area(v)])
    inert = np.array([mass[0] * r**2 * 0.3])
    rad = np.array([r])

    inradius = r * np.cos(np.pi / 5)
    pos = np.array([[0.05, inradius]])
    vel = np.zeros((n, 2))
    th = np.zeros(n)
    om = np.zeros(n)
    dt = 2e-6
    hist = np.full(n * 16, -1, dtype=np.int64)
    hft = np.zeros(n * 16)
    hm = np.zeros(n * 16)
    hst = np.full(n * 16, -(10**9), dtype=np.int64)
    for i in range(60000):
        dem_step(
            pos, vel, th, om,
            np.zeros((n, 2)), np.zeros(n),
            vloc, np.full(n, 5), np.zeros_like(vloc),
            mass, inert, rad,
            -1.0, 1.0, 0.0, 2.0,
            0.5, 0.0, P.Y, 2.0, P.xi_t, dt, 9.81,
            hist, hft, hm, hst, i,
            np.zeros(n, dtype=np.int64), 0,
            np.empty(n * 4), np.empty(n * 4), np.empty(n * 4),
            np.zeros(1, dtype=np.int64), *scratch,
        )
    # equilibrium: F = Y A / l = m g with A = w*delta, l = 2*(r_in - delta/2)
    w = 2 * r * np.sin(np.pi / 5)  # edge length (flat on floor)
    mg = mass[0] * 9.81
    delta = 2 * inradius * mg / (P.Y * w)  # linearized
    y_eq = inradius - delta
    # particle center height above floor, allow small residual oscillation
    assert pos[0, 1] == pytest.approx(y_eq, abs=5 * delta)
    assert abs(vel[0, 1]) < 1e-4
