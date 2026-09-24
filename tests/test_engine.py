import numpy as np
import pytest

from angularity_dem.analysis import eta_from_phi, friction_param
from angularity_dem.engine import fabric_aniso
from angularity_dem.geometry import polygon_area, regular_polygon
from angularity_dem.protocol import Params, build_system, run_simulation


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
    from angularity_dem.engine import dem_step

    P = Params(corners=5)
    r = 2.5e-3
    v = regular_polygon(5, r, 0.3)
    n = 1
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
            0.5, P.Y, P.gamma, P.xi_t, dt, 9.81,
            np.full(n * 16, -1, dtype=np.int64), np.zeros(n * 16),
            np.full(n * 16, -(10**9), dtype=np.int64), i,
            np.zeros(n, dtype=np.int64), 0,
            np.empty(n * 4), np.empty(n * 4), np.empty(n * 4),
            np.zeros(1, dtype=np.int64),
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
    from angularity_dem.engine import dem_step

    P = Params(corners=5)
    r = 2.5e-3
    v = regular_polygon(5, r, np.pi * 1.3)  # flat bottom edge (vertices 234/306 deg)
    n = 1
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
    hst = np.full(n * 16, -(10**9), dtype=np.int64)
    for i in range(60000):
        dem_step(
            pos, vel, th, om,
            np.zeros((n, 2)), np.zeros(n),
            vloc, np.full(n, 5), np.zeros_like(vloc),
            mass, inert, rad,
            -1.0, 1.0, 0.0, 2.0,
            0.5, P.Y, 2.0, P.xi_t, dt, 9.81,
            hist, hft, hst, i,
            np.zeros(n, dtype=np.int64), 0,
            np.empty(n * 4), np.empty(n * 4), np.empty(n * 4),
            np.zeros(1, dtype=np.int64),
        )
    # equilibrium: F = Y A / l = m g with A = w*delta, l = 2*(r_in - delta/2)
    w = 2 * r * np.sin(np.pi / 5)  # edge length (flat on floor)
    mg = mass[0] * 9.81
    delta = 2 * inradius * mg / (P.Y * w)  # linearized
    y_eq = inradius - delta
    # particle center height above floor, allow small residual oscillation
    assert pos[0, 1] == pytest.approx(y_eq, abs=5 * delta)
    assert abs(vel[0, 1]) < 1e-4
