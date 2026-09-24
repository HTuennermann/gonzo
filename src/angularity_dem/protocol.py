"""Experiment protocol: assembly generation, deposition, consolidation, biaxial shear.

Follows paper Section 2.3: bidisperse frictionless deposition under gravity,
trim, isotropic consolidation via servo walls to sigma_c, then biaxial shear
(constant lid velocity, constant lateral confining pressure).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from .engine import (
    MAX_SLOTS,
    contact_medians,
    dem_step,
    fabric_aniso,
    particle_metrics,
)
from .geometry import polygon_area, polygon_inertia, regular_polygon


@dataclass
class Params:
    corners: int = 5
    r_small: float = 2.5e-3
    r_large: float = 5.0e-3
    frac_small: float = 0.8
    rho: float = 2830.0
    Y: float = 1e9
    mu: float = 0.5
    gamma: float = 1.0  # paper unspecified; see README
    xi_t: float = 2.0 / 7.0  # kT = xi_t * kN
    width: float = 0.08
    aspect: float = 1.2  # h0 / w
    sigma_c: float = 2e4
    v_y: float = 0.03  # lid velocity during shear (paper: 0.01; keep quasi-static)
    dt: float = 2e-6
    eps_end: float = 0.15
    solid_target: float = 0.80
    rsa_fraction: float = 0.45  # circumscribed-circle fraction for placement
    settle_time: float = 0.3
    settle_max: float = 0.8
    servo_v: float = 0.3  # must exceed ~0.6*(w/h)*v_y or walls lag and jam
    servo_gain: float = 15.0
    servo_tol: float = 0.03
    smooth_tau: float = 50.0  # steps
    log_every: int = 100
    seed: int = 1
    label: str = ""
    out: str = ""

    @classmethod
    def paper(cls, **kw):
        p = cls(
            width=0.25,
            v_y=0.01,
            servo_v=0.1,
            eps_end=1.0 - (0.3 - 5.0 * 0.01) / 0.3,
            settle_time=2.0,
            settle_max=3.0,
            log_every=500,
        )
        for k, v in kw.items():
            setattr(p, k, v)
        return p

    @classmethod
    def demo(cls, **kw):
        p = cls()
        for k, v in kw.items():
            setattr(p, k, v)
        return p


def mean_particle_area(P: Params) -> float:
    a_s = polygon_area(regular_polygon(P.corners, P.r_small))
    a_l = polygon_area(regular_polygon(P.corners, P.r_large))
    return P.frac_small * a_s + (1.0 - P.frac_small) * a_l


def build_system(P: Params, rng: np.random.Generator) -> dict:
    """Bidisperse RSA assembly of regular polygons in the open box."""
    N = P.corners
    A_mean = mean_particle_area(P)
    h_box = P.aspect * P.width
    n_total = int(round(P.width * h_box * P.solid_target / A_mean))
    n_large = int(round((1.0 - P.frac_small) * n_total))
    n_small = n_total - n_large

    radii = np.concatenate([np.full(n_large, P.r_large), np.full(n_small, P.r_small)])
    rng.shuffle(radii)
    n = radii.shape[0]

    circle_area = float(np.sum(np.pi * radii**2))
    h_fill = circle_area / (P.rsa_fraction * P.width)

    pos = np.full((n, 2), 1e18)
    placed = 0
    for i in range(n):
        for _ in range(5000):
            x = radii[i] + rng.random() * (P.width - 2 * radii[i])
            y = radii[i] + rng.random() * (h_fill - 2 * radii[i])
            ok = True
            for k in range(i):  # scan all previous rows (skip unplaced 1e18)
                if pos[k, 0] > 1e17:
                    continue
                dx = pos[k, 0] - x
                dy = pos[k, 1] - y
                rr = radii[k] + radii[i]
                if dx * dx + dy * dy < rr * rr:
                    ok = False
                    break
            if ok:
                pos[i, 0] = x
                pos[i, 1] = y
                placed += 1
                break
    keep = pos[:, 0] < 1e17
    if not np.all(keep):
        print(f"  RSA: dropped {int(np.sum(~keep))} particles")
    pos = pos[keep]
    radii = radii[keep]
    n = pos.shape[0]

    K = N
    vloc = np.zeros((n, K, 2))
    mass = np.empty(n)
    inert = np.empty(n)
    for i in range(n):
        v = regular_polygon(N, radii[i], phase=2.0 * np.pi * rng.random())
        vloc[i, :, 0] = v[:, 0]
        vloc[i, :, 1] = v[:, 1]
        mass[i] = P.rho * polygon_area(v)
        inert[i] = polygon_inertia(v, P.rho)

    return {
        "pos": pos,
        "vel": np.zeros((n, 2)),
        "th": 2.0 * np.pi * rng.random(n),
        "om": np.zeros(n),
        "rad": radii,
        "nverts": np.full(n, N, dtype=np.int64),
        "vloc": vloc,
        "wverts": np.zeros_like(vloc),
        "mass": mass,
        "inert": inert,
        "frc": np.zeros((n, 2)),
        "trq": np.zeros(n),
        "ccount": np.zeros(n, dtype=np.int64),
        "hist_part": np.full(n * MAX_SLOTS, -1, dtype=np.int64),
        "hist_ft": np.zeros(n * MAX_SLOTS),
        "hist_stp": np.full(n * MAX_SLOTS, -(10**9), dtype=np.int64),
        "K": K,
        "sum_area": float(np.sum(mass) / P.rho),
    }


def _rebuild_after_trim(sysd: dict, keep: np.ndarray, rho: float) -> None:
    n = int(np.sum(keep))
    for k in ("pos", "vel", "th", "om", "rad", "nverts", "vloc", "mass", "inert"):
        sysd[k] = sysd[k][keep].copy()
    sysd["wverts"] = np.zeros_like(sysd["vloc"])
    sysd["frc"] = np.zeros((n, 2))
    sysd["trq"] = np.zeros(n)
    sysd["ccount"] = np.zeros(n, dtype=np.int64)
    sysd["hist_part"] = np.full(n * MAX_SLOTS, -1, dtype=np.int64)
    sysd["hist_ft"] = np.zeros(n * MAX_SLOTS)
    sysd["hist_stp"] = np.full(n * MAX_SLOTS, -(10**9), dtype=np.int64)
    sysd["sum_area"] = float(np.sum(sysd["mass"]) / rho)


class _Runner:
    def __init__(self, P: Params, sysd: dict):
        self.P = P
        self.s = sysd
        self.step = 0
        self.xl = 0.0
        self.xr = P.width
        self.yt = 1e18
        self.s_l = self.s_r = self.s_lid = 0.0
        n = sysd["pos"].shape[0]
        self.m_ang = np.empty(n * 6)
        self.m_fn = np.empty(n * 6)
        self.m_ft = np.empty(n * 6)
        self.m_count = np.zeros(1, dtype=np.int64)

    def step_once(self, mu, grav, want_metrics=0):
        fl, fr, flid = dem_step(
            self.s["pos"], self.s["vel"], self.s["th"], self.s["om"],
            self.s["frc"], self.s["trq"], self.s["vloc"], self.s["nverts"],
            self.s["wverts"], self.s["mass"], self.s["inert"], self.s["rad"],
            self.xl, self.xr, 0.0, self.yt,
            mu, self.P.Y, self.P.gamma, self.P.xi_t, self.P.dt, grav,
            self.s["hist_part"], self.s["hist_ft"], self.s["hist_stp"], self.step,
            self.s["ccount"], want_metrics,
            self.m_ang, self.m_fn, self.m_ft, self.m_count,
        )
        self.step += 1
        a = 1.0 / self.P.smooth_tau
        h = self.yt if self.yt < 1e17 else 1.0
        self.s_l += a * (fl / h - self.s_l)
        self.s_r += a * (fr / h - self.s_r)
        self.s_lid += a * (flid / max(self.xr - self.xl, 1e-12) - self.s_lid)
        return fl, fr, flid

    def servo_walls(self, do_lid=True):
        P = self.P
        e_l = np.clip(P.servo_gain * (P.sigma_c - self.s_l) / P.sigma_c, -1, 1)
        e_r = np.clip(P.servo_gain * (P.sigma_c - self.s_r) / P.sigma_c, -1, 1)
        w0 = P.width
        self.xl = min(max(self.xl + P.servo_v * e_l * P.dt, -0.1 * w0), 0.45 * w0)
        self.xr = max(min(self.xr - P.servo_v * e_r * P.dt, 1.1 * w0), self.xl + 0.2 * w0)
        if do_lid:
            e_t = np.clip(P.servo_gain * (P.sigma_c - self.s_lid) / P.sigma_c, -1, 1)
            self.yt = max(self.yt - P.servo_v * e_t * P.dt, 0.05 * P.width)


class _Log:
    KEYS = ("t", "eps", "xl", "xr", "yt", "s1", "s3", "eta", "porosity",
            "Z", "medrot", "madrot", "medfn", "medft", "aniso", "ke")

    def __init__(self, P: Params):
        self.rows = {k: [] for k in self.KEYS}
        self.log_every = P.log_every
        self.h0 = None
        self.th0 = None

    def record(self, R: _Runner, phase: int):
        s = R.s
        t = R.step * R.P.dt
        w = R.xr - R.xl
        h = R.yt
        eps = 1.0 - h / self.h0 if (phase == 2 and self.h0) else 0.0
        s3 = 0.5 * (R.s_l + R.s_r)
        self.rows["t"].append(t)
        self.rows["eps"].append(eps)
        self.rows["xl"].append(R.xl)
        self.rows["xr"].append(R.xr)
        self.rows["yt"].append(h)
        self.rows["s1"].append(R.s_lid)
        self.rows["s3"].append(s3)
        self.rows["eta"].append(R.s_lid / max(s3, 1e-9))
        self.rows["porosity"].append(1.0 - s["sum_area"] / (w * h))
        th_ref = self.th0 if self.th0 is not None else s["th"]
        Z, med, mad = particle_metrics(s["ccount"], s["th"], th_ref)
        self.rows["Z"].append(Z)
        self.rows["medrot"].append(med)
        self.rows["madrot"].append(mad)
        medfn, medft = contact_medians(R.m_fn, R.m_ft, R.m_count[0])
        self.rows["medfn"].append(medfn)
        self.rows["medft"].append(medft)
        self.rows["aniso"].append(fabric_aniso(R.m_ang, R.m_count[0]))
        self.rows["ke"].append(
            float(np.sum(s["mass"] * np.sum(s["vel"] ** 2, axis=1))
                  + np.sum(s["inert"] * s["om"] ** 2))
        )


def run_simulation(P: Params, verbose: bool = True) -> dict:
    rng = np.random.default_rng(P.seed)
    sysd = build_system(P, rng)
    if verbose:
        n0 = sysd["pos"].shape[0]
        print(f"[N={P.corners}] {n0} particles (paper: ~2200-2750 at w=0.25 m)")

    R = _Runner(P, sysd)
    log = _Log(P)

    # ---- phase 1: deposition (frictionless, gravity, open top) ----
    settled = False
    while R.step * P.dt < P.settle_max:
        R.step_once(0.0, 9.81)
        if R.step % 2000 == 0:
            if R.step * P.dt > P.settle_time:
                vmax = float(np.max(np.abs(sysd["vel"])))
                if vmax < 0.02:
                    settled = True
                    break
    if verbose:
        print(f"  deposition done t={R.step*P.dt:.3f} s settled={settled}")

    # ---- trim + close lid ----
    # remove particles that intersect the cut plane (centre above, or poking
    # above with any part of the circumcircle), then rest the lid exactly there
    h_cut = min(P.aspect * P.width, float(np.max(sysd["pos"][:, 1] + sysd["rad"])))
    keep = sysd["pos"][:, 1] + sysd["rad"] <= h_cut + 1e-12
    if not np.all(keep):
        _rebuild_after_trim(sysd, keep, P.rho)
        if verbose:
            print(f"  trimmed to h={h_cut:.4f} m -> {sysd['pos'].shape[0]} particles")
    R.xr = P.width
    R.yt = h_cut
    R.s_l = R.s_r = R.s_lid = 0.0

    # ---- phase 2: isotropic consolidation ----
    t0 = R.step * P.dt
    reached_at = None
    hold_until = None
    while R.step * P.dt < t0 + 3.0:
        want = int(reached_at is not None and R.step % log.log_every == 0)
        R.step_once(P.mu, 0.0, want_metrics=want)
        R.servo_walls()
        if R.yt < 1e-4 or R.xr - R.xl < 1e-4:
            raise RuntimeError("walls collapsed - instability")
        if R.step % 1000 == 0 and not np.all(np.isfinite(sysd["pos"])):
            raise RuntimeError("NaN positions - blow-up")
        ok = (
            abs(R.s_l / P.sigma_c - 1) < P.servo_tol
            and abs(R.s_r / P.sigma_c - 1) < P.servo_tol
            and abs(R.s_lid / P.sigma_c - 1) < P.servo_tol
        )
        tnow = R.step * P.dt
        if ok and reached_at is None:
            reached_at = tnow
            hold_until = reached_at + max(0.05, 0.5 * (reached_at - t0))
        if want:
            log.record(R, phase=1)
        if reached_at is not None and tnow >= hold_until:
            break
    if verbose:
        print(
            f"  consolidated t={R.step*P.dt:.3f} s "
            f"(sl={R.s_l/P.sigma_c:.3f} sr={R.s_r/P.sigma_c:.3f} "
            f"st={R.s_lid/P.sigma_c:.3f})"
        )

    # pre-shear contact snapshot
    R.step_once(P.mu, 0.0, want_metrics=1)
    pre = {
        "ang": R.m_ang[: R.m_count[0]].copy(),
        "fn": R.m_fn[: R.m_count[0]].copy(),
        "ft": R.m_ft[: R.m_count[0]].copy(),
    }

    # ---- phase 3: biaxial shear ----
    log.h0 = R.yt
    log.th0 = sysd["th"].copy()
    eps = 0.0
    while eps < P.eps_end:
        want = int(R.step % log.log_every == 0)
        R.step_once(P.mu, 0.0, want_metrics=want)
        R.yt -= P.v_y * P.dt
        R.servo_walls(do_lid=False)
        if R.xr - R.xl < 1e-4:
            raise RuntimeError("walls collapsed - instability")
        if want:
            log.record(R, phase=2)
        if R.step % 1000 == 0 and not np.all(np.isfinite(sysd["pos"])):
            raise RuntimeError("NaN positions - blow-up")
        eps = 1.0 - R.yt / log.h0
    if verbose:
        print(f"  sheared to eps={eps:.3f} t={R.step*P.dt:.3f} s")

    R.step_once(P.mu, 0.0, want_metrics=1)
    end = {
        "ang": R.m_ang[: R.m_count[0]].copy(),
        "fn": R.m_fn[: R.m_count[0]].copy(),
        "ft": R.m_ft[: R.m_count[0]].copy(),
    }

    res = {k: np.array(v) for k, v in log.rows.items()}
    res["th0"] = log.th0
    res["th_end"] = sysd["th"].copy()
    res["pos_end"] = sysd["pos"].copy()
    res["rad_end"] = sysd["rad"].copy()
    res["pre_ang"] = pre["ang"]
    res["pre_fn"] = pre["fn"]
    res["pre_ft"] = pre["ft"]
    res["end_ang"] = end["ang"]
    res["end_fn"] = end["fn"]
    res["end_ft"] = end["ft"]
    if P.out:
        np.savez_compressed(P.out, params_json=json.dumps(asdict(P)), **res)
        if verbose:
            print(f"  saved {P.out}")
    return res
