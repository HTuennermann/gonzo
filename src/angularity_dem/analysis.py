"""Analysis: critical-state metrics, Eq. (6)/(7) fit, figures, verification report.

Reproduces the quantitative results of Krengel et al. (2023):
- Fig. 4  stress ratio / porosity vs strain
- Fig. 5  phi/mu vs N with power-law fit (Eq. 7: 3.425 N^-1.215 + 0.573)
- Fig. 6  critical-state porosity vs N
- Fig. 9/10 rotation
- Fig. 11/12 coordination number (paper: Z -> 3.24)
- Fig. 14 contact-normal anisotropy
- Fig. 16/17 contact force distributions and medians (paper: F_N ~ 203.7 N)
"""

from __future__ import annotations

import json
import os

import numpy as np

EPS_CS_ONSET = 0.10  # paper: critical state starts at eps = 0.1

PAPER = {
    "eq7": {"a": 3.425, "b": -1.215, "c": 0.573},
    "Z_sat": 3.24,
    "medFN_crit": (203.684, 7.117),
    "phi_mu_limit": 0.6,
}


def friction_param(s1: float, s3: float) -> float:
    """Eq. (6): phi = tan(asin((s1-s3)/(s1+s3)))."""
    q = (s1 - s3) / (s1 + s3)
    return float(np.tan(np.arcsin(np.clip(q, -0.999999, 0.999999))))


def eta_from_phi(phi: float) -> float:
    q = np.sin(np.arctan(phi))
    return float((1.0 + q) / (1.0 - q))


def load_run(path: str) -> dict:
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["params"] = json.loads(str(d["params_json"]))
    return out


def critical_state(run: dict) -> dict:
    """Average metrics over the critical-state window (eps >= 0.1).

    At demo scale, small angular samples exhibit global stick-slip jams that
    inflate the lateral pressure beyond the servo target (large-system runs in
    the paper avoid this). Rows where the servo tracks the target confining
    pressure (|s3/sigma_c - 1| <= 0.15) therefore represent the physical
    critical state and are used preferentially, with the tracked fraction
    reported. Falls back to the full window if tracking is rare.
    """
    eps = run["eps"]
    sc = run["params"]["sigma_c"]
    m = eps >= min(EPS_CS_ONSET, 0.6 * eps.max())
    if not np.any(m):
        m = eps >= 0.5 * eps.max()
    tracked = np.abs(run["s3"] / sc - 1.0) <= 0.15
    mt = m & tracked
    frac = float(mt.sum()) / max(int(m.sum()), 1)
    if frac < 0.05:
        mt = m
        frac = float("nan")
    s1 = float(np.mean(run["s1"][mt]))
    s3 = float(np.mean(run["s3"][mt]))
    phi = friction_param(s1, s3)
    return {
        "N": run["params"]["corners"],
        "n_samples_rows": int(m.sum()),
        "servo_ok_fraction": frac,
        "eta": s1 / s3,
        "phi": phi,
        "phi_mu": phi / run["params"]["mu"],
        "porosity": float(np.nanmean(run["porosity"][mt])),
        "porosity_init": float(run["porosity"][1]) if len(run["porosity"]) > 1 else np.nan,
        "Z": float(np.nanmean(run["Z"][mt])),
        "medFN": float(np.nanmean(run["medfn"][mt])),
        "medFT": float(np.nanmean(run["medft"][mt])),
        "aniso": float(np.nanmean(run["aniso"][mt])),
        "medrot_end": float(run["medrot"][-1]),
        "madrot_end": float(run["madrot"][-1]),
        "phi_std": float(np.std([friction_param(a, b)
                                 for a, b in zip(run["s1"][mt], run["s3"][mt])])),
    }


def fit_eq7(cs_list: list[dict]):
    """Fit phi/mu = a N^b + c (paper Eq. 7)."""
    from scipy.optimize import curve_fit

    Ns = np.array([c["N"] for c in cs_list], dtype=float)
    y = np.array([c["phi_mu"] for c in cs_list])
    sig = np.array([max(c["phi_std"] / c["phi"], 0.02) for c in cs_list])

    def f(N, a, b, c):
        return a * N**b + c

    p0 = (3.4, -1.2, 0.6)
    popt, pcov = curve_fit(f, Ns, y, p0=p0, sigma=sig, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def make_figures(runs: dict[int, dict], cs: dict[int, dict], figdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    Ns_all = sorted(runs.keys())

    # ---- Fig 4: stress ratio + porosity vs strain ----
    fig, axes = plt.subplots(2, 1, figsize=(6, 7), sharex=True)
    for N in Ns_all:
        r = runs[N]
        m = r["eps"] > 0
        axes[0].plot(r["eps"][m], r["eta"][m], label=f"N={N}", lw=0.8)
        axes[1].plot(r["eps"][m], r["porosity"][m], lw=0.8)
    axes[0].set_ylabel(r"stress ratio $\eta=\sigma_1/\sigma_3$")
    axes[0].legend(fontsize=8)
    axes[0].axhline(1.0, color="k", lw=0.4)
    axes[1].set_ylabel("porosity n")
    axes[1].set_xlabel(r"axial strain $\epsilon$")
    fig.suptitle("paper Fig. 4 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig04_stress_strain.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 5: phi/mu vs N + fit + paper curve ----
    popt, perr = fit_eq7([cs[N] for N in Ns_all])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    Nf = np.linspace(min(Ns_all) * 0.9, max(Ns_all) * 1.1, 200)
    ax.plot(Nf, PAPER["eq7"]["a"] * Nf**PAPER["eq7"]["b"] + PAPER["eq7"]["c"],
            "k--", label="paper Eq. (7)")
    ax.plot(Nf, popt[0] * Nf**popt[1] + popt[2], "r-",
            label=(f"repro fit: {popt[0]:.2f} N$^{{{popt[1]:.2f}}}$ + {popt[2]:.2f}"))
    for N in Ns_all:
        c = cs[N]
        ax.errorbar(N, c["phi_mu"], yerr=max(c["phi_std"] / c["phi"] * c["phi_mu"], 0),
                    fmt="o", color="C0", capsize=3)
    ax.set_xlabel("corner number N")
    ax.set_ylabel(r"$\phi/\mu$")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_title("paper Fig. 5 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig05_phi_fit.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 6: porosity vs N ----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([cs[N]["N"] for N in Ns_all], [cs[N]["porosity_init"] for N in Ns_all],
            "s--", label="initial (after consolidation)")
    ax.plot([cs[N]["N"] for N in Ns_all], [cs[N]["porosity"] for N in Ns_all],
            "o-", label="critical state")
    ax.axhspan(0.187, 0.205, color="0.85", label="paper initial range")
    ax.set_xlabel("corner number N")
    ax.set_ylabel("porosity n")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.set_title("paper Figs. 2/6 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig06_porosity.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 9/10: rotation ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar([cs[N]["N"] for N in Ns_all],
                     [np.degrees(cs[N]["medrot_end"]) for N in Ns_all],
                     yerr=[np.degrees(cs[N]["madrot_end"]) for N in Ns_all],
                     fmt="o-", capsize=3)
    axes[0].set_xlabel("corner number N")
    axes[0].set_ylabel(r"median $|\theta|$ [deg] at $\epsilon_{end}$")
    axes[0].set_title("paper Fig. 9 analogue")
    for N in Ns_all:
        r = runs[N]
        m = r["eps"] > 0
        axes[1].plot(r["eps"][m], np.degrees(r["medrot"][m]), lw=0.8, label=f"N={N}")
    axes[1].set_xlabel(r"axial strain $\epsilon$")
    axes[1].set_ylabel(r"median $|\theta|$ [deg]")
    axes[1].legend(fontsize=8)
    axes[1].set_title("paper Fig. 10 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig09_10_rotation.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 7: rotation map + Fig 8 distribution ----
    fig, axes = plt.subplots(1, len(Ns_all), figsize=(3 * len(Ns_all), 3.4),
                             sharex=True, sharey=True)
    for ax, N in zip(np.atleast_1d(axes), Ns_all):
        r = runs[N]
        rot = np.degrees(r["th_end"] - r["th0"])
        sc = ax.scatter(r["pos_end"][:, 0], r["pos_end"][:, 1], c=rot, s=8,
                        cmap="RdBu_r", vmin=-90, vmax=90)
        ax.set_title(f"N={N}")
        ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=axes, shrink=0.85)
    cb.set_label("cumulative rotation [deg]")
    fig.suptitle("paper Fig. 7 analogue (blue CW, red CCW)")
    fig.savefig(os.path.join(figdir, "fig07_rotation_map.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for N in Ns_all:
        r = runs[N]
        rot = np.degrees(np.abs(r["th_end"] - r["th0"]))
        ax.hist(rot, bins=24, histtype="step", lw=1.2, label=f"N={N}", density=True)
    ax.set_xlabel(r"cumulative rotation magnitude $|\theta|$ [deg]")
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8)
    ax.set_title("paper Fig. 8 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig08_rotation_dist.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 11/12: coordination number ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for N in Ns_all:
        r = runs[N]
        m = r["eps"] > 0
        axes[0].plot(r["eps"][m], r["Z"][m], lw=0.8, label=f"N={N}")
    axes[0].axhline(PAPER["Z_sat"], color="k", ls="--", lw=0.8, label="paper limit 3.24")
    axes[0].set_xlabel(r"axial strain $\epsilon$")
    axes[0].set_ylabel("coordination number Z")
    axes[0].legend(fontsize=8)
    axes[0].set_title("paper Fig. 11 analogue")
    axes[1].plot([cs[N]["N"] for N in Ns_all], [cs[N]["Z"] for N in Ns_all], "o-")
    axes[1].axhline(PAPER["Z_sat"], color="k", ls="--", lw=0.8, label="paper limit 3.24")
    axes[1].set_xlabel("corner number N")
    axes[1].set_ylabel("critical Z")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    axes[1].set_title("paper Fig. 12 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig11_12_coordination.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 13/14: contact normals + anisotropy ----
    fig, axes = plt.subplots(1, len(Ns_all), figsize=(3 * len(Ns_all), 3.6),
                             subplot_kw={"projection": "polar"})
    for ax, N in zip(np.atleast_1d(axes), Ns_all):
        r = runs[N]
        for key, col, lab in (("pre_ang", "0.7", "pre"), ("end_ang", "0.3", "critical")):
            a = np.mod(r[key], np.pi)
            h, _ = np.histogram(a, bins=36, range=(0, np.pi))
            ax.plot(np.linspace(0, np.pi, 36), h, color=col, lw=1, label=lab)
        ax.set_title(f"N={N}", pad=12)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("paper Fig. 13 analogue (contact-normal directions)")
    fig.savefig(os.path.join(figdir, "fig13_contact_normals.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([cs[N]["N"] for N in Ns_all], [cs[N]["aniso"] for N in Ns_all], "o-")
    ax.set_xlabel("corner number N")
    ax.set_ylabel("fitted ellipse aspect ratio")
    ax.set_xscale("log")
    ax.set_title("paper Fig. 14 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig14_anisotropy.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 16: normal force distributions ----
    fig, axes = plt.subplots(1, len(Ns_all), figsize=(3 * len(Ns_all), 3.2),
                             sharey=True)
    fmax = max(float(np.max(runs[N]["end_fn"])) for N in Ns_all)
    for ax, N in zip(np.atleast_1d(axes), Ns_all):
        r = runs[N]
        ax.hist(r["pre_fn"], bins=32, range=(0, fmax), histtype="step",
                color="0.7", lw=1.2, density=True, label="pre")
        ax.hist(r["end_fn"], bins=32, range=(0, fmax), histtype="step",
                color="k", lw=1.2, density=True, label="critical")
        ax.set_title(f"N={N}")
        ax.set_xlabel("$F_N$ [N]")
    np.atleast_1d(axes)[0].set_ylabel("frequency")
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle("paper Fig. 16 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig16_force_hist.png"), dpi=150)
    plt.close(fig)

    # ---- Fig 17: median forces vs N ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    Nsc = PAPER["medFN_crit"]
    axes[0].plot([cs[N]["N"] for N in Ns_all], [cs[N]["medFN"] for N in Ns_all], "o-",
                 label="critical state")
    axes[0].axhspan(Nsc[0] - Nsc[1], Nsc[0] + Nsc[1], color="0.85",
                    label=f"paper {Nsc[0]:.0f}±{Nsc[1]:.0f} N")
    pre_fn = []
    for N in Ns_all:
        r = runs[N]
        pre_fn.append(float(np.median(r["pre_fn"])) if len(r["pre_fn"]) else np.nan)
    axes[0].plot([cs[N]["N"] for N in Ns_all], pre_fn, "s--", label="pre-shear")
    axes[0].set_xlabel("corner number N")
    axes[0].set_ylabel(r"median $F_N$ [N]")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[1].plot([cs[N]["N"] for N in Ns_all], [cs[N]["medFT"] for N in Ns_all], "o-",
                 label="critical state")
    axes[1].set_xlabel("corner number N")
    axes[1].set_ylabel(r"median $F_T$ [N]")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    fig.suptitle("paper Fig. 17 analogue")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig17_median_forces.png"), dpi=150)
    plt.close(fig)

    return popt, perr


def _spearman(x: dict) -> float:
    """Spearman rank correlation of values vs corner number (1 = increasing)."""
    ks = sorted(x)
    rx = np.argsort(np.argsort([x[k] for k in ks])).astype(float)
    ry = np.arange(len(ks), dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def verification_report(runs: dict[int, dict], cs: dict[int, dict],
                        popt, perr, path: str) -> str:
    lines = []
    add = lines.append
    add("VERIFICATION REPORT - reproduction of Krengel et al. (2023)")
    add("=" * 64)
    add("")
    add("Note: demo preset (w=0.08 m, ~260 particles; paper: 0.25 m, ~2500).")
    add("Values are expected to show the paper's TRENDS; exact constants need")
    add("paper-scale runs. Critical-state rows are conditioned on the lateral")
    add("servo tracking sigma_c (last column shows the tracked fraction).")
    add("")
    add(f"{'N':>4} {'eta_cs':>8} {'phi/mu':>8} {'n_cs':>7} {'Z_cs':>6} "
        f"{'anisot':>7} {'medFN':>8} {'medFT':>8} {'medRot[deg]':>11} {'servo%':>7}")
    for N in sorted(cs.keys()):
        c = cs[N]
        sf = c.get("servo_ok_fraction", 1.0)
        sf = 100.0 * sf if sf == sf else 0.0
        add(f"{N:>4} {c['eta']:>8.3f} {c['phi_mu']:>8.3f} {c['porosity']:>7.4f} "
            f"{c['Z']:>6.3f} {c['aniso']:>7.3f} {c['medFN']:>8.1f} {c['medFT']:>8.1f} "
            f"{np.degrees(c['medrot_end']):>11.2f} {sf:>6.0f}%")
    add("")
    add("CHECKS vs paper")
    add("-" * 64)
    a, b, c = popt
    pa, pb, pc = PAPER["eq7"]["a"], PAPER["eq7"]["b"], PAPER["eq7"]["c"]

    def check(name, ok, detail):
        add(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    check("Eq. (7) fit shape", b < 0 and a > 0,
          f"repro a={a:.3f} b={b:.3f} c={c:.3f} vs paper a={pa} b={pb} c={pc}")
    check("Eq. (7) limit c ~ 0.573-0.6", 0.4 < c < 0.8, f"c = {c:.3f}")
    phi_mu = {N: cs[N]["phi_mu"] for N in cs}
    if 5 in phi_mu:
        check("phi/mu(N=5) > 1", phi_mu[5] > 1.0, f"{phi_mu[5]:.3f}")
    Nmax = max(cs)
    Nmin = min(cs)
    check(f"phi/mu decreases with N (Spearman <= -0.8)",
          _spearman(phi_mu) <= -0.8, f"rho = {_spearman(phi_mu):.2f}")
    check("angular >> round shear strength (ratio > 1.5)",
          phi_mu[Nmin] / phi_mu[Nmax] > 1.5,
          f"{phi_mu[Nmin]:.3f} / {phi_mu[Nmax]:.3f} = {phi_mu[Nmin] / phi_mu[Nmax]:.2f}")
    n_cs = {N: cs[N]["porosity"] for N in cs}
    check("critical porosity decreases with N (Spearman <= -0.8)",
          _spearman(n_cs) <= -0.8, f"rho = {_spearman(n_cs):.2f}")
    if all(k in n_cs for k in (5, 6, 7)):
        hex_dip = n_cs[6] < n_cs[5] and n_cs[6] < n_cs[7]
        check("hexagon porosity anomaly (N=6 local dip)", hex_dip,
              f"n(6)={n_cs[6]:.4f} vs n(5)={n_cs[5]:.4f}, n(7)={n_cs[7]:.4f}")
    rot = {N: np.degrees(cs[N]["medrot_end"]) for N in cs}
    check("median rotation increases with N (Spearman >= 0.9)",
          _spearman(rot) >= 0.9, f"rho = {_spearman(rot):.2f}")
    check("round particles rotate > 2x angular ones",
          rot[Nmax] > 2.0 * rot[Nmin], f"{rot[Nmax]:.1f} vs {rot[Nmin]:.1f} deg")
    check(f"critical Z near paper limit {PAPER['Z_sat']}",
          abs(cs[Nmax]["Z"] - PAPER["Z_sat"]) < 0.4,
          f"Z(N={Nmax}) = {cs[Nmax]['Z']:.3f}")
    check("Z increases with N (saturates)", _spearman({N: cs[N]["Z"] for N in cs}) >= 0.8,
          f"Z {cs[Nmin]['Z']:.2f} -> {cs[Nmax]['Z']:.2f}")
    med = np.mean([cs[N]["medFN"] for N in cs])
    lo, hi = PAPER["medFN_crit"]
    check(f"median F_N in critical state ~ {lo:.0f} N", 0.5 * lo < med < 1.7 * lo,
          f"mean over N = {med:.1f} N")
    check("median F_N roughly N-independent (spread < 35%; paper: 3.5%)",
          (max(cs[N]["medFN"] for N in cs) / min(cs[N]["medFN"] for N in cs)) < 1.35,
          str({N: round(cs[N]["medFN"]) for N in sorted(cs)}))
    an = {N: cs[N]["aniso"] for N in cs}
    check("contact anisotropy decreases with N (Spearman <= -0.8)",
          _spearman(an) <= -0.8, f"rho = {_spearman(an):.2f}")
    report = "\n".join(lines)
    with open(path, "w") as f:
        f.write(report + "\n")
    return report


def _average_cs(cs_list: list[dict]) -> dict:
    """Average critical-state metrics over several samples (paper: 5 per N)."""
    out = dict(cs_list[0])
    for k, v in cs_list[0].items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals = [c[k] for c in cs_list if c[k] == c[k]]
            if k in ("N", "n_samples_rows"):
                continue
            if k == "phi_std":
                out[k] = float(np.sqrt(np.mean([c[k] ** 2 for c in cs_list])))
            elif k == "servo_ok_fraction":
                out[k] = float(np.mean(vals))
            else:
                out[k] = float(np.mean(vals))
    return out


def analyze_directory(resultdir: str, figdir: str | None = None,
                      report_path: str | None = None) -> str:
    import re

    runs = {}
    for fn in sorted(os.listdir(resultdir)):
        m = re.match(r"run_n(\d+)_s(\d+)\.npz$", fn)
        if not m:
            continue
        r = load_run(os.path.join(resultdir, fn))
        runs.setdefault(int(m.group(1)), []).append(r)
    if not runs:
        raise FileNotFoundError(f"no .npz runs in {resultdir}")
    cs = {N: _average_cs([critical_state(r) for r in rs]) for N, rs in runs.items()}
    first = {N: rs[0] for N, rs in runs.items()}
    figdir = figdir or os.path.join(os.path.dirname(resultdir), "figures")
    popt, perr = make_figures(first, cs, figdir)
    report_path = report_path or os.path.join(figdir, "..", "verification_report.txt")
    return verification_report(first, cs, popt, perr, report_path)
