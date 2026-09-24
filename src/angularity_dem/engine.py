"""Numba DEM engine: cell-list broad phase, polygon contact forces (Eqs. 1-5),
Cundall-Strack tangential springs, servo walls, semi-implicit integration.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .contact import char_length, contact_frame, overlap_area_centroid, polygon_overlap

MAX_SLOTS = 16  # per-particle Cundall-Strack history slots
STALE = 3  # history entries older than this many steps are ignored


@njit(cache=True, fastmath=True, inline="always")
def _hist_get(part, fts, stps, owner, other, now):
    base = owner * MAX_SLOTS
    for s in range(MAX_SLOTS):
        if part[base + s] == other and now - stps[base + s] <= STALE:
            return fts[base + s]
    return 0.0


@njit(cache=True, fastmath=True, inline="always")
def _hist_set(part, fts, stps, owner, other, ft, now):
    base = owner * MAX_SLOTS
    empty = -1
    worst = 0
    worst_val = 1e300
    for s in range(MAX_SLOTS):
        p = part[base + s]
        if p == other:
            fts[base + s] = ft
            stps[base + s] = now
            return
        a = fts[base + s]
        if a < 0.0:
            a = -a
        if stps[base + s] < now - STALE:
            a = 0.0  # stale slots are free real estate
        if p == -1 and empty == -1:
            empty = s
        if a < worst_val:
            worst_val = a
            worst = s
    idx = empty if empty >= 0 else worst
    part[base + idx] = other
    fts[base + idx] = ft
    stps[base + idx] = now


@njit(cache=True, fastmath=True)
def dem_step(
    pos, vel, th, om, frc, trq,
    vloc, nverts, wverts,
    mass, inert, rad,
    xl, xr, yb, yt,
    mu, Y, gamma, xiT, dt, grav,
    hist_part, hist_ft, hist_stp, step_now,
    ccount,
    want_metrics, m_ang, m_fn, m_ft, m_count,
):
    """One DEM step. Returns (F_left, F_right, F_lid) wall reactions (inward > 0).

    Particles are convex CCW polygons; walls axis-aligned; yt=1e18 disables lid.
    Metrics arrays are filled when want_metrics != 0 (m_count[0] = n contacts).
    """
    n = pos.shape[0]
    K = vloc.shape[1]

    # world vertices
    for i in range(n):
        ct = np.cos(th[i])
        st = np.sin(th[i])
        nv = nverts[i]
        for k in range(nv):
            x = vloc[i, k, 0]
            y = vloc[i, k, 1]
            wverts[i, k, 0] = x * ct - y * st + pos[i, 0]
            wverts[i, k, 1] = x * st + y * ct + pos[i, 1]

    for i in range(n):
        frc[i, 0] = 0.0
        frc[i, 1] = -mass[i] * grav  # gravity acts toward -y (floor at y = 0)
        trq[i] = 0.0
        ccount[i] = 0

    # ---- broad phase: linked cell list ----
    rmax = rad[0]
    for i in range(n):
        if rad[i] > rmax:
            rmax = rad[i]
    cell = 2.0 * rmax * 1.000001
    xmin = 1e300
    xmax = -1e300
    ymin = 1e300
    ymax = -1e300
    for i in range(n):
        if pos[i, 0] < xmin:
            xmin = pos[i, 0]
        if pos[i, 0] > xmax:
            xmax = pos[i, 0]
        if pos[i, 1] < ymin:
            ymin = pos[i, 1]
        if pos[i, 1] > ymax:
            ymax = pos[i, 1]
    ncx = int((xmax - xmin) / cell) + 1
    ncy = int((ymax - ymin) / cell) + 1
    # safety cap: enlarge cells if the span is pathological (runaway particle)
    while ncx * ncy > 4_000_000:
        cell *= 2.0
        ncx = int((xmax - xmin) / cell) + 1
        ncy = int((ymax - ymin) / cell) + 1
    if ncx < 1:
        ncx = 1
    if ncy < 1:
        ncy = 1
    ncell = ncx * ncy
    head = np.full(ncell, -1, dtype=np.int64)
    nxt = np.empty(n, dtype=np.int64)
    for i in range(n):
        ci = int((pos[i, 0] - xmin) / cell)
        cj = int((pos[i, 1] - ymin) / cell)
        if ci < 0:
            ci = 0
        elif ci >= ncx:
            ci = ncx - 1
        if cj < 0:
            cj = 0
        elif cj >= ncy:
            cj = ncy - 1
        c = ci + ncx * cj
        nxt[i] = head[c]
        head[c] = i

    buf1 = np.empty((2 * K + 8, 2))
    buf2 = np.empty((2 * K + 8, 2))
    m_n = 0

    # half-stencil so every cell pair is visited once
    for ci in range(ncx):
        for cj in range(ncy):
            c = ci + ncx * cj
            for offs in range(5):
                if offs == 0:
                    dcx = 0
                    dcy = 0
                elif offs == 1:
                    dcx = 1
                    dcy = 0
                elif offs == 2:
                    dcx = 0
                    dcy = 1
                elif offs == 3:
                    dcx = 1
                    dcy = 1
                else:
                    dcx = -1
                    dcy = 1
                ni = ci + dcx
                nj = cj + dcy
                if ni < 0 or ni >= ncx or nj < 0 or nj >= ncy:
                    continue
                c2 = ni + ncx * nj
                i = head[c]
                while i != -1:
                    j = head[c2]
                    while j != -1:
                        if c == c2 and j <= i:
                            j = nxt[j]
                            continue
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        rs = rad[i] + rad[j]
                        if dx * dx + dy * dy < rs * rs:
                            # canonical order (a < b) for stable contact frame/history
                            if i < j:
                                a = i
                                b = j
                            else:
                                a = j
                                b = i
                            m = polygon_overlap(
                                wverts[a], nverts[a], wverts[b], nverts[b], buf1, buf2
                            )
                            if m >= 3:
                                A, cx, cy = overlap_area_centroid(buf1, m)
                                if A > 1e-20:
                                    ux, uy, wchord = contact_frame(buf1, m, cx, cy)
                                    nx_ = -uy
                                    ny_ = ux
                                    dxb = pos[b, 0] - pos[a, 0]
                                    dyb = pos[b, 1] - pos[a, 1]
                                    if nx_ * dxb + ny_ * dyb < 0.0:
                                        nx_ = -nx_
                                        ny_ = -ny_
                                    rax = cx - pos[a, 0]
                                    ray = cy - pos[a, 1]
                                    rbx = cx - pos[b, 0]
                                    rby = cy - pos[b, 1]
                                    l_ = char_length(rax, ray, rbx, rby)
                                    lmin = 0.1 * rad[a]  # deep-penetration safety
                                    if l_ < lmin:
                                        l_ = lmin
                                    FN = Y * A / l_
                                    # relative velocity at contact (b w.r.t. a)
                                    vrx = (
                                        vel[b, 0] - om[b] * rby - vel[a, 0] + om[a] * ray
                                    )
                                    vry = (
                                        vel[b, 1] + om[b] * rbx - vel[a, 1] - om[a] * rax
                                    )
                                    vn = vrx * nx_ + vry * ny_
                                    adot = -vn * wchord
                                    mm = 1.0 / (1.0 / mass[a] + 1.0 / mass[b])
                                    FN += gamma * np.sqrt(mm * Y) * adot / l_
                                    if FN < 0.0:
                                        FN = 0.0
                                    tx_ = -ny_
                                    ty_ = nx_
                                    vt = vrx * tx_ + vry * ty_
                                    kT = xiT * (Y * wchord / l_)
                                    FT = _hist_get(hist_part, hist_ft, hist_stp, a, b, step_now)
                                    FT -= kT * dt * vt
                                    ftm = mu * FN
                                    if FT > ftm:
                                        FT = ftm
                                    elif FT < -ftm:
                                        FT = -ftm
                                    _hist_set(hist_part, hist_ft, hist_stp, a, b, FT, step_now)
                                    Fx = FN * nx_ + FT * tx_
                                    Fy = FN * ny_ + FT * ty_
                                    frc[a, 0] -= Fx
                                    frc[a, 1] -= Fy
                                    frc[b, 0] += Fx
                                    frc[b, 1] += Fy
                                    trq[a] -= rax * Fy - ray * Fx
                                    trq[b] += rbx * Fy - rby * Fx
                                    ccount[a] += 1
                                    ccount[b] += 1
                                    if want_metrics != 0:
                                        m_ang[m_n] = np.arctan2(ny_, nx_)
                                        m_fn[m_n] = FN
                                        m_ft[m_n] = FT
                                        m_n += 1
                        j = nxt[j]
                    i = nxt[i]

    # ---- wall contacts ----
    fl = 0.0
    fr = 0.0
    flid = 0.0
    for i in range(n):
        nv = nverts[i]
        # left wall: keep x <= xl, normal on particle (+1, 0)
        if pos[i, 0] - rad[i] < xl:
            fl += _wall_force(
                wverts[i], nv, pos, vel, om, frc, trq, mass, rad, i,
                xl, 0.0, 0.0, 1.0, 1.0, 0.0, Y, gamma, dt, buf1, buf2
            )
        # right wall: keep x >= xr, normal (-1, 0)
        if pos[i, 0] + rad[i] > xr:
            fr += _wall_force(
                wverts[i], nv, pos, vel, om, frc, trq, mass, rad, i,
                xr, 0.0, 0.0, -1.0, -1.0, 0.0, Y, gamma, dt, buf1, buf2
            )
        # floor: keep y <= yb, normal (0, +1)
        if pos[i, 1] - rad[i] < yb:
            _wall_force(
                wverts[i], nv, pos, vel, om, frc, trq, mass, rad, i,
                0.0, yb, -1.0, 0.0, 0.0, 1.0, Y, gamma, dt, buf1, buf2
            )
        # lid: keep y >= yt, normal (0, -1)
        if pos[i, 1] + rad[i] > yt:
            flid += _wall_force(
                wverts[i], nv, pos, vel, om, frc, trq, mass, rad, i,
                0.0, yt, 1.0, 0.0, 0.0, -1.0, Y, gamma, dt, buf1, buf2
            )

    # ---- integrate (semi-implicit Euler) ----
    for i in range(n):
        vel[i, 0] += frc[i, 0] / mass[i] * dt
        vel[i, 1] += frc[i, 1] / mass[i] * dt
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        om[i] += trq[i] / inert[i] * dt
        th[i] += om[i] * dt

    m_count[0] = m_n
    return fl, fr, flid


@njit(cache=True, fastmath=True, inline="always")
def _wall_force(
    wv, nv, pos, vel, om, frc, trq, mass, rad, i,
    ox, oy, ex, ey, nx_, ny_, Y, gamma, dt, buf1, buf2,
):
    """Clip particle i against wall half-plane, apply normal force, return FN.

    Wall friction is zero (paper: particle-wall friction set to zero).
    """
    m = _clip_poly(wv, nv, ox, oy, ex, ey, buf1)
    if m < 3:
        return 0.0
    A, cx, cy = overlap_area_centroid(buf1, m)
    if A <= 1e-20:
        return 0.0
    rax = cx - pos[i, 0]
    ray = cy - pos[i, 1]
    l_ = char_length(rax, ray, rax, ray)  # |rb| = |ra| for walls -> l = 2|ra|
    lmin = 0.1 * rad[i]  # deep-penetration safety bound
    if l_ < lmin:
        l_ = lmin
    FN = Y * A / l_
    # contact-point velocity of particle; wall is static
    vcx = vel[i, 0] - om[i] * ray
    vcy = vel[i, 1] + om[i] * rax
    vn = vcx * nx_ + vcy * ny_  # < 0 approaching
    # chord width along the wall
    pmin = 1e300
    pmax = -1e300
    for k in range(m):
        p = buf1[k, 0] * (-ny_) + buf1[k, 1] * nx_  # along-wall coordinate
        if p < pmin:
            pmin = p
        if p > pmax:
            pmax = p
    wchord = pmax - pmin
    adot = -vn * wchord
    FN += gamma * np.sqrt(mass[i] * Y) * adot / l_
    if FN < 0.0:
        FN = 0.0
    frc[i, 0] += FN * nx_
    frc[i, 1] += FN * ny_
    trq[i] += rax * (FN * ny_) - ray * (FN * nx_)
    return FN


@njit(cache=True, fastmath=True, inline="always")
def _clip_poly(pin, nin, ox, oy, ex, ey, pout):
    """Clip polygon against one half-plane (keep cross(e, p-o) >= 0)."""
    nout = 0
    for i in range(nin):
        x1 = pin[i, 0]
        y1 = pin[i, 1]
        j = i + 1
        if j >= nin:
            j = 0
        x2 = pin[j, 0]
        y2 = pin[j, 1]
        d1 = ex * (y1 - oy) - ey * (x1 - ox)
        d2 = ex * (y2 - oy) - ey * (x2 - ox)
        if d1 >= 0.0:
            pout[nout, 0] = x1
            pout[nout, 1] = y1
            nout += 1
        if (d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0):
            t = d1 / (d1 - d2)
            pout[nout, 0] = x1 + t * (x2 - x1)
            pout[nout, 1] = y1 + t * (y2 - y1)
            nout += 1
    return nout


@njit(cache=True, fastmath=True)
def particle_metrics(ccount, th, th0):
    """Z (excluding particles with <2 contacts), median |theta-theta0|, MAD."""
    n = ccount.shape[0]
    nload = 0
    ssum = 0
    for i in range(n):
        if ccount[i] >= 2:
            nload += 1
            ssum += ccount[i]
    Z = ssum / nload if nload > 0 else 0.0

    rot = np.empty(n)
    for i in range(n):
        rot[i] = abs(th[i] - th0[i])
    # median
    s = np.sort(rot)
    if n % 2 == 1:
        med = s[n // 2]
    else:
        med = 0.5 * (s[n // 2 - 1] + s[n // 2])
    dev = np.empty(n)
    for i in range(n):
        dev[i] = abs(rot[i] - med)
    s2 = np.sort(dev)
    if n % 2 == 1:
        mad = s2[n // 2]
    else:
        mad = 0.5 * (s2[n // 2 - 1] + s2[n // 2])
    return Z, med, mad


@njit(cache=True, fastmath=True)
def contact_medians(m_fn, m_ft, m_count):
    """Median FN and FT over current contacts."""
    if m_count == 0:
        return 0.0, 0.0
    fn = np.sort(m_fn[:m_count])
    ft = np.sort(m_ft[:m_count])
    h = m_count // 2
    if m_count % 2 == 1:
        return fn[h], ft[h]
    return 0.5 * (fn[h - 1] + fn[h]), 0.5 * (ft[h - 1] + ft[h])


@njit(cache=True, fastmath=True)
def fabric_aniso(m_ang, m_count):
    """Contact-normal anisotropy = fitted-ellipse aspect ratio (paper Fig. 14).

    Least-squares fit of f(t) = a + b cos2t + c sin2t to unit contact masses is
    equivalent to the 2nd-order fabric tensor: with Phi = (1/m) sum n (x) n,
    r = 2 sqrt(((Pxx-Pyy)/2)^2 + Pxy^2), aspect ratio = (1+r)/(1-r).
    Contacts are orientation lines, so theta is folded mod pi.
    """
    if m_count == 0:
        return 1.0
    fxx = 0.0
    fyy = 0.0
    fxy = 0.0
    for k in range(m_count):
        t = m_ang[k]
        if t < 0.0:
            t += np.pi
        if t >= np.pi:
            t -= np.pi
        c = np.cos(t)
        s = np.sin(t)
        fxx += c * c
        fyy += s * s
        fxy += c * s
    m = float(m_count)
    fxx /= m
    fyy /= m
    fxy /= m
    r = 2.0 * np.sqrt(0.25 * (fxx - fyy) ** 2 + fxy * fxy)
    if r > 0.999:
        return 1000.0
    return (1.0 + r) / (1.0 - r)
