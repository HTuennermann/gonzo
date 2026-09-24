"""Diagnostic: which stage of dem_step burns time? level: 0 broad, 1 +clip, 2 full."""

import numpy as np
from numba import njit

from angularity_dem.contact import (
    char_length,
    contact_frame,
    overlap_area_centroid,
    polygon_overlap,
)
from angularity_dem.engine import MAX_SLOTS, STALE, _hist_get, _hist_set


@njit(cache=True, fastmath=True)
def dem_step_level(
    pos, vel, th, om, frc, trq, vloc, nverts, wverts, mass, inert, rad,
    xl, xr, yb, yt, mu, Y, gamma, xiT, dt, grav,
    hist_part, hist_ft, hist_stp, step_now, ccount, level,
):
    n = pos.shape[0]
    K = vloc.shape[1]
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
        frc[i, 1] = -mass[i] * grav
        trq[i] = 0.0
        ccount[i] = 0

    rmax = rad[0]
    for i in range(n):
        if rad[i] > rmax:
            rmax = rad[i]
    cell = 2.0 * rmax * 1.000001
    xmin = 1e300; xmax = -1e300; ymin = 1e300; ymax = -1e300
    for i in range(n):
        if pos[i, 0] < xmin: xmin = pos[i, 0]
        if pos[i, 0] > xmax: xmax = pos[i, 0]
        if pos[i, 1] < ymin: ymin = pos[i, 1]
        if pos[i, 1] > ymax: ymax = pos[i, 1]
    ncx = int((xmax - xmin) / cell) + 1
    ncy = int((ymax - ymin) / cell) + 1
    while ncx * ncy > 4_000_000:
        cell *= 2.0
        ncx = int((xmax - xmin) / cell) + 1
        ncy = int((ymax - ymin) / cell) + 1
    head = np.full(ncx * ncy, -1, dtype=np.int64)
    nxt = np.empty(n, dtype=np.int64)
    for i in range(n):
        ci = int((pos[i, 0] - xmin) / cell)
        cj = int((pos[i, 1] - ymin) / cell)
        if ci < 0: ci = 0
        elif ci >= ncx: ci = ncx - 1
        if cj < 0: cj = 0
        elif cj >= ncy: cj = ncy - 1
        c = ci + ncx * cj
        nxt[i] = head[c]
        head[c] = i

    buf1 = np.empty((2 * K + 8, 2))
    buf2 = np.empty((2 * K + 8, 2))
    nc = 0

    for ci in range(ncx):
        for cj in range(ncy):
            c = ci + ncx * cj
            for offs in range(5):
                if offs == 0: dcx = 0; dcy = 0
                elif offs == 1: dcx = 1; dcy = 0
                elif offs == 2: dcx = 0; dcy = 1
                elif offs == 3: dcx = 1; dcy = 1
                else: dcx = -1; dcy = 1
                ni = ci + dcx; nj = cj + dcy
                if ni < 0 or ni >= ncx or nj < 0 or nj >= ncy:
                    continue
                c2 = ni + ncx * nj
                i = head[c]
                while i != -1:
                    j = head[c2]
                    while j != -1:
                        if c == c2 and j <= i:
                            j = nxt[j]; continue
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        rs = rad[i] + rad[j]
                        if dx * dx + dy * dy < rs * rs:
                            if level >= 1:
                                if i < j: a = i; b = j
                                else: a = j; b = i
                                m = polygon_overlap(wverts[a], nverts[a], wverts[b], nverts[b], buf1, buf2)
                                if m >= 3 and level >= 2:
                                    A, cx, cy = overlap_area_centroid(buf1, m)
                                    if A > 1e-20:
                                        ux, uy, wchord = contact_frame(buf1, m, cx, cy)
                                        nx_ = -uy; ny_ = ux
                                        dxb = pos[b, 0] - pos[a, 0]
                                        dyb = pos[b, 1] - pos[a, 1]
                                        if nx_ * dxb + ny_ * dyb < 0.0:
                                            nx_ = -nx_; ny_ = -ny_
                                        rax = cx - pos[a, 0]; ray = cy - pos[a, 1]
                                        rbx = cx - pos[b, 0]; rby = cy - pos[b, 1]
                                        l_ = char_length(rax, ray, rbx, rby)
                                        lmin = 0.1 * rad[a]
                                        if l_ < lmin: l_ = lmin
                                        FN = Y * A / l_
                                        vrx = vel[b, 0] - om[b] * rby - vel[a, 0] + om[a] * ray
                                        vry = vel[b, 1] + om[b] * rbx - vel[a, 1] - om[a] * rax
                                        vn = vrx * nx_ + vry * ny_
                                        adot = -vn * wchord
                                        mm = 1.0 / (1.0 / mass[a] + 1.0 / mass[b])
                                        FN += gamma * np.sqrt(mm * Y) * adot / l_
                                        if FN < 0.0: FN = 0.0
                                        tx_ = -ny_; ty_ = nx_
                                        vt = vrx * tx_ + vry * ty_
                                        kT = xiT * (Y * wchord / l_)
                                        FT = _hist_get(hist_part, hist_ft, hist_stp, a, b, step_now)
                                        FT -= kT * dt * vt
                                        ftm = mu * FN
                                        if FT > ftm: FT = ftm
                                        elif FT < -ftm: FT = -ftm
                                        _hist_set(hist_part, hist_ft, hist_stp, a, b, FT, step_now)
                                        Fx = FN * nx_ + FT * tx_
                                        Fy = FN * ny_ + FT * ty_
                                        frc[a, 0] -= Fx; frc[a, 1] -= Fy
                                        frc[b, 0] += Fx; frc[b, 1] += Fy
                                        trq[a] -= rax * Fy - ray * Fx
                                        trq[b] += rbx * Fy - rby * Fx
                                        ccount[a] += 1; ccount[b] += 1
                                        nc += 1
                        j = nxt[j]
                    i = nxt[i]

    for i in range(n):
        vel[i, 0] += frc[i, 0] / mass[i] * dt
        vel[i, 1] += frc[i, 1] / mass[i] * dt
        pos[i, 0] += vel[i, 0] * dt
        pos[i, 1] += vel[i, 1] * dt
        om[i] += trq[i] / inert[i] * dt
        th[i] += om[i] * dt
    return nc
