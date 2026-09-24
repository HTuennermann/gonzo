"""Polygon contact geometry (numba): clipping, overlap area/centroid, contact frame.

Implements the Matuttis-Chen style contact used in the paper:
- overlap region of two convex polygons (Sutherland-Hodgman clipping),
- elastic force F_N = Y A / l, Eq. (1),
- characteristic length l = 4|ra||rb| / (|ra|+|rb|), Eq. (2).
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True, inline="always")
def _clip_halfplane(pin, nin, ox, oy, ex, ey, pout):
    """Clip polygon (keep left of directed line o + t*e, i.e. cross(e, p-o) >= 0).

    Returns number of output vertices (SH algorithm).
    """
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
def polygon_overlap(Pa, na, Pb, nb, buf1, buf2):
    """Overlap polygon of convex CCW polygons Pa (na verts) and Pb (nb verts).

    Clips Pa against all half-planes of Pb (edges ordered far-to-near for early
    rejection). Result (m>=3 vertices) is copied into buf1; returns m (0 if empty).
    """
    # order Pb edges by decreasing distance of edge midpoint along (Pb.c - Pa.c)
    # approximated by projection of edge midpoint minus Pa centroid
    cax = 0.0
    cay = 0.0
    for k in range(na):
        cax += Pa[k, 0]
        cay += Pa[k, 1]
    cax /= na
    cay /= na
    cbx = 0.0
    cby = 0.0
    for k in range(nb):
        cbx += Pb[k, 0]
        cby += Pb[k, 1]
    cbx /= nb
    cby /= nb
    dx = cbx - cax
    dy = cby - cay

    # selection sort of edge indices by decreasing support (nb is small, <=64)
    supp = np.empty(nb)
    order = np.empty(nb, dtype=np.int64)
    for e in range(nb):
        j = e + 1
        if j >= nb:
            j = 0
        supp[e] = 0.5 * (Pb[e, 0] + Pb[j, 0]) * dx + 0.5 * (Pb[e, 1] + Pb[j, 1]) * dy
        order[e] = e
    for i in range(nb):
        imax = i
        for k in range(i + 1, nb):
            if supp[k] > supp[imax]:
                imax = k
        if imax != i:
            tmp = supp[i]
            supp[i] = supp[imax]
            supp[imax] = tmp
            it = order[i]
            order[i] = order[imax]
            order[imax] = it

    cur = buf1
    oth = buf2
    for k in range(na):
        cur[k, 0] = Pa[k, 0]
        cur[k, 1] = Pa[k, 1]
    n = na
    swapped = 0
    for s in range(nb):
        e = order[s]
        j = e + 1
        if j >= nb:
            j = 0
        ox = Pb[e, 0]
        oy = Pb[e, 1]
        ex = Pb[j, 0] - ox
        ey = Pb[j, 1] - oy
        n2 = _clip_halfplane(cur, n, ox, oy, ex, ey, oth)
        if n2 < 3:
            return 0
        n = n2
        swp = cur
        cur = oth
        oth = swp
        swapped += 1
    if swapped % 2 == 1:  # result ended in buf2 -> copy into buf1
        for k in range(n):
            buf1[k, 0] = cur[k, 0]
            buf1[k, 1] = cur[k, 1]
    return n


@njit(cache=True, fastmath=True, inline="always")
def overlap_area_centroid(P, m):
    """Shoelace area and centroid of CCW polygon P[0:m]. Returns (A, cx, cy)."""
    a2 = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(m):
        j = i + 1
        if j >= m:
            j = 0
        x1 = P[i, 0]
        y1 = P[i, 1]
        x2 = P[j, 0]
        y2 = P[j, 1]
        cr = x1 * y2 - x2 * y1
        a2 += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    a = 0.5 * a2
    if a <= 0.0:
        return 0.0, 0.0, 0.0
    return a, cx / (6.0 * a), cy / (6.0 * a)


@njit(cache=True, fastmath=True, inline="always")
def contact_frame(P, m, cx, cy):
    """Contact line direction u (unit) and chord width w from the overlap sliver.

    u = major principal axis of the vertex covariance (thin sliver -> contact
    line connecting the intersection points, Matuttis & Chen 2014); w = extent
    of the overlap along u.
    """
    ixx = 0.0
    iyy = 0.0
    ixy = 0.0
    for i in range(m):
        dx = P[i, 0] - cx
        dy = P[i, 1] - cy
        ixx += dx * dx
        iyy += dy * dy
        ixy += dx * dy
    th = 0.5 * np.arctan2(2.0 * ixy, ixx - iyy)  # major-axis direction
    ux = np.cos(th)
    uy = np.sin(th)
    pmin = 1e300
    pmax = -1e300
    for i in range(m):
        p = P[i, 0] * ux + P[i, 1] * uy
        if p < pmin:
            pmin = p
        if p > pmax:
            pmax = p
    return ux, uy, pmax - pmin


@njit(cache=True, fastmath=True, inline="always")
def char_length(rax, ray, rbx, rby):
    """Eq. (2): l = 4|ra||rb| / (|ra|+|rb|). For walls use |rb| = |ra|."""
    la = np.sqrt(rax * rax + ray * ray)
    lb = np.sqrt(rbx * rbx + rby * rby)
    return 4.0 * la * lb / (la + lb)
