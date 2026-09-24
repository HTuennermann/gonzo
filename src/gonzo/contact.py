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
def polygon_overlap(Pa, na, Pb, nb, buf1, buf2, ra, rb):
    """Overlap polygon of convex CCW regular polygons Pa (na) and Pb (nb).

    ra, rb are the circumradii. When the circumcircles properly intersect
    (thin-contact regime), only Pb's edges whose lines can cut the circle lens
    are visited: normals within alpha_b + pi/nb of the near direction, where
    cos(alpha_b) = (d^2 + rb^2 - ra^2) / (2 d rb). This window is exact for
    regular polygons (edge lines are incircle tangents at uniform spacing);
    every other edge's half-plane contains all of Pa, so skipping them cannot
    change the result. Deep overlaps fall back to a full far-to-near ordered
    walk with early exit.

    Result (m>=3 vertices) is copied into buf1; returns m (0 if empty).
    """
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
    dx = cax - cbx  # near direction (from b towards a)
    dy = cay - cby
    dist = np.sqrt(dx * dx + dy * dy)

    dth = 2.0 * np.pi / nb
    base = np.arctan2(Pb[0, 1] - cby, Pb[0, 0] - cbx)

    use_window = (dist > 1e-12) and (abs(ra - rb) < dist) and (dist < ra + rb)
    if use_window:
        cosab = (dist * dist + rb * rb - ra * ra) / (2.0 * dist * rb)
        if cosab > 1.0:
            cosab = 1.0
        elif cosab < -1.0:
            cosab = -1.0
        alpha = np.arccos(cosab)
        # near-face edge: outward normal nearest to (ca - cb)
        e0 = int(np.round((np.arctan2(dy, dx) - base) / dth - 0.5)) % nb
        w = int(np.ceil(alpha / dth + 0.5))
        if w > nb:
            w = nb  # deep overlap: revisit is idempotent, just cap the array
        edges = np.empty(2 * w + 1, dtype=np.int64)
        ne = 0
        edges[ne] = e0
        ne += 1
        for k in range(1, w + 1):
            edges[ne] = (e0 + k) % nb
            ne += 1
            edges[ne] = (e0 - k) % nb
            ne += 1
    else:
        # full walk, far-to-near (support-ordered) for early exit
        dx = -dx
        dy = -dy
        e0 = int(np.round((np.arctan2(dy, dx) - base) / dth - 0.5)) % nb
        edges = np.empty(nb, dtype=np.int64)
        ne = 0
        ii = (e0 + 1) % nb
        jj = (e0 - 1) % nb
        left = nb
        while left > 0:
            if left == nb:
                e = e0
            else:
                j = ii + 1
                if j >= nb:
                    j = 0
                si = (Pb[ii, 0] + Pb[j, 0]) * dx + (Pb[ii, 1] + Pb[j, 1]) * dy
                jm = jj - 1
                if jm < 0:
                    jm = nb - 1
                sj = (Pb[jj, 0] + Pb[jm, 0]) * dx + (Pb[jj, 1] + Pb[jm, 1]) * dy
                if si >= sj:
                    e = ii
                    ii = j
                else:
                    e = jj
                    jj = jm
            left -= 1
            edges[ne] = e
            ne += 1

    cur = buf1
    oth = buf2
    for k in range(na):
        cur[k, 0] = Pa[k, 0]
        cur[k, 1] = Pa[k, 1]
    n = na
    swapped = 0
    for s in range(ne):
        e = edges[s]
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
