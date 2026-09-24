"""Regular-polygon geometry: shapes, Fig. 1 metrics, mass properties.

Reference: Krengel, Chen & Kikumoto (2023), Comput. Geotech. 164, 105812,
Section 2.2 "Particle shape".
"""

from __future__ import annotations

import numpy as np


def regular_polygon(N: int, r: float, phase: float = 0.0) -> np.ndarray:
    """CCW vertices of a regular convex N-gon with circumradius r."""
    ang = phase + 2.0 * np.pi * np.arange(N) / N
    return np.column_stack((r * np.cos(ang), r * np.sin(ang)))


def polygon_area(verts: np.ndarray) -> float:
    """Signed shoelace area (CCW -> positive)."""
    x = verts[:, 0]
    y = verts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * float(np.sum(x * y2 - x2 * y))


def polygon_perimeter(verts: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(verts, -1, axis=0) - verts, axis=1)))


def polygon_centroid(verts: np.ndarray) -> tuple[float, float]:
    a = polygon_area(verts)
    x = verts[:, 0]
    y = verts[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cr = x * y2 - x2 * y
    cx = np.sum((x + x2) * cr) / (6.0 * a)
    cy = np.sum((y + y2) * cr) / (6.0 * a)
    return float(cx), float(cy)


def polygon_inertia(verts: np.ndarray, rho: float) -> float:
    """Mass moment of inertia (per unit thickness, 2D density rho) about the centroid."""
    a = polygon_area(verts)
    cx, cy = polygon_centroid(verts)
    v = verts - np.array([cx, cy])
    x = v[:, 0]
    y = v[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    cr = x * y2 - x2 * y
    # integral of (x^2+y^2) dA over polygon = (1/12) sum cr (p1.p1 + p1.p2 + p2.p2)
    second = np.sum(cr * (x * x + y * y + x * x2 + y * y2 + x2 * x2 + y2 * y2)) / 12.0
    m = rho * a
    return m * second / a  # rho * integral


def area_equivalent_diameter(N: int, r: float) -> float:
    """d_area = 2 sqrt(A_polygon / pi)  (paper Fig. 1)."""
    a = N / 2.0 * r**2 * np.sin(2.0 * np.pi / N)
    return 2.0 * np.sqrt(a / np.pi)


def circularity(N: int, r: float) -> float:
    """Wadell circularity = P_circle / P_polygon for equal-area circle (paper Fig. 1)."""
    a = N / 2.0 * r**2 * np.sin(2.0 * np.pi / N)
    p = 2.0 * N * r * np.sin(np.pi / N)
    return 2.0 * np.sqrt(np.pi * a) / p


def fig1_table(corners=(5, 6, 7, 10, 13, 17, 32, 64), r: float = 1.0) -> dict:
    """Reproduce the values behind paper Fig. 1 (unit circumradius)."""
    return {
        N: {
            "d_area_over_r": area_equivalent_diameter(N, r) / r,
            "circularity": circularity(N, r),
        }
        for N in corners
    }
