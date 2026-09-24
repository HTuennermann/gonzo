import numpy as np
import pytest
from numba import njit

from angularity_dem.contact import (
    char_length,
    contact_frame,
    overlap_area_centroid,
    polygon_overlap,
)
from angularity_dem.geometry import regular_polygon

shapely = pytest.importorskip("shapely")


def _overlap_py(pa, pb):
    a = shapely.Polygon(pa)
    b = shapely.Polygon(pb)
    inter = a.intersection(b)
    assert not inter.is_empty
    c = inter.centroid
    return inter.area, c.x, c.y


def _run_overlap(pa, pb):
    bufs = [np.zeros((len(pa) + len(pb) + 8, 2)) for _ in range(2)]
    ra = float(np.max(np.linalg.norm(pa - pa.mean(axis=0), axis=1)))
    rb = float(np.max(np.linalg.norm(pb - pb.mean(axis=0), axis=1)))
    m = polygon_overlap(np.ascontiguousarray(pa), len(pa),
                        np.ascontiguousarray(pb), len(pb), bufs[0], bufs[1], ra, rb)
    assert m >= 3
    return overlap_area_centroid(bufs[0], m)


def test_overlap_matches_shapely_random():
    rng = np.random.default_rng(42)
    for _ in range(60):
        Na = rng.integers(3, 12)
        Nb = rng.integers(3, 12)
        ra = 0.5 + rng.random()
        rb = 0.5 + rng.random()
        pa = regular_polygon(Na, ra, rng.random() * 7) + rng.random(2)
        pb = regular_polygon(Nb, rb, rng.random() * 7) + rng.random(2)
        # place b near a so they overlap
        pb = pb + (pa.mean(axis=0) - pb.mean(axis=0)) + (rng.random(2) - 0.5) * 0.6
        A1, x1, y1 = _run_overlap(pa, pb)
        A2, x2, y2 = _overlap_py(pa, pb)
        assert A1 == pytest.approx(A2, rel=1e-9, abs=1e-12)
        assert x1 == pytest.approx(x2, abs=1e-9)
        assert y1 == pytest.approx(y2, abs=1e-9)


def test_overlap_symmetric_and_exact_square():
    # two unit squares offset by delta -> overlap delta x 1 exactly
    s1 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    d = 0.137
    s2 = s1 + [d, 0.0]
    A1, _, _ = _run_overlap(s1, s2)
    assert A1 == pytest.approx(1.0 - d, rel=1e-12)
    A2, _, _ = _run_overlap(s2, s1)
    assert A1 == pytest.approx(A2, rel=1e-12)  # clipping is symmetric


def test_no_overlap_separated():
    p1 = regular_polygon(5, 1.0)
    p2 = regular_polygon(5, 1.0) + [5.0, 5.0]
    bufs = [np.zeros((24, 2)) for _ in range(2)]
    m = polygon_overlap(p1, 5, p2, 5, bufs[0], bufs[1], 1.0, 1.0)
    assert m == 0


def test_contact_frame_sliver():
    # thin sliver: contact line should be ~horizontal (x), chord ~ 1
    p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1e-4], [0.0, 1e-4]])
    A, cx, cy = overlap_area_centroid(p, 4)
    ux, uy, w = contact_frame(p, 4, cx, cy)
    assert abs(uy) < 0.01
    assert w == pytest.approx(1.0, abs=1e-6)


def test_char_length():
    # equal arms -> 2r; wall case (rb = ra) -> 2r as well
    assert char_length(1.0, 0.0, 1.0, 0.0) == pytest.approx(2.0)
    # formula check 4*2*1/(2+1) = 8/3
    assert char_length(2.0, 0.0, 1.0, 0.0) == pytest.approx(8.0 / 3.0)
