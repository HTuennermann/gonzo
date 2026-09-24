import numpy as np
import pytest

from gonzo.geometry import (
    area_equivalent_diameter,
    circularity,
    fig1_table,
    polygon_area,
    polygon_centroid,
    polygon_inertia,
    polygon_perimeter,
    regular_polygon,
)


def test_regular_polygon_closed_form():
    for N, r in [(5, 2.5e-3), (6, 5e-3), (13, 1.0), (64, 1.0)]:
        v = regular_polygon(N, r)
        assert polygon_area(v) == pytest.approx(
            N / 2 * r**2 * np.sin(2 * np.pi / N), rel=1e-12
        )
        assert polygon_perimeter(v) == pytest.approx(
            2 * N * r * np.sin(np.pi / N), rel=1e-12
        )
        cx, cy = polygon_centroid(v)
        assert abs(cx) < 1e-12 and abs(cy) < 1e-12


def test_fig1_values_paper():
    """Closed-form values read off paper Fig. 1 (unit circumradius)."""
    assert area_equivalent_diameter(5, 1.0) == pytest.approx(1.739916, abs=1e-5)
    assert circularity(5, 1.0) == pytest.approx(0.929950, abs=1e-5)
    assert circularity(64, 1.0) == pytest.approx(0.999598, abs=1e-5)
    tab = fig1_table()
    circs = [tab[N]["circularity"] for N in (5, 6, 7, 10, 13, 17, 32, 64)]
    assert circs == sorted(circs)  # monotone increase with corner number
    ds = [tab[N]["d_area_over_r"] for N in (5, 6, 7, 10, 13, 17, 32, 64)]
    assert ds == sorted(ds)
    assert ds[-1] == pytest.approx(2.0, abs=2e-3)  # -> circle of radius r


def test_inertia_positive_and_scaling():
    v = regular_polygon(7, 2.0)
    i1 = polygon_inertia(v, 2830.0)
    v2 = regular_polygon(7, 4.0)  # double size -> 16x mass, 4x length scale
    i2 = polygon_inertia(v2, 2830.0)
    assert i2 / i1 == pytest.approx(16.0, rel=1e-9)  # I ~ rho r^4
    assert i1 > 0
