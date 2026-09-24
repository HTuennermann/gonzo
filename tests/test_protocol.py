import numpy as np
import pytest

from gonzo.protocol import Params, run_simulation


@pytest.mark.slow
def test_tiny_full_protocol():
    """Full protocol on a micro system: completes, stresses sane, files saved."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "run.npz")
        P = Params.demo(
            corners=6,
            width=0.03,
            aspect=1.2,
            solid_target=0.75,
            eps_end=0.05,
            settle_time=0.15,
            settle_max=0.25,
            seed=7,
            out=out,
        )
        res = run_simulation(P, verbose=True)
        d = np.load(out)
        assert "params_json" in d.files
        # consolidation brought lateral stress near sigma_c (within 15%)
        s3_end = res["s3"][-1]
        assert abs(s3_end / P.sigma_c - 1) < 0.15
        # shearing: axial stress at least sigma_c (eta >= ~1 in this window)
        assert res["s1"].max() > P.sigma_c
        assert res["eps"][-1] >= P.eps_end - 1e-3
        # porosity physically sensible
        assert 0.1 < np.nanmean(res["porosity"][-5:]) < 0.4
