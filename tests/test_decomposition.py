import masknmf
import numpy as np

from imgui_masknmf_manim.decomposition import lag1_correlation, variance_shares


def test_background_plus_signal_is_masknmfs_reconstruction(run, d):
    pmd = run.compression.pmd
    full = masknmf.CompressionArray.from_flyweight(pmd.shape, pmd.flyweight, rescale=True, include_trend=True)
    reconstruction = np.asarray(full[:])
    scale = float(np.abs(reconstruction).max())
    np.testing.assert_allclose(d.background + d.signal, reconstruction, atol=1e-4 * scale)


def test_residual_is_masknmfs_residual(run, d):
    pmd = run.compression.pmd
    residual = np.asarray(masknmf.CompressionResidualArray(d.registered, pmd)[:])
    np.testing.assert_allclose(d.residual, residual, atol=1e-4 * float(np.abs(d.registered).max()))


def test_pieces_add_up_to_the_registered_movie(d):
    np.testing.assert_allclose(d.background + d.signal + d.residual, d.registered, atol=1e-3)


def test_trend_has_zero_mean_over_time(d):
    if d.trend_temporal.shape[0] == 0:
        return
    trend = d.background - d.mean_image
    assert float(np.abs(trend.mean(axis=0)).max()) < 1e-3 * float(np.abs(d.registered).max())


def test_residual_is_closer_to_white_than_the_fluctuations(d):
    assert np.median(lag1_correlation(d.residual)) < 0.1 < np.median(lag1_correlation(d.fluctuations))


def test_variance_shares_are_fractions(d):
    shares = variance_shares(d)
    assert set(shares) == {"background", "signal", "residual"}
    assert all(0 <= v <= 1 for v in shares.values())


def test_stored_fraction_matches_what_u_and_v_hold(run, d):
    pmd = run.compression.pmd
    stored = pmd.spatial_compressed._nnz() + pmd.temporal_compressed.numel()
    assert d.stored_fraction == stored / d.registered.size
