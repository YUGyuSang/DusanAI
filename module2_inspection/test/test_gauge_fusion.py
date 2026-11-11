"""Temporal fusion of gauge readings (gauge_reader.GaugeFusion): weighted median + MAD outlier
rejection over the last N samples / T seconds."""
import numpy as np
import pytest

from module2_inspection.gauge_reader import GaugeFusion, weighted_median


def test_weighted_median():
    assert weighted_median([1.0, 2.0, 3.0], [1, 1, 1]) == 2.0
    assert weighted_median([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1]) == 2.5
    assert weighted_median([1.0, 2.0, 10.0], [1, 1, 5]) == 10.0      # heavy sample dominates
    assert weighted_median([5.0], [0.3]) == 5.0


def test_outliers_rejected_and_value_is_inlier_mean():
    """A 0..10 bar gauge at 6.2: noisy reads (0.3 % FS) + two gross misreads (needle tail / wrong
    end of the scale) -> the misreads are flagged and do not move the fused value."""
    f = GaugeFusion(window_n=15, window_s=10.0, mad_k=3.0)
    rng = np.random.default_rng(0)
    vals = list(6.2 + rng.normal(0, 0.03, 10))
    vals[3] = 9.8
    vals[7] = 0.4
    fr = None
    for k, v in enumerate(vals):
        fr = f.add("G01", 0.2 * k, v, 0.9, 10.0)
    assert fr.n_total == 10 and fr.outliers == 2 and fr.n_samples == 8
    assert abs(fr.value - 6.2) < 0.03
    assert 0.5 < fr.confidence <= 1.0


def test_confidence_weighting():
    f = GaugeFusion(window_n=10, window_s=10.0, mad_k=3.0, min_sigma_fs=0.05)
    f.add("G02", 0.0, 10.0, 0.9, 16.0)
    fr = f.add("G02", 0.1, 10.4, 0.1, 16.0)           # both inliers (5 % FS floor)
    assert fr.n_samples == 2
    assert fr.value == pytest.approx((10.0 * 0.9 + 10.4 * 0.1) / 1.0)


def test_window_by_time_and_count():
    f = GaugeFusion(window_n=5, window_s=2.0)
    for k in range(8):
        f.add("G03", float(k) * 0.1, float(k), 1.0, 100.0)
    assert f.fused("G03").n_total == 5                  # count window
    f2 = GaugeFusion(window_n=50, window_s=2.0)
    f2.add("G03", 0.0, 1.0, 1.0, 6.0)
    f2.add("G03", 1.0, 1.0, 1.0, 6.0)
    fr = f2.add("G03", 2.5, 2.0, 1.0, 6.0)             # the sample at t=0 is older than 2 s
    assert fr.n_total == 2
    assert f2.fused("G03", now=10.0) is None           # everything expired


def test_value_change_is_followed():
    """A real pressure change: once the new value dominates the window it becomes the median."""
    f = GaugeFusion(window_n=9, window_s=100.0)
    for k in range(9):
        f.add("G04", float(k), 18.5, 0.9, 25.0)
    for k in range(9, 15):
        fr = f.add("G04", float(k), 21.0, 0.9, 25.0)
    assert abs(fr.value - 21.0) < 1e-9 and fr.n_samples == 6 and fr.outliers == 3


def test_sim_reset_clears_history_and_ids_are_independent():
    f = GaugeFusion()
    for k in range(5):
        f.add("G05", 100.0 + k, 3.7, 0.9, 10.0)
        f.add("G06", 100.0 + k, 1.15, 0.9, 4.0)
    fr = f.add("G05", 3.0, 5.0, 0.9, 10.0)             # clock jumped backwards (Gazebo reset)
    assert fr.n_total == 1 and fr.value == 5.0
    assert f.fused("G06").n_total == 5 and abs(f.fused("G06").value - 1.15) < 1e-12
    f.reset("G06")
    assert f.fused("G06") is None
    f.reset()
    assert f.fused("G05") is None


def test_single_and_two_sample_behaviour():
    f = GaugeFusion()
    fr = f.add("G07", 0.0, 4.8, 0.8, 10.0)
    assert fr.n_samples == 1 and fr.value == 4.8 and fr.confidence < 0.8   # one sample: damped
    fr = f.add("G07", 0.2, 7.9, 0.8, 10.0)
    # two disagreeing samples: no majority -> the most confident, then most recent sample
    assert fr.n_samples == 1 and fr.outliers == 1 and fr.value == 7.9
    fr = f.add("G07", 0.4, 4.85, 0.8, 10.0)
    # a third sample agreeing with the first forms a majority: 4.8 / 4.85 win, 7.9 is an outlier
    assert fr.n_samples == 2 and fr.outliers == 1 and abs(fr.value - 4.825) < 1e-9
