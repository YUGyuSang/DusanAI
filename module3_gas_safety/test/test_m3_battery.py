"""Battery model: drain, charging, voltage, dock detection."""
import pytest

from module3_gas_safety.battery_model import (STATUS_CHARGING, STATUS_DISCHARGING, STATUS_FULL,
                                              BatteryConfig, BatteryModel, DockDetector)


def test_idle_drain_matches_power_over_capacity():
    b = BatteryModel(BatteryConfig(capacity_wh=100.0, initial_pct=80.0, r_internal_ohm=1e-4))
    p = b.power_draw(0.0, 0.0, "stand")
    assert p == pytest.approx(25.0 + 15.0 + 45.0)
    for _ in range(3600):
        b.step(1.0, 0.0, 0.0, "stand")
    # 85 W for one hour from a 100 Wh pack = 85 % ... clamped at 0 from 80 %
    assert b.percentage == pytest.approx(0.0, abs=1e-9)
    b2 = BatteryModel(BatteryConfig(capacity_wh=100.0, initial_pct=80.0, r_internal_ohm=1e-4))
    for _ in range(600):
        b2.step(1.0)
    assert 80.0 - b2.percentage == pytest.approx(85.0 * 600 / 3600 / 100 * 100, rel=1e-3)


def test_speed_and_turning_increase_drain():
    a, b, c = BatteryModel(), BatteryModel(), BatteryModel()
    for _ in range(1000):
        a.step(0.1, 0.0, 0.0, "stand")
        b.step(0.1, 0.35, 0.0, "walk")
        c.step(0.1, 0.35, 0.8, "walk")
    assert a.percentage > b.percentage > c.percentage
    s = BatteryModel()
    for _ in range(1000):
        s.step(0.1, 0.0, 0.0, "sit")
    assert s.percentage > a.percentage


def test_voltage_and_current_conventions():
    b = BatteryModel()
    v_hi = b.ocv(0.9)
    v_lo = b.ocv(0.1)
    assert v_hi > v_lo and 40.0 < v_lo < v_hi < 60.0
    b.step(0.1, 0.3, 0.0, "walk")
    assert b.current_a < 0.0                         # discharging -> negative (ROS convention)
    assert b.voltage_v < b.ocv()                     # IR drop under load
    assert b.status == STATUS_DISCHARGING
    b.step(0.1, docked=True)
    assert b.current_a > 0.0 and b.status == STATUS_CHARGING


def test_charging_cc_cv_reaches_95_and_full():
    b = BatteryModel(BatteryConfig(initial_pct=15.0, charge_rate_pct_per_s=0.5, cv_threshold_pct=80.0))
    t = 0.0
    t95 = None
    while t < 2000.0 and b.status != STATUS_FULL:
        b.step(0.5, docked=True)
        t += 0.5
        if t95 is None and b.percentage >= 95.0:
            t95 = t
    # CC: 65 % at 0.5 %/s = 130 s, CV taper to 95 %: 0.2/0.005 ln(0.2/0.05) = 55 s
    assert t95 == pytest.approx(130.0 + 55.5, abs=3.0)
    assert b.status == STATUS_FULL
    rate_cc = BatteryModel(BatteryConfig(initial_pct=50.0)).charge_rate()
    rate_cv = BatteryModel(BatteryConfig(initial_pct=90.0)).charge_rate()
    assert rate_cv == pytest.approx(0.5 * rate_cc)


def test_set_level_clamps_and_time_scale():
    b = BatteryModel()
    b.set_level(150.0)
    assert b.percentage == 100.0
    b.set_level(-3.0)
    assert b.percentage == 0.0
    fast = BatteryModel(BatteryConfig(time_scale=10.0))
    slow = BatteryModel()
    fast.step(10.0)
    for _ in range(10):
        slow.step(10.0)
    assert fast.percentage == pytest.approx(slow.percentage, abs=0.05)


def test_energy_prediction():
    b = BatteryModel(BatteryConfig(capacity_wh=200.0))
    e = b.energy_for_distance(60.0, 0.3)
    p_walk = b.power_draw(0.3, 0.0, "walk")
    assert e == pytest.approx(p_walk * 200.0 / 3600.0)
    assert b.pct_for_energy(e) == pytest.approx(100.0 * e / 200.0)
    assert b.runtime_s(0.0) > b.runtime_s(0.35)


def test_dock_detector():
    d = DockDetector((61.9, 0.0), tolerance=0.35, speed_max=0.05, settle_s=1.0)
    assert not d.update(0.0, 60.0, 0.0, 0.0)                 # far away
    assert not d.update(1.0, 61.8, 0.1, 0.3)                 # inside but moving
    assert not d.update(2.0, 61.8, 0.1, 0.0)
    assert not d.update(2.5, 61.8, 0.1, 0.0)
    assert d.update(3.1, 61.8, 0.1, 0.0)                     # settled for > 1 s
    assert d.update(3.5, 61.8, 0.1, 0.2)                     # small motion on the pad keeps it docked
    assert not d.update(4.0, 62.5, 0.0, 0.0)                 # left the pad
