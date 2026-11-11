"""
Battery state-of-charge model for the quadruped (Module 3, ROS free).

Power demand
------------
    P = P_compute + P_sensors + P_posture(mode) + k_v |v| + k_w |w_z|

* ``P_posture``: standing (servo holding torque) ``p_stand_w``, sitting /
  e-stop ``p_sit_w``;
* ``k_v |v|``: locomotion power proportional to the body speed
  (cost of transport of a ~15 kg legged robot);
* ``k_w |w_z|``: turning on the spot.

Electrical model
----------------
Open-circuit voltage OCV(SoC) of an ``cells_series`` Li-ion pack (tabulated
per-cell curve) with an internal resistance R.  For a load P the pack current
solves P = (OCV - I R) I:

    I = (OCV - sqrt(OCV^2 - 4 R P)) / (2 R),   V_term = OCV - I R

and  dSoC/dt = -I V_term / (3600 E_cap)  (E_cap in Wh, i.e. energy based).

Charging (on the dock): constant-current phase at ``charge_rate_pct_per_s``
up to ``cv_threshold_pct``, then a linear constant-voltage taper down to
``taper_min_frac`` of the rate at 100 %.  The charger also feeds the robot's
own loads, so the SoC rate is the net charge rate.

``time_scale`` > 1 accelerates the drain for demos (documented, default 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Li-ion cell open-circuit voltage vs. state of charge (typical NMC curve)
_OCV_SOC = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00])
_OCV_V = np.array([3.00, 3.30, 3.45, 3.57, 3.63, 3.68, 3.73, 3.79, 3.87, 3.95, 4.05, 4.20])

# sensor_msgs/BatteryState power_supply_status constants (kept here so the
# ROS-free model and the node agree without importing ROS)
STATUS_UNKNOWN, STATUS_CHARGING, STATUS_DISCHARGING, STATUS_NOT_CHARGING, STATUS_FULL = 0, 1, 2, 3, 4
STATUS_NAMES = {0: "unknown", 1: "charging", 2: "discharging", 3: "not_charging", 4: "full"}


@dataclass
class BatteryConfig:
    capacity_wh: float = 200.0
    initial_pct: float = 80.0
    cells_series: int = 14
    r_internal_ohm: float = 0.08
    p_compute_w: float = 25.0
    p_sensors_w: float = 15.0
    p_stand_w: float = 45.0
    p_sit_w: float = 8.0
    p_per_mps_w: float = 110.0
    p_per_radps_w: float = 20.0
    charge_rate_pct_per_s: float = 0.5
    cv_threshold_pct: float = 80.0
    taper_min_frac: float = 0.1
    full_pct: float = 99.5
    time_scale: float = 1.0


class BatteryModel:
    def __init__(self, cfg: Optional[BatteryConfig] = None):
        self.cfg = cfg or BatteryConfig()
        self.soc = float(np.clip(self.cfg.initial_pct / 100.0, 0.0, 1.0))
        self.power_w = 0.0
        self.current_a = 0.0
        self.voltage_v = self.ocv()
        self.charging = False
        self.energy_used_wh = 0.0

    # ----------------------------------------------------------- electrical
    def ocv(self, soc: Optional[float] = None) -> float:
        s = self.soc if soc is None else soc
        return float(np.interp(np.clip(s, 0.0, 1.0), _OCV_SOC, _OCV_V)) * self.cfg.cells_series

    @property
    def percentage(self) -> float:
        return 100.0 * self.soc

    def set_level(self, pct: float) -> None:
        self.soc = float(np.clip(pct / 100.0, 0.0, 1.0))
        self.voltage_v = self.ocv()

    def power_draw(self, speed: float = 0.0, yaw_rate: float = 0.0, mode: str = "stand") -> float:
        """Electrical load [W] for a body speed [m/s], yaw rate [rad/s] and
        posture mode ('walk'/'stand' -> standing load, 'sit'/'estop'/'fallen'
        -> sitting load)."""
        c = self.cfg
        m = (mode or "stand").lower()
        posture = c.p_sit_w if m in ("sit", "estop", "fallen", "off") else c.p_stand_w
        return (c.p_compute_w + c.p_sensors_w + posture
                + c.p_per_mps_w * abs(speed) + c.p_per_radps_w * abs(yaw_rate))

    def _solve_current(self, power: float) -> tuple:
        """Discharge current / terminal voltage for a load (quadratic in I)."""
        v0 = self.ocv()
        r = max(self.cfg.r_internal_ohm, 1e-6)
        disc = v0 * v0 - 4.0 * r * power
        if disc <= 0.0:   # load beyond the maximum power point: clamp
            i = v0 / (2.0 * r)
        else:
            i = (v0 - math.sqrt(disc)) / (2.0 * r)
        return i, v0 - i * r

    # -------------------------------------------------------------- update
    def charge_rate(self) -> float:
        """Net SoC rate [1/s] while docked (CC then linear CV taper)."""
        c = self.cfg
        r = c.charge_rate_pct_per_s / 100.0
        p = self.percentage
        if p >= c.cv_threshold_pct:
            f = 1.0 - (p - c.cv_threshold_pct) / max(100.0 - c.cv_threshold_pct, 1e-6)
            r *= max(c.taper_min_frac, f)
        return r

    def step(self, dt: float, speed: float = 0.0, yaw_rate: float = 0.0, mode: str = "stand",
             docked: bool = False) -> None:
        """Advance the model by dt seconds (sim time)."""
        if dt <= 0.0:
            return
        dt_eff = dt * max(self.cfg.time_scale, 0.0)
        self.charging = bool(docked)
        if docked:
            if self.percentage < self.cfg.full_pct:
                rate = self.charge_rate()
                self.soc = min(1.0, self.soc + rate * dt_eff)
                # charger current into the pack (positive = charging)
                p_in = rate * 3600.0 * self.cfg.capacity_wh
                self.voltage_v = self.ocv() + 0.5 * self.cfg.r_internal_ohm * p_in / max(self.ocv(), 1.0)
                self.current_a = p_in / max(self.voltage_v, 1.0)
            else:
                self.voltage_v = self.ocv()
                self.current_a = 0.0
            self.power_w = 0.0
            return
        p = self.power_draw(speed, yaw_rate, mode)
        i, v = self._solve_current(p)
        self.power_w = p
        self.current_a = -i               # ROS convention: negative while discharging
        self.voltage_v = v
        e = i * v * dt_eff / 3600.0       # Wh drawn from the pack
        self.energy_used_wh += e
        self.soc = max(0.0, self.soc - e / max(self.cfg.capacity_wh, 1e-6))

    @property
    def status(self) -> int:
        if self.charging:
            return STATUS_FULL if self.percentage >= self.cfg.full_pct else STATUS_CHARGING
        return STATUS_DISCHARGING

    # ------------------------------------------------------- predictions
    def energy_for_distance(self, distance_m: float, speed: float, overhead_s: float = 0.0) -> float:
        """Energy [Wh] to walk ``distance_m`` at ``speed`` (+ ``overhead_s`` of
        standing, e.g. docking manoeuvres)."""
        speed = max(speed, 1e-3)
        t = distance_m / speed
        return (self.power_draw(speed, 0.0, "walk") * t + self.power_draw(0.0, 0.0, "stand") * overhead_s) / 3600.0

    def pct_for_energy(self, wh: float) -> float:
        return 100.0 * wh / max(self.cfg.capacity_wh, 1e-6)

    def runtime_s(self, speed: float = 0.0, mode: str = "stand") -> float:
        """Remaining runtime at a constant load (ignores voltage sag)."""
        p = max(self.power_draw(speed, 0.0, mode), 1e-6)
        return self.soc * self.cfg.capacity_wh * 3600.0 / p


class DockDetector:
    """'Docked' = robot within ``tolerance`` of the dock and (almost) still
    for ``settle_s`` seconds.  Leaving the tolerance radius undocks at once."""

    def __init__(self, dock_xy, tolerance: float = 0.35, speed_max: float = 0.05, settle_s: float = 1.0):
        self.dock_xy = (float(dock_xy[0]), float(dock_xy[1]))
        self.tolerance = float(tolerance)
        self.speed_max = float(speed_max)
        self.settle_s = float(settle_s)
        self._since: Optional[float] = None
        self.docked = False

    def distance(self, x: float, y: float) -> float:
        return math.hypot(x - self.dock_xy[0], y - self.dock_xy[1])

    def update(self, t: float, x: float, y: float, speed: float) -> bool:
        if self.distance(x, y) > self.tolerance:
            self._since = None
            self.docked = False
            return False
        if abs(speed) <= self.speed_max:
            if self._since is None:
                self._since = t
            if t - self._since >= self.settle_s:
                self.docked = True
        elif not self.docked:
            self._since = None
        return self.docked
