"""
ROS parameter declarations of the Module-3 nodes (ROS free).

Every node declares exactly the parameters listed here (name -> default; the
default fixes the ROS parameter type) and the YAML files in ``config/`` set
the same keys.  ``test/test_m3_config.py`` checks names and types so a typo or an
int-for-double mistake (which rclpy rejects at start-up) is caught on Windows.

Sentinel defaults meaning "take the value from world_layout.yaml":
``source_override=False`` / ``wind_override=False``, ``dock_tolerance <= 0``,
``rssi_min_dbm >= 0``, ``path_loss_exponent <= 0``, ``wall_attenuation_db < 0``.
"""
from __future__ import annotations

from typing import Dict

from dusan_common.obstacle_layer import OBSTACLE_DEFAULTS, OBSTACLE_NODE_KEYS

from .battery_model import BatteryConfig
from .gas_model import GasConfig, rects_from_flat
from .rth_logic import RthConfig
from .source_seeking import SeekerConfig

_COMMON = {
    "layout_file": "",
    "map_yaml": "",
    "frame_id": "map",
    "odom_topic": "/odom",
}

GAS_PARAMS: Dict[str, object] = dict(_COMMON, **{
    # model geometry
    "resolution": 0.25,
    "obstacle_fraction": 0.15,
    # leak
    "source_override": False,
    "source_x": -3.2,
    "source_y": 2.0,
    "emission_rate": 170.0,
    "source_sigma": 0.25,
    "decay_rate": 0.012,
    "diffusivity": 0.03,
    # wind
    "wind_override": False,
    "wind_x": 0.40,
    "wind_y": -0.10,
    "corridor_speed": 0.20,
    "factory_x_max": 14.9,
    "inflow_rects": [-10.4, -8.9, -9.2, -1.0, -10.4, 8.45, -9.2, 8.95, -3.5, 8.7, 14.8, 9.3],
    "outflow_rects": [63.3, -1.95, 63.95, 1.95, 14.6, -9.2, 15.2, -3.4, 14.6, 3.4, 15.2, 9.0,
                      -10.0, -9.4, 14.85, -8.8],
    "cfl": 0.9,
    "max_substeps": 40,
    "projection_tol": 1.0e-6,
    "projection_max_iter": 20000,
    # gusts
    "gusts_enabled": True,
    "gust_speed_amp": 0.15,
    "gust_speed_period": 47.0,
    "gust_dir_amp_deg": 8.0,
    "gust_dir_period": 83.0,
    "gust_noise_speed": 0.08,
    "gust_noise_dir_deg": 5.0,
    "gust_tau": 8.0,
    "eddy_rms": 0.06,
    "eddy_scale": 2.5,
    "eddy_tau": 15.0,
    "eddy_modes": 6,
    "eddy_wall_taper": 0.75,
    "seed": 7,
    # start-up
    "spinup_seconds": 400.0,
    "spinup_dt": 0.5,
    "spinup_gust_seconds": 30.0,
    # rates
    "update_rate": 10.0,
    "grid_rate": 2.0,
    "max_dt": 0.5,
    # sensors
    "sensor_tau": 1.0,
    "sensor_noise_ppm": 0.3,
    "sensor_noise_rel": 0.03,
    "sensor_quantum": 0.1,
    "nose_offset": [0.42, 0.0],
    "array_offsets": [0.30, 0.15, 0.30, -0.15, -0.30, 0.15, -0.30, -0.15],
    "wind_sensor_noise": 0.03,
    "wind_sensor_tau": 0.5,
    "sensor_seed": 3,
    # visualisation
    "vis_max_ppm": 1000.0,
    "vis_min_ppm": 0.5,
    "vis_log_scale": True,
})

_SEEKER_CFG = SeekerConfig()
SEEKER_PARAMS: Dict[str, object] = dict(_COMMON, **{
    "control_rate": 10.0,
    "status_rate": 2.0,
    "path_rate": 1.0,
    "use_map_topic": True,
    "sensor_timeout": 1.0,
    "ground_truth_override": False,
    "ground_truth_x": -3.2,
    "ground_truth_y": 2.0,
    # layout debris the seeker treats as obstacles (comma separated types, "" = none): its gradient
    # steps / local A* see only the nav map, where debris is free (Gazebo: falls in the F-OBS5 rubble,
    # over the F-OBS4 pipe); an item whose blocking would close a passage (F-OBS3 across the
    # partition doorway) stays free
    "debris_blocked_kinds": "rubble,pipe",
})
for _k, _v in vars(_SEEKER_CFG).items():
    if _k == "nose_offset":
        SEEKER_PARAMS[_k] = [float(v) for v in _v]
    elif _k == "array_offsets":
        SEEKER_PARAMS[_k] = [float(v) for o in _v for v in o]
    else:
        SEEKER_PARAMS[_k] = _v
# novel-obstacle input selection (str node parameters; the numeric memory keys are SeekerConfig fields)
SEEKER_PARAMS.update({k: OBSTACLE_DEFAULTS[k] for k in OBSTACLE_NODE_KEYS})

_BAT_CFG = BatteryConfig()
BATTERY_PARAMS: Dict[str, object] = {
    "layout_file": "",
    "odom_topic": "/odom",
    "update_rate": 10.0,
    "publish_rate": 2.0,
    "charge_on_dock": True,
    "dock_tolerance": 0.0,
    "dock_speed_max": 0.05,
    "dock_settle_s": 1.0,
    "speed_filter_tau": 0.5,
    "location": "base",
    "serial_number": "DUSAN-SIM-BAT-01",
}
BATTERY_PARAMS.update(vars(_BAT_CFG))

COMM_PARAMS: Dict[str, object] = dict(_COMMON, **{
    "map_resolution": 0.5,
    "publish_rate": 5.0,
    "map_republish_period": 10.0,
    "antenna_height": 0.6,
    "path_loss_exponent": 0.0,
    "wall_attenuation_db": -1.0,
    "rf_min_obstacle_area_m2": 2.5,
    "fading_sigma_db": 2.0,
    "fading_tau_s": 2.0,
    "seed": 11,
    "vis_min_dbm": -100.0,
    "vis_max_dbm": -30.0,
})

RTH_PARAMS: Dict[str, object] = dict(_COMMON, **{
    "rate": 10.0,
    "status_rate": 2.0,
    "active_rate": 5.0,
    "use_map_topic": True,
    "local_comm_model": True,
    "comm_map_resolution": 0.5,
    "rf_min_obstacle_area_m2": 2.5,     # keep in sync with comm_simulator (checked by the tests)
    # battery model used for the energy-to-home prediction (keep in sync
    # with the battery_simulator section; checked by the unit tests)
    "capacity_wh": _BAT_CFG.capacity_wh,
    "p_compute_w": _BAT_CFG.p_compute_w,
    "p_sensors_w": _BAT_CFG.p_sensors_w,
    "p_stand_w": _BAT_CFG.p_stand_w,
    "p_sit_w": _BAT_CFG.p_sit_w,
    "p_per_mps_w": _BAT_CFG.p_per_mps_w,
    "p_per_radps_w": _BAT_CFG.p_per_radps_w,
})
RTH_PARAMS.update(vars(RthConfig()))
RTH_PARAMS["rssi_min_dbm"] = 0.0            # >= 0 -> layout comm.rssi_min_dbm
# novel-obstacle input selection (str node parameters; rth_config() only reads RthConfig fields)
RTH_PARAMS.update({k: OBSTACLE_DEFAULTS[k] for k in OBSTACLE_NODE_KEYS})


# =============================================================================
# conversions parameter dict -> model configs
# =============================================================================
def gas_config(p: dict, layout=None) -> GasConfig:
    leak = layout.gas_leak if layout is not None else {}
    src = (float(p["source_x"]), float(p["source_y"]))
    if not p.get("source_override") and leak.get("source"):
        src = (float(leak["source"][0]), float(leak["source"][1]))
    wind = (float(p["wind_x"]), float(p["wind_y"]))
    if not p.get("wind_override") and leak.get("wind_base"):
        wind = (float(leak["wind_base"][0]), float(leak["wind_base"][1]))
    names = set(vars(GasConfig()).keys())
    kw = {k: p[k] for k in names if k in p}
    kw.update(source_xy=src, wind_base=wind,
              inflow_rects=rects_from_flat(p["inflow_rects"]),
              outflow_rects=rects_from_flat(p["outflow_rects"]))
    for k in ("max_substeps", "projection_max_iter", "eddy_modes", "seed"):
        kw[k] = int(kw[k])
    for k, v in list(kw.items()):
        if isinstance(v, int) and not isinstance(v, bool) and isinstance(getattr(GasConfig(), k), float):
            kw[k] = float(v)
    return GasConfig(**kw)


def seeker_config(p: dict) -> SeekerConfig:
    base = SeekerConfig()
    kw = {}
    for k, default in vars(base).items():
        if k not in p:
            continue
        v = p[k]
        if k == "nose_offset":
            v = (float(v[0]), float(v[1]))
        elif k == "array_offsets":
            f = [float(x) for x in v]
            v = tuple((f[i], f[i + 1]) for i in range(0, len(f) - 1, 2))
        elif isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int):
            v = int(v)
        elif isinstance(default, float):
            v = float(v)
        kw[k] = v
    return SeekerConfig(**kw)


def battery_config(p: dict) -> BatteryConfig:
    base = BatteryConfig()
    kw = {}
    for k, default in vars(base).items():
        if k in p:
            kw[k] = int(p[k]) if isinstance(default, int) else float(p[k])
    return BatteryConfig(**kw)


def rth_config(p: dict, layout=None) -> RthConfig:
    base = RthConfig()
    kw = {}
    for k, default in vars(base).items():
        if k not in p:
            continue
        v = p[k]
        if isinstance(default, bool):
            v = bool(v)
        elif isinstance(default, int):
            v = int(v)
        else:
            v = float(v)
        kw[k] = v
    if kw.get("rssi_min_dbm", 0.0) >= 0.0:
        kw["rssi_min_dbm"] = float((layout.comm if layout is not None else {}).get("rssi_min_dbm", -80.0))
    return RthConfig(**kw)
