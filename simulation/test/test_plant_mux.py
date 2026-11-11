"""cmd_vel mux priority / freshness tests (ROS free)."""
import math

import pytest

from plant_simulation.mux_logic import MUX_DEFAULTS, ZERO, CmdVelMux, sanitize_cmd


def make(**kw):
    p = dict(MUX_DEFAULTS)
    p.update(kw)
    return CmdVelMux.from_params(p)


def test_nothing_fresh_is_zero():
    m = make()
    assert m.select(0.0) == ("none", ZERO)


def test_priority_order_teleop_rth_seek_patrol():
    m = make()
    t = 10.0
    m.update("patrol", (0.1, 0.0, 0.0), t)
    assert m.select(t) == ("patrol", (0.1, 0.0, 0.0))
    m.update("seek", (0.2, 0.0, 0.0), t)
    assert m.select(t)[0] == "seek"
    m.update("rth", (0.3, 0.0, 0.0), t)
    assert m.select(t)[0] == "rth"
    m.update("teleop", (0.4, 0.0, 0.1), t)
    assert m.select(t) == ("teleop", (0.4, 0.0, 0.1))


def test_freshness_timeout_falls_back_to_lower_priority():
    m = make()
    m.update("teleop", (0.4, 0.0, 0.0), 0.0)
    m.update("patrol", (0.1, 0.0, 0.0), 0.5)
    assert m.select(0.45)[0] == "teleop"
    assert m.select(0.55) == ("patrol", (0.1, 0.0, 0.0))   # teleop older than 0.5 s
    assert m.select(1.2) == ("none", ZERO)                  # patrol stale too


def test_teleop_active_flag_holds_zero_and_blocks_autonomy():
    m = make()
    m.set_flag("teleop_active", True, 0.0)
    m.update("patrol", (0.3, 0.0, 0.0), 1.0)
    m.update("rth", (0.3, 0.0, 0.0), 1.0)
    assert m.select(1.0) == ("teleop", ZERO)          # manual override, no key pressed
    m.update("teleop", (0.0, 0.2, 0.0), 1.1)
    assert m.select(1.1) == ("teleop", (0.0, 0.2, 0.0))
    m.set_flag("teleop_active", False, 1.2)
    m.update("rth", (0.3, 0.0, 0.0), 1.5)
    assert m.select(1.55)[0] == "teleop"              # last key press still fresh
    assert m.select(1.65)[0] == "rth"


def test_rth_flag_beats_seek_and_patrol_even_when_rth_cmd_stale():
    m = make()
    m.set_flag("rth_active", True, 4.9)               # return_to_home heartbeats the flag at 5 Hz
    m.update("seek", (0.2, 0.0, 0.0), 5.0)
    m.update("patrol", (0.1, 0.0, 0.0), 5.0)
    assert m.select(5.0) == ("rth", ZERO)             # RTH planning: hold still
    m.update("rth", (0.25, 0.0, 0.1), 5.1)
    assert m.select(5.2) == ("rth", (0.25, 0.0, 0.1))
    m.update("teleop", (0.1, 0.0, 0.0), 5.3)
    assert m.select(5.3)[0] == "teleop"               # teleop still on top


def test_rth_flag_without_hold_lets_lower_sources_through():
    m = make(hold_on_stale_flag=False)
    m.set_flag("rth_active", True, 0.0)
    m.update("seek", (0.2, 0.0, 0.0), 1.0)
    assert m.select(1.0)[0] == "seek"


def test_flag_timeout():
    m = make(flag_timeout=1.0)
    m.set_flag("teleop_active", True, 0.0)
    assert m.flag("teleop_active", 0.5)
    assert not m.flag("teleop_active", 1.5)
    assert m.select(1.5) == ("none", ZERO)


def test_dead_teleop_releases_manual_override():
    """keyboard_teleop SIGKILLed while the override is on: no 'false' is ever
    sent, the heartbeat simply stops -> after flag_timeout the autonomy drives
    again instead of a zero 'teleop' command forever."""
    m = make()
    assert m.flag_timeout == pytest.approx(1.5)
    t = 10.0
    while t < 13.0:                                    # 2 Hz heartbeat while alive
        m.set_flag("teleop_active", True, t)
        m.update("patrol", (0.3, 0.0, 0.0), t)
        assert m.select(t) == ("teleop", ZERO)
        t += 0.5
    last = t - 0.5                                     # teleop dies here
    for k in range(1, 40):
        t = last + 0.05 * k
        m.update("patrol", (0.3, 0.0, 0.0), t)
        src = m.select(t)[0]
        assert src == ("teleop" if t - last <= 1.5 else "patrol"), (t, src)
    # rth flag expires the same way
    m.set_flag("rth_active", True, t)
    m.update("patrol", (0.3, 0.0, 0.0), t + 1.0)
    assert m.select(t + 1.0) == ("rth", ZERO)          # RTH planning: hold still
    m.update("patrol", (0.3, 0.0, 0.0), t + 1.6)
    assert m.select(t + 1.6)[0] == "patrol"            # return_to_home gone -> flag expired


def test_flag_survives_a_sim_reset_for_one_timeout():
    m = make()
    m.set_flag("teleop_active", True, 500.0)
    assert m.select(0.2)[0] == "teleop"               # clock jumped back: window restarted at 0.2
    assert m.select(1.6)[0] == "teleop"
    assert m.select(1.8) == ("none", ZERO)            # no heartbeat since -> released
    m.set_flag("teleop_active", True, 2.0)
    assert m.select(2.1)[0] == "teleop"


def test_zero_flag_timeout_keeps_latched_behaviour():
    m = make(flag_timeout=0.0)
    m.set_flag("teleop_active", True, 0.0)
    assert m.select(1000.0) == ("teleop", ZERO)


def test_clamp_and_nan_rejection():
    m = make(max_linear=0.5, max_angular=1.0)
    m.update("patrol", (2.0, -3.0, 5.0), 0.0)
    assert m.select(0.0)[1] == (0.5, -0.5, 1.0)
    m.update("patrol", (float("nan"), 0.0, 0.0), 0.1)
    assert m.select(0.1)[1] == ZERO
    assert sanitize_cmd(("a", 0, 0)) == ZERO
    assert sanitize_cmd((0.1, math.inf, 0)) == ZERO


def test_time_going_backwards_is_not_fresh():
    m = make()
    m.update("patrol", (0.1, 0.0, 0.0), 100.0)
    assert m.select(5.0) == ("none", ZERO)            # sim reset: stale stamps from the future


def test_per_source_timeout_and_unknown_names():
    m = make(seek_timeout=2.0)
    m.update("seek", (0.1, 0.0, 0.0), 0.0)
    assert m.select(1.5)[0] == "seek"
    with pytest.raises(KeyError):
        m.update("joystick", (0, 0, 0), 0.0)
    with pytest.raises(KeyError):
        m.set_flag("estop", True, 0.0)
    st = m.status(1.5)
    assert st["fresh"]["seek"] and not st["fresh"]["patrol"]
