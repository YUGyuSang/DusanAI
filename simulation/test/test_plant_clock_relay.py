"""/clock decimation rule of the clock relay (ROS free)."""
from plant_simulation.clock_relay import CLOCK_RELAY_DEFAULTS, ClockDecimator

MS = 1000000


def forwarded(dec, times_ms, wall=0.0, dwall=0.001):
    out = []
    for k, t in enumerate(times_ms):
        if dec.accept(int(t * MS), wall + k * dwall):
            out.append(t)
    return out


def test_defaults():
    assert CLOCK_RELAY_DEFAULTS["rate"] == 200.0
    assert ClockDecimator(200.0).period_ns == 5 * MS


def test_1khz_steps_become_200hz_on_the_5ms_grid():
    dec = ClockDecimator(200.0, 0.5)
    out = forwarded(dec, range(1003, 1100))            # first sample, then every grid crossing
    assert out[0] == 1003
    assert out[1:] == list(range(1005, 1100, 5))
    assert dec.n_in == 97 and dec.n_out == len(out)
    # a 100 Hz timer created on the grid fires exactly every 10 ms
    assert all((t - 1005) % 5 == 0 for t in out[1:])


def test_dropped_samples_forward_the_next_one():
    dec = ClockDecimator(200.0, 0.5)
    out = forwarded(dec, [0, 1, 2, 7, 8, 9, 13, 21, 22])
    assert out == [0, 7, 13, 21]


def test_reset_and_pause():
    dec = ClockDecimator(200.0, 0.5)
    assert forwarded(dec, [500000, 500001]) == [500000]
    assert dec.accept(2 * MS, 1.0)                     # backward jump (world reset) is forwarded
    assert not dec.accept(2 * MS, 1.2)                 # paused ...
    assert dec.accept(2 * MS, 1.5)                     # ... keepalive after 0.5 s wall
    assert not dec.accept(2 * MS, 1.7)


def test_rate_zero_forwards_everything_and_keepalive_zero_disables_resend():
    dec = ClockDecimator(0.0, 0.5)
    assert forwarded(dec, [0, 1, 1, 2]) == [0, 1, 1, 2]
    dec = ClockDecimator(200.0, 0.0)
    assert dec.accept(0, 0.0)
    assert not dec.accept(0, 100.0)
