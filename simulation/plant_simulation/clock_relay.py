"""
ROS-free decimation rule of the /clock relay (clock_relay_node.py).

Gazebo publishes its clock on every physics step: with the 1 ms step of
plant.sdf that is 1000 msg/s.  Every rclpy node started with
``use_sim_time: true`` subscribes to /clock in Python, so bridging that
stream 1:1 makes ~17 Python processes wake up 1000 times per second next to
Gazebo.  sim.launch.py therefore remaps the bridge's output to /clock_gz and
this relay republishes it on /clock at ``rate`` Hz of *simulation* time.

Decimation is done on sim-time buckets of 1/rate s: a sample is forwarded
when it falls into a different bucket than the last forwarded one.  With
1 ms physics steps and 200 Hz the forwarded times are exact multiples of
5 ms, so every timer whose period is a multiple of 5 ms (the 100 Hz
controller, 20/10/5 Hz loops) still fires exactly on time.  A backward jump
(simulation reset) is a different bucket too and is forwarded at once.
While the simulation is paused (sim time constant) the latest sample is
re-sent every ``keepalive`` s of wall time, so nodes started during a pause
still get a clock.
"""

CLOCK_RELAY_DEFAULTS = {
    "rate": 200.0,        # Hz of sim time forwarded on /clock (<= 0: forward every sample)
    "keepalive": 0.5,     # s (wall): re-send the last sample this often while sim time stands still
    "log_period": 30.0,   # s (wall): relay statistics in the log (<= 0: off)
}


class ClockDecimator:
    """Decide which clock samples to forward.

    ``accept(sim_ns, wall_s)`` -> True when the sample must be published.
    ``sim_ns`` is the sample's simulation time in integer nanoseconds,
    ``wall_s`` a monotonic wall-clock time in seconds.
    """

    def __init__(self, rate=200.0, keepalive=0.5):
        rate = float(rate)
        self.period_ns = int(round(1e9 / rate)) if rate > 0.0 else 0
        self.keepalive = max(0.0, float(keepalive))
        self._bucket = None
        self._wall = None
        self.n_in = 0
        self.n_out = 0

    def accept(self, sim_ns, wall_s):
        self.n_in += 1
        sim_ns = int(sim_ns)
        bucket = sim_ns // self.period_ns if self.period_ns > 0 else sim_ns
        if self.period_ns <= 0:
            forward = True
        elif self._bucket is None or bucket != self._bucket:
            forward = True
        else:
            forward = self.keepalive > 0.0 and (wall_s - self._wall) >= self.keepalive
        if forward:
            self._bucket = bucket
            self._wall = float(wall_s)
            self.n_out += 1
        return forward
