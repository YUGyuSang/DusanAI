"""
Emulation of rclpy's per-call-site logger rules for the stubbed-node tests.

rclpy's ``RcutilsLogger.log`` keys every call site (file, line and function of
the first frame outside rclpy) and raises ``ValueError`` when the same site
logs again with another severity, or with other filters / filter parameters
(``throttle_duration_sec``, ``once``, ``skip_first``).  A stub logger that only
appends lines hides such bugs: on 2026-10-26 the mission manager logged its
events through one line with varying severity and died on ROS 2 Lyrical the
first time a warning followed info events.

Stub loggers call :meth:`CallSiteRules.check` from their methods, passing their
own file in ``skip_files`` so that the call site found is the node's line.
"""
import inspect
import os

FILTER_KEYS = ("throttle_duration_sec", "once", "skip_first")


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


class CallSiteRules:
    def __init__(self, skip_files=()):
        self._sites = {}
        self._skip = {_norm(f) for f in skip_files} | {_norm(__file__)}

    def check(self, severity, **kwargs):
        frame = inspect.currentframe().f_back
        while frame is not None and _norm(frame.f_code.co_filename) in self._skip:
            frame = frame.f_back
        if frame is None:
            return
        key = (_norm(frame.f_code.co_filename), frame.f_lineno, frame.f_code.co_name)
        filters = {k: kwargs[k] for k in FILTER_KEYS if k in kwargs}
        prev = self._sites.get(key)
        if prev is None:
            self._sites[key] = (severity, filters)
            return
        if prev[0] != severity:
            raise ValueError("Logger severity cannot be changed between calls.")
        if set(prev[1]) != set(filters):
            raise ValueError("Requested logging filters cannot be changed between calls.")
        if any(prev[1][k] != filters[k] for k in filters):
            raise ValueError("Logging filter parameters cannot be changed between calls.")
