# Makes the in-tree ROS 2 python packages importable when running the
# algorithm unit tests WITHOUT a colcon build (e.g. `python -m pytest`).
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _pkg in ("dusan_common", "module1_locomotion", "module2_inspection", "module3_gas_safety", "simulation", "tools"):
    _p = os.path.join(_ROOT, _pkg)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
