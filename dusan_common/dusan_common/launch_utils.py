"""
Helpers shared by the launch files of the repository packages (ROS free).

``respawn_kwargs`` builds the Node(...) keyword arguments that restart a dead
process.  launch_ros ``Node`` passes them on to launch's ``ExecuteProcess``,
which hands them to its base class ``ExecuteLocal``: from Jazzy on
``respawn_max_retries`` is a parameter of ``ExecuteLocal.__init__`` (default
-1 = restart forever), NOT of ``ExecuteProcess.__init__`` (only ``cmd``,
``prefix``, ``name``, ``cwd``, ``env``, ``additional_env`` and ``**kwargs``),
so the support check walks the whole class hierarchy.  Humble's launch has
no such parameter at all - passing it there would raise a TypeError.
"""
from __future__ import annotations

import inspect
from typing import Optional


def accepts_parameter(cls, name: str) -> bool:
    """True when ``__init__`` of ``cls`` or of one of its base classes declares
    the parameter ``name`` (a subclass forwarding ``**kwargs`` to its base
    accepts what the base declares)."""
    for c in inspect.getmro(cls) if inspect.isclass(cls) else ():
        init = c.__dict__.get("__init__")
        if init is None:
            continue
        try:
            params = inspect.signature(init).parameters
        except (TypeError, ValueError):
            continue
        if name in params and params[name].kind not in (inspect.Parameter.VAR_POSITIONAL,
                                                        inspect.Parameter.VAR_KEYWORD):
            return True
    return False


def respawn_kwargs(process_cls, enabled: bool = True, delay: float = 3.0,
                   max_retries: Optional[int] = 10) -> dict:
    """Node(...) keyword arguments that restart a dead process: ``respawn`` /
    ``respawn_delay`` (every distro) plus ``respawn_max_retries`` where the
    launch version has it (``process_cls`` = launch.actions.ExecuteProcess),
    so a node that crashes in its constructor is not restarted forever.
    ``enabled`` False: no respawn at all ({})."""
    if not enabled:
        return {}
    kw = {"respawn": True, "respawn_delay": float(delay)}
    if max_retries is not None and accepts_parameter(process_cls, "respawn_max_retries"):
        kw["respawn_max_retries"] = int(max_retries)
    return kw
