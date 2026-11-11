"""dusan_common.launch_utils: respawn_max_retries support is looked up through the whole class
hierarchy of launch's ExecuteProcess (Jazzy and later: a parameter of the base class ExecuteLocal)."""
from dusan_common.launch_utils import accepts_parameter, respawn_kwargs


class ExecuteLocal:                       # launch jazzy / rolling (execute_local.py)
    def __init__(self, *, process_description=None, shell=False, respawn=False, respawn_delay=None,
                 respawn_max_retries=-1, **kwargs):
        pass


class ExecuteProcess(ExecuteLocal):       # launch jazzy / rolling (execute_process.py)
    def __init__(self, *, cmd, prefix=None, name=None, cwd=None, env=None, additional_env=None, **kwargs):
        super().__init__(**kwargs)


class Node(ExecuteProcess):               # launch_ros: forwards **kwargs
    def __init__(self, *, executable, package=None, **kwargs):
        super().__init__(cmd=[executable], **kwargs)


class HumbleExecuteProcess:               # launch humble: respawn / respawn_delay only
    def __init__(self, *, cmd, name=None, respawn=False, respawn_delay=None, **kwargs):
        pass


def test_parameter_of_a_base_class_counts():
    assert accepts_parameter(ExecuteProcess, "respawn_max_retries")
    assert accepts_parameter(Node, "respawn_max_retries")
    assert accepts_parameter(ExecuteProcess, "cmd") and accepts_parameter(ExecuteProcess, "respawn")
    assert not accepts_parameter(HumbleExecuteProcess, "respawn_max_retries")
    assert not accepts_parameter(ExecuteProcess, "kwargs")          # **kwargs is not a named parameter
    assert not accepts_parameter(None, "respawn_max_retries")


def test_respawn_kwargs_per_launch_version():
    assert respawn_kwargs(ExecuteProcess, False) == {}
    assert respawn_kwargs(ExecuteProcess) == {"respawn": True, "respawn_delay": 3.0, "respawn_max_retries": 10}
    assert respawn_kwargs(ExecuteProcess, True, 5.0, 3) == {"respawn": True, "respawn_delay": 5.0,
                                                           "respawn_max_retries": 3}
    assert respawn_kwargs(HumbleExecuteProcess) == {"respawn": True, "respawn_delay": 3.0}
    assert respawn_kwargs(ExecuteProcess, True, max_retries=None) == {"respawn": True, "respawn_delay": 3.0}
    # the kwargs really are accepted by the real-shaped classes
    Node(executable="x", **respawn_kwargs(Node))
    HumbleExecuteProcess(cmd=["x"], **respawn_kwargs(HumbleExecuteProcess))
