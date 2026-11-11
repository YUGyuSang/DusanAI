"""Tests of the Ubuntu helper scripts in scripts/ that can run without ROS.

* workspace selection / linking of scripts/common.sh (`ensure_ws_src`, used by
  build.sh), exercised with a throw-away copy of common.sh in a fake repository;
* install_deps.sh installs the ROS 2 CLI verbs the run scripts need and
  check_install.sh verifies them.

The functional tests need a POSIX bash (Git Bash on Windows, /bin/bash on Linux).
On Windows without native symlinks `ln -s` copies the (tiny) fake repository, so
the tests that depend on real symlinks are skipped there.
"""
import os
import shutil
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, "scripts")


def _read(name):
    with open(os.path.join(SCRIPTS, name), "r", encoding="utf-8") as f:
        return f.read()


def _bash():
    exe = shutil.which("bash")
    if exe is None:
        return None
    if os.name == "nt" and "system32" in exe.lower():   # WSL launcher, not a real bash
        return None
    return exe


needs_bash = pytest.mark.skipif(_bash() is None, reason="bash not available")


def _make_repo(root, rel):
    """Fake repository <root>/<rel> containing only scripts/common.sh."""
    repo = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.join(repo, "scripts"))
    shutil.copy(os.path.join(SCRIPTS, "common.sh"), os.path.join(repo, "scripts", "common.sh"))
    return repo


def _run(root, repo_rel, dusan_ws=None):
    """Source <repo_rel>/scripts/common.sh in bash (cwd = root) and call ensure_ws_src."""
    ws_line = "unset DUSAN_WS" if dusan_ws is None else "export DUSAN_WS='%s'" % dusan_ws
    script = "\n".join([
        "set -e",
        'export HOME="$PWD/home"',
        ws_line,
        '. "%s/scripts/common.sh"' % repo_rel,
        "ensure_ws_src",
        "printf 'WS=%s\\n' \"$WS_DIR\"",
    ])
    env = {k: v for k, v in os.environ.items() if k != "DUSAN_WS"}
    os.makedirs(os.path.join(root, "home"), exist_ok=True)
    r = subprocess.run([_bash(), "-c", script], cwd=root, env=env, capture_output=True, text=True, timeout=60)
    ws = None
    for line in r.stdout.splitlines():
        if line.startswith("WS="):
            ws = line[3:]
    return r, ws


def _native_symlinks(root):
    probe = os.path.join(root, "probe")
    os.makedirs(os.path.join(probe, "a"))
    r = subprocess.run([_bash(), "-c", 'ln -s "$PWD/probe/a" probe/b 2>/dev/null && [ -L probe/b ]'],
                       cwd=root, capture_output=True, text=True, timeout=30)
    shutil.rmtree(probe, ignore_errors=True)
    return r.returncode == 0


def _linked(ws_dir):
    return os.path.isfile(os.path.join(ws_dir, "src", "DusanAI", "scripts", "common.sh"))


# ---------------------------------------------------------------- ensure_ws_src
@needs_bash
def test_dusan_ws_outside_repo_gets_link(tmp_path):
    root = str(tmp_path)
    _make_repo(root, "repo")
    r, ws = _run(root, "repo", dusan_ws="ws")
    assert r.returncode == 0, r.stderr
    assert ws is not None and ws.endswith("/ws")
    assert _linked(os.path.join(root, "ws"))
    if _native_symlinks(root):
        assert os.path.islink(os.path.join(root, "ws", "src", "DusanAI"))
        r2, _ = _run(root, "repo", dusan_ws="ws")        # second build: link already there
        assert r2.returncode == 0, r2.stderr


@needs_bash
def test_dusan_ws_trailing_slash_and_tilde(tmp_path):
    root = str(tmp_path)
    _make_repo(root, "repo")
    r, ws = _run(root, "repo", dusan_ws="ws/")
    assert r.returncode == 0, r.stderr
    assert ws.endswith("/ws")
    r, ws = _run(root, "repo", dusan_ws="~/myws")
    assert r.returncode == 0, r.stderr
    assert ws.endswith("/home/myws")
    assert _linked(os.path.join(root, "home", "myws"))


@needs_bash
def test_repo_already_inside_dusan_ws_src_is_not_linked(tmp_path):
    root = str(tmp_path)
    _make_repo(root, "ws2/src/group/DusanAI")
    r, ws = _run(root, "ws2/src/group/DusanAI", dusan_ws="ws2")
    assert r.returncode == 0, r.stderr
    assert ws.endswith("/ws2")
    assert not os.path.exists(os.path.join(root, "ws2", "src", "DusanAI"))


@needs_bash
def test_default_workspace_is_parent_of_src(tmp_path):
    root = str(tmp_path)
    _make_repo(root, "ws3/src/DusanAI")
    r, ws = _run(root, "ws3/src/DusanAI")
    assert r.returncode == 0, r.stderr
    assert ws.endswith("/ws3")
    assert sorted(os.listdir(os.path.join(root, "ws3", "src"))) == ["DusanAI"]
    assert not os.path.exists(os.path.join(root, "home", "dusan_ws"))


@needs_bash
def test_default_workspace_in_home_gets_link(tmp_path):
    root = str(tmp_path)
    _make_repo(root, "repo")
    r, ws = _run(root, "repo")
    assert r.returncode == 0, r.stderr
    assert ws.endswith("/home/dusan_ws")
    assert _linked(os.path.join(root, "home", "dusan_ws"))


@needs_bash
@pytest.mark.parametrize("dusan_ws", ["repo", "repo/", "repo/ws"])
def test_workspace_inside_repo_is_rejected(tmp_path, dusan_ws):
    root = str(tmp_path)
    repo = _make_repo(root, "repo")
    r, ws = _run(root, "repo", dusan_ws=dusan_ws)
    assert r.returncode != 0
    assert ws is None
    assert "inside it" in r.stderr
    assert sorted(os.listdir(repo)) == ["scripts"]          # no src/ or ws/ left behind


@needs_bash
def test_foreign_directory_at_link_path_is_rejected(tmp_path):
    root = str(tmp_path)
    _make_repo(root, "repo")
    other = os.path.join(root, "ws", "src", "DusanAI")
    os.makedirs(other)
    with open(os.path.join(other, "marker"), "w") as f:
        f.write("old copy")
    r, ws = _run(root, "repo", dusan_ws="ws")
    assert r.returncode != 0
    assert "not this repository" in r.stderr
    assert os.listdir(other) == ["marker"]                  # left untouched


@needs_bash
def test_stale_symlink_is_repointed(tmp_path):
    root = str(tmp_path)
    if not _native_symlinks(root):
        pytest.skip("bash 'ln -s' does not create real symlinks here")
    _make_repo(root, "repo")
    os.makedirs(os.path.join(root, "old_copy"))
    os.makedirs(os.path.join(root, "ws", "src"))
    r0 = subprocess.run([_bash(), "-c", 'ln -s "$PWD/old_copy" ws/src/DusanAI'], cwd=root,
                        capture_output=True, text=True, timeout=30)
    assert r0.returncode == 0, r0.stderr
    r, ws = _run(root, "repo", dusan_ws="ws")
    assert r.returncode == 0, r.stderr
    assert "re-linking" in r.stderr
    assert _linked(os.path.join(root, "ws"))


# ----------------------------------------------------------------- static checks
def test_build_sh_uses_ensure_ws_src():
    text = _read("build.sh")
    assert "ensure_ws_src" in text
    # the old condition skipped the link whenever DUSAN_WS was set
    assert '[ -z "${DUSAN_WS:-}" ]' not in text
    assert text.index("ensure_ws_src") < text.index('cd "$WS_DIR"')


def test_install_deps_installs_ros2_cli_verbs():
    text = _read("install_deps.sh")
    start = text.index("\nPKGS=(\n")
    pkgs = text[start:text.index("\n)\n", start)]
    for pkg in ('"ros-$ROS_DISTRO-ros-base"', '"ros-$ROS_DISTRO-ros2launch"', '"ros-$ROS_DISTRO-ros2run"'):
        assert pkg in pkgs, pkg


def test_check_install_tests_launch_and_run_verbs():
    text = _read("check_install.sh")
    assert 'ros2 "$verb" -h' in text
    assert "for verb in launch run" in text
    # missing launch/run verbs are failures, not warnings
    assert "launch|run) need=fail" in text


def _setup_entry_points():
    """{(package, executable)} of every console_scripts entry point in the repository."""
    import re
    found = set()
    for d in sorted(os.listdir(REPO)):
        path = os.path.join(REPO, d, "setup.py")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        m = re.search(r'package_name\s*=\s*["\']([\w-]+)["\']', text)
        assert m, path
        for exe in re.findall(r'["\']\s*([\w-]+)\s*=\s*[\w.]+:main\s*["\']', text):
            found.add((m.group(1), exe))
    return found


def test_check_install_checks_every_entry_point():
    import re
    text = _read("check_install.sh")
    m = re.search(r'EXECUTABLES="([^"]*)"', text)
    assert m, "EXECUTABLES list missing in check_install.sh"
    listed = set(tuple(pe.split(":", 1)) for pe in m.group(1).split())
    expected = _setup_entry_points()
    assert ("plant_simulation", "clock_relay") in expected   # started by sim.launch.py by default
    assert len(expected) == 21
    assert listed == expected
    # the loop really queries `ros2 pkg executables` per package and fails on a missing one
    assert 'ros2 pkg executables "$pk"' in text
    assert 'fail "executable $pk/$e missing"' in text


def test_demo_hud_dependencies_are_installed_and_checked():
    """The demo HUD writes Korean text with Pillow + a Korean font (it falls back
    to English without them): package.xml declares python3-pil for rosdep,
    install_deps.sh installs it with fonts-noto-cjk, check_install.sh warns
    (never fails) when either is missing."""
    import xml.etree.ElementTree as ET
    deps = {e.text.strip() for e in ET.parse(os.path.join(REPO, "simulation", "package.xml")).getroot()
            .iter("exec_depend")}
    assert "python3-pil" in deps
    text = _read("install_deps.sh")
    start = text.index("\nPKGS=(\n")
    pkgs = text[start:text.index("\n)\n", start)].split()
    assert "python3-pil" in pkgs and "fonts-noto-cjk" in pkgs
    check = _read("check_install.sh")
    assert 'python3 -c "import PIL"' in check
    for line in check.splitlines():
        if "python3-pil" in line or "fonts-noto-cjk" in line:
            assert "fail " not in line, line


# ---------------------------------------------------------------------------
# --low-memory flag of run_bringup.sh / run_demo.sh (parse_launch_flags in common.sh)
# ---------------------------------------------------------------------------
def _launch_flags(tmp_path, args, env=None, meminfo_kb=None):
    """Source common.sh, run parse_launch_flags <args>; -> (LAUNCH_ARGS list, stderr)."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       %d kB\nMemFree:        1000 kB\n" % (meminfo_kb or 32 * 1048576))
    quoted = " ".join("'%s'" % a for a in args)
    script = "\n".join([
        "set -e",
        'export HOME="$PWD"',
        'export DUSAN_MEMINFO="%s"' % str(meminfo).replace("\\", "/"),
        '. "%s/common.sh"' % SCRIPTS.replace("\\", "/"),
        "parse_launch_flags %s" % quoted,
        'for a in "${LAUNCH_ARGS[@]}"; do printf "ARG=%s\\n" "$a"; done',
    ])
    e = {k: v for k, v in os.environ.items() if k not in ("DUSAN_WS", "DUSAN_LOW_MEMORY")}
    e.update(env or {})
    r = subprocess.run([_bash(), "-c", script], cwd=str(tmp_path), env=e, capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return [line[4:] for line in r.stdout.splitlines() if line.startswith("ARG=")], r.stderr


@needs_bash
def test_low_memory_flag_becomes_a_launch_argument(tmp_path):
    out, _ = _launch_flags(tmp_path, ["--low-memory", "rviz:=false", "follow_offset:=-3.0 0.0 1.8"])
    assert out == ["rviz:=false", "follow_offset:=-3.0 0.0 1.8", "low_memory:=true"]
    out, _ = _launch_flags(tmp_path, ["--lowmem"])
    assert out == ["low_memory:=true"]
    out, _ = _launch_flags(tmp_path, ["demo:=true"], env={"DUSAN_LOW_MEMORY": "1"})
    assert out == ["demo:=true", "low_memory:=true"]
    # an explicit launch argument wins over the flag / the environment
    out, _ = _launch_flags(tmp_path, ["--low-memory", "low_memory:=false"], env={"DUSAN_LOW_MEMORY": "1"})
    assert out == ["low_memory:=false"]
    out, err = _launch_flags(tmp_path, [])
    assert out == [] and "--low-memory" not in err                         # 32 GB: no hint


@needs_bash
def test_small_ram_without_the_profile_prints_a_hint(tmp_path):
    _, err = _launch_flags(tmp_path, ["demo:=true"], meminfo_kb=7_800_000)  # an "8 GB" laptop
    assert "--low-memory" in err and "8 GB" in err
    _, err = _launch_flags(tmp_path, ["--low-memory"], meminfo_kb=7_800_000)
    assert "--low-memory" not in err


def test_run_scripts_pass_the_translated_arguments():
    for name, fixed in (("run_bringup.sh", ""), ("run_demo.sh", "demo:=true ")):
        text = _read(name)
        assert 'parse_launch_flags "$@"' in text
        assert text.index('parse_launch_flags "$@"') < text.index("source_ws")
        assert ('exec ros2 launch plant_simulation bringup.launch.py %s"${LAUNCH_ARGS[@]}"' % fixed) in text
        assert '"$@"' not in text.split('parse_launch_flags "$@"', 1)[1]      # nothing bypasses the translation
        assert "--low-memory" in text
