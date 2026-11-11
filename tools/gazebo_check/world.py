"""Harness robot description + test world for the headless Gazebo check.

Pure python (ElementTree, optional ``xacro``): importable without gz.

* :func:`harness_urdf` - spot.urdf.xacro expanded with ``enable_cameras:=false
  enable_thermal:=false`` (the camera / thermal sensors need the ogre2 render
  system, which the Sensors system of plant.sdf provides but a headless
  physics check does not need).  Without the ``xacro`` package the committed
  simulation/urdf/spot.urdf is used and its camera sensors are removed.  The
  mesh visuals are dropped (package:// URIs need GZ_SIM_RESOURCE_PATH and play
  no role in the physics).  Everything else - links, inertias, collisions,
  foot friction, the 12 JointPositionController blocks, JointStatePublisher,
  OdometryPublisher and the IMU sensor - is exactly what the launch file
  spawns.  ``servo`` / ``foot_mu`` overrides exist for what-if studies.
* :func:`world_sdf` - world "plant" with the ``<physics>`` block and the system
  plugins of simulation/worlds/plant.sdf (minus Sensors / ParticleEmitter),
  a flat floor (mu 0.9 like the plant floor) and optional obstacles; the robot
  is included from the harness URDF (sdformat converts URDF -> SDF exactly
  like ``ros_gz_sim create`` does with robot_description).
* :func:`plant_world_sdf` (terrain ``plant``) - the REAL simulation/worlds/
  plant.sdf: every collision of the building, corridor, machines, obstacles
  and included models unchanged, includes inlined, visuals / lights and the
  Sensors / ParticleEmitter systems removed; robot at world_layout.yaml
  robot_spawn.
* :class:`HeightField` - ground-truth terrain height of a world from its
  collision primitives (vertical ray cast), the harness's stand-in for the
  depth camera -> terrain_mapper elevation map and for the fall metric.
"""
from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
XACRO = os.path.join(ROOT, "simulation", "urdf", "spot.urdf.xacro")
URDF = os.path.join(ROOT, "simulation", "urdf", "spot.urdf")
PLANT_SDF = os.path.join(ROOT, "simulation", "worlds", "plant.sdf")

# world systems of plant.sdf that need rendering / are irrelevant headless
SKIP_SYSTEMS = ("gz-sim-sensors-system", "gz-sim-particle-emitter-system")
RENDER_SENSORS = ("camera", "depth_camera", "thermal", "rgbd_camera", "gpu_lidar", "segmentation_camera")
FLOOR_MU = 0.9          # plant.sdf factory floor / corridor surfaces


# ---------------------------------------------------------------------------
# robot
# ---------------------------------------------------------------------------
def _expand_xacro(mappings: Dict[str, str]) -> Optional[ET.Element]:
    try:
        import xacro  # type: ignore
    except ImportError:
        return None
    doc = xacro.process_file(XACRO, mappings=dict(mappings))
    return ET.fromstring(doc.toxml())


def _strip_render_sensors(robot: ET.Element) -> None:
    for gz in list(robot.findall("gazebo")):
        for s in list(gz.findall("sensor")):
            if s.get("type") in RENDER_SENSORS:
                gz.remove(s)
        if len(gz) == 0:
            robot.remove(gz)


def _set_text(parent: ET.Element, tag: str, value) -> None:
    el = parent.find(tag)
    if el is None:
        el = ET.SubElement(parent, tag)
    el.text = str(value)


def robot_tree(servo: Optional[Dict[str, float]] = None, foot_mu: Optional[float] = None,
               keep_visuals: bool = False, use_xacro: bool = True) -> ET.Element:
    """Harness robot description (ElementTree root ``<robot>``).

    ``servo``: optional overrides of every JointPositionController block,
    keys ``p_gain``, ``d_gain``, ``i_gain``, ``cmd_scale`` (cmd_max / cmd_min
    as a factor of the joint effort) - what-if studies only."""
    robot = _expand_xacro({"enable_cameras": "false", "enable_thermal": "false"}) if use_xacro else None
    if robot is None:
        robot = ET.parse(URDF).getroot()
    _strip_render_sensors(robot)
    if not keep_visuals:
        for link in robot.findall("link"):
            for v in list(link.findall("visual")):
                link.remove(v)
    efforts = {j.get("name"): float(j.find("limit").get("effort"))
               for j in robot.findall("joint") if j.find("limit") is not None}
    for gz in robot.findall("gazebo"):
        for pl in gz.findall("plugin"):
            if pl.get("name") != "gz::sim::systems::JointPositionController" or not servo:
                continue
            for k in ("p_gain", "d_gain", "i_gain"):
                if k in servo and servo[k] is not None:
                    _set_text(pl, k, float(servo[k]))
            if servo.get("cmd_scale") is not None:
                eff = efforts[pl.findtext("joint_name")] * float(servo["cmd_scale"])
                _set_text(pl, "cmd_max", eff)
                _set_text(pl, "cmd_min", -eff)
        if foot_mu is not None and (gz.get("reference") or "").endswith("_foot"):
            for k in ("mu1", "mu2"):
                _set_text(gz, k, float(foot_mu))
    return robot


def servo_params(robot: ET.Element) -> Dict[str, Dict[str, float]]:
    """JointPositionController parameters per joint (as gz reads them)."""
    out = {}
    for gz in robot.findall("gazebo"):
        for pl in gz.findall("plugin"):
            if pl.get("name") == "gz::sim::systems::JointPositionController":
                j = pl.findtext("joint_name")
                out[j] = {k: float(pl.findtext(k)) for k in ("p_gain", "i_gain", "d_gain", "cmd_max", "cmd_min",
                                                               "initial_position") if pl.find(k) is not None}
                out[j]["use_velocity_commands"] = (pl.findtext("use_velocity_commands") or "false").strip() == "true"
    return out


def write_harness_urdf(path: str, **kw) -> str:
    robot = robot_tree(**kw)
    ET.ElementTree(robot).write(path, encoding="unicode", xml_declaration=False)
    return path


# ---------------------------------------------------------------------------
# world
# ---------------------------------------------------------------------------
def plant_physics() -> Tuple[str, Sequence[str], str]:
    """(<physics> xml, [system plugin xml], gravity) copied from plant.sdf."""
    world = ET.parse(PLANT_SDF).getroot().find("world")
    physics = ET.tostring(world.find("physics"), encoding="unicode").strip()
    plugins = [ET.tostring(p, encoding="unicode").strip() for p in world.findall("plugin")
               if p.get("filename") not in SKIP_SYSTEMS]
    gravity = (world.findtext("gravity") or "0 0 -9.80665").strip()
    return physics, plugins, gravity


def _box(name: str, pos, size, mu: float = FLOOR_MU) -> str:
    return (f'      <collision name="{name}">\n'
            f'        <pose>{pos[0]} {pos[1]} {pos[2]} 0 0 0</pose>\n'
            f'        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>\n'
            f'        <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>\n'
            f'      </collision>\n')


def _cyl_y(name: str, pos, radius: float, length: float, mu: float = FLOOR_MU) -> str:
    return (f'      <collision name="{name}">\n'
            f'        <pose>{pos[0]} {pos[1]} {pos[2]} 1.5707963 0 0</pose>\n'
            f'        <geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>\n'
            f'        <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>\n'
            f'      </collision>\n')


def _box_pitched(name: str, pos, size, pitch: float, mu: float = FLOOR_MU) -> str:
    return (f'      <collision name="{name}">\n'
            f'        <pose>{pos[0]} {pos[1]} {pos[2]} 0 {pitch} 0</pose>\n'
            f'        <geometry><box><size>{size[0]} {size[1]} {size[2]}</size></box></geometry>\n'
            f'        <surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>\n'
            f'      </collision>\n')


TERRAINS = ("flat", "obstacles", "pipe", "ramp")
PIPE_X, PIPE_HEIGHT = 2.0, 0.08            # "pipe": a 0.08 m pipe (radius 0.04) across the path
RAMP_X0, RAMP_LENGTH = 1.5, 3.0            # "ramp": slope starts at x = 1.5 m, 3 m long ...
RAMP_ANGLE = 0.2064                        # ... 11.8 deg like the plant corridor ramp (plant.sdf)
RAMP_TOP = RAMP_LENGTH * math.sin(RAMP_ANGLE)                 # plateau height (0.615 m)
RAMP_X1 = RAMP_X0 + RAMP_LENGTH * math.cos(RAMP_ANGLE)        # plateau start (4.44 m)


def _ramp_collisions() -> str:
    a, L, t = RAMP_ANGLE, RAMP_LENGTH, 0.1
    cx = RAMP_X0 + 0.5 * L * math.cos(a) + 0.5 * t * math.sin(a)
    cz = 0.5 * L * math.sin(a) - 0.5 * t * math.cos(a)
    out = _box_pitched("ramp", (round(cx, 5), 0.0, round(cz, 5)), (L, 3.0, t), round(-a, 6))
    out += _box("plateau", (round(RAMP_X1 + 2.0, 5), 0.0, round(RAMP_TOP - 0.15, 5)), (4.0, 3.0, 0.3))
    return out


def terrain_height(terrain: str, x, y):
    """Ground-truth terrain height (vectorised) of the test worlds - the
    harness's stand-in for /elevation_map/points (``--map``)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.zeros(np.broadcast(x, y).shape)
    inside = np.abs(y) <= 1.5
    if terrain in ("obstacles", "pipe"):
        r = 0.05 if terrain == "obstacles" else 0.5 * PIPE_HEIGHT
        px = 2.0 if terrain == "obstacles" else PIPE_X
        d = np.abs(x - px)
        z = np.where(inside & (d < r), r + np.sqrt(np.maximum(r * r - d * d, 0.0)), z)
        if terrain == "obstacles":
            z = np.where(inside & (x >= 4.0) & (x <= 8.0), np.maximum(z, 0.10), z)
    elif terrain == "ramp":
        slope = (x - RAMP_X0) * math.tan(RAMP_ANGLE)
        z = np.where(inside & (x >= RAMP_X0) & (x < RAMP_X1), slope, z)
        z = np.where(inside & (x >= RAMP_X1) & (x <= RAMP_X1 + 4.0), RAMP_TOP, z)
    return z


def _extra_specs(extra) -> list:
    """Normalised extra (novel) object specs: dicts {name, center [x, y], size
    [sx, sy, sz], yaw, floor_z}; a missing name becomes novel_box_<k>."""
    out = []
    for k, e in enumerate(extra or ()):
        e = dict(e)
        out.append({"name": str(e.get("name") or f"novel_box_{k}"),
                    "center": [float(v) for v in e["center"][:2]],
                    "size": [float(v) for v in e["size"][:3]],
                    "yaw": float(e.get("yaw", 0.0)), "floor_z": float(e.get("floor_z", 0.0))})
    return out


def extra_models_xml(extra) -> str:
    """Static ``<model>`` elements of extra (novel) box objects - in the
    Gazebo world (physics, depth camera) but NOT in the nav map."""
    out = []
    for e in _extra_specs(extra):
        (x, y), (sx, sy, sz) = e["center"], e["size"]
        z = e["floor_z"] + 0.5 * sz
        geo = f"<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>"
        fr = f"<surface><friction><ode><mu>{FLOOR_MU}</mu><mu2>{FLOOR_MU}</mu2></ode></friction></surface>"
        out.append(f'<model name="{e["name"]}"><static>true</static><pose>{x} {y} {z} 0 0 {e["yaw"]}</pose>'
                   f'<link name="link"><collision name="collision">{geo}{fr}</collision>'
                   f'<visual name="visual">{geo}</visual></link></model>')
    return "\n    ".join(out)


def extra_shapes(extra) -> list:
    """Collision primitives (:func:`collision_shapes` format) of extra objects."""
    shapes = []
    for e in _extra_specs(extra):
        (x, y), (sx, sy, sz) = e["center"], e["size"]
        shapes.append({"name": f"{e['name']}/collision", "type": "box",
                       "p": np.array([x, y, e["floor_z"] + 0.5 * sz]), "R": _rpy_matrix(0.0, 0.0, e["yaw"]),
                       "size": np.array([sx, sy, sz])})
    return shapes


def world_sdf(urdf_path: str, terrain: str = "flat", spawn: Sequence[float] = (0.0, 0.0, 0.56, 0.0),
              world_name: str = "plant", robot_name: str = "spot", extra=()) -> str:
    """Test world SDF text (see module docstring).

    terrain ``flat``: 60 x 60 m floor; ``obstacles``: additionally a 0.10 m pipe
    across the path at x = 2 m and a 0.10 m step from x = 4 m; ``pipe``: a
    0.08 m pipe across the path at x = 2 m; ``ramp``: an 11.8 deg slope (the
    plant corridor ramp) from x = 1.5 m up to a 0.615 m plateau from x = 4.44 m.
    ``extra``: novel static boxes (:func:`extra_models_xml`)."""
    if terrain not in TERRAINS:
        raise ValueError(f"terrain must be one of {TERRAINS}")
    physics, plugins, gravity = plant_physics()
    cols = _box("floor", (0.0, 0.0, -0.15), (60.0, 60.0, 0.3))
    if terrain == "obstacles":
        cols += _cyl_y("pipe", (2.0, 0.0, 0.05), 0.05, 3.0)
        cols += _box("step", (6.0, 0.0, 0.05), (4.0, 3.0, 0.10))
    elif terrain == "pipe":
        r = 0.5 * PIPE_HEIGHT
        cols += _cyl_y("pipe", (PIPE_X, 0.0, r), r, 3.0)
    elif terrain == "ramp":
        cols += _ramp_collisions()
    uri = os.path.abspath(urdf_path).replace(os.sep, "/")
    x, y, z, yaw = (float(v) for v in spawn)
    plug = "\n    ".join(plugins)
    extras = extra_models_xml(extra)
    return f"""<?xml version="1.0"?>
<!-- GENERATED by tools/gazebo_check (physics + systems copied from simulation/worlds/plant.sdf) -->
<sdf version="1.9">
  <world name="{world_name}">
    {physics}
    <gravity>{gravity}</gravity>
    {plug}
    <model name="ground">
      <static>true</static>
      <link name="link">
{cols}      </link>
    </model>
    {extras}
    <include>
      <uri>{uri}</uri>
      <name>{robot_name}</name>
      <pose>{x} {y} {z} 0 0 {yaw}</pose>
    </include>
  </world>
</sdf>
"""


def build(out_dir: str, terrain: str = "flat", spawn: Sequence[float] = (0.0, 0.0, 0.56, 0.0),
          servo: Optional[Dict[str, float]] = None, foot_mu: Optional[float] = None, tag: str = "",
          extra=()) -> Tuple[str, str]:
    """Write harness URDF + world SDF into out_dir -> (world_path, urdf_path).

    terrain ``plant``: the REAL simulation/worlds/plant.sdf (:func:`plant_world_sdf`).
    ``extra``: novel static boxes added to the world (not to the nav map)."""
    os.makedirs(out_dir, exist_ok=True)
    urdf = write_harness_urdf(os.path.join(out_dir, f"spot_harness{tag}.urdf"), servo=servo, foot_mu=foot_mu)
    world = os.path.join(out_dir, f"gazebo_check_{terrain}{tag}.sdf")
    text = plant_world_sdf(urdf, spawn, extra=extra) if terrain == PLANT else world_sdf(urdf, terrain, spawn,
                                                                                           extra=extra)
    with open(world, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return world, urdf


# ---------------------------------------------------------------------------
# the real plant world (simulation/worlds/plant.sdf)
# ---------------------------------------------------------------------------
PLANT = "plant"
MODELS_DIR = os.path.join(ROOT, "simulation", "models")
LAYOUT = os.path.join(ROOT, "simulation", "config", "world_layout.yaml")


def layout_spawn(path: str = LAYOUT) -> Tuple[float, float, float, float]:
    """robot_spawn [x, y, z, yaw] of world_layout.yaml (what sim.launch.py spawns at)."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return tuple(float(v) for v in data["robot_spawn"][:4])


def _model_dir(uri: str) -> str:
    if not uri.startswith("model://"):
        raise ValueError(f"unsupported include uri {uri!r}")
    return os.path.join(MODELS_DIR, uri[len("model://"):].strip("/").split("/")[0])


def _strip_visuals(el: ET.Element) -> int:
    """Remove ``<visual>`` and ``<particle_emitter>`` (gas leak smoke) elements."""
    n = 0
    for parent in el.iter():
        for tag in ("visual", "particle_emitter"):
            for v in list(parent.findall(tag)):
                parent.remove(v)
                n += 1
    return n


def plant_world_tree(drop_visuals: bool = True) -> ET.Element:
    """plant.sdf as an ElementTree ``<sdf>`` root, prepared for a headless server:

    * the systems that need rendering (Sensors) or are irrelevant (ParticleEmitter)
      are removed, every other world plugin / the physics block stay;
    * every ``<include>`` of a model://<name> is replaced by the ``<model>`` of
      simulation/models/<name>/model.sdf (name and pose of the include) - the
      model:// URIs need GZ_SIM_RESOURCE_PATH, which the gz python TestFixture
      does not resolve on this Windows build;
    * ``drop_visuals``: all ``<visual>`` elements (meshes, thermal tags), the gas
      leak ``<particle_emitter>`` and the lights are removed (no physics).

    Every ``<collision>`` (pose, geometry, surface) is kept byte for byte."""
    root = ET.parse(PLANT_SDF).getroot()
    world = root.find("world")
    for p in list(world.findall("plugin")):
        if p.get("filename") in SKIP_SYSTEMS:
            world.remove(p)
    if drop_visuals:
        for light in list(world.findall("light")):
            world.remove(light)
    children = list(world)
    for i, el in enumerate(children):
        if el.tag != "include":
            continue
        mdir = _model_dir(el.findtext("uri").strip())
        model = copy.deepcopy(ET.parse(os.path.join(mdir, "model.sdf")).getroot().find("model"))
        if el.findtext("name"):
            model.set("name", el.findtext("name").strip())
        pose = el.find("pose")
        old = model.find("pose")
        if old is not None:
            model.remove(old)
        if pose is not None:
            new = ET.Element("pose")
            new.text = pose.text
            model.insert(0, new)
        if el.find("static") is not None:
            _set_text(model, "static", el.findtext("static").strip())
        idx = list(world).index(el)
        world.remove(el)
        world.insert(idx, model)
    if drop_visuals:
        for model in world.findall("model"):
            _strip_visuals(model)
    return root


def add_extra_models(root: ET.Element, extra) -> ET.Element:
    """Append the extra (novel) box models to the ``<world>`` of an SDF root."""
    world = root.find("world")
    for e in _extra_specs(extra):
        world.append(ET.fromstring(extra_models_xml([e])))
    return root


def plant_world_sdf(urdf_path: str, spawn: Optional[Sequence[float]] = None, robot_name: str = "spot",
                    extra=()) -> str:
    """The real plant world (:func:`plant_world_tree`) with the harness robot
    included at ``spawn`` (default: world_layout.yaml robot_spawn, as sim.launch.py)
    and the optional ``extra`` novel boxes."""
    root = add_extra_models(plant_world_tree(), extra)
    world = root.find("world")
    x, y, z, yaw = (float(v) for v in (spawn if spawn is not None else layout_spawn()))
    inc = ET.SubElement(world, "include")
    _set_text(inc, "uri", os.path.abspath(urdf_path).replace(os.sep, "/"))
    _set_text(inc, "name", robot_name)
    _set_text(inc, "pose", f"{x} {y} {z} 0 0 {yaw}")
    text = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0"?>\n<!-- GENERATED by tools/gazebo_check from simulation/worlds/plant.sdf '
            '(includes inlined, visuals / Sensors / ParticleEmitter removed, collisions unchanged) -->\n' + text + "\n")


# ---------------------------------------------------------------------------
# ground truth height field of a world (stand-in for the depth camera -> terrain_mapper map)
# ---------------------------------------------------------------------------
def _rpy_matrix(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _pose(el: Optional[ET.Element]) -> Tuple[np.ndarray, np.ndarray]:
    if el is None or not (el.text or "").strip():
        return np.zeros(3), np.eye(3)
    v = [float(s) for s in el.text.split()]
    v += [0.0] * (6 - len(v))
    return np.array(v[:3]), _rpy_matrix(*v[3:6])


def collision_shapes(root: ET.Element, skip_models: Iterable[str] = ("spot",)) -> list:
    """World-frame collision primitives of the static models of an SDF world:
    dicts {name, type box|cylinder|sphere, p (3,), R (3, 3), size | radius / length}.
    (Model / link / collision poses composed; no relative_to frames in plant.sdf.)"""
    out = []
    skip = set(skip_models)
    for model in root.find("world").findall("model"):
        if model.get("name") in skip:
            continue
        pm, Rm = _pose(model.find("pose"))
        for link in model.findall("link"):
            pl, Rl = _pose(link.find("pose"))
            p1, R1 = pm + Rm @ pl, Rm @ Rl
            for col in link.findall("collision"):
                pc, Rc = _pose(col.find("pose"))
                p, R = p1 + R1 @ pc, R1 @ Rc
                g = col.find("geometry")
                name = f"{model.get('name')}/{col.get('name')}"
                if g.find("box") is not None:
                    size = np.array([float(s) for s in g.findtext("box/size").split()])
                    out.append({"name": name, "type": "box", "p": p, "R": R, "size": size})
                elif g.find("cylinder") is not None:
                    out.append({"name": name, "type": "cylinder", "p": p, "R": R,
                                "radius": float(g.findtext("cylinder/radius")),
                                "length": float(g.findtext("cylinder/length"))})
                elif g.find("sphere") is not None:
                    out.append({"name": name, "type": "sphere", "p": p, "R": R,
                                "radius": float(g.findtext("sphere/radius"))})
    return out


def _shape_aabb(s) -> Tuple[np.ndarray, np.ndarray]:
    if s["type"] == "box":
        h = 0.5 * s["size"]
    elif s["type"] == "cylinder":
        h = np.array([s["radius"], s["radius"], 0.5 * s["length"]])
    else:
        h = np.full(3, s["radius"])
    ext = np.abs(s["R"]) @ h
    return s["p"] - ext, s["p"] + ext


def ray_hit(s, O: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Ray parameter t >= 0 of the first hit of the rays O + t D (O, D: (..., 3)
    world frame, D not necessarily normalised, either may be a single (3,)
    vector) with the collision primitive ``s`` of :func:`collision_shapes`:
    0 when the origin lies inside, inf when the ray misses."""
    R = s["R"]
    O = np.asarray(O, dtype=float)
    D = np.asarray(D, dtype=float)
    shape = np.broadcast_shapes(O.shape, D.shape)[:-1]
    o = np.broadcast_to((O - s["p"]) @ R, shape + (3,))          # R^T (O - p), row vectors
    d = np.broadcast_to(D @ R, shape + (3,))
    inf = np.full(shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        if s["type"] == "box":
            h = 0.5 * s["size"]
            t0, t1 = np.full(shape, -np.inf), np.full(shape, np.inf)
            for k in range(3):
                dk, ok_ = d[..., k], o[..., k]
                par = np.abs(dk) < 1e-12
                a, b = (-h[k] - ok_) / dk, (h[k] - ok_) / dk
                lo, hi = np.minimum(a, b), np.maximum(a, b)
                lo = np.where(par, np.where(np.abs(ok_) <= h[k], -np.inf, np.inf), lo)
                hi = np.where(par, np.where(np.abs(ok_) <= h[k], np.inf, -np.inf), hi)
                t0, t1 = np.maximum(t0, lo), np.minimum(t1, hi)
            hit = (t0 <= t1) & (t1 >= 0.0)
            return np.where(hit, np.maximum(t0, 0.0), inf)
        if s["type"] == "sphere":
            r = s["radius"]
            a = np.sum(d * d, axis=-1)
            b = np.sum(o * d, axis=-1)
            c = np.sum(o * o, axis=-1) - r * r
            disc = b * b - a * c
            t = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
            return np.where(c <= 0.0, 0.0, np.where((disc >= 0.0) & (t >= 0.0), t, inf))
        # cylinder (axis = local z, |z| <= length / 2)
        r, hl = s["radius"], 0.5 * s["length"]
        best = inf.copy()
        inside = (o[..., 0] ** 2 + o[..., 1] ** 2 <= r * r) & (np.abs(o[..., 2]) <= hl)
        a = d[..., 0] ** 2 + d[..., 1] ** 2
        b = o[..., 0] * d[..., 0] + o[..., 1] * d[..., 1]
        c = o[..., 0] ** 2 + o[..., 1] ** 2 - r * r
        disc = b * b - a * c
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
        zc = o[..., 2] + t * d[..., 2]
        ok = (a > 1e-12) & (disc >= 0.0) & (t >= 0.0) & (np.abs(zc) <= hl)
        best = np.where(ok, np.minimum(best, t), best)
        for zcap in (-hl, hl):
            t = (zcap - o[..., 2]) / d[..., 2]
            xc, yc = o[..., 0] + t * d[..., 0], o[..., 1] + t * d[..., 1]
            ok = (np.abs(d[..., 2]) > 1e-12) & (t >= 0.0) & (xc * xc + yc * yc <= r * r)
            best = np.where(ok, np.minimum(best, t), best)
        return np.where(inside, 0.0, best)


def _ray_down(s, x: np.ndarray, y: np.ndarray, z0: float) -> np.ndarray:
    """Distance t >= 0 along (0, 0, -1) from (x, y, z0) to the shape surface
    (0 when the start point lies inside, inf when the ray misses)."""
    O = np.stack([x, y, np.full_like(x, z0)], axis=-1)
    return ray_hit(s, O, np.array([0.0, 0.0, -1.0]))


def shape_aabb(s) -> Tuple[np.ndarray, np.ndarray]:
    """World-frame axis-aligned bounding box (lo, hi) of a collision primitive."""
    return _shape_aabb(s)


class HeightField:
    """Ground-truth terrain height of a world on a regular grid: the first
    collision surface hit by a vertical ray cast down from ``z_top`` (walls and
    anything taller than z_top read as z_top).  Used as the harness's stand-in
    for /elevation_map/points (``--map``) and for the fall metric (base height
    above the ground)."""

    def __init__(self, shapes: list, x_range: Tuple[float, float], y_range: Tuple[float, float],
                 res: float = 0.02, z_top: float = 1.2, z_floor: float = -5.0):
        self.res = float(res)
        self.x0, self.y0 = float(x_range[0]), float(y_range[0])
        nx = int(math.ceil((x_range[1] - x_range[0]) / res)) + 1
        ny = int(math.ceil((y_range[1] - y_range[0]) / res)) + 1
        self.z_top, self.z_floor = float(z_top), float(z_floor)
        Z = np.full((ny, nx), self.z_floor)
        for s in shapes:
            lo, hi = _shape_aabb(s)
            if lo[2] > z_top:
                continue
            c0, c1 = int(math.floor((lo[0] - self.x0) / res)), int(math.ceil((hi[0] - self.x0) / res))
            r0, r1 = int(math.floor((lo[1] - self.y0) / res)), int(math.ceil((hi[1] - self.y0) / res))
            c0, r0 = max(c0, 0), max(r0, 0)
            c1, r1 = min(c1, nx - 1), min(r1, ny - 1)
            if c1 < c0 or r1 < r0:
                continue
            X, Y = np.meshgrid(self.x0 + res * np.arange(c0, c1 + 1), self.y0 + res * np.arange(r0, r1 + 1))
            t = _ray_down(s, X, Y, self.z_top)
            zs = np.where(np.isfinite(t), self.z_top - t, -np.inf)
            Z[r0:r1 + 1, c0:c1 + 1] = np.maximum(Z[r0:r1 + 1, c0:c1 + 1], zs)
        self.Z = Z

    @classmethod
    def plant(cls, res: float = 0.02, z_top: float = 1.2, extra=()) -> "HeightField":
        """plant.sdf (all static collisions incl. the inlined models, plus the
        ``extra`` novel boxes), whole world."""
        shapes = collision_shapes(plant_world_tree()) + extra_shapes(extra)
        return cls(shapes, (-10.6, 64.2), (-9.6, 9.6), res=res, z_top=z_top)

    def height(self, x, y) -> np.ndarray:
        """Nearest-cell height (vectorised); z_floor outside the grid."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        c = np.rint((x - self.x0) / self.res).astype(int)
        r = np.rint((y - self.y0) / self.res).astype(int)
        ok = (c >= 0) & (c < self.Z.shape[1]) & (r >= 0) & (r < self.Z.shape[0])
        out = np.full(np.broadcast(x, y).shape, self.z_floor)
        out[ok] = self.Z[r[ok], c[ok]]
        return out

    def ground(self, x: float, y: float, radius: float = 0.25) -> float:
        """Ground height under a point: median of the cells within ``radius``
        (robust against a pipe / rubble block right below the base)."""
        n = int(radius / self.res)
        c, r = int(round((x - self.x0) / self.res)), int(round((y - self.y0) / self.res))
        sub = self.Z[max(r - n, 0):r + n + 1, max(c - n, 0):c + n + 1]
        return float(np.median(sub)) if sub.size else self.z_floor


def terrain_height_fn(terrain: str, extra=()):
    """Vectorised ground-truth height function (x, y) -> z of a harness terrain
    (``extra`` novel boxes included: truth map / fall metric see them)."""
    if terrain == PLANT:
        hf = HeightField.plant(extra=extra)
        return hf.height, hf
    specs = _extra_specs(extra)
    if not specs:
        return (lambda x, y, _t=terrain: terrain_height(_t, x, y)), None

    def fn(x, y, _t=terrain, _s=specs):
        z = terrain_height(_t, x, y)
        X, Y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        for e in _s:
            c, s = math.cos(e["yaw"]), math.sin(e["yaw"])
            u = c * (X - e["center"][0]) + s * (Y - e["center"][1])
            v = -s * (X - e["center"][0]) + c * (Y - e["center"][1])
            inside = (np.abs(u) <= 0.5 * e["size"][0]) & (np.abs(v) <= 0.5 * e["size"][1])
            z = np.where(inside, np.maximum(z, e["floor_z"] + e["size"][2]), z)
        return z
    return fn, None
