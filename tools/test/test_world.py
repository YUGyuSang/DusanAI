"""
Tests of the generated virtual plant (tools/generate_world.py output) and of
the geometry helpers of the generator.  They run without ROS / Gazebo:

    python -m pytest tools/test/test_world.py -q

They check the committed artefacts (simulation/worlds/plant.sdf, the models,
simulation/maps/plant_map.*, simulation/config/generated/world_geometry.yaml)
against the layout and the project requirements.
"""
import math
import os
import re
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml

import generate_world as gw
from convert_environment import read_obj
from dusan_common import gauge_design
from dusan_common.grid_map import OccupancyGridMap
from dusan_common.layout import Layout
from dusan_common.planning import plan

REPO = gw.REPO
WORLD = gw.WORLD_PATH
MODELS = gw.MODELS_DIR
GEOM = os.path.join(gw.GENERATED_DIR, "world_geometry.yaml")


@pytest.fixture(scope="module")
def layout():
    return Layout.load(gw.LAYOUT_PATH)


@pytest.fixture(scope="module")
def world():
    return ET.parse(WORLD).getroot()


@pytest.fixture(scope="module")
def geom():
    with open(GEOM, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def nav(layout):
    return OccupancyGridMap.load(layout.nav_map_yaml())


def _pose(text):
    v = [float(x) for x in (text or "0 0 0 0 0 0").split()]
    return v + [0.0] * (6 - len(v))


def _includes(world):
    return {inc.findtext("name"): inc for inc in world.iter("include")}


# =============================================================================
# SDF structure
# =============================================================================
REQUIRED_PLUGINS = {
    "gz-sim-physics-system": "gz::sim::systems::Physics",
    "gz-sim-user-commands-system": "gz::sim::systems::UserCommands",
    "gz-sim-scene-broadcaster-system": "gz::sim::systems::SceneBroadcaster",
    "gz-sim-sensors-system": "gz::sim::systems::Sensors",
    "gz-sim-imu-system": "gz::sim::systems::Imu",
    "gz-sim-contact-system": "gz::sim::systems::Contact",
    "gz-sim-apply-link-wrench-system": "gz::sim::systems::ApplyLinkWrench",
    "gz-sim-particle-emitter-system": "gz::sim::systems::ParticleEmitter",
}

# children allowed by the SDF 1.9 spec for the elements the generator writes
ALLOWED_CHILDREN = {
    "world": {"physics", "gravity", "plugin", "scene", "atmosphere", "light", "include", "model",
              "magnetic_field", "wind", "spherical_coordinates", "gui", "audio", "frame", "joint",
              "actor", "road", "population", "state"},
    "physics": {"max_step_size", "real_time_factor", "real_time_update_rate", "max_contacts", "dart", "ode",
                "bullet", "simbody"},
    "scene": {"ambient", "background", "shadows", "grid", "origin_visual", "sky", "fog"},
    "atmosphere": {"temperature", "pressure", "temperature_gradient"},
    "light": {"pose", "cast_shadows", "diffuse", "specular", "attenuation", "direction", "spot", "intensity",
              "visualize", "light_on"},
    "attenuation": {"range", "constant", "linear", "quadratic"},
    "spot": {"inner_angle", "outer_angle", "falloff"},
    "include": {"uri", "name", "pose", "static", "placement_frame", "plugin", "experimental:params"},
    "model": {"static", "pose", "link", "self_collide", "allow_auto_disable", "include", "model", "joint",
              "plugin", "gripper", "frame", "enable_wind"},
    "link": {"pose", "inertial", "collision", "visual", "sensor", "particle_emitter", "gravity", "kinematic",
             "must_be_base_link", "velocity_decay", "self_collide", "enable_wind", "projector", "audio_sink",
             "audio_source", "battery", "light"},
    "collision": {"pose", "geometry", "surface", "laser_retro", "max_contacts"},
    "visual": {"pose", "geometry", "material", "plugin", "cast_shadows", "transparency", "laser_retro",
               "meta", "visibility_flags"},
    "geometry": {"box", "cylinder", "sphere", "mesh", "plane", "capsule", "ellipsoid", "heightmap",
                 "polyline", "empty"},
    "box": {"size"},
    "cylinder": {"radius", "length"},
    "mesh": {"uri", "scale", "submesh"},
    "material": {"ambient", "diffuse", "specular", "emissive", "pbr", "script", "shader", "lighting",
                 "render_order", "double_sided"},
    "pbr": {"metal", "specular"},
    "metal": {"metalness", "roughness", "albedo_map", "roughness_map", "metalness_map", "normal_map",
              "environment_map", "ambient_occlusion_map", "emissive_map", "light_map"},
    "surface": {"friction", "contact", "bounce", "soft_contact"},
    "friction": {"ode", "bullet", "torsional"},
    "ode": {"mu", "mu2", "fdir1", "slip1", "slip2"},
    "particle_emitter": {"emitting", "duration", "size", "particle_size", "lifetime", "rate", "min_velocity",
                         "max_velocity", "scale_rate", "color_start", "color_end", "color_range_image", "topic",
                         "particle_scatter_ratio", "pose", "material"},
}


def _check_schema(elem, path=""):
    errors = []
    allowed = ALLOWED_CHILDREN.get(elem.tag)
    if allowed is not None and not (elem.tag == "ode" and path.endswith("physics")):
        for c in elem:
            if c.tag not in allowed:
                errors.append(f"{path}/{elem.tag}: unexpected <{c.tag}>")
    # sibling name uniqueness per element type
    seen = {}
    for c in elem:
        nm = c.get("name")
        if nm is not None and c.tag in ("model", "link", "visual", "collision", "light", "particle_emitter"):
            key = (c.tag, nm)
            if key in seen:
                errors.append(f"{path}/{elem.tag}: duplicate {c.tag} name '{nm}'")
            seen[key] = True
        if c.tag == "plugin":            # plugin parameters are free-form
            continue
        errors += _check_schema(c, f"{path}/{elem.tag}")
    return errors


def test_world_sdf_well_formed(world, layout):
    assert world.tag == "sdf" and float(world.get("version")) >= 1.9
    worlds = world.findall("world")
    assert len(worlds) == 1 and worlds[0].get("name") == layout.get("world_name", "plant") == "plant"
    w = worlds[0]
    ph = w.find("physics")
    assert ph.get("type") == "dart"
    assert float(ph.findtext("max_step_size")) == pytest.approx(0.001)
    assert float(ph.findtext("real_time_factor")) == pytest.approx(1.0)
    plugins = {p.get("filename"): p for p in w.findall("plugin")}
    for fn, nm in REQUIRED_PLUGINS.items():
        assert fn in plugins, fn
        assert plugins[fn].get("name") == nm
    assert plugins["gz-sim-sensors-system"].findtext("render_engine") == "ogre2"
    atm = w.find("atmosphere")
    assert atm.get("type") == "adiabatic"
    assert float(atm.findtext("temperature")) == pytest.approx(layout.thermal["ambient_k"])
    # lighting: indoor lamps, one shadowless directional, shadows mostly off
    lights = w.findall("light")
    assert len(lights) >= 10
    dirs = [li for li in lights if li.get("type") == "directional"]
    assert len(dirs) == 1 and dirs[0].findtext("cast_shadows") == "false"
    assert sum(li.findtext("cast_shadows") == "true" for li in lights) <= 2
    # the robot is spawned by the launch file, never part of the world
    names = [m.get("name") for m in w.findall("model")] + [i.findtext("name") for i in w.findall("include")]
    assert layout["robot_name"] not in names
    assert len(names) == len(set(names)), "duplicate top level model names"


def test_sdf_schema_and_unique_names(world):
    errors = _check_schema(world)
    for d in os.listdir(MODELS):
        p = os.path.join(MODELS, d, "model.sdf")
        if os.path.isfile(p):
            errors += [f"{d}: {e}" for e in _check_schema(ET.parse(p).getroot())]
    assert not errors, "\n".join(errors[:20])


def test_positive_sizes(world):
    for g in world.iter("geometry"):
        b = g.find("box")
        if b is not None:
            assert all(float(v) > 0 for v in b.findtext("size").split())
        c = g.find("cylinder")
        if c is not None:
            assert float(c.findtext("radius")) > 0 and float(c.findtext("length")) > 0


# =============================================================================
# Files referenced by the world
# =============================================================================
def _model_path(uri):
    assert uri.startswith("model://"), uri
    rest = uri[len("model://"):]
    name, _, sub = rest.partition("/")
    return name, os.path.join(MODELS, name, *sub.split("/")) if sub else os.path.join(MODELS, name)


def _exists_exact(path):
    """os.path.isfile with exact-case path components below the repo: the
    files are written on Windows (case-insensitive) but loaded by Gazebo on
    Linux, where 'Foo.jpg' != 'foo.jpg'."""
    if not os.path.isfile(path):
        return False
    rel = os.path.relpath(os.path.normpath(path), REPO)
    cur = REPO
    for part in rel.split(os.sep):
        if part not in os.listdir(cur):
            return False
        cur = os.path.join(cur, part)
    return True


def _check_obj(path):
    assert _exists_exact(path), path
    base = os.path.dirname(path)
    mtls = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("mtllib "):
                mtls.append(line.split(None, 1)[1].strip())
    assert mtls, f"{path}: no mtllib"
    for m in mtls:
        mp = os.path.join(base, m)
        assert _exists_exact(mp), mp
        with open(mp, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("map_Kd"):
                    tex = line.split()[-1]
                    assert " " not in line.strip().split(None, 1)[1], f"space in texture name: {line}"
                    assert _exists_exact(os.path.join(base, tex)), f"{mp}: missing texture {tex} (exact case)"


def test_every_uri_exists(world):
    uris = [u.text.strip() for u in world.iter("uri")]
    assert uris
    checked_models = set()
    for uri in uris:
        name, path = _model_path(uri)
        mdir = os.path.join(MODELS, name)
        assert os.path.isdir(mdir) and name in os.listdir(MODELS), uri
        assert _exists_exact(os.path.join(mdir, "model.config")), uri
        if path.endswith(".obj"):
            _check_obj(path)
        if name not in checked_models:
            checked_models.add(name)
            cfg = ET.parse(os.path.join(mdir, "model.config")).getroot()
            sdf_file = cfg.find("sdf").text.strip()
            msdf = ET.parse(os.path.join(mdir, sdf_file)).getroot()
            assert msdf.find("model").get("name") == name
            for u in msdf.iter("uri"):
                n2, p2 = _model_path(u.text.strip())
                assert n2 == name, f"{name} references another model: {u.text}"
                if p2.endswith(".obj"):
                    _check_obj(p2)
            for a in msdf.iter("albedo_map"):
                assert _exists_exact(_model_path(a.text.strip())[1]), a.text
    for want in ("factory_hall", "corridor", "pallet", "gas_leak_valve", "charging_station", "access_point"):
        assert want in checked_models
    assert {f"gauge_g0{k}" for k in range(1, 9)} <= checked_models
    # machines are inline models with scaled mesh visuals
    meshes = {m.findtext("uri"): m.findtext("scale") for m in world.iter("mesh")}
    assert "model://turbine_machine/meshes/turbine_machine.obj" in meshes
    assert meshes["model://gas_tank/meshes/gas_tank.obj"].split()[0] == "0.8"


def test_converted_meshes_have_normals_uvs_and_size_budget():
    total = 0
    for name in ("factory_hall", "corridor", "turbine_machine", "gas_tank", "pallet"):
        m = read_obj(os.path.join(MODELS, name, "meshes", name + ".obj"))
        assert m.n_tris > 100 and len(m.uv) > 0 and len(m.vn) > 0
        assert np.all(m.fuv >= 0) and np.all(m.fn >= 0)
        assert any(mat.texture_file for mat in m.materials), name
        for root, _d, files in os.walk(os.path.join(MODELS, name)):
            total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    assert total < 80e6, f"converted models are {total / 1e6:.1f} MB"


def test_textures_downscaled():
    import cv2
    for name in ("factory_hall", "corridor", "turbine_machine", "gas_tank", "pallet"):
        d = os.path.join(MODELS, name, "meshes")
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED)
                assert img is not None and max(img.shape[:2]) <= 1024, f


def test_models_readme_lists_every_model():
    with open(os.path.join(MODELS, "README.md"), "r", encoding="utf-8") as f:
        text = f.read()
    for d in sorted(os.listdir(MODELS)):
        if os.path.isdir(os.path.join(MODELS, d)) and d != "spot":
            assert f"`{d}`" in text, d


# =============================================================================
# Gauges: dial frame, orientation, texture
# =============================================================================
def _disc_uv_at(local_pt, mesh):
    """UV of a point lying in the dial plane (barycentric in the containing tri)."""
    tri = mesh.v[mesh.fv]
    p = np.asarray(local_pt)
    for k in range(len(tri)):
        a, b, c = tri[k]
        n = np.cross(b - a, c - a)
        nn = n @ n
        w1 = np.cross(c - b, p - b) @ n / nn
        w2 = np.cross(a - c, p - c) @ n / nn
        w3 = 1.0 - w1 - w2
        if min(w1, w2, w3) >= -1e-9:
            uv = mesh.uv[mesh.fuv[k]]
            return w1 * uv[0] + w2 * uv[1] + w3 * uv[2]
    return None


@pytest.mark.parametrize("gid", [f"G0{k}" for k in range(1, 9)])
def test_gauge_dial_pose_and_orientation(gid, layout, world):
    """Dial centre = layout position, face normal = layout normal, front facing
    winding, and the texture seen from the front is upright and not mirrored:
    the scale ends (min at 7:30, max at 4:30 with the red band) and the top of
    the dial land where gauge_design puts them for a viewer in front."""
    g = layout.gauge(gid)
    inc = _includes(world)[f"gauge_{gid}"]
    x, y, z, r, p, yw = _pose(inc.findtext("pose"))
    R = gw.rot_rpy(r, p, yw)
    c = np.array([x, y, z])
    n = np.asarray(g["normal"], float)
    n /= np.linalg.norm(n)
    assert np.allclose(c, g["position"], atol=1e-4)
    assert np.allclose(R[:, 0], n, atol=1e-4)
    name = f"gauge_{gid.lower()}"
    mesh = read_obj(os.path.join(MODELS, name, "meshes", "dial.obj"))
    assert np.allclose(mesh.v[:, 0], 0.0)                       # dial plane through the origin
    tw = (mesh.v @ R.T + c)[mesh.fv]
    fn = np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0])
    assert np.all(fn @ n > 0), "dial faces must be counter-clockwise seen from the front"
    radius = np.abs(mesh.v[:, 1:]).max()
    D = float(g["dial_diameter"])
    assert radius * 2 == pytest.approx(D, rel=1e-3)
    import cv2
    img = cv2.imread(os.path.join(MODELS, name, "meshes", f"{name}_dial.png"))
    S = img.shape[0]
    # viewer in front looking along -n with world up: his right = (-n) x up
    up_w = np.array([0.0, 0.0, 1.0])
    up = up_w - (up_w @ n) * n
    up /= np.linalg.norm(up)
    right = np.cross(-n, up)
    # module2 convention (gauge_synth / gauge_association): the texture side
    # spans dial_diameter metres -> a point rho metres from the centre lies
    # rho / D * S texture pixels from the texture centre (white face radius
    # = gauge_design.R_FACE * D in the world).
    for phi_deg in (gauge_design.PHI_MIN_DEG, gauge_design.PHI_MAX_DEG, 90.0, 0.0, 180.0):
        for rho in (0.8 * radius, gauge_design.R_FACE * D):
            a = math.radians(phi_deg)
            pw = c + rho * (math.cos(a) * right + math.sin(a) * up)
            local = R.T @ (pw - c)
            uv = _disc_uv_at(local, mesh)
            assert uv is not None
            px = np.array([uv[0] * S, (1.0 - uv[1]) * S])
            r_t = rho / D * S
            expect = np.array([S / 2 + r_t * math.cos(a), S / 2 - r_t * math.sin(a)])
            # tolerance: chord of the 72-segment disc (<= 0.1 % of the radius) + 1 px
            assert np.linalg.norm(px - expect) < 1.5, (gid, phi_deg, rho, px, expect)
    # the red danger band is on the viewer's lower right (max end of the scale)
    phi = math.radians(gauge_design.PHI_MAX_DEG + 0.3 * gauge_design.DANGER_FRACTION * gauge_design.SWEEP_DEG)
    rho = (gauge_design.R_TICK_OUT - 0.012) * D
    pw = c + rho * (math.cos(phi) * right + math.sin(phi) * up)
    uv = _disc_uv_at(R.T @ (pw - c), mesh)
    b, gch, rch = img[int((1 - uv[1]) * S), int(uv[0] * S)].astype(int)
    assert rch > 150 and gch < 90 and b < 90, "red band not at the viewer's lower right"
    assert (pw - c) @ right > 0


def test_gauge_textures_match_layout_values(layout):
    import cv2
    for g in layout.gauges:
        name = f"gauge_{g['id'].lower()}"
        img = cv2.imread(os.path.join(MODELS, name, "meshes", f"{name}_dial.png"))
        ref = gauge_design.render_gauge_face(g["value"], g["min"], g["max"], g.get("unit", ""), g.get("tag", ""),
                                             size=img.shape[0])
        assert np.abs(img.astype(int) - ref.astype(int)).max() <= 1, g["id"]


def test_gauge_stands_reach_the_floor(geom):
    for gf in geom["gauges"]:
        sdf = ET.parse(os.path.join(MODELS, gf["model"], "model.sdf")).getroot()
        posts = [v for v in sdf.iter("visual") if v.get("name") == "post_visual"]
        if gf["mounted_on"]:
            assert not posts
            continue
        assert posts, gf["id"]
        x, y, z, *_ = _pose(posts[0].findtext("pose"))
        length = float(posts[0].find("geometry/cylinder/length").text)
        bottom_world = gf["position"][2] + z - length / 2
        assert bottom_world == pytest.approx(gf["floor_z"], abs=0.01), gf["id"]


# =============================================================================
# Thermal, obstacles, leak, dock
# =============================================================================
def _thermal_visuals(root):
    out = {}
    for v in root.iter("visual"):
        for p in v.findall("plugin"):
            if p.get("filename") == "gz-sim-thermal-system":
                assert p.get("name") == "gz::sim::systems::Thermal"
                out[v.get("name")] = float(p.findtext("temperature"))
    return out


def test_hot_spots_and_pipe_temperatures(world, layout):
    th = _thermal_visuals(world)
    for h in layout.hotspots:
        key = f"hotspot_{h['id'].lower()}_{h['kind']}_visual"
        assert th.get(key) == pytest.approx(h["temp_k"]), key
    assert th["rack_pipe_visual"] == pytest.approx(layout.thermal["pipe_body_k"])
    assert th["mesh_visual"] == pytest.approx(gw.MACHINE_TEMP_K)
    # hot spot primitives exactly at the layout positions
    poses = {v.get("name"): _pose(v.findtext("pose")) for v in world.iter("visual")}
    for h in layout.hotspots:
        p = poses[f"hotspot_{h['id'].lower()}_{h['kind']}_visual"]
        assert np.allclose(p[:3], h["position"], atol=1e-3)


def test_obstacles_present(world, layout):
    models = {m.get("name"): m for m in world.iter("model")}
    incs = _includes(world)
    n_area = {"corridor": 0, "factory": 0}
    for ob in layout.obstacles:
        mid = "obstacle_" + ob["id"].lower().replace("-", "_")
        if ob["type"] == "pallet":
            stack = [k for k in incs if k.startswith(mid + "_")]
            assert len(stack) == int(ob.get("stack", 1)), ob["id"]
        else:
            assert mid in models, ob["id"]
            if ob["type"] == "rubble":
                chunks = models[mid].findall("link/collision")
                assert len(chunks) == ob["count"]
            if ob["type"] == "plate":
                mu = float(models[mid].find("link/collision/surface/friction/ode/mu").text)
                assert mu == pytest.approx(ob["mu"])
        n_area[ob["area"]] += 1
    assert n_area["corridor"] >= 3 and n_area["factory"] >= 5


def test_corridor_rubble_leaves_north_passage(geom, layout):
    """C-OBS3 (blocks 2-6 cm, not crossable yet) keeps >= 0.9 m of free floor along the
    north wall: every block (bounding circle) south of the rect edge, and the strip between
    the rect and the measured north wall face wider than 0.9 m; the blocks still span
    most of the rest of the corridor (it stays an obstacle)."""
    ob = next(o for o in layout.obstacles if o["id"] == "C-OBS3")
    x0, y0, x1, y1 = ob["rect"]
    north = geom["architecture"]["corridor"]["wall_faces"]["north_upper"]
    assert north - y1 >= 0.9
    chunks = [p for p in geom["primitives"] if p["model"] == "obstacle_c_obs3"]
    assert len(chunks) == ob["count"]
    ymax = max(p["pose"][1] + 0.5 * math.hypot(p["size"][0], p["size"][1]) for p in chunks)
    ymin = min(p["pose"][1] - 0.5 * math.hypot(p["size"][0], p["size"][1]) for p in chunks)
    assert ymax <= y1 + 1e-6 and north - ymax >= 0.9
    assert ymin < y0 + 0.2 and ymax - ymin > 1.4                 # still blocks the south 1.5 m
    assert all(ob["h_min"] - 1e-6 <= p["size"][2] <= ob["h_max"] + 1e-6 for p in chunks)


def test_gas_leak_emitter_at_source(world, layout):
    inc = _includes(world)["gas_leak_valve"]
    mp = _pose(inc.findtext("pose"))
    sdf = ET.parse(os.path.join(MODELS, "gas_leak_valve", "model.sdf")).getroot()
    em = sdf.find("model/link/particle_emitter")
    assert em is not None
    ep = _pose(em.findtext("pose"))
    src = np.array(mp[:3]) + np.array(ep[:3])                 # include has no rotation
    assert np.allclose(src, layout.gas_leak["source"], atol=1e-3)
    w = layout.gas_leak["wind_base"]
    assert ep[5] == pytest.approx(math.atan2(w[1], w[0]), abs=1e-4)
    assert float(em.findtext("particle_scatter_ratio")) == 0.0
    assert np.allclose(mp[:2], layout.gas_leak["valve_prop"][:2], atol=1e-6)


def test_dock_marking_points_along_dock_heading(world, layout):
    """The pad marking (texture top = docking arrow) points along the dock
    heading in the world and is not mirrored seen from above."""
    m = read_obj(os.path.join(MODELS, "charging_station", "meshes", "dock_marking.obj"))
    tri = m.v[m.fv]
    assert np.all(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] > 0), "marking must face up"
    # affine map uv -> local xy from the quad corners (least squares, exact for the quad)
    A = np.c_[m.uv[m.fuv].reshape(-1, 2), np.ones(m.fuv.size)]
    sol, *_ = np.linalg.lstsq(A, m.v[m.fv].reshape(-1, 3)[:, :2], rcond=None)
    du, dv = sol[0], sol[1]                                    # local xy per unit u / v
    assert du[0] * dv[1] - du[1] * dv[0] > 0, "texture mirrored seen from above"
    yaw = _pose(_includes(world)["charging_station"].findtext("pose"))[5]
    Rz = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    heading = np.array([math.cos(layout.dock_pose[2]), math.sin(layout.dock_pose[2])])
    up_world = Rz @ (dv / np.linalg.norm(dv))                  # image up (arrow) in the world
    assert up_world @ heading > 0.999


def test_robot_spawn_is_clear_and_just_above_the_floor(geom, layout):
    """The launch file spawns the robot with straight legs at layout robot_spawn:
    no collision primitive may intersect the robot there and the feet must be
    within a few cm above the walkable surface."""
    from dusan_common import robot_model as rm
    x, y, z, yaw = layout.spawn
    leg = rm.UPPER_LEG_LENGTH + rm.LOWER_LEG_LENGTH + rm.FOOT_RADIUS
    prims = [_prim_from_record(p) for p in geom["primitives"]]
    tops = [float(p.top_z_at(x, y)) for p in prims
            if p.nav == "walkable" and p.kind in ("floor", "tread", "ramp", "landing")
            and _inside(p.footprint(), x, y)]
    assert tops, "no floor under the spawn point"
    gap = (z - leg) - max(tops)
    assert 0.0 <= gap < 0.03, gap
    # robot envelope: trunk box + legs (hip_x .. leg plane), from the floor to the trunk top
    hx = max(rm.BASE_BOX[0] / 2, rm.HIP_X_OFFSET + 0.06)
    hy = max(rm.BASE_BOX[1] / 2, rm.LEG_PLANE_Y + 0.05)
    c, s = math.cos(yaw), math.sin(yaw)
    rect = np.array([[x + c * a - s * b, y + s * a + c * b] for a, b in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])
    z0, z1 = max(tops) + 0.01, z + rm.BASE_BOX[2] / 2 + 0.05
    import cv2
    for rec, p in zip(geom["primitives"], prims):
        # walkable surfaces are below z0; visual-only details (dock contacts, LEDs) have no physics
        if p.nav == "walkable" or (not rec["collision"] and p.nav == "ignore"):
            continue
        zr = p.z_range()
        if zr[1] < z0 or zr[0] > z1:
            continue
        fp = p.footprint().astype(np.float32)
        inter, _ = cv2.intersectConvexConvex(fp, rect.astype(np.float32))
        assert inter < 1e-6, f"{p.name} intersects the robot at spawn"


def _inside(poly, x, y):
    import cv2
    return cv2.pointPolygonTest(np.asarray(poly, np.float32).reshape(-1, 1, 2), (float(x), float(y)), False) >= 0


def test_machines_inline_with_layout_pose_scale_and_collisions(world, layout, geom):
    models = {m.get("name"): m for m in world.iter("model")}
    recs = {r["id"]: r for r in geom["architecture"]["machines"]}
    for mach in layout["machines"]:
        mid = mach["id"].lower().replace("-", "_")
        m = models.get(f"machine_{mid}")
        assert m is not None, mach["id"]
        p = _pose(m.findtext("pose"))
        assert np.allclose(p[:3], mach["pose"][:3], atol=1e-4) and p[5] == pytest.approx(mach["pose"][3], abs=1e-4)
        mesh = m.find("link/visual/geometry/mesh")
        assert mesh.findtext("uri") == f"model://{mach['model']}/meshes/{mach['model']}.obj"
        assert [float(v) for v in mesh.findtext("scale").split()] == pytest.approx([mach.get("scale", 1.0)] * 3)
        cols = m.findall("link/collision")
        assert len(cols) >= 1
        # every collision box lies inside the placed mesh bounding box (world frame)
        (bx0, by0, _), (bx1, by1, bz1) = recs[mach["id"]]["mesh_bbox"]
        R = gw.rot_rpy(0, 0, p[5])
        for col in cols:
            q = _pose(col.findtext("pose"))
            size = [float(v) for v in col.findtext("geometry/box/size").split()]
            pr = gw.Prim("c", "box", tuple(q), tuple(size), "m")
            pts = pr.sample_points() @ R.T + np.array(p[:3])
            assert pts[:, 0].min() >= bx0 - 0.06 and pts[:, 0].max() <= bx1 + 0.06, col.get("name")
            assert pts[:, 1].min() >= by0 - 0.06 and pts[:, 1].max() <= by1 + 0.06, col.get("name")
            assert pts[:, 2].max() <= bz1 + 0.01


def test_dock_and_access_points(world, layout):
    incs = _includes(world)
    p = _pose(incs["charging_station"].findtext("pose"))
    cs = layout["charging_station"]
    assert np.allclose(p[:2], cs["pad"][:2]) and p[5] == pytest.approx(cs["dock_pose"][2], abs=1e-4)
    assert abs(p[2] - cs["pad"][2]) < 0.01                     # flush with the lower floor
    for ap in layout.comm["access_points"]:
        q = _pose(incs[f"access_point_{ap['id']}"].findtext("pose"))
        assert np.allclose(q[:3], ap["position"])
    ap_sdf = ET.parse(os.path.join(MODELS, "access_point", "model.sdf")).getroot()
    assert not list(ap_sdf.iter("collision")), "access points are visual only"


# =============================================================================
# Collision geometry vs. the visual mesh (what the depth camera sees)
# =============================================================================
def _mesh_top(tris, x, y, zmax):
    """Highest mesh surface at (x, y) below zmax (vertical ray cast)."""
    xs, ys = tris[:, :, 0], tris[:, :, 1]
    sel = (xs.min(1) <= x) & (xs.max(1) >= x) & (ys.min(1) <= y) & (ys.max(1) >= y)
    best = None
    for t in tris[sel]:
        (ax, ay), (bx, by), (cx, cy) = t[0, :2], t[1, :2], t[2, :2]
        d = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(d) < 1e-12:
            continue
        l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / d
        l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / d
        l3 = 1 - l1 - l2
        if min(l1, l2, l3) < -1e-9:
            continue
        z = l1 * t[0, 2] + l2 * t[1, 2] + l3 * t[2, 2]
        if z < zmax and (best is None or z > best):
            best = z
    return best


def _prim_from_record(p):
    return gw.Prim(p["name"], p["shape"], tuple(p["pose"]), tuple(p["size"]), p["model"], nav=p["nav"],
                   kind=p["kind"])


def test_walkable_collisions_coincide_with_mesh(geom, layout):
    cor = layout["corridor"]
    C = gw.load_placed_mesh(cor["mesh_model"], cor["pose"])
    F = gw.load_placed_mesh(layout["factory"]["mesh_model"], layout["factory"]["pose"])
    prims = [_prim_from_record(p) for p in geom["primitives"]]
    walk = {p.name: p for p in prims if p.nav == "walkable"}
    checks = 0
    for name, p in walk.items():
        if not (name.startswith("corridor_stair_") or name in ("corridor_ramp", "corridor_floor_upper",
                                                                 "corridor_floor_lower", "factory_floor")):
            continue
        fp = p.footprint()
        lo, hi = fp.min(0), fp.max(0)
        mesh = F if name == "factory_floor" else C
        for fx, fy in ((0.3, 0.3), (0.5, 0.5), (0.7, 0.6), (0.4, 0.75)):
            x = lo[0] + fx * (hi[0] - lo[0])
            y = lo[1] + fy * (hi[1] - lo[1])
            zc = float(p.top_z_at(x, y))
            zm = _mesh_top(mesh.tris, x, y, zc + 0.15)
            assert zm is not None, (name, x, y)
            if name == "corridor_ramp":
                # perforated grating: either a bar (on the plane) or a hole (deck / floor below)
                if abs(zm - zc) > 0.02:
                    continue
            assert abs(zm - zc) < 0.015, (name, x, y, zm, zc)
            checks += 1
    assert checks >= 20


def test_measured_architecture(geom):
    a = geom["architecture"]
    st = a["corridor"]["stairs"]["treads"]
    assert len(st) == 4
    rises = -np.diff([a["corridor"]["upper_floor_z"]] + [t["z"] for t in st] + [a["corridor"]["lower_floor_z"]])
    assert np.all((rises > 0.14) & (rises < 0.21)), rises
    r = a["corridor"]["ramp"]
    assert 8.0 < r["slope_deg"] < 20.0 and r["residual_max"] < 0.02
    assert a["corridor"]["lower_floor_z"] < a["corridor"]["upper_floor_z"] - 0.8
    assert len(a["factory"]["stair_treads"]) >= 10
    assert a["factory"]["interior_walls"] and a["factory"]["interior_walls"][0]["doors"]


def test_requirement_dimensions(geom):
    a = geom["architecture"]
    cx, cy = a["corridor"]["interior_size"]
    assert cx >= 45.0 and cy >= 3.5
    (x0, y0, _), (x1, y1, _) = a["factory"]["mesh_bbox"]
    assert x1 - x0 >= 25.0 and y1 - y0 >= 18.0              # building footprint
    fx, fy = a["factory"]["interior_size"]
    assert fx >= 24.5 and fy >= 17.5                          # clear interior


# =============================================================================
# Navigation map
# =============================================================================
def test_nav_map_metadata(nav, layout):
    nb = layout["nav_map"]["bounds"]
    assert nav.resolution == pytest.approx(layout["nav_map"]["resolution"])
    assert nav.origin_x == pytest.approx(nb[0]) and nav.origin_y == pytest.approx(nb[1])
    assert nav.width == round((nb[2] - nb[0]) / nav.resolution)
    assert nav.height == round((nb[3] - nb[1]) / nav.resolution)
    assert set(np.unique(nav.occ)) <= {0, 100}


def test_nav_map_semantics(nav, layout, geom):
    fw = geom["architecture"]["factory"]["wall_faces"]
    free = [layout.dock_pose[:2], (0.0, 0.0), (47.45, -1.0), (48.4, -1.0), (48.0, 0.8), (50.5, 0.8),
            (-8.5, 0.0),                                        # under the mezzanine
            (2.0, 6.4)]                                         # under the pipe rack pipe, between supports
    for ob in layout.obstacles:                                  # low debris is crossable
        if ob["type"] in ("rubble", "plate", "platform"):
            x0, y0, x1, y1 = ob["rect"]
            free.append(((x0 + x1) / 2, (y0 + y1) / 2))
        if ob["type"] == "pipe":
            free.append(tuple(ob["center"]))
    for x, y in free:
        assert nav.is_free(x, y), (x, y)
    occ = [(0.0, fw["north"] + 0.1), (fw["west"] - 0.1, 0.0), (30.0, 3.0), (5.0, 9.05)]
    for ob in layout.obstacles:
        if ob["type"] in ("drum", "pallet"):
            occ.append(tuple(ob["center"]))
    for k in layout["pipe_rack"]["supports_x"]:
        occ.append((k, layout["pipe_rack"]["pipe"]["start"][1]))
    for m in layout["machines"]:
        occ.append(tuple(m["pose"][:2]))
    occ.append(tuple(layout.gas_leak["valve_prop"][:2]))
    for x, y in occ:
        ix, iy = nav.world_to_grid(x, y)
        assert not nav.in_bounds(ix, iy) or nav.occ[iy, ix] == 100, (x, y)
    # drop-off between the ramp and the lower stairs is blocked
    xs = np.arange(48.6, 50.0, 0.1)
    ys = np.arange(-0.4, 0.05, 0.1)
    blocked_cols = [any(not nav.is_free(x, y) for y in ys) for x in xs]
    assert all(blocked_cols)


def test_astar_reachability(nav, layout):
    """Requirement: from the dock to the factory centre, 1.5 m in front of every
    gauge and to 0.8 m from the gas source (robot radius 0.30 m)."""
    start = layout.dock_pose[:2]
    goals = gw.test_goals(layout)
    assert len(goals) == 1 + len(layout.gauges) + 1
    for name, goal in goals.items():
        path = plan(nav, start, goal, robot_radius=0.30)
        assert path is not None, name
        end = np.asarray(path[-1])
        assert np.linalg.norm(end - goal) < 0.5, (name, end, goal)
        for x, y in path:
            assert nav.is_free(x, y), (name, x, y)


# =============================================================================
# GUI config
# =============================================================================
def test_gui_config():
    path = os.path.join(gw.SIM_DIR, "config", "gui.config")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    body = text.split("?>", 1)[1]
    root = ET.fromstring("<root>" + body + "</root>")
    plugins = [p.get("filename") for p in root.findall("plugin")]
    for need in ("MinimalScene", "GzSceneManager", "InteractiveViewControl", "CameraTracking", "MarkerManager",
                 "SelectEntities", "Spawn", "VisualizationCapabilities", "WorldControl", "WorldStats",
                 "EntityTree", "ComponentInspector", "KeyPublisher"):
        assert need in plugins, need
    topics = [p.findtext("topic") for p in root.findall("plugin") if p.get("filename") == "ImageDisplay"]
    assert sorted(topics) == ["/demo/hud", "/inspection_camera/image", "/thermal_camera/image"]
    pose = [float(v) for v in root.find("plugin/camera_pose").text.split()]
    assert 50 < pose[0] < 61.9 and abs(pose[1]) < 1.7       # inside the lower corridor, west of the dock
    for comment in re.findall(r"<!--(.*?)-->", text, re.S):
        assert "--" not in comment                             # well-formed XML comments


# =============================================================================
# Generator helpers (synthetic cases)
# =============================================================================
def test_clip_polygon_z():
    tri = [(0, 0, 0), (1, 0, 1), (0, 1, 2)]
    out = gw.clip_polygon_z(tri, 0.5, 1.5)
    assert len(out) >= 3
    assert all(0.5 - 1e-9 <= p[2] <= 1.5 + 1e-9 for p in out)
    assert gw.clip_polygon_z(tri, 3.0, 4.0) == []


def test_band_occupied_relative_to_floor():
    vox = np.zeros((10, 2, 2), np.uint8)       # bands of 0.1 from z = 0
    vox[3, 0, 0] = 1                           # z in [0.3, 0.4)
    floor = np.array([[0.0, 0.0], [np.nan, 0.25]])
    m = gw.band_occupied(vox, floor, 0.2, 0.7, z_lo=0.0, dz=0.1)
    assert m[0, 0] and not m[0, 1] and not m[1, 0]
    m2 = gw.band_occupied(vox, np.full((2, 2), 0.25), 0.2, 0.7, z_lo=0.0, dz=0.1)
    assert not m2[0, 0]                        # 0.35 is only 0.10 above that floor


def test_dropoff_mask():
    hm = np.zeros((5, 6))
    hm[:, 3:] = -0.5
    hm[0, 0] = np.nan
    d = gw.dropoff_mask(hm, 0.22)
    assert d[:, 2].all() and d[:, 3].all() and not d[:, 0].any() and not d[:, 5].any()
    hm2 = np.cumsum(np.full((4, 8), 0.18), axis=1)              # stairs: 0.18 risers stay walkable
    assert not gw.dropoff_mask(hm2, 0.22).any()


def test_decompose_rectangles_covers_mask():
    m = np.zeros((20, 30), bool)
    m[2:8, 3:25] = True
    m[8:18, 3:9] = True
    rects = gw.decompose_rectangles(m, min_cells=1, min_coverage=1.0)
    cov = np.zeros_like(m)
    for r0, r1, c0, c1 in rects:
        assert m[r0:r1 + 1, c0:c1 + 1].all()
        cov[r0:r1 + 1, c0:c1 + 1] = True
    assert np.array_equal(cov, m) and len(rects) == 2


def test_stair_flights_rejects_planks():
    flight = [{"z": 0.2 * (k + 1), "x0": -0.3 * (k + 1), "x1": -0.3 * k, "y0": 0.0, "y1": 1.0, "area": 0.3}
              for k in range(5)]
    planks = [{"z": 1.15, "x0": -3.0, "x1": -2.0, "y0": 2.0 + 0.3 * k, "y1": 2.25 + 0.3 * k, "area": 0.25}
              for k in range(4)]
    out = gw.stair_flights(flight + planks)
    assert len(out) == 5 and all(t["y0"] == 0.0 for t in out)


def test_prim_footprint_and_ramp_plane():
    p = gw.box("ramp", (0.0, 0.0, 0.0), (2.0, 1.0, 0.1), "g", pitch=math.radians(10))
    fp = p.footprint()
    assert fp[:, 0].max() == pytest.approx(math.cos(math.radians(10)) + 0.05 * math.sin(math.radians(10)), abs=1e-6)
    # top surface descends towards +x for positive pitch
    assert p.top_z_at(0.5, 0.0) < p.top_z_at(-0.5, 0.0)
    assert p.top_z_at(0.0, 0.0) == pytest.approx(0.05 / math.cos(math.radians(10)), abs=1e-9)


def test_dial_disc_mapping():
    v, uv, tris = gw.dial_disc_vertices(0.15, 36)
    # +Y edge of the disc -> right edge of the texture's inscribed circle, +Z -> top (v up);
    # texture side == dial diameter (module2 convention)
    k_y = np.argmax(v[:, 1])
    k_z = np.argmax(v[:, 2])
    assert uv[k_y, 0] == pytest.approx(1.0) and uv[k_y, 1] == pytest.approx(0.5)
    assert uv[k_z, 1] == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(uv - 0.5, v[:, 1:] / 0.30)
    n = np.cross(v[tris[:, 1]] - v[tris[:, 0]], v[tris[:, 2]] - v[tris[:, 0]])
    assert np.all(n[:, 0] > 0)


# =============================================================================
# Converter helpers (tools/convert_environment.py, synthetic cases)
# =============================================================================
def _grid_mesh(n=20, size=1.0):
    """Flat n x n quad grid (2 n^2 triangles) with per-corner UVs and normals."""
    import convert_environment as ce
    xs = np.linspace(0, size, n + 1)
    X, Y = np.meshgrid(xs, xs)
    v = np.c_[X.ravel(), Y.ravel(), np.zeros(X.size)]
    uv = v[:, :2] / size
    idx = np.arange(X.size).reshape(n + 1, n + 1)
    f = []
    for r in range(n):
        for c in range(n):
            a, b, d, e = idx[r, c], idx[r, c + 1], idx[r + 1, c + 1], idx[r + 1, c]
            f += [[a, b, d], [a, d, e]]
    f = np.asarray(f)
    return ce.Mesh(v, uv, np.array([[0.0, 0.0, 1.0]]), f, f.copy(), np.zeros_like(f), np.zeros(len(f), np.int64),
                   [ce.Material("m")])


def test_cluster_decimation_preserves_attributes():
    import convert_environment as ce
    m = _grid_mesh(40)
    d = ce.decimate_to(m, 800, log=lambda *_: None)
    assert 0 < d.n_tris <= 800
    lo, hi = d.bbox()
    assert np.allclose(lo[:2], 0.0, atol=0.03) and np.allclose(hi[:2], 1.0, atol=0.03)
    # surviving faces keep valid uv / normal indices and their orientation (+Z)
    assert d.fuv.min() >= 0 and d.fuv.max() < len(d.uv) and np.all(d.fn >= 0)
    assert np.all(d.face_normals()[:, 2] > 0.99)
    assert d.face_areas().sum() == pytest.approx(1.0, abs=0.05)


def test_double_sided_and_transform_winding():
    import convert_environment as ce
    m = _grid_mesh(2)
    ds = ce.make_double_sided(m, np.array([True]))
    assert ds.n_tris == 2 * m.n_tris
    n = ds.face_normals()
    assert np.sum(n[:, 2] > 0) == m.n_tris and np.sum(n[:, 2] < 0) == m.n_tris
    assert np.allclose(ds.vn[ds.fn[m.n_tris:]].mean((0, 1)), [0, 0, -1])
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])        # a mirroring transform must flip the winding
    t = m.transformed(mirror)
    assert np.all(t.face_normals()[:, 2] > 0)


def test_obj_roundtrip(tmp_path):
    import convert_environment as ce
    m = _grid_mesh(3)
    m.materials[0].texture = np.full((8, 8, 3), 120, np.uint8)
    m.materials[0].texture_key = "k"
    m.materials[0].texture_label = "tex"
    ce.write_obj(m, str(tmp_path), "grid")
    back = ce.read_obj(str(tmp_path / "grid.obj"), load_textures=True)
    assert back.n_tris == m.n_tris
    assert np.allclose(back.triangles(), m.triangles(), atol=1e-5)
    assert np.allclose(back.uv[back.fuv], m.uv[m.fuv], atol=1e-5)
    assert back.materials[0].texture_file == "grid_tex.jpg" and back.materials[0].texture is not None


def test_unity_material_parsing():
    import convert_environment as ce
    text = """--- !u!21 &2100000
Material:
  serializedVersion: 8
  m_Name: Floor_01
  m_SavedProperties:
    m_TexEnvs:
    - _BaseColorMap:
        m_Texture: {fileID: 2800000, guid: 1550f57882e97bc48aa527d55a4c4c0f, type: 3}
        m_Scale: {x: 2, y: 3}
        m_Offset: {x: 0.5, y: 0}
    - _BumpMap:
        m_Texture: {fileID: 0}
        m_Scale: {x: 1, y: 1}
        m_Offset: {x: 0, y: 0}
    m_Floats:
    - _DoubleSidedEnable: 1
    - _SurfaceType: 0
    m_Colors:
    - _BaseColor: {r: 0.8, g: 0.7, b: 0.6, a: 1}
"""
    info = ce.parse_material(text)
    assert info["name"] == "Floor_01"
    guid, sc, off = info["textures"]["_BaseColorMap"]
    assert guid == "1550f57882e97bc48aa527d55a4c4c0f" and sc == (2.0, 3.0) and off == (0.5, 0.0)
    assert "_BumpMap" not in info["textures"]
    assert info["floats"]["_DoubleSidedEnable"] == 1.0
    assert info["colors"]["_BaseColor"][:3] == (0.8, 0.7, 0.6)
    docs = ce.parse_unity_yaml_docs(text)
    assert docs[0][0] == 21 and docs[0][1] == 2100000
