#!/usr/bin/env python3
"""
Convert the Unity Spot rig into per-link Gazebo / RViz meshes.

Input : Doosanenerbility/.../robot_spot.unitypackage
        -> SpotHPRIG_Rigging_02.fbx  (skinned body + two rigid accessories)
        -> Material.001_Base_Color.png (4096^2 16-bit albedo of the body)
Output: simulation/models/spot/meshes/<link>.obj + spot.mtl
        simulation/models/spot/conversion_report.json
        docs/images/spot_model_preview.png

Pipeline
--------
1. Read the FBX with tools/fbx_reader.py (pure numpy).  Vertices are moved to
   the "G frame" (metres, x_G = robot right, y_G = robot forward, z up) with the
   file axis matrix and then to the robot base frame
       x_r = y_G - y0 ,  y_r = -(x_G - x0) ,  z_r = z_G - z0
   where (x0, y0, z0) is the centre of the four hip_x bones of the BIND pose
   (~ (0, 0.0145, 0.365)).
2. The body is rigidly skinned (every control point has exactly one cluster
   with weight 1).  Each vertex is assigned to the link of its max-weight bone
   and each triangle to the majority link of its corners -> 13 link meshes.
   The camera / EAP accessories are rigid on the body bone -> base_link.
3. The rest (bind) pose joint angles are recovered from the bind-pose bone
   positions (hip_x roll of the leg plane, thigh angle theta = atan2(-dx, -dz),
   calf angle phi, knee = phi - theta).  Each link's vertices are expressed in
   its own frame:  v_link = T_rest^-1 v_robot, where T_rest is the link pose
   given by forward kinematics (dusan_common.robot_model geometry) at the rest
   angles.  At zero joint angles the legs then hang straight down.
4. Colours: the albedo texture is sampled at every face's UV centroid,
   quantised with weighted k-means (CIE-Lab) into <= 6 flat materials and
   cleaned with a face-adjacency majority filter.  Accessories keep the
   diffuse colours of their FBX materials.
5. Decimation: batched quadric-error-metric (Garland & Heckbert) edge
   collapse, fully vectorised in numpy.  Every round collapses an independent
   set of cheap edges (each edge is the cheapest in its 2-ring, so no face is
   touched by two collapses), rejecting collapses that violate the link
   condition (non-manifold) or flip a face.  Open boundaries and colour
   borders are protected by constraint quadrics.
6. Per-corner normals are recomputed with a crease angle (smooth inside
   panels, sharp at hard edges); OBJ/MTL are written with flat Kd colours.
7. Verification: the decimated OBJ files are read back, assembled with FK at
   the rest angles and compared with the original high-poly vertices
   (point-to-surface distance); bone positions are compared with the FK joint
   positions; previews (rest / zero / stand pose, side + top + 3/4 view) are
   rendered with a small painter's-algorithm rasteriser (OpenCV).

Run (Windows or Linux; numpy + opencv, optional `pip install xacro`):
    python tools/convert_spot.py                 # meshes + report + preview + spot.urdf
    python tools/convert_spot.py --no-preview    # skip the preview sheet
    python tools/convert_spot.py --urdf-only     # only re-expand spot.urdf from the xacro
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, os.path.join(REPO, "dusan_common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dusan_common import robot_model as rm  # noqa: E402

MODEL_DIR = os.path.join(REPO, "simulation", "models", "spot")
MESH_DIR = os.path.join(MODEL_DIR, "meshes")
REPORT_PATH = os.path.join(MODEL_DIR, "conversion_report.json")
PREVIEW_PATH = os.path.join(REPO, "docs", "images", "spot_model_preview.png")
URDF_XACRO = os.path.join(REPO, "simulation", "urdf", "spot.urdf.xacro")
URDF_PATH = os.path.join(REPO, "simulation", "urdf", "spot.urdf")
MTL_NAME = "spot.mtl"

FBX_NAME = "SpotHPRIG_Rigging_02.fbx"
TEXTURE_NAME = "Material.001_Base_Color.png"

# ---------------------------------------------------------------------------
# Rig -> robot mapping
# ---------------------------------------------------------------------------
LEGS = rm.LEG_NAMES                       # ("fl", "fr", "rl", "rr")
# FBX side / segment names per leg.  "R" bones live at +x_G = robot right.
_LEG_BONES = {
    "fl": ("L", "Shoulder", "UpperArm", "Arm", "Hand"),
    "fr": ("R", "Shoulder", "UpperArm", "Arm", "Hand"),
    "rl": ("L", "Pelvis", "Thigh", "Calf", "Foot"),
    "rr": ("R", "Pelvis", "Thigh", "Calf", "Foot"),
}
LINK_SEGMENTS = ("hip", "upper_leg", "lower_leg")


def leg_bone_names(leg: str) -> dict[str, str]:
    """FBX bone names of a leg: keys hip, upper_leg, lower_leg, foot."""
    side, hip, upper, lower, foot = _LEG_BONES[leg]
    return {
        "hip": f"Spot_{side}_{hip}_Bone_01",
        "upper_leg": f"Spot_{side}_{upper}_Bone_01",
        "lower_leg": f"Spot_{side}_{lower}_Bone_01",
        "foot": f"Spot_{side}_{foot}_Bone_00",
    }


BODY_BONE = "Spot_Body_Bone_01"


def bone_to_link_map() -> dict[str, str]:
    """Skin-cluster bone name -> URDF link name."""
    out = {BODY_BONE: "base_link"}
    for leg in LEGS:
        names = leg_bone_names(leg)
        for seg in LINK_SEGMENTS:
            out[names[seg]] = f"{leg}_{seg}"
    return out


LINK_NAMES = ("base_link",) + tuple(f"{leg}_{seg}" for leg in LEGS for seg in LINK_SEGMENTS)

# Triangle budgets after decimation (total ~73k).
TRIANGLE_TARGETS = {
    "base_link": 19000,
    "base_cam": 4500,
    "base_eap": 1500,
    "hip": 3000,
    "upper_leg": 5000,
    "lower_leg": 4000,
}

N_BODY_COLOURS = 6
CREASE_ANGLE_DEG = 40.0


def target_for(mesh_name: str) -> int:
    if mesh_name in TRIANGLE_TARGETS:
        return TRIANGLE_TARGETS[mesh_name]
    return TRIANGLE_TARGETS[mesh_name.split("_", 1)[1]]


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (deliberately self-contained)
# ---------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def homogeneous(R: np.ndarray | None = None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to (N,3) points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def inv_rigid(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    return homogeneous(R.T, -R.T @ T[:3, 3])


# ---------------------------------------------------------------------------
# Kinematics (robot_model geometry; zero angles = legs straight down)
# ---------------------------------------------------------------------------
def leg_link_frames(leg_index: int, q) -> dict[str, np.ndarray]:
    """Poses (4x4, base frame) of hip / upper_leg / lower_leg / foot links.

    Mirrors the URDF joint chain:
      base -> hip_x (axis +X) at (sx*HIP_X_OFFSET, sy*HIP_Y_OFFSET, 0)
           -> hip_y (axis +Y) at (0, sy*HIP_TO_THIGH_Y, 0)
           -> knee  (axis +Y) at (0, 0, -UPPER_LEG_LENGTH)
           -> foot  (fixed)   at (0, 0, -LOWER_LEG_LENGTH)
    """
    q0, q1, q2 = (float(v) for v in q)
    sy = rm.LEG_SIGN_Y[leg_index]
    leg = LEGS[leg_index]
    T_hip = homogeneous(rot_x(q0), rm.hip_x_position(leg_index))
    T_upper = T_hip @ homogeneous(rot_y(q1), (0.0, sy * rm.HIP_TO_THIGH_Y, 0.0))
    T_lower = T_upper @ homogeneous(rot_y(q2), (0.0, 0.0, -rm.UPPER_LEG_LENGTH))
    T_foot = T_lower @ homogeneous(None, (0.0, 0.0, -rm.LOWER_LEG_LENGTH))
    return {f"{leg}_hip": T_hip, f"{leg}_upper_leg": T_upper,
            f"{leg}_lower_leg": T_lower, f"{leg}_foot": T_foot}


def all_link_frames(joint_angles: dict[str, tuple]) -> dict[str, np.ndarray]:
    """Link poses for every link given {leg: (q_hip_x, q_hip_y, q_knee)}."""
    frames = {"base_link": np.eye(4)}
    for i, leg in enumerate(LEGS):
        frames.update(leg_link_frames(i, joint_angles.get(leg, (0.0, 0.0, 0.0))))
    return frames


def leg_ik_sagittal(leg_index: int, foot_base) -> tuple[float, float, float]:
    """Closed-form IK for a foot position in the base frame (knee bent backwards).

    Used for the preview "stand" pose only (module1 owns the real IK)."""
    sy = rm.LEG_SIGN_Y[leg_index]
    p = np.asarray(foot_base, dtype=float) - np.asarray(rm.hip_x_position(leg_index))
    # hip_x: foot lies in the leg plane at lateral distance HIP_TO_THIGH_Y
    r_yz = math.hypot(p[1], p[2])
    d = rm.HIP_TO_THIGH_Y
    h = math.sqrt(max(r_yz * r_yz - d * d, 1e-12))
    q0 = math.atan2(p[2], p[1]) - math.atan2(-h, sy * d)
    q0 = (q0 + math.pi) % (2 * math.pi) - math.pi
    # foot in the hip_y frame (x forward, z down the leg)
    x, z = p[0], -h
    L1, L2 = rm.UPPER_LEG_LENGTH, rm.LOWER_LEG_LENGTH
    r2 = x * x + z * z
    c2 = np.clip((r2 - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0)
    q2 = -math.acos(c2)                              # knee < 0 (bent backwards)
    # foot = -L1*[sin q1, cos q1] - L2*[sin(q1+q2), cos(q1+q2)]  (x, z)
    k1 = L1 + L2 * math.cos(q2)
    k2 = L2 * math.sin(q2)
    q1 = math.atan2(-x, -z) - math.atan2(k2, k1)
    return q0, q1, q2


def foot_position(leg_index: int, q) -> np.ndarray:
    return leg_link_frames(leg_index, q)[f"{LEGS[leg_index]}_foot"][:3, 3]


# ---------------------------------------------------------------------------
# Rig analysis
# ---------------------------------------------------------------------------
def rest_angles_from_bones(bones: dict[str, np.ndarray]) -> dict[str, tuple]:
    """Recover (hip_x, hip_y, knee) per leg from bind-pose bone positions
    (robot base frame).

    hip_x : roll of the leg plane - the foot must lie at lateral distance
            HIP_TO_THIGH_Y from the hip_x axis in the rolled plane.
    hip_y : theta = atan2(-t_x, -t_z) of the thigh vector t (hip_y -> knee)
            expressed in the rolled hip frame.
    knee  : phi - theta with phi = atan2(-c_x, -c_z) of the calf vector c.
    """
    out = {}
    for i, leg in enumerate(LEGS):
        names = leg_bone_names(leg)
        p_hipx = np.asarray(rm.hip_x_position(i))
        knee = bones[names["lower_leg"]]
        thigh_root = bones[names["upper_leg"]]
        foot = bones[names["foot"]]
        sy = rm.LEG_SIGN_Y[i]
        p = foot - p_hipx
        d = rm.HIP_TO_THIGH_Y
        h = math.sqrt(max(p[1] ** 2 + p[2] ** 2 - d * d, 1e-12))
        q0 = math.atan2(p[2], p[1]) - math.atan2(-h, sy * d)
        q0 = (q0 + math.pi) % (2 * math.pi) - math.pi
        Rinv = rot_x(q0).T
        t = Rinv @ (knee - thigh_root)
        c = Rinv @ (foot - knee)
        theta = math.atan2(-t[0], -t[2])
        phi = math.atan2(-c[0], -c[2])
        out[leg] = (q0, theta, phi - theta)
    return out


@dataclass
class MeshPart:
    """Triangle mesh with one integer material label per face."""
    name: str
    vertices: np.ndarray                 # (N,3) float64
    faces: np.ndarray                    # (M,3) int64
    labels: np.ndarray                   # (M,) int64 -> materials[label]
    materials: list = field(default_factory=list)   # [(name, (r,g,b))]
    source_index: np.ndarray | None = None          # original vertex id of every vertex

    @property
    def n_faces(self) -> int:
        return int(len(self.faces))


@dataclass
class Rig:
    origin_g: np.ndarray                 # base origin in the G frame
    bones: dict                          # bind-pose bone name -> (3,) robot frame
    body_vertices: np.ndarray            # (N,3) robot frame (bind pose)
    body_faces: np.ndarray               # (M,3)
    body_face_uv: np.ndarray             # (M,2) UV centroid per face
    body_vertex_link: np.ndarray         # (N,) index into LINK_NAMES
    accessories: dict                    # name -> MeshPart (robot frame)
    texture_path: str


def g_to_robot(p_g: np.ndarray, origin_g: np.ndarray) -> np.ndarray:
    """G frame (x right, y forward, z up) -> robot base frame (x fwd, y left)."""
    p = np.asarray(p_g, dtype=float) - origin_g
    return np.stack([p[..., 1], -p[..., 0], p[..., 2]], axis=-1)


def _triangulate(geom: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fan-triangulate FBX polygons.

    Returns (faces (M,3) control-point indices, corner (M,3) polygon-vertex
    indices, polygon index per triangle)."""
    starts, ends = geom["starts"], geom["ends"]
    sizes = ends - starts + 1
    pv = geom["poly_vertex"]
    if np.all(sizes == 3):
        corner = np.stack([starts, starts + 1, starts + 2], axis=1)
        poly = np.arange(len(starts))
    else:
        tri_counts = np.maximum(sizes - 2, 0)
        poly = np.repeat(np.arange(len(starts)), tri_counts)
        first = np.cumsum(tri_counts) - tri_counts
        k = np.arange(len(poly)) - np.repeat(first, tri_counts)
        s = starts[poly]
        corner = np.stack([s, s + k + 1, s + k + 2], axis=1)
    return pv[corner], corner, poly


def load_rig(package: str = "robot_spot") -> Rig:
    """Read the FBX + texture from the Unity package and express everything
    in the robot base frame (bind pose)."""
    from fbx_reader import FBXScene
    from unitypackage import UnityPackage

    pkg = UnityPackage(package)
    fbx_path = pkg.find(FBX_NAME)
    tex_path = pkg.find(TEXTURE_NAME)
    if fbx_path is None or tex_path is None:
        raise FileNotFoundError(f"{FBX_NAME} / {TEXTURE_NAME} not found in {pkg.path}")
    sc = FBXScene(fbx_path)
    A = sc.axis_matrix()

    # --- bind pose of every node (the mesh vertices are in this pose; a few
    #     front-left bones were moved after binding, so world matrices of the
    #     current pose must NOT be used)
    bind = {}
    for oid, kind in sc.kind.items():
        if kind != "Pose":
            continue
        for pn in sc.objects[oid].children:
            if pn.name != "PoseNode":
                continue
            node = pn.find("Node").props[0]
            M = np.asarray(pn.find("Matrix").props[0], dtype=float).reshape(4, 4).T
            bind[sc.name(node)] = A @ M
    model_ids = {sc.name(m): m for m in sc.ids_of("Model")}
    for name, mid in model_ids.items():           # fall back to the scene pose
        bind.setdefault(name, A @ sc.world_matrix(mid))

    hip_bones = [leg_bone_names(leg)["hip"] for leg in LEGS]
    origin_g = np.mean([bind[b][:3, 3] for b in hip_bones], axis=0)
    bones = {name: g_to_robot(M[:3, 3], origin_g) for name, M in bind.items()}

    meshes = {}
    for gid in sc.ids_of("Geometry"):
        mids = sc.parent_ids(gid, "Model")
        if not mids:
            continue
        mid = mids[0]
        mesh_name = sc.name(mid)
        geom = sc.geometry(gid)
        if geom is None:
            continue
        W = bind.get(mesh_name, A @ sc.world_matrix(mid)) @ sc.geometric_matrix(mid)
        v_g = geom["vertices"] @ W[:3, :3].T + W[:3, 3]
        faces, corner, poly = _triangulate(geom)
        meshes[mesh_name] = dict(mid=mid, gid=gid, geom=geom, v=g_to_robot(v_g, origin_g),
                                 faces=faces, corner=corner, poly=poly)

    # --- body: skin clusters -> link per vertex
    body = meshes["Spot_Body_01"]
    nv = len(body["v"])
    b2l = bone_to_link_map()
    best_w = np.full(nv, -1.0)
    vlink = np.full(nv, -1, dtype=np.int64)
    for bone, idx, w, _tl in sc.skin_clusters(body["gid"]):
        bname = sc.name(bone)
        if bname not in b2l:
            raise ValueError(f"unexpected skin bone {bname}")
        li = LINK_NAMES.index(b2l[bname])
        better = w > best_w[idx]
        best_w[idx[better]] = w[better]
        vlink[idx[better]] = li
    if np.any(vlink < 0):
        # unskinned control points follow the body
        vlink[vlink < 0] = 0
    uvs = body["geom"]["uvs"]
    face_uv = uvs[body["corner"]].mean(axis=1) if uvs is not None else np.zeros((len(body["faces"]), 2))

    # --- rigid accessories (skinned 100 % to the body bone) -> base_link
    accessories = {}
    for mesh_name, out_name in (("Spot_Cam_01", "base_cam"), ("Spot_EAP2_01", "base_eap")):
        m = meshes[mesh_name]
        mats = [sc.material_info(x) for x in sc.materials_of_model(m["mid"])]
        poly_mat = m["geom"]["materials"][m["poly"]]
        used = np.unique(poly_mat)
        remap = -np.ones(max(len(mats), int(poly_mat.max()) + 1), dtype=np.int64)
        materials = []
        for new, old in enumerate(used):
            info = mats[old] if old < len(mats) else {"name": f"mat{old}", "diffuse": (0.6, 0.6, 0.6)}
            remap[old] = new
            materials.append((_sanitize(info["name"]), tuple(float(c) for c in info["diffuse"])))
        accessories[out_name] = MeshPart(out_name, m["v"], m["faces"].astype(np.int64),
                                         remap[poly_mat], materials)

    return Rig(origin_g=origin_g, bones=bones, body_vertices=body["v"],
               body_faces=body["faces"].astype(np.int64), body_face_uv=face_uv,
               body_vertex_link=vlink, accessories=accessories, texture_path=tex_path)


def _sanitize(name: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return out.strip("_") or "material"


# ---------------------------------------------------------------------------
# Mesh topology helpers
# ---------------------------------------------------------------------------
def face_normals(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit face normals (right-hand rule on the corner order) and areas."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    dbl = np.linalg.norm(n, axis=1)
    unit = n / np.maximum(dbl, 1e-300)[:, None]
    unit[dbl <= 0] = 0.0
    return unit, 0.5 * dbl


@dataclass
class EdgeTopology:
    edges: np.ndarray        # (K,2) vertex ids, edges[:,0] < edges[:,1]
    counts: np.ndarray       # (K,) number of incident faces
    face0: np.ndarray        # (K,) first incident face
    face1: np.ndarray        # (K,) second incident face (-1 if none)
    he_edge: np.ndarray      # (3M,) edge id of every half-edge (face-major)


def edge_topology(F: np.ndarray, n_vertices: int) -> EdgeTopology:
    """Unique undirected edges of a triangle mesh with their incident faces."""
    m = len(F)
    he = F[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    lo = np.minimum(he[:, 0], he[:, 1])
    hi = np.maximum(he[:, 0], he[:, 1])
    key = lo * np.int64(n_vertices) + hi
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    edges = np.stack([uniq // n_vertices, uniq % n_vertices], axis=1)
    he_face = np.repeat(np.arange(m), 3)
    order = np.argsort(inv, kind="stable")
    first = np.cumsum(counts) - counts
    face0 = he_face[order[first]]
    face1 = np.full(len(uniq), -1, dtype=np.int64)
    two = counts >= 2
    face1[two] = he_face[order[first[two] + 1]]
    return EdgeTopology(edges, counts, face0, face1, inv)


def _csr(src: np.ndarray, dst: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compressed adjacency: neighbours of v are data[ptr[v]:ptr[v+1]]."""
    order = np.argsort(src, kind="stable")
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=ptr[1:])
    return ptr, dst[order]


def _gather(ptr: np.ndarray, data: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate data[ptr[r]:ptr[r+1]] for every r in rows.

    Returns (values, owner) where owner[i] is the index into `rows`."""
    deg = ptr[rows + 1] - ptr[rows]
    total = int(deg.sum())
    owner = np.repeat(np.arange(len(rows)), deg)
    offs = np.arange(total) - np.repeat(np.cumsum(deg) - deg, deg)
    return data[np.repeat(ptr[rows], deg) + offs], owner


def remove_unreferenced(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop vertices not used by any face. Returns (V', F', old index of each kept vertex)."""
    used = np.unique(F)
    remap = -np.ones(len(V), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return V[used], remap[F], used


def connected_components(F: np.ndarray, n_vertices: int) -> np.ndarray:
    """Component id per face (faces sharing a vertex are connected).

    Vectorised union-find: hook every root onto the smallest root seen across
    an edge, then pointer-jump to the roots; repeat until stable."""
    parent = np.arange(n_vertices)
    a = np.concatenate([F[:, 0], F[:, 1]])
    b = np.concatenate([F[:, 1], F[:, 2]])
    while True:
        ra, rb = parent[a], parent[b]
        if np.array_equal(ra, rb):
            break
        lo = np.minimum(ra, rb)
        np.minimum.at(parent, ra, lo)
        np.minimum.at(parent, rb, lo)
        while True:
            pp = parent[parent]
            if np.array_equal(pp, parent):
                break
            parent = pp
    return parent[F[:, 0]]


# ---------------------------------------------------------------------------
# Colours: texture sampling + weighted k-means in CIE-Lab
# ---------------------------------------------------------------------------
def load_texture_rgb(path: str, size: int = 1024) -> np.ndarray:
    """Albedo texture as uint8 RGB, area-downsampled to size x size.

    Downsampling (4x4 texel averaging) plus a median filter removes the fine
    print (logos, warning labels) that would otherwise turn into colour
    speckles on the decimated mesh."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 257.0).round().astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = np.ascontiguousarray(img[:, :, :3])
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = cv2.medianBlur(img, 5)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def sample_texture(tex_rgb: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-texel lookup; FBX v=0 is the bottom image row, UVs wrap."""
    h, w = tex_rgb.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    col = np.clip((u * w).astype(np.int64), 0, w - 1)
    row = np.clip(((1.0 - v) * h).astype(np.int64), 0, h - 1)
    return tex_rgb[row, col]


def rgb_to_lab(rgb_u8: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIE-Lab (OpenCV float convention: L 0..100)."""
    import cv2
    x = (np.asarray(rgb_u8, dtype=np.float32) / 255.0).reshape(-1, 1, 3)
    return cv2.cvtColor(x, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64)


def weighted_kmeans(X: np.ndarray, w: np.ndarray, k: int, iters: int = 50,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means with sample weights and k-means++ seeding (deterministic).

    Returns (centres (k',d), labels (N,)), k' <= k (empty clusters dropped)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    w = w / w.sum()
    n = len(X)
    sub = rng.choice(n, size=20000, replace=True, p=w) if n > 20000 else np.arange(n)
    Xs = X[sub]
    centres = [Xs[rng.integers(len(Xs))]]
    for _ in range(1, k):
        d2 = np.min(((Xs[:, None, :] - np.asarray(centres)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() <= 0:
            break
        centres.append(Xs[rng.choice(len(Xs), p=d2 / d2.sum())])
    C = np.asarray(centres)
    labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
    for _ in range(iters):
        wsum = np.bincount(labels, weights=w, minlength=len(C))
        keep = wsum > 0
        C = np.stack([np.bincount(labels, weights=w * X[:, j], minlength=len(C))
                      for j in range(X.shape[1])], axis=1)[keep] / wsum[keep, None]
        new = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        if np.array_equal(new, labels) and keep.all():
            break
        labels = new
    return C, labels


def smooth_face_labels(F: np.ndarray, labels: np.ndarray, n_vertices: int, iters: int = 3) -> np.ndarray:
    """Majority filter over edge-adjacent faces: a face whose label differs
    from all of its neighbours while >= 2 neighbours agree is relabelled."""
    topo = edge_topology(F, n_vertices)
    he_edge = topo.he_edge.reshape(-1, 3)
    own = np.arange(len(F))[:, None]
    other = np.where(topo.face0[he_edge] == own, topo.face1[he_edge], topo.face0[he_edge])
    lab = labels.copy()
    for _ in range(iters):
        nl = np.where(other >= 0, lab[np.maximum(other, 0)], -1)          # (M,3)
        a01 = (nl[:, 0] == nl[:, 1]) & (nl[:, 0] >= 0)
        a02 = (nl[:, 0] == nl[:, 2]) & (nl[:, 0] >= 0)
        a12 = (nl[:, 1] == nl[:, 2]) & (nl[:, 1] >= 0)
        majority = np.where(a01 | a02, nl[:, 0], np.where(a12, nl[:, 1], -1))
        isolated = np.all(nl != lab[:, None], axis=1) & (majority >= 0)
        if not isolated.any():
            break
        lab[isolated] = majority[isolated]
    return lab


_COLOUR_NAMES = (
    ("yellow", (0.95, 0.70, 0.05)), ("black", (0.05, 0.05, 0.05)),
    ("dark_gray", (0.24, 0.24, 0.24)), ("gray", (0.55, 0.56, 0.57)),
    ("light_gray", (0.78, 0.79, 0.80)), ("green", (0.05, 0.55, 0.32)),
    ("cyan", (0.05, 0.75, 0.75)), ("white", (0.97, 0.97, 0.97)),
    ("orange", (0.95, 0.45, 0.05)), ("red", (0.80, 0.10, 0.10)),
)


def colour_name(rgb) -> str:
    """Closest descriptive name (Lab distance) for a 0..1 RGB colour."""
    lab = rgb_to_lab(np.clip(np.asarray(rgb, dtype=float)[None] * 255.0, 0, 255))[0]
    refs = rgb_to_lab(np.array([c for _n, c in _COLOUR_NAMES]) * 255.0)
    return _COLOUR_NAMES[int(np.argmin(((refs - lab) ** 2).sum(1)))][0]


def merge_close_clusters(C: np.ndarray, labels: np.ndarray, w: np.ndarray, max_k: int,
                         min_delta_e: float, min_area_frac: float = 0.0025) -> np.ndarray:
    """Agglomerative clean-up of k-means clusters (Lab centres C):

    1. merge the closest pair while it is closer than min_delta_e (CIE76),
    2. fold clusters covering < min_area_frac of the surface into their
       nearest cluster (tiny decals are not worth a material),
    3. merge the closest pair while more than max_k clusters remain.
    Merged centres are weight averaged.  Returns the new labels."""
    C = [np.asarray(c, dtype=float) for c in C]
    wt = [float(x) for x in np.bincount(labels, weights=w, minlength=len(C))]
    alive = [i for i in range(len(C)) if wt[i] > 0]
    lab = labels.copy()
    total = sum(wt)

    def closest_pair(cands=None):
        best, pair = np.inf, None
        for i in alive if cands is None else cands:
            for j in alive:
                if j == i or (cands is None and j < i):
                    continue
                d = float(np.linalg.norm(C[i] - C[j]))
                if d < best:
                    best, pair = d, (j, i) if cands is not None else (i, j)
        return best, pair

    def merge(keep, gone):
        tot = wt[keep] + wt[gone]
        C[keep] = (C[keep] * wt[keep] + C[gone] * wt[gone]) / max(tot, 1e-30)
        wt[keep] = tot
        lab[lab == gone] = keep
        alive.remove(gone)

    while len(alive) > 1:                                   # 1. near duplicates
        d, pair = closest_pair()
        if d >= min_delta_e:
            break
        merge(*pair)
    while len(alive) > 1:                                   # 2. tiny clusters
        small = [i for i in alive if wt[i] < min_area_frac * total]
        if not small:
            break
        s = min(small, key=lambda i: wt[i])
        _d, (keep, gone) = closest_pair([s])
        merge(keep, gone)
    while len(alive) > max_k:                               # 3. palette size
        _d, pair = closest_pair()
        merge(*pair)
    return lab


def quantise_body_colours(V: np.ndarray, F: np.ndarray, face_uv: np.ndarray, tex_path: str,
                          k: int = N_BODY_COLOURS) -> tuple[np.ndarray, list, np.ndarray]:
    """Per-face material labels + palette [(name, rgb01)] for the textured
    body (+ the raw sampled sRGB per face, used for the 'original' preview).

    k-means starts with k+2 clusters; merge_close_clusters() then joins
    near-identical colours (CIE76 < 10), folds decals covering < 0.25 % of
    the surface into their nearest colour and caps the palette at k."""
    tex = load_texture_rgb(tex_path)
    rgb = sample_texture(tex, face_uv)
    _n, area = face_normals(V, F)
    w = area + 1e-12
    C, labels = weighted_kmeans(rgb_to_lab(rgb), w, k + 2)
    labels = merge_close_clusters(C, labels, w, k, 10.0)
    labels = smooth_face_labels(F, labels, len(V))
    # order the palette by usage (most used first) and name the colours
    usage = np.bincount(labels, weights=w)
    order = [int(i) for i in np.argsort(-usage) if usage[i] > 0]
    relabel = -np.ones(len(usage), dtype=np.int64)
    pal, names = [], set()
    for new, old in enumerate(order):
        relabel[old] = new
        sel = labels == old
        c = (rgb[sel].astype(np.float64) * w[sel, None]).sum(0) / w[sel].sum() / 255.0
        name = base = "spot_" + colour_name(c)
        j = 2
        while name in names:
            name, j = f"{base}_{j}", j + 1
        names.add(name)
        pal.append((name, tuple(round(float(x), 4) for x in c)))
    return relabel[labels], pal, rgb


# ---------------------------------------------------------------------------
# Mesh cleanup
# ---------------------------------------------------------------------------
def weld_vertices(V: np.ndarray, F: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that snap to the same tol-grid cell (exact duplicates
    at UV / smoothing seams); faces are re-indexed, not removed."""
    key = np.round(V / tol).astype(np.int64)
    _u, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.reshape(-1)[F]


def clean_faces(F: np.ndarray, labels: np.ndarray, V: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Remove faces with repeated corners, zero area (if V given) and exact
    duplicates (same corners in the same cyclic order)."""
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    if V is not None:
        _n, area = face_normals(V, F)
        ok &= area > 1e-14
    F, labels = F[ok], labels[ok]
    if len(F) == 0:
        return F, labels
    # canonical rotation (smallest index first) keeps the orientation
    r = np.argmin(F, axis=1)
    idx = (r[:, None] + np.arange(3)[None]) % 3
    canon = np.take_along_axis(F, idx, axis=1)
    _u, first = np.unique(canon, axis=0, return_index=True)
    first = np.sort(first)
    return F[first], labels[first]


# ---------------------------------------------------------------------------
# Quadric error metric decimation (batched edge collapse, numpy only)
# ---------------------------------------------------------------------------
# A symmetric 4x4 quadric is stored as its 10 upper-triangle entries
#   q = [a2, ab, ac, ad, b2, bc, bd, c2, cd, d2]  for the plane (a, b, c, d)
_QIDX = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))


def plane_quadrics(normals: np.ndarray, points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Fundamental error quadrics K_p = w * p p^T of the planes n.x + d = 0
    through `points`."""
    d = -np.einsum("ij,ij->i", normals, points)
    p = np.column_stack([normals, d])
    return weights[:, None] * np.stack([p[:, i] * p[:, j] for i, j in _QIDX], axis=1)


def _accumulate(Q: np.ndarray, idx: np.ndarray, q: np.ndarray) -> None:
    n = len(Q)
    for j in range(10):
        Q[:, j] += np.bincount(idx, weights=q[:, j], minlength=n)


def quadric_cost(Q: np.ndarray, x: np.ndarray) -> np.ndarray:
    """v^T Q v with v = (x, 1) for every row."""
    X, Y, Z = x[:, 0], x[:, 1], x[:, 2]
    return (Q[:, 0] * X * X + 2 * Q[:, 1] * X * Y + 2 * Q[:, 2] * X * Z + 2 * Q[:, 3] * X
            + Q[:, 4] * Y * Y + 2 * Q[:, 5] * Y * Z + 2 * Q[:, 6] * Y
            + Q[:, 7] * Z * Z + 2 * Q[:, 8] * Z + Q[:, 9])


def quadric_optimum(Q: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal collapse position and its cost for every edge (pa, pb).

    Minimises v^T Q v.  Solved relative to the edge midpoint m with Tikhonov
    regularisation, (A + lam I) delta = -(A m + b): directions in which the
    quadric is flat (planar / straight-crease neighbourhoods) keep the
    midpoint, so the solution never runs away.  The optimum is compared with
    both endpoints and the midpoint and the cheapest candidate wins."""
    A = np.empty((len(Q), 3, 3))
    A[:, 0, 0], A[:, 0, 1], A[:, 0, 2] = Q[:, 0], Q[:, 1], Q[:, 2]
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2] = Q[:, 1], Q[:, 4], Q[:, 5]
    A[:, 2, 0], A[:, 2, 1], A[:, 2, 2] = Q[:, 2], Q[:, 5], Q[:, 7]
    bvec = Q[:, [3, 6, 8]]
    m = 0.5 * (pa + pb)
    rhs = -(np.einsum("kij,kj->ki", A, m) + bvec)
    tr = A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]
    lam = 1e-7 * tr + 1e-30
    Areg = A + lam[:, None, None] * np.eye(3)[None]
    delta = np.linalg.solve(Areg, rhs[..., None])[..., 0]
    elen = np.linalg.norm(pb - pa, axis=1)
    far = ~np.isfinite(delta).all(axis=1) | (np.linalg.norm(delta, axis=1) > 1.5 * elen)
    delta[far] = 0.0
    cands = np.stack([m + delta, pa, pb, m], axis=1)              # (K,4,3)
    costs = np.stack([quadric_cost(Q, cands[:, i]) for i in range(4)], axis=1)
    best = np.argmin(costs, axis=1)
    rows = np.arange(len(Q))
    return cands[rows, best], np.maximum(costs[rows, best], 0.0)


def initial_quadrics(V: np.ndarray, F: np.ndarray, labels: np.ndarray,
                     boundary_penalty: float = 10.0, label_penalty: float = 4.0) -> np.ndarray:
    """Vertex quadrics: area weighted face planes + constraint planes along
    open boundaries and along colour borders (plane through the edge,
    perpendicular to the adjacent face, weight penalty * |e|^2)."""
    n = len(V)
    fn, area = face_normals(V, F)
    Q = np.zeros((n, 10))
    face_q = plane_quadrics(fn, V[F[:, 0]], area)
    for k in range(3):
        _accumulate(Q, F[:, k], face_q)
    topo = edge_topology(F, n)
    E = topo.edges
    border = (topo.counts == 2) & (labels[topo.face0] != labels[np.maximum(topo.face1, 0)])
    for mask, pen, faces in ((topo.counts == 1, boundary_penalty, (topo.face0,)),
                             (border, label_penalty, (topo.face0, topo.face1))):
        if not mask.any():
            continue
        e = E[mask]
        evec = V[e[:, 1]] - V[e[:, 0]]
        l2 = (evec ** 2).sum(1)
        for fsel in faces:
            nrm = np.cross(evec, fn[fsel[mask]])
            nl = np.linalg.norm(nrm, axis=1)
            ok = nl > 1e-20
            if not ok.any():
                continue
            q = plane_quadrics(nrm[ok] / nl[ok, None], V[e[ok, 0]], pen * l2[ok] / len(faces))
            _accumulate(Q, e[ok, 0], q)
            _accumulate(Q, e[ok, 1], q)
    return Q


def decimate(V: np.ndarray, F: np.ndarray, labels: np.ndarray, target_faces: int, *,
             candidate_fraction: float = 0.35, max_rounds: int = 500,
             min_component_faces: int = 4, verbose: bool = False
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quadric-error-metric simplification to ~target_faces triangles.

    Each round:
      1. unique edges + incident face counts; edges touching a non-manifold
         edge and interior edges joining two boundary vertices are frozen;
      2. optimal position + cost for every edge (quadric_optimum); a tiny
         |e|^2 term breaks ties in flat regions in favour of short edges;
      3. edges are tested, cheapest first, for the link condition (the mesh
         stays manifold; triangular holes may be closed) and for face flips /
         slivers in their 1-rings; the cheapest `candidate_fraction` of the
         collapsible edges become candidates;
      4. an edge is collapsed only if its rank is the minimum over the 2-ring
         of both endpoints.  Two selected edges are then never adjacent, so
         no face is modified by two collapses and the flip test is exact;
      5. collapse: v_a <- x, Q_a <- Q_a + Q_b, faces relinked, degenerate
         faces dropped (material labels travel with the surviving faces).
    """
    V = np.asarray(V, dtype=np.float64).copy()
    F = np.asarray(F, dtype=np.int64).copy()
    labels = np.asarray(labels, dtype=np.int64).copy()
    n = len(V)
    Q = initial_quadrics(V, F, labels)
    _fn, area0 = face_normals(V, F)
    alpha = 1e-4 * float(np.median(area0)) if len(area0) else 0.0
    t0 = time.time()
    rounds = 0
    while len(F) > target_faces and rounds < max_rounds:
        rounds += 1
        m = len(F)
        topo = edge_topology(F, n)
        E, cnt = topo.edges, topo.counts
        boundary_v = np.zeros(n, dtype=bool)
        boundary_v[E[cnt == 1].ravel()] = True
        frozen_v = np.zeros(n, dtype=bool)
        frozen_v[E[cnt > 2].ravel()] = True
        ok = (cnt <= 2) & ~frozen_v[E[:, 0]] & ~frozen_v[E[:, 1]]
        ok &= ~((cnt == 2) & boundary_v[E[:, 0]] & boundary_v[E[:, 1]])
        opposite = F[topo.face0].sum(axis=1) - E[:, 0] - E[:, 1]
        opposite2 = np.where(topo.face1 >= 0,
                             F[np.maximum(topo.face1, 0)].sum(axis=1) - E[:, 0] - E[:, 1], -1)
        eidx = np.nonzero(ok)[0]
        if len(eidx) == 0:
            break
        ea, eb = E[eidx, 0], E[eidx, 1]
        x, cost = quadric_optimum(Q[ea] + Q[eb], V[ea], V[eb])
        cost = cost + alpha * ((V[eb] - V[ea]) ** 2).sum(1)
        # Validity is tested on a generous pool of the cheapest edges first;
        # only if too few of them are collapsible is the whole set tested
        # (prevents stalling on cheap but blocked edges).
        order = np.argsort(cost, kind="stable")
        pool_size = min(len(order), max(64, int(2.5 * candidate_fraction * len(order))))
        while True:
            pool = order[:pool_size]
            valid = _link_condition_ok(E, cnt, eidx[pool], n, opposite, opposite2)
            valid &= _no_flip(V, F, ea[pool], eb[pool], x[pool], n)
            good = pool[valid]
            want = max(1, int(candidate_fraction * len(order)))
            if len(good) >= want or pool_size == len(order):
                break
            pool_size = len(order)
        cand = good[:want]                      # cheapest valid edges (cost order)
        sel = cand[_independent_minima(E, ea[cand], eb[cand], cost[cand], n)]
        if len(sel) == 0:
            break
        # do not overshoot: an interior collapse removes two faces
        need = int(math.ceil((m - target_faces) / 2.0))
        if len(sel) > need:
            sel = sel[np.argsort(cost[sel], kind="stable")[:need]]
        a, b = ea[sel], eb[sel]
        V[a] = x[sel]
        Q[a] += Q[b]
        remap = np.arange(n)
        remap[b] = a
        F = remap[F]
        keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F, labels = F[keep], labels[keep]
        if verbose and rounds % 10 == 0:
            print(f"    round {rounds:3d}: {m:7d} -> {len(F):7d} faces ({len(sel)} collapses, "
                  f"{time.time() - t0:5.1f} s)")
    F, labels = clean_faces(F, labels, V)
    if min_component_faces > 1 and len(F):
        comp = connected_components(F, n)
        _u, inv, cnts = np.unique(comp, return_inverse=True, return_counts=True)
        keep = cnts[inv.reshape(-1)] >= min_component_faces
        F, labels = F[keep], labels[keep]
    V2, F2, _used = remove_unreferenced(V, F)
    if verbose:
        print(f"    decimated to {len(F2)} faces / {len(V2)} vertices in {rounds} rounds "
              f"({time.time() - t0:.1f} s)")
    return V2, F2, labels


def _link_condition_ok(E: np.ndarray, cnt: np.ndarray, cidx: np.ndarray, n: int,
                       opposite: np.ndarray | None = None,
                       opposite2: np.ndarray | None = None) -> np.ndarray:
    """Topological safety of collapsing edge (a,b) (Dey et al. link condition).

    * The common neighbours of a and b must be exactly the opposite vertices
      of the faces on (a,b).
    * Interior edge whose two opposite vertices c, d are themselves adjacent
      (tetrahedron-like fin): refused, the collapse would fold (a,c,d) onto
      (b,c,d).  Needs `opposite2` (third vertex of the second face).
    * Relaxation for boundary edges: an extra common neighbour c is allowed
      when (a,c) and (b,c) are boundary edges too, i.e. a-b-c is a
      triangular hole - the collapse then just closes the hole (zips (a,c)
      and (b,c) into one interior edge) and the mesh stays manifold.  Without
      it, meshes made of many open panels stall with thousands of 3-vertex
      holes.  Needs `opposite` (third vertex of the first face).
    """
    src = np.concatenate([E[:, 0], E[:, 1]])
    dst = np.concatenate([E[:, 1], E[:, 0]])
    ptr, nbr = _csr(src, dst, n)
    a, b = E[cidx, 0], E[cidx, 1]
    na, oa = _gather(ptr, nbr, a)
    nb, ob = _gather(ptr, nbr, b)
    keys = np.sort(np.concatenate([oa * np.int64(n) + na, ob * np.int64(n) + nb]))
    dup = keys[1:] == keys[:-1]
    common_keys = keys[1:][dup]
    owner = common_keys // n
    common = np.bincount(owner, minlength=len(cidx))
    ekeys = E[:, 0] * np.int64(n) + E[:, 1]                  # sorted (np.unique order)

    def edge_count(u, v):
        k = np.minimum(u, v) * np.int64(n) + np.maximum(u, v)
        pos = np.clip(np.searchsorted(ekeys, k), 0, len(ekeys) - 1)
        return np.where(ekeys[pos] == k, cnt[pos], 0)

    if opposite is not None:
        w = common_keys % n
        bnd = (cnt[cidx][owner] == 1) & (w != opposite[cidx][owner])
        if bnd.any():
            o = owner[bnd]
            hole = (edge_count(a[o], w[bnd]) == 1) & (edge_count(b[o], w[bnd]) == 1)
            common = common - np.bincount(o[hole], minlength=len(cidx))
    ok = common == cnt[cidx]
    if opposite is not None and opposite2 is not None:
        inner = cnt[cidx] == 2
        if inner.any():
            c, d = opposite[cidx][inner], opposite2[cidx][inner]
            ok[np.nonzero(inner)[0][edge_count(c, d) > 0]] = False
    return ok


def _triangle_quality(P: np.ndarray, dbl_area: np.ndarray) -> np.ndarray:
    """4*sqrt(3)*area / sum(edge^2): 1 for equilateral, -> 0 for slivers."""
    s = (((P[:, 1] - P[:, 0]) ** 2).sum(1) + ((P[:, 2] - P[:, 1]) ** 2).sum(1)
         + ((P[:, 0] - P[:, 2]) ** 2).sum(1))
    return 2.0 * math.sqrt(3.0) * dbl_area / np.maximum(s, 1e-30)


def _no_flip(V: np.ndarray, F: np.ndarray, a: np.ndarray, b: np.ndarray, x: np.ndarray, n: int,
             min_cos: float = 0.2, min_quality: float = 0.02) -> np.ndarray:
    """True for collapses (a,b)->x that neither flip nor crush a 1-ring face."""
    m = len(F)
    ptr, vf = _csr(F.ravel(), np.repeat(np.arange(m), 3), n)
    fa, oa = _gather(ptr, vf, a)
    fb, ob = _gather(ptr, vf, b)
    faces = np.concatenate([fa, fb])
    owner = np.concatenate([oa, ob])
    moved = np.concatenate([a[oa], b[ob]])
    other = np.concatenate([b[oa], a[ob]])
    Fc = F[faces]
    keep = ~np.any(Fc == other[:, None], axis=1)        # faces on the edge vanish
    Fc, owner, moved = Fc[keep], owner[keep], moved[keep]
    P = V[Fc]
    n_old = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    mask = (Fc == moved[:, None])[..., None]
    Pn = np.where(mask, x[owner][:, None, :], P)
    n_new = np.cross(Pn[:, 1] - Pn[:, 0], Pn[:, 2] - Pn[:, 0])
    lo = np.linalg.norm(n_old, axis=1)
    ln = np.linalg.norm(n_new, axis=1)
    bad = np.einsum("ij,ij->i", n_old, n_new) <= min_cos * lo * ln
    bad |= ln <= 1e-12 * np.maximum(lo, 1e-30)
    qn = _triangle_quality(Pn, ln)
    bad |= (qn < min_quality) & (qn < 0.5 * _triangle_quality(P, lo))
    return np.bincount(owner[bad], minlength=len(a)) == 0


def _independent_minima(E: np.ndarray, a: np.ndarray, b: np.ndarray, cost: np.ndarray,
                        n: int) -> np.ndarray:
    """Indices of candidate edges whose cost rank is the minimum over the
    2-ring of both endpoints (the selected edges are pairwise non-adjacent:
    if a vertex of e1 neighboured a vertex of e2, each rank would have to be
    <= the other)."""
    if len(a) == 0:
        return np.zeros(0, dtype=np.int64)
    rank = np.empty(len(a), dtype=np.int64)
    rank[np.argsort(cost, kind="stable")] = np.arange(len(a))
    big = np.iinfo(np.int64).max
    v1 = np.full(n, big, dtype=np.int64)
    np.minimum.at(v1, a, rank)
    np.minimum.at(v1, b, rank)
    v2 = v1.copy()
    np.minimum.at(v2, E[:, 0], v1[E[:, 1]])
    np.minimum.at(v2, E[:, 1], v1[E[:, 0]])
    return np.nonzero((rank == v2[a]) & (rank == v2[b]))[0]


# ---------------------------------------------------------------------------
# Normals and OBJ / MTL I/O
# ---------------------------------------------------------------------------
def corner_normals(V: np.ndarray, F: np.ndarray, crease_deg: float = CREASE_ANGLE_DEG
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Crease-aware per-corner normals.

    The normal of corner (f, v) is the area weighted sum of the normals of
    the faces around v whose normal is within `crease_deg` of the normal of
    f: smooth shading on curved panels, sharp edges where the dihedral angle
    is large.  Returns (unique normals (K,3), normal index per corner (M,3))."""
    m = len(F)
    fn, area = face_normals(V, F)
    wn = fn * area[:, None]
    cv = F.ravel()
    cf = np.repeat(np.arange(m), 3)
    ptr, vf = _csr(cv, cf, len(V))
    g, owner = _gather(ptr, vf, cv)                 # faces around each corner's vertex
    f_own = cf[owner]
    ok = np.einsum("ij,ij->i", fn[f_own], fn[g]) >= math.cos(math.radians(crease_deg))
    acc = np.stack([np.bincount(owner[ok], weights=wn[g[ok], j], minlength=3 * m)
                    for j in range(3)], axis=1)
    ln = np.linalg.norm(acc, axis=1)
    fallback = ln < 1e-20
    acc[fallback] = fn[cf[fallback]]
    ln = np.linalg.norm(acc, axis=1)
    acc = acc / np.maximum(ln, 1e-20)[:, None]
    q = np.round(acc * 1e4).astype(np.int64)
    _u, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    normals = acc[first]
    normals /= np.maximum(np.linalg.norm(normals, axis=1), 1e-20)[:, None]
    return normals, inv.reshape(m, 3)


def write_obj(path: str, V: np.ndarray, F: np.ndarray, labels: np.ndarray, materials: list,
              normals: np.ndarray, nidx: np.ndarray, mtl_file: str = MTL_NAME,
              header: str = "") -> None:
    """Write an OBJ with per-corner normals, faces grouped by material."""
    lines = [f"# {line}" for line in header.splitlines()]
    lines.append(f"mtllib {mtl_file}")
    lines.append(f"o {os.path.splitext(os.path.basename(path))[0]}")
    lines.extend(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}" for p in V.tolist())
    lines.extend(f"vn {q[0]:.4f} {q[1]:.4f} {q[2]:.4f}" for q in normals.tolist())
    lines.append("s off")
    for li in np.unique(labels):
        lines.append(f"usemtl {materials[li][0]}")
        sel = np.nonzero(labels == li)[0]
        fv = (F[sel] + 1).tolist()
        fn = (nidx[sel] + 1).tolist()
        lines.extend(f"f {a}//{na} {b}//{nb} {c}//{nc}"
                     for (a, b, c), (na, nb, nc) in zip(fv, fn))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_mtl(path: str, materials: list, header: str = "") -> None:
    """Flat colour materials (no textures).  Ka = Kd so ambient light keeps the hue."""
    lines = [f"# {line}" for line in header.splitlines()]
    for name, rgb in materials:
        r, g, b = (min(max(float(c), 0.0), 1.0) for c in rgb)
        lines += ["", f"newmtl {name}", f"Ka {r:.4f} {g:.4f} {b:.4f}", f"Kd {r:.4f} {g:.4f} {b:.4f}",
                  "Ks 0.1500 0.1500 0.1500", "Ns 40.0", "d 1.0", "illum 2"]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def read_mtl(path: str) -> dict:
    """{material name: Kd rgb} of an MTL file."""
    out, cur = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if not tok:
                continue
            if tok[0] == "newmtl":
                cur = tok[1]
                out[cur] = (0.8, 0.8, 0.8)
            elif tok[0] == "Kd" and cur is not None:
                out[cur] = tuple(float(t) for t in tok[1:4])
    return out


def read_obj(path: str) -> dict:
    """Minimal OBJ reader (v / vn / f / usemtl / mtllib); polygons are
    fan-triangulated.  Returns dict(vertices, faces, normals,
    face_normal_idx, face_material (names), mtllib)."""
    vs, vns, faces, fnorm, fmat = [], [], [], [], []
    mtllib, cur = None, ""
    with open(path, encoding="utf-8") as f:
        for line in f:
            tok = line.split()
            if not tok or tok[0].startswith("#"):
                continue
            t = tok[0]
            if t == "v":
                vs.append([float(c) for c in tok[1:4]])
            elif t == "vn":
                vns.append([float(c) for c in tok[1:4]])
            elif t == "f":
                idx = [c.split("/") for c in tok[1:]]
                vi = [int(c[0]) for c in idx]
                ni = [int(c[2]) if len(c) > 2 and c[2] else 0 for c in idx]
                vi = [i - 1 if i > 0 else len(vs) + i for i in vi]
                ni = [i - 1 if i > 0 else len(vns) + i for i in ni]
                for k in range(1, len(vi) - 1):
                    faces.append([vi[0], vi[k], vi[k + 1]])
                    fnorm.append([ni[0], ni[k], ni[k + 1]])
                    fmat.append(cur)
            elif t == "usemtl":
                cur = tok[1] if len(tok) > 1 else ""
            elif t == "mtllib":
                mtllib = tok[1] if len(tok) > 1 else None
    return {"vertices": np.asarray(vs, dtype=np.float64).reshape(-1, 3),
            "faces": np.asarray(faces, dtype=np.int64).reshape(-1, 3),
            "normals": np.asarray(vns, dtype=np.float64).reshape(-1, 3),
            "face_normal_idx": np.asarray(fnorm, dtype=np.int64).reshape(-1, 3),
            "face_material": fmat, "mtllib": mtllib}


# ---------------------------------------------------------------------------
# Verification helpers: nearest neighbours + point-to-triangle distance
# ---------------------------------------------------------------------------
def knn(points: np.ndarray, queries: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """k nearest points for every query -> (indices (Q,k), distances (Q,k)).

    Uses OpenCV's FLANN kd-tree when available (128 checks, practically
    exact for this use), otherwise chunked brute force."""
    points = np.asarray(points, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    k = min(k, len(points))
    try:
        import cv2
        index = cv2.flann_Index(points.astype(np.float32), dict(algorithm=1, trees=4))
        idx, _d2 = index.knnSearch(queries.astype(np.float32), k, params=dict(checks=128))
        idx = idx.astype(np.int64).reshape(len(queries), k)
        dist = np.linalg.norm(points[idx] - queries[:, None, :], axis=2)
        order = np.argsort(dist, axis=1)
        return np.take_along_axis(idx, order, 1), np.take_along_axis(dist, order, 1)
    except Exception:  # pragma: no cover - no OpenCV / FLANN binding: exact brute force
        out_i = np.empty((len(queries), k), dtype=np.int64)
        out_d = np.empty((len(queries), k))
        p2 = (points ** 2).sum(1)
        for s in range(0, len(queries), 1024):
            q = queries[s:s + 1024]
            d2 = np.maximum((q ** 2).sum(1)[:, None] - 2 * q @ points.T + p2[None], 0.0)
            part = np.argpartition(d2, k - 1, axis=1)[:, :k] if k < len(points) else \
                np.tile(np.arange(len(points)), (len(q), 1))
            dd = np.take_along_axis(d2, part, 1)
            o = np.argsort(dd, axis=1)
            out_i[s:s + 1024] = np.take_along_axis(part, o, 1)
            out_d[s:s + 1024] = np.sqrt(np.take_along_axis(dd, o, 1))
        return out_i, out_d


def point_triangle_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Exact distance from points P to triangles (A,B,C), all (K,3).

    Closest-point-on-triangle by Voronoi region classification
    (Ericson, Real-Time Collision Detection, 5.1.5), vectorised."""
    ab, ac, ap = B - A, C - A, P - A
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = P - B
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = P - C
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    eps = 1e-300
    # default: interior (barycentric projection)
    denom = va + vb + vc
    v = vb / np.where(np.abs(denom) > eps, denom, eps)
    w = vc / np.where(np.abs(denom) > eps, denom, eps)
    Q = A + ab * v[:, None] + ac * w[:, None]
    # edge regions
    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t_bc = (d4 - d3) / np.maximum((d4 - d3) + (d5 - d6), eps)
    Q = np.where(m_bc[:, None], B + (C - B) * t_bc[:, None], Q)
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t_ac = d2 / np.maximum(d2 - d6, eps)
    Q = np.where(m_ac[:, None], A + ac * t_ac[:, None], Q)
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t_ab = d1 / np.maximum(d1 - d3, eps)
    Q = np.where(m_ab[:, None], A + ab * t_ab[:, None], Q)
    # vertex regions
    Q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, Q)
    Q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, Q)
    Q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, Q)
    return np.linalg.norm(P - Q, axis=1)


def surface_distance(queries: np.ndarray, V: np.ndarray, F: np.ndarray, k: int = 6) -> np.ndarray:
    """Approximate unsigned distance from points to a triangle mesh: exact
    point-triangle distances to every face incident to the k nearest mesh
    vertices (exact whenever the closest face touches one of them, which is
    the case for the well-shaped meshes produced here)."""
    if len(F) == 0:
        return np.full(len(queries), np.inf)
    idx, _d = knn(V, queries, k)
    ptr, vf = _csr(F.ravel(), np.repeat(np.arange(len(F)), 3), len(V))
    rows = idx.reshape(-1)
    faces, owner = _gather(ptr, vf, rows)
    qi = owner // idx.shape[1]
    d = point_triangle_distance(queries[qi], V[F[faces, 0]], V[F[faces, 1]], V[F[faces, 2]])
    out = np.full(len(queries), np.inf)
    np.minimum.at(out, qi, d)
    return out


# ---------------------------------------------------------------------------
# Preview renderer (orthographic, painter's algorithm, OpenCV)
# ---------------------------------------------------------------------------
@dataclass
class RenderMesh:
    vertices: np.ndarray        # (N,3) world (base) frame
    faces: np.ndarray           # (M,3)
    face_rgb: np.ndarray        # (M,3) float 0..1


def _view_basis(forward, up) -> np.ndarray:
    f = np.asarray(forward, dtype=float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, f])          # rows: screen-right, screen-up, depth


VIEWS = {
    "side": _view_basis((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),       # from robot's right, nose -> right
    "top": _view_basis((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),       # nose -> right, left side up
    "3/4": _view_basis((-0.75, 0.85, -0.55), (0.0, 0.0, 1.0)),   # from front-right, above
}


def render(meshes: list, basis: np.ndarray, size=(420, 300), bounds=None, ss: int = 2,
           ground_z: float | None = None, background=(250, 250, 250)) -> np.ndarray:
    """Flat-shaded orthographic rendering -> BGR uint8 image.

    bounds = (xmin, xmax, ymin, ymax) in screen coordinates (metres) so that
    several panels can share one scale; faces are drawn back to front."""
    import cv2
    W, H = size[0] * ss, size[1] * ss
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:] = background
    allV = [m.vertices @ basis.T for m in meshes if len(m.faces)]
    if bounds is None:
        pts = np.concatenate(allV)
        bounds = (pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max())
    x0, x1, y0, y1 = bounds
    s = 0.92 * min(W / max(x1 - x0, 1e-6), H / max(y1 - y0, 1e-6))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

    def to_px(p):
        return np.stack([W / 2 + (p[..., 0] - cx) * s, H / 2 - (p[..., 1] - cy) * s], axis=-1)

    if ground_z is not None:
        g = np.array([[-5.0, 0.0, ground_z], [5.0, 0.0, ground_z]]) @ basis.T
        gp = to_px(g)
        if abs(basis[1, 2]) > 0.3:     # the ground is visible as a line only when z is up on screen
            yv = int(round(gp[0, 1]))
            img[min(max(yv, 0), H - 1):, :] = (225, 228, 230)
            cv2.line(img, (0, yv), (W - 1, yv), (150, 150, 150), ss)
    light = -basis[2] * 0.75 + basis[1] * 0.55 + basis[0] * 0.35
    light /= np.linalg.norm(light)
    light_s = basis @ light                                     # light in screen coordinates
    tris, depth, cols = [], [], []
    for m, Vs in zip([m for m in meshes if len(m.faces)], allV):
        P = Vs[m.faces]                                        # (M,3,3) screen coords
        n, _a = face_normals(m.vertices, m.faces)
        n_s = n @ basis.T
        n_s[n_s[:, 2] > 0] *= -1.0                             # two-sided: face the viewer
        lam = 0.32 + 0.68 * np.clip(n_s @ light_s, 0.0, 1.0)            # Lambert + ambient
        tris.append(to_px(P[:, :, :2]))
        depth.append(P[:, :, 2].mean(axis=1))
        cols.append(np.clip(m.face_rgb * lam[:, None], 0, 1))
    if not tris:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    tris = np.concatenate(tris)
    depth = np.concatenate(depth)
    cols = (np.concatenate(cols)[:, ::-1] * 255).astype(np.int64)  # BGR
    order = np.argsort(-depth)                                     # far -> near
    pix = np.round(tris * 16).astype(np.int32)                     # 4 fractional bits
    for i in order.tolist():
        c = cols[i]
        cv2.fillConvexPoly(img, pix[i], (int(c[0]), int(c[1]), int(c[2])), cv2.LINE_8, 4)
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def screen_bounds(meshes: list, basis: np.ndarray, extra_pts=None) -> tuple:
    pts = [m.vertices @ basis.T for m in meshes if len(m.faces)]
    if extra_pts is not None:
        pts.append(np.asarray(extra_pts) @ basis.T)
    p = np.concatenate(pts)
    return (p[:, 0].min(), p[:, 0].max(), p[:, 1].min(), p[:, 1].max())


def sphere_mesh(radius: float, n_lat: int = 10, n_lon: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """UV sphere (outward CCW winding)."""
    verts = [(0.0, 0.0, radius)]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * math.pi * j / n_lon
            verts.append((radius * math.sin(th) * math.cos(ph), radius * math.sin(th) * math.sin(ph),
                          radius * math.cos(th)))
    verts.append((0.0, 0.0, -radius))
    faces = []
    for j in range(n_lon):
        faces.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            a, b = r0 + j, r0 + (j + 1) % n_lon
            c, d = r1 + j, r1 + (j + 1) % n_lon
            faces += [(a, c, b), (b, c, d)]
    last = len(verts) - 1
    r0 = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append((last, r0 + (j + 1) % n_lon, r0 + j))
    return np.asarray(verts), np.asarray(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------
FOOT_VISUAL_RGB = (0.06, 0.06, 0.06)      # rubber foot sphere (URDF visual)


def split_body(rig: Rig, labels: np.ndarray, palette: list, rest: dict) -> dict:
    """High-poly link meshes (link frames) + accessories (base frame)."""
    fl = rig.body_vertex_link[rig.body_faces]
    # majority vote of the three corners (ties -> first corner)
    face_link = np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0])
    frames = all_link_frames(rest)
    parts = {}
    for li, ln in enumerate(LINK_NAMES):
        sel = face_link == li
        V, F, used = remove_unreferenced(rig.body_vertices, rig.body_faces[sel])
        V = transform(inv_rigid(frames[ln]), V)
        parts[ln] = MeshPart(ln, V, F, labels[sel], palette, source_index=used)
    for name, acc in rig.accessories.items():
        parts[name] = MeshPart(name, acc.vertices.copy(), acc.faces.copy(), acc.labels.copy(),
                               list(acc.materials))
    return parts


OBJ_DECIMALS = 5          # 10 um vertex resolution in the OBJ files


def simplify_part(part: MeshPart, target: int, verbose: bool = False) -> MeshPart:
    """Weld + clean + QEM-decimate one part, then snap it to the OBJ grid
    (vertices that coincide after rounding are merged and faces that become
    degenerate are dropped, so the written file has no zero-area faces)."""
    V, F = weld_vertices(part.vertices, part.faces)
    F, L = clean_faces(F, part.labels, V)
    V2, F2, L2 = decimate(V, F, L, target, verbose=verbose)
    V2 = np.round(V2, OBJ_DECIMALS)
    V2, F2 = weld_vertices(V2, F2, tol=10.0 ** -OBJ_DECIMALS / 2)
    _n, area = face_normals(V2, F2)
    keep = area > 1e-12
    F2, L2 = clean_faces(F2[keep], L2[keep], V2)
    V2, F2, _used = remove_unreferenced(V2, F2)
    return MeshPart(part.name, V2, F2, L2, part.materials)


def mesh_frame_of(name: str) -> str:
    """URDF link a mesh file belongs to."""
    return "base_link" if name.startswith("base") else name


def export_meshes(parts: dict, mesh_dir: str) -> dict:
    """Write <name>.obj for every part + one shared spot.mtl. Returns stats."""
    os.makedirs(mesh_dir, exist_ok=True)
    mtl = {}
    for p in parts.values():
        for name, rgb in p.materials:
            mtl.setdefault(name, rgb)
    hdr = ("Spot quadruped - generated by tools/convert_spot.py from SpotHPRIG_Rigging_02.fbx\n"
           "Units: metres.  Frame: URDF link frame (x forward, y left, z up, zero joint angles).")
    write_mtl(os.path.join(mesh_dir, MTL_NAME), sorted(mtl.items()), hdr)
    stats = {}
    for name, p in parts.items():
        normals, nidx = corner_normals(p.vertices, p.faces)
        path = os.path.join(mesh_dir, f"{name}.obj")
        write_obj(path, p.vertices, p.faces, p.labels, p.materials, normals, nidx,
                  header=f"{hdr}\nLink: {mesh_frame_of(name)}  faces: {len(p.faces)}")
        stats[name] = {
            "file": f"meshes/{name}.obj", "link": mesh_frame_of(name),
            "faces": int(len(p.faces)), "vertices": int(len(p.vertices)),
            "normals": int(len(normals)), "bytes": os.path.getsize(path),
            "bbox_min": [round(float(v), 5) for v in p.vertices.min(0)],
            "bbox_max": [round(float(v), 5) for v in p.vertices.max(0)],
            "materials": sorted({p.materials[i][0] for i in np.unique(p.labels)}),
        }
    return stats


def load_exported(mesh_dir: str) -> dict:
    """Read every exported OBJ back -> {name: RenderMesh (link frame)}."""
    mats = read_mtl(os.path.join(mesh_dir, MTL_NAME))
    out = {}
    for fname in sorted(os.listdir(mesh_dir)):
        if not fname.endswith(".obj"):
            continue
        d = read_obj(os.path.join(mesh_dir, fname))
        rgb = np.array([mats.get(m, (0.7, 0.7, 0.7)) for m in d["face_material"]]).reshape(-1, 3)
        out[fname[:-4]] = RenderMesh(d["vertices"], d["faces"], rgb)
    return out


def assemble(exported: dict, joint_angles: dict, foot_spheres: bool = True) -> list:
    """Place link-frame meshes with FK -> list of RenderMesh in the base frame."""
    frames = all_link_frames(joint_angles)
    out = []
    for name, m in exported.items():
        T = frames[mesh_frame_of(name)]
        out.append(RenderMesh(transform(T, m.vertices), m.faces, m.face_rgb))
    if foot_spheres:
        sv, sf = sphere_mesh(rm.FOOT_RADIUS)
        for leg in LEGS:
            T = frames[f"{leg}_foot"]
            out.append(RenderMesh(transform(T, sv), sf, np.tile(FOOT_VISUAL_RGB, (len(sf), 1))))
    return out


def verify(rig: Rig, parts_hi: dict, exported: dict, rest: dict, n_samples: int = 12000,
           seed: int = 0) -> dict:
    """Assemble the exported (decimated, re-read from disk) meshes with FK at
    the rest angles and compare them with the ORIGINAL bind-pose vertices of
    the FBX (robot frame), in both directions (point-to-surface)."""
    rng = np.random.default_rng(seed)
    frames = all_link_frames(rest)
    res = {"per_mesh": {}}
    all_fwd, all_bwd = [], []
    roundtrip = 0.0
    for name, hi in parts_hi.items():
        T = frames[mesh_frame_of(name)]
        if name in rig.accessories:
            orig = rig.accessories[name].vertices
            orig_faces = rig.accessories[name].faces
        else:
            orig = rig.body_vertices[hi.source_index]
            orig_faces = hi.faces
            roundtrip = max(roundtrip, float(np.abs(transform(T, hi.vertices) - orig).max()))
        ex = exported[name]
        dec = transform(T, ex.vertices)
        q = orig[rng.choice(len(orig), size=min(n_samples, len(orig)), replace=False)]
        fwd = surface_distance(q, dec, ex.faces)               # original -> decimated
        bwd = surface_distance(dec, orig, orig_faces)          # decimated -> original
        res["per_mesh"][name] = {
            "orig_to_dec_mean_mm": round(1e3 * float(np.mean(fwd)), 3),
            "orig_to_dec_p95_mm": round(1e3 * float(np.percentile(fwd, 95)), 3),
            "orig_to_dec_max_mm": round(1e3 * float(np.max(fwd)), 3),
            "dec_to_orig_mean_mm": round(1e3 * float(np.mean(bwd)), 3),
            "dec_to_orig_max_mm": round(1e3 * float(np.max(bwd)), 3),
        }
        all_fwd.append(fwd)
        all_bwd.append(bwd)
    fwd, bwd = np.concatenate(all_fwd), np.concatenate(all_bwd)
    res["orig_to_dec_mean_mm"] = round(1e3 * float(np.mean(fwd)), 3)
    res["orig_to_dec_p95_mm"] = round(1e3 * float(np.percentile(fwd, 95)), 3)
    res["orig_to_dec_p99_mm"] = round(1e3 * float(np.percentile(fwd, 99)), 3)
    res["orig_to_dec_max_mm"] = round(1e3 * float(np.max(fwd)), 3)
    res["dec_to_orig_mean_mm"] = round(1e3 * float(np.mean(bwd)), 3)
    res["dec_to_orig_max_mm"] = round(1e3 * float(np.max(bwd)), 3)
    res["hausdorff_mm"] = max(res["orig_to_dec_max_mm"], res["dec_to_orig_max_mm"])
    res["link_frame_roundtrip_max_mm"] = round(1e3 * roundtrip, 6)
    # joint placement: FK (robot_model geometry at the rest angles) vs bind-pose bones
    joints = {}
    for i, leg in enumerate(LEGS):
        names = leg_bone_names(leg)
        Fk = leg_link_frames(i, rest[leg])

        def err(bone, link):
            return round(1e3 * float(np.linalg.norm(rig.bones[bone] - Fk[link][:3, 3])), 3)

        joints[leg] = {"hip_x_mm": err(names["hip"], f"{leg}_hip"),
                       "hip_y_mm": err(names["upper_leg"], f"{leg}_upper_leg"),
                       "knee_mm": err(names["lower_leg"], f"{leg}_lower_leg"),
                       "foot_mm": err(names["foot"], f"{leg}_foot")}
    res["joint_fit"] = joints
    res["joint_fit_max_mm"] = max(v for j in joints.values() for v in j.values())
    return res


# ---------------------------------------------------------------------------
# Preview sheet
# ---------------------------------------------------------------------------
def stand_pose(body_height: float = rm.NOMINAL_BODY_HEIGHT) -> dict:
    """Joint angles with the feet under the hips at the given body height."""
    feet = rm.nominal_foot_positions(body_height)
    return {leg: leg_ik_sagittal(i, feet[i]) for i, leg in enumerate(LEGS)}


def make_preview(rig: Rig, face_rgb: np.ndarray, exported: dict, rest: dict, path: str,
                 stats_line: str = "") -> None:
    """Grid of renders: rows = side / top / 3-4 view, columns = original FBX
    (bind pose, textured colours) and the converted model at the rest, zero
    and stand poses (assembled from the OBJ files with FK)."""
    import cv2
    original = [RenderMesh(rig.body_vertices, rig.body_faces, face_rgb.astype(np.float64) / 255.0)]
    for acc in rig.accessories.values():
        cols = np.array([acc.materials[i][1] for i in acc.labels])
        original.append(RenderMesh(acc.vertices, acc.faces, cols))
    zero = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
    stand = stand_pose()
    n_orig = sum(len(m.faces) for m in original)
    n_conv = sum(len(m.faces) for m in exported.values())

    def lowest_foot(q):
        return min(foot_position(i, q[leg])[2] for i, leg in enumerate(LEGS)) - rm.FOOT_RADIUS

    columns = [
        (f"original FBX - bind pose ({n_orig // 1000}k tris)", original, -rig.origin_g[2]),
        (f"converted - rest pose ({n_conv // 1000}k tris)", assemble(exported, rest), lowest_foot(rest)),
        ("converted - zero pose (URDF q=0)", assemble(exported, zero), lowest_foot(zero)),
        (f"converted - stand pose (h={rm.NOMINAL_BODY_HEIGHT:.2f} m)", assemble(exported, stand),
         lowest_foot(stand)),
    ]
    pw, ph, head, left, gap = 420, 300, 46, 64, 22
    W = left + len(columns) * pw
    H = head + len(VIEWS) * (ph + gap) + 26
    sheet = np.full((H, W, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, "Spot model conversion (tools/convert_spot.py): ground line = lowest foot-sphere bottom "
                "(original: FBX floor)", (left, 28), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    for r, (vname, basis) in enumerate(VIEWS.items()):
        extra = [[0.0, 0.0, g] for _t, _m, g in columns]
        bounds = None
        for _t, meshes, _g in columns:
            b = screen_bounds(meshes, basis, extra)
            bounds = b if bounds is None else (min(bounds[0], b[0]), max(bounds[1], b[1]),
                                               min(bounds[2], b[2]), max(bounds[3], b[3]))
        y0 = head + r * (ph + gap) + gap
        cv2.putText(sheet, vname, (8, y0 + ph // 2), font, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        for c, (title, meshes, g) in enumerate(columns):
            img = render(meshes, basis, (pw, ph), bounds, ground_z=g if vname != "top" else None)
            x0 = left + c * pw
            sheet[y0:y0 + ph, x0:x0 + pw] = img
            cv2.rectangle(sheet, (x0, y0), (x0 + pw - 1, y0 + ph - 1), (200, 200, 200), 1)
            if r == 0:
                cv2.putText(sheet, title, (x0 + 6, y0 - 6), font, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    if stats_line:
        cv2.putText(sheet, stats_line, (left, H - 9), font, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, sheet)


# ---------------------------------------------------------------------------
# URDF-driven preview (what the URDF actually describes)
# ---------------------------------------------------------------------------
COLLISION_PREVIEW_PATH = os.path.join(REPO, "docs", "images", "spot_collision_preview.png")


def _xyz_rpy(el) -> np.ndarray:
    o = el.find("origin") if el is not None else None
    xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
    rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
    return homogeneous(rot_z(rpy[2]) @ rot_y(rpy[1]) @ rot_x(rpy[0]), xyz)


def urdf_link_frames(root, q: dict) -> dict:
    """Pose of every URDF link in the root link frame for joint angles q."""
    joints = {}
    for j in root.findall("joint"):
        joints[j.find("child").get("link")] = j
    frames = {}

    def pose(link):
        if link in frames:
            return frames[link]
        j = joints.get(link)
        if j is None:
            frames[link] = np.eye(4)
            return frames[link]
        T = pose(j.find("parent").get("link")) @ _xyz_rpy(j)
        if j.get("type") in ("revolute", "continuous"):
            ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            ax /= np.linalg.norm(ax)
            a = q.get(j.get("name"), 0.0)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            T = T @ homogeneous(np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K)
        frames[link] = T
        return T

    for lk in root.findall("link"):
        pose(lk.get("name"))
    return frames


def primitive_mesh(geom) -> tuple[np.ndarray, np.ndarray] | None:
    """Triangle mesh of a URDF <geometry> box / cylinder / sphere."""
    if geom.find("box") is not None:
        sx, sy, sz = (float(v) / 2 for v in geom.find("box").get("size").split())
        V = np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)])
        F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
        return V, F
    if geom.find("cylinder") is not None:
        c = geom.find("cylinder")
        r, h, n = float(c.get("radius")), float(c.get("length")), 24
        ang = 2 * np.pi * np.arange(n) / n
        ring = np.column_stack([r * np.cos(ang), r * np.sin(ang)])
        V = np.vstack([np.column_stack([ring, np.full(n, -h / 2)]), np.column_stack([ring, np.full(n, h / 2)]),
                       [[0, 0, -h / 2], [0, 0, h / 2]]])
        F = []
        for i in range(n):
            j = (i + 1) % n
            F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        return V, np.array(F)
    if geom.find("sphere") is not None:
        return sphere_mesh(float(geom.find("sphere").get("radius")))
    return None


_COLLISION_COLOURS = {"base": (0.30, 0.55, 0.85), "upper": (0.95, 0.60, 0.20), "lower": (0.35, 0.75, 0.40),
                      "foot": (0.85, 0.20, 0.20)}


def urdf_render_meshes(root, q: dict, exported: dict, kind: str) -> list:
    """RenderMeshes of the URDF visuals ("visual") or collisions ("collision")
    placed with the URDF joint chain; visual mode adds the camera axes."""
    frames = urdf_link_frames(root, q)
    mats = {m.get("name"): m for m in root.findall("material")}
    out = []
    for lk in root.findall("link"):
        T = frames[lk.get("name")]
        for el in lk.findall(kind):
            Tg = T @ _xyz_rpy(el)
            geom = el.find("geometry")
            mesh = geom.find("mesh")
            if mesh is not None:
                key = os.path.splitext(os.path.basename(mesh.get("filename")))[0]
                m = exported[key]
                out.append(RenderMesh(transform(Tg, m.vertices), m.faces, m.face_rgb))
                continue
            prim = primitive_mesh(geom)
            if prim is None:
                continue
            V, F = prim
            if kind == "collision":
                name = lk.get("name")
                part = ("foot" if name.endswith("foot") else "lower" if "lower" in name
                        else "upper" if "upper" in name else "base")
                rgb = _COLLISION_COLOURS[part]
            else:
                mref = el.find("material")
                col = mref.find("color") if mref is not None else None
                if col is None and mref is not None and mref.get("name") in mats:
                    col = mats[mref.get("name")].find("color")
                rgb = tuple(float(v) for v in col.get("rgba").split()[:3]) if col is not None else (0.6, 0.6, 0.6)
            out.append(RenderMesh(transform(Tg, V), F, np.tile(rgb, (len(F), 1))))
    if kind == "visual":                         # camera viewing directions (+X of the camera links)
        for link, rgb in (("depth_camera_link", (0.1, 0.4, 1.0)), ("inspection_camera_link", (0.1, 0.8, 0.2)),
                          ("thermal_camera_link", (0.9, 0.1, 0.1))):
            if link not in frames:
                continue
            V, F = primitive_mesh(_box_geometry(0.25, 0.008, 0.008))
            V = V + [0.125, 0, 0]
            out.append(RenderMesh(transform(frames[link], V), F, np.tile(rgb, (len(F), 1))))
    return out


def _box_geometry(x: float, y: float, z: float):
    """<geometry><box size=.../></geometry> element (for primitive_mesh)."""
    import xml.etree.ElementTree as ET
    g = ET.Element("geometry")
    ET.SubElement(g, "box", size=f"{x} {y} {z}")
    return g


def make_collision_preview(urdf_path: str, exported: dict, path: str = COLLISION_PREVIEW_PATH) -> None:
    """Visual vs collision geometry of the expanded URDF at the zero and the
    stand pose (+ camera viewing axes: depth blue, inspection green, thermal red)."""
    import cv2
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()
    poses = {"zero pose": {}, f"stand pose (h={rm.NOMINAL_BODY_HEIGHT:.2f} m)": {}}
    for leg, q in stand_pose().items():
        for suffix, v in zip(rm.JOINT_SUFFIXES, q):
            poses[f"stand pose (h={rm.NOMINAL_BODY_HEIGHT:.2f} m)"][f"{leg}_{suffix}"] = v
    columns = []
    for pname, q in poses.items():
        frames = urdf_link_frames(root, q)
        ground = min(frames[f"{leg}_foot"][2, 3] for leg in LEGS) - rm.FOOT_RADIUS
        for kind in ("visual", "collision"):
            columns.append((f"URDF {kind} - {pname}", urdf_render_meshes(root, q, exported, kind), ground))
    views = {"side": VIEWS["side"], "3/4": VIEWS["3/4"]}
    pw, ph, head, left, gap = 420, 300, 46, 64, 22
    W = left + len(columns) * pw
    H = head + len(views) * (ph + gap) + 10
    sheet = np.full((H, W, 3), 255, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(sheet, "spot.urdf: visual vs collision (box trunk, box thighs, cylinder shins, sphere feet); "
                "camera axes: depth=blue inspection=green thermal=red", (left, 28), font, 0.5, (30, 30, 30), 1,
                cv2.LINE_AA)
    for r, (vname, basis) in enumerate(views.items()):
        bounds = None
        for _t, meshes, g in columns:
            b = screen_bounds(meshes, basis, [[0, 0, g]])
            bounds = b if bounds is None else (min(bounds[0], b[0]), max(bounds[1], b[1]),
                                               min(bounds[2], b[2]), max(bounds[3], b[3]))
        y0 = head + r * (ph + gap) + gap
        cv2.putText(sheet, vname, (8, y0 + ph // 2), font, 0.6, (60, 60, 60), 1, cv2.LINE_AA)
        for c, (title, meshes, g) in enumerate(columns):
            x0 = left + c * pw
            sheet[y0:y0 + ph, x0:x0 + pw] = render(meshes, basis, (pw, ph), bounds, ground_z=g)
            cv2.rectangle(sheet, (x0, y0), (x0 + pw - 1, y0 + ph - 1), (200, 200, 200), 1)
            if r == 0:
                cv2.putText(sheet, title, (x0 + 6, y0 - 6), font, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, sheet)


def dir_size(path: str) -> int:
    total = 0
    for root, _d, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def convert(preview: bool = True, verbose: bool = True, preview_path: str = PREVIEW_PATH) -> dict:
    """Run the full pipeline; returns the report dict (also written as JSON)."""
    t0 = time.time()

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:6.1f} s] {msg}", flush=True)

    rig = load_rig()
    log(f"loaded rig: body {len(rig.body_faces)} tris, accessories "
        f"{ {k: v.n_faces for k, v in rig.accessories.items()} }, origin_G {np.round(rig.origin_g, 5)}")
    labels, palette, face_rgb = quantise_body_colours(rig.body_vertices, rig.body_faces,
                                                      rig.body_face_uv, rig.texture_path)
    log(f"palette: {palette}")
    rest = rest_angles_from_bones(rig.bones)
    log("rest angles (deg): " + ", ".join(f"{k}=" + str(np.round(np.degrees(v), 2)) for k, v in rest.items()))
    parts_hi = split_body(rig, labels, palette, rest)
    parts_lo = {}
    for name, part in parts_hi.items():
        tgt = target_for(name)
        parts_lo[name] = simplify_part(part, tgt, verbose=False)
        log(f"{name:14s} {part.n_faces:7d} -> {parts_lo[name].n_faces:6d} tris (target {tgt})")
    os.makedirs(MESH_DIR, exist_ok=True)
    for f in os.listdir(MESH_DIR):                     # rerunnable: drop stale outputs
        if f.endswith((".obj", ".mtl")):
            os.remove(os.path.join(MESH_DIR, f))
    stats = export_meshes(parts_lo, MESH_DIR)
    exported = load_exported(MESH_DIR)
    log("verifying (decimated OBJ files re-read and assembled with FK at the rest angles) ...")
    ver = verify(rig, parts_hi, exported, rest)
    log(f"surface error original->converted mean {ver['orig_to_dec_mean_mm']} mm, p95 "
        f"{ver['orig_to_dec_p95_mm']} mm, max {ver['orig_to_dec_max_mm']} mm; Hausdorff "
        f"{ver['hausdorff_mm']} mm; joint fit max {ver['joint_fit_max_mm']} mm; "
        f"link-frame round trip {ver['link_frame_roundtrip_max_mm']} mm")
    total_faces = int(sum(s["faces"] for s in stats.values()))
    report = {
        "generator": "tools/convert_spot.py",
        "source": {"package": "Doosanenerbility/Doosanenerbility/Modeling/object/robot_spot.unitypackage",
                   "fbx": FBX_NAME, "texture": TEXTURE_NAME,
                   "original_triangles": {"Spot_Body_01": int(len(rig.body_faces)),
                                          **{k: v.n_faces for k, v in rig.accessories.items()}}},
        "frame": {"origin_G": [round(float(v), 6) for v in rig.origin_g],
                  "transform": "x_r = y_G - origin_G.y ; y_r = -(x_G - origin_G.x) ; z_r = z_G - origin_G.z",
                  "G": "FBX file axes in metres: x = robot right, y = robot forward, z = up"},
        "rest_angles_rad": {k: [round(float(a), 6) for a in v] for k, v in rest.items()},
        "rest_angles_deg": {k: [round(math.degrees(a), 3) for a in v] for k, v in rest.items()},
        "palette": [{"name": n, "rgb": list(c)} for n, c in palette],
        "triangle_targets": TRIANGLE_TARGETS,
        "meshes": stats,
        "total_triangles": total_faces,
        "verification": ver,
        "foot_visual": {"type": "sphere", "radius": rm.FOOT_RADIUS, "rgb": list(FOOT_VISUAL_RGB),
                        "lowest_mesh_point_lower_leg_frame_z":
                            round(float(min(stats[f"{leg}_lower_leg"]["bbox_min"][2] for leg in LEGS)), 5),
                        "foot_centre_lower_leg_frame_z": -rm.LOWER_LEG_LENGTH},
    }
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    if preview:
        log("rendering preview ...")
        line = (f"{total_faces} triangles in {len(stats)} OBJ files; surface error orig->conv mean "
                f"{ver['orig_to_dec_mean_mm']:.2f} mm / p95 {ver['orig_to_dec_p95_mm']:.2f} mm; joint fit "
                f"<= {ver['joint_fit_max_mm']:.1f} mm; rest angles front "
                f"{np.degrees(rest['fl'][1]):.1f}/{np.degrees(rest['fl'][2]):.1f} deg, rear "
                f"{np.degrees(rest['rl'][1]):.1f}/{np.degrees(rest['rl'][2]):.1f} deg")
        make_preview(rig, face_rgb, exported, rest, preview_path, line)
        log(f"preview -> {os.path.relpath(preview_path, REPO)}")
    size = dir_size(MODEL_DIR)
    report["model_dir_bytes"] = size
    with open(REPORT_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    log(f"done: {total_faces} triangles, simulation/models/spot = {size / 1e6:.2f} MB")
    return report


def expand_urdf(xacro_path: str = URDF_XACRO, out_path: str = URDF_PATH, mappings: dict | None = None) -> str:
    """Expand simulation/urdf/spot.urdf.xacro into the committed spot.urdf
    (needs the pure-python `xacro` package: pip install xacro).  On Ubuntu the
    launch files run xacro themselves; the .urdf is a convenience /
    validation artefact.  Returns the output path."""
    import xacro
    doc = xacro.process_file(xacro_path, mappings=dict(mappings or {}))
    text = doc.toprettyxml(indent="  ")
    # keep the machine-specific absolute path out of the committed banner
    rel = os.path.relpath(xacro_path, REPO).replace(os.sep, "/")
    text = text.replace(xacro_path, rel)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return out_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-preview", action="store_true", help="skip rendering docs/images/spot_model_preview.png")
    ap.add_argument("--preview-path", default=PREVIEW_PATH, help="output path of the preview sheet")
    ap.add_argument("--urdf-only", action="store_true",
                    help="only re-expand simulation/urdf/spot.urdf from the xacro (no mesh conversion)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if not args.urdf_only:
        convert(preview=not args.no_preview, verbose=not args.quiet, preview_path=args.preview_path)
    try:
        out = expand_urdf()
    except ImportError:
        print("python package 'xacro' not installed: spot.urdf not regenerated (pip install xacro)")
        return 1 if args.urdf_only else 0
    if not args.quiet:
        print(f"expanded {os.path.relpath(URDF_XACRO, REPO)} -> {os.path.relpath(out, REPO)}")
    if not args.no_preview:
        make_collision_preview(out, load_exported(MESH_DIR))
        if not args.quiet:
            print(f"collision preview -> {os.path.relpath(COLLISION_PREVIEW_PATH, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
