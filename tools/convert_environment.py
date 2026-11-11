#!/usr/bin/env python3
"""
Convert the Unity environment assets (turbine hall, corridor, machines, pallet)
into Gazebo models:  model.config + model.sdf + meshes/<name>.obj/.mtl + JPG
textures, plus verification renders.

Inputs (read through tools/unitypackage.py, never modified):
    factory_inner_1.unitypackage  Factory_03.fbx          -> factory_hall
    factory_hall.unitypackage     Corridor_Wall_02.fbx    -> corridor
    fac_gastank.unitypackage      Main_Machine_01.fbx     -> turbine_machine
                                  (Main_Machine_01 + Gauge_01_003/004/005)
                                                          -> gas_tank (Main_Machine_02)
    obs_palette.unitypackage      Plastic_Pallet_01.fbx   -> pallet

Outputs:
    simulation/models/<name>/model.config, model.sdf
    simulation/models/<name>/meshes/<name>.obj, <name>.mtl, <name>_<texture>.jpg
    simulation/config/generated/model_manifest.yaml   (sources, sizes, materials)
    docs/images/world_model_<name>.png                (top/side/iso renders)

Pipeline
--------
1. Geometry: tools/fbx_reader.py parses the binary FBX.  Every polygon is
   fan-triangulated; control points go through the full FBX transform chain
   and FBXScene.axis_matrix() ("G frame": metres, Z up, X = file X).  The
   axis matrix is a proper rotation (det = +1) so the FBX counter-clockwise
   winding is kept; per-corner normals are transformed with the inverse
   transpose.  UVs are copied unchanged (FBX and OBJ both have v pointing up
   from the bottom of the image).
2. Materials: Unity decides the material of each sub-mesh through the prefab
   (MeshRenderer.m_Materials).  Unity orders sub-meshes by the FIRST USE of
   each FBX material index in the polygon list, so slot k of the renderer
   belongs to the k-th material in first-use order (verified on all three
   prefabs: e.g. FBX 'Material.004' -> slot 10 -> Material.001_BaseColor.mat,
   whose texture is the one the FBX itself references).  When no prefab
   renderer matches, the .mat asset with the same name is used (Unity's
   "search by name" import rule), then the texture embedded in / referenced by
   the FBX, then the FBX diffuse colour.  From the .mat we read the albedo
   texture GUID (HDRP _BaseColorMap, then _MainTex, _BaseMap), its tiling
   (baked into the UVs), the tint (_BaseColor/_Color), emissive colour,
   transparency and the double-sided flag.  The texture file is
   <cache>/<guid>/asset.
3. Building shells are made double-sided (duplicated faces with reversed
   winding): several walls of Factory_03 face outwards and are only visible
   from the inside in Unity because their HDRP materials are double sided;
   ogre2 culls back faces.
4. Machines / pallet are recentred: origin at the bbox centre in x, y and
   z_min = 0 (before any scale).  The layout scale (gas tank 0.8) is applied
   by the world generator through <mesh><scale>.
5. Optional decimation (vertex clustering that keeps the per-corner UVs and
   normals of the surviving triangles) for models above a triangle budget.
   The full-resolution meshes fit comfortably in the size budget, so it is
   disabled by default (see MODEL_SPECS / --max-tris).
6. Textures are downscaled to <= 1024 px (INTER_AREA) and written as JPG
   (PNG only for genuinely transparent materials).  File names are prefixed
   with the model name so that ogre2's texture cache can never confuse two
   different images with the same base name.
7. Verification renders: a small z-buffer rasteriser (numpy) draws the
   converted OBJ (read back from disk) with its textures from the top, the
   side and an isometric direction (building roofs cut away).

Usage:
    python tools/convert_environment.py                    # everything
    python tools/convert_environment.py --models pallet,gas_tank
    python tools/convert_environment.py --no-preview --max-tex 512
"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MODELS_DIR = os.path.join(REPO, "simulation", "models")
GENERATED_DIR = os.path.join(REPO, "simulation", "config", "generated")
IMAGES_DIR = os.path.join(REPO, "docs", "images")
SDF_VERSION = "1.9"

# Building shells: every material is rendered double sided (see module doc).
# Machines follow the Unity material flag.
MODEL_SPECS = {
    "factory_hall": {
        "package": "factory_inner_1",
        "fbx": "Factory_03.fbx",
        "prefab": "factory_inner_1.prefab",
        "models": None,                 # all meshes of the file
        "recentre": False,              # world frame == G frame of this file
        "double_sided": "all",
        "max_tris": None,
        "description": "Turbine hall shell (walls, floor, roof, mezzanine, back room) from "
                       "Factory_03.fbx (factory_inner_1.unitypackage). Visual only; the world "
                       "generator adds primitive collisions.",
    },
    "corridor": {
        "package": "factory_hall",
        "fbx": "Corridor_Wall_02.fbx",
        "prefab": "factory_hall_1.prefab",
        "models": None,
        "recentre": False,              # placed at layout corridor.pose (native coordinates)
        "double_sided": "all",
        "max_tris": None,
        "description": "Service corridor with stairs + ramp descent from Corridor_Wall_02.fbx "
                       "(factory_hall.unitypackage). Visual only.",
    },
    "turbine_machine": {
        "package": "fac_gastank",
        "fbx": "Main_Machine_01.fbx",
        "prefab": "fac_gastank_1.prefab",
        "models": ["Main_Machine_01", "Gauge_01_003", "Gauge_01_004", "Gauge_01_005"],
        "recentre": True,
        "double_sided": "material",
        "max_tris": None,
        "thermal_k": 303.15,
        "description": "Turbine / main machine (Main_Machine_01 + its three small gauges) from "
                       "Main_Machine_01.fbx (fac_gastank.unitypackage). Origin = bbox centre, "
                       "z_min = 0.",
    },
    "gas_tank": {
        "package": "fac_gastank",
        "fbx": "Main_Machine_01.fbx",
        "prefab": "fac_gastank_1.prefab",
        "models": ["Main_Machine_02"],
        "recentre": True,
        "double_sided": "material",
        "max_tris": None,
        "thermal_k": 303.15,
        "description": "Gas tank (Main_Machine_02) from Main_Machine_01.fbx "
                       "(fac_gastank.unitypackage). Origin = bbox centre, z_min = 0; the world "
                       "scales it by the layout factor.",
    },
    "pallet": {
        "package": "obs_palette",
        "fbx": "Plastic_Pallet_01.fbx",
        "prefab": "obs_palette_1.prefab",
        "models": None,
        "recentre": True,
        "double_sided": "material",
        "max_tris": None,
        "collision": "bbox",
        "description": "Plastic pallet from Plastic_Pallet_01.fbx (obs_palette.unitypackage). "
                       "Mesh visual + bounding-box collision.",
    },
}

ALBEDO_KEYS = ("_BaseColorMap", "_MainTex", "_BaseMap")
MIN_GLASS_ALPHA = 0.30


# =============================================================================
# Small helpers
# =============================================================================
def slug(name: str) -> str:
    """Lower-case identifier safe for MTL material names and file names."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    return s or "unnamed"


def fmt(v: float, nd: int = 4) -> str:
    """Compact float formatting for SDF text ('-0' avoided)."""
    s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fmt_vec(vals, nd: int = 4) -> str:
    return " ".join(fmt(v, nd) for v in vals)


def imread_any(path_or_bytes) -> np.ndarray | None:
    """Read an image file (format sniffed from the content) or encoded bytes ->
    uint8 BGR or BGRA array, or None."""
    import cv2
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(bytes(path_or_bytes), dtype=np.uint8)
    else:
        buf = np.fromfile(path_or_bytes, dtype=np.uint8)
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def downscale(img: np.ndarray, max_size: int) -> np.ndarray:
    import cv2
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_size:
        return img
    f = max_size / float(s)
    return cv2.resize(img, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                      interpolation=cv2.INTER_AREA)


# =============================================================================
# Mesh container
# =============================================================================
@dataclass
class Material:
    """Render material of one OBJ sub-mesh."""
    name: str
    tint: tuple = (0.7, 0.7, 0.7)          # Kd (multiplies the texture)
    alpha: float = 1.0
    texture: np.ndarray | None = None      # BGR(A) uint8 (already downscaled)
    texture_key: str | None = None         # source identity (dedupe textures)
    texture_label: str | None = None       # base name used for the output file
    texture_file: str | None = None        # file name written next to the OBJ
    double_sided: bool = False
    emissive: tuple | None = None
    transparent: bool = False
    source: str = ""                       # human readable provenance

    @property
    def mean_colour(self) -> np.ndarray:
        """Average RGB 0..1 (texture mean x tint) - used by the previews."""
        tint = np.asarray(self.tint, float)
        if self.texture is None:
            return tint
        bgr = self.texture[..., :3].reshape(-1, 3).mean(0) / 255.0
        return np.clip(bgr[::-1] * tint, 0, 1)


@dataclass
class Mesh:
    """Indexed triangle mesh with independent position / uv / normal indices
    (exactly the OBJ data model)."""
    v: np.ndarray                   # (V,3) float64 positions
    uv: np.ndarray                  # (U,2) float64
    vn: np.ndarray                  # (N,3) float64 unit normals
    fv: np.ndarray                  # (F,3) int64 position indices
    fuv: np.ndarray                 # (F,3) int64 uv indices (-1 = none)
    fn: np.ndarray                  # (F,3) int64 normal indices (-1 = none)
    fmat: np.ndarray                # (F,) int64 index into materials
    materials: list = field(default_factory=list)

    @property
    def n_tris(self) -> int:
        return int(len(self.fv))

    def triangles(self) -> np.ndarray:
        return self.v[self.fv]

    def bbox(self) -> tuple[np.ndarray, np.ndarray]:
        used = np.unique(self.fv)
        p = self.v[used]
        return p.min(0), p.max(0)

    def transformed(self, T: np.ndarray) -> "Mesh":
        """Copy with a 4x4 affine transform applied (normals: inverse transpose;
        winding flipped automatically for mirroring transforms)."""
        R = T[:3, :3]
        v = self.v @ R.T + T[:3, 3]
        Rn = np.linalg.inv(R).T
        vn = self.vn @ Rn.T
        vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
        fv, fuv, fn = self.fv, self.fuv, self.fn
        if np.linalg.det(R) < 0:
            fv, fuv, fn = fv[:, ::-1], fuv[:, ::-1], fn[:, ::-1]
        return Mesh(v, self.uv.copy(), vn, fv.copy(), fuv.copy(), fn.copy(), self.fmat.copy(),
                    list(self.materials))

    def face_normals(self) -> np.ndarray:
        t = self.triangles()
        n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    def face_areas(self) -> np.ndarray:
        t = self.triangles()
        return 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)

    def material_names(self) -> np.ndarray:
        """Per-face material name array."""
        names = np.array([m.name for m in self.materials] + [""], dtype=object)
        return names[self.fmat]


def concat_meshes(meshes: list[Mesh]) -> Mesh:
    """Concatenate meshes; materials that are identical (same name, source,
    tint and texture) are merged, distinct materials sharing a name are kept."""
    v, uv, vn, fv, fuv, fn, fmat = [], [], [], [], [], [], []
    mats: list[Material] = []
    index: dict[tuple, int] = {}
    ov = ou = on = 0
    for m in meshes:
        remap = []
        for mat in m.materials:
            key = (mat.name, mat.source, tuple(np.round(mat.tint, 4)), mat.texture_key,
                   mat.double_sided, mat.transparent)
            if key not in index:
                index[key] = len(mats)
                mats.append(mat)
            remap.append(index[key])
        remap = np.asarray(remap + [0], dtype=np.int64)
        v.append(m.v); uv.append(m.uv); vn.append(m.vn)
        fv.append(m.fv + ov)
        fuv.append(np.where(m.fuv >= 0, m.fuv + ou, -1))
        fn.append(np.where(m.fn >= 0, m.fn + on, -1))
        fmat.append(remap[m.fmat])
        ov += len(m.v); ou += len(m.uv); on += len(m.vn)
    return Mesh(np.concatenate(v), np.concatenate(uv) if uv else np.zeros((0, 2)),
                np.concatenate(vn) if vn else np.zeros((0, 3)), np.concatenate(fv),
                np.concatenate(fuv), np.concatenate(fn), np.concatenate(fmat), mats)


def _dedupe_rows(arr: np.ndarray, decimals: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique rows after rounding -> (unique array, inverse index)."""
    if len(arr) == 0:
        return arr, np.zeros(0, dtype=np.int64)
    key = np.round(arr, decimals)
    key[key == 0] = 0.0                      # merge -0.0 and 0.0
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    return uniq, inv.reshape(-1)


def compact(mesh: Mesh) -> Mesh:
    """Drop unused / duplicate vertices, uvs and normals and degenerate faces."""
    t = mesh.triangles()
    area2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    keep = (area2 > 1e-14) & (mesh.fv[:, 0] != mesh.fv[:, 1]) & (mesh.fv[:, 1] != mesh.fv[:, 2]) \
        & (mesh.fv[:, 0] != mesh.fv[:, 2])
    fv, fuv, fn, fmat = mesh.fv[keep], mesh.fuv[keep], mesh.fn[keep], mesh.fmat[keep]
    v, inv_v = _dedupe_rows(mesh.v, 6)
    fv = inv_v[fv]
    used, fv_new = np.unique(fv, return_inverse=True)
    v = v[used]
    fv = fv_new.reshape(-1, 3)

    def remap_attr(arr, fidx, decimals):
        if len(arr) == 0 or np.all(fidx < 0):
            return np.zeros((0, arr.shape[1] if arr.ndim == 2 else 2)), np.full_like(fidx, -1)
        u, inv = _dedupe_rows(arr, decimals)
        f2 = np.where(fidx >= 0, inv[np.clip(fidx, 0, None)], -1)
        usedi = np.unique(f2[f2 >= 0])
        lut = np.full(len(u), -1, dtype=np.int64)
        lut[usedi] = np.arange(len(usedi))
        return u[usedi], np.where(f2 >= 0, lut[np.clip(f2, 0, None)], -1)

    uv, fuv = remap_attr(mesh.uv, fuv, 5)
    vn, fn = remap_attr(mesh.vn, fn, 4)
    # drop materials that are no longer used
    used_m = np.unique(fmat)
    lut = np.full(len(mesh.materials), -1, dtype=np.int64)
    lut[used_m] = np.arange(len(used_m))
    mats = [mesh.materials[i] for i in used_m]
    return Mesh(v, uv, vn, fv, fuv, fn, lut[fmat], mats)


def make_double_sided(mesh: Mesh, material_mask: np.ndarray) -> Mesh:
    """Append back faces (reversed winding, negated normals) for the faces whose
    material is flagged in `material_mask` (bool per material)."""
    sel = material_mask[mesh.fmat]
    if not np.any(sel):
        return mesh
    nv = len(mesh.vn)
    vn = np.concatenate([mesh.vn, -mesh.vn])
    fn_back = np.where(mesh.fn[sel] >= 0, mesh.fn[sel] + nv, -1)[:, ::-1]
    return Mesh(mesh.v, mesh.uv, vn,
                np.concatenate([mesh.fv, mesh.fv[sel][:, ::-1]]),
                np.concatenate([mesh.fuv, mesh.fuv[sel][:, ::-1]]),
                np.concatenate([mesh.fn, fn_back]),
                np.concatenate([mesh.fmat, mesh.fmat[sel]]),
                mesh.materials)


# =============================================================================
# Decimation (vertex clustering, attribute preserving)
# =============================================================================
def cluster_decimate(mesh: Mesh, cell: float) -> Mesh:
    """Vertex clustering on a cubic grid of size `cell` (Rossignac-Borrel).

    Every vertex moves to the mean of its cell; faces whose corners collapse
    onto fewer than 3 cells disappear.  Surviving faces keep their original
    per-corner UV and normal indices (OBJ allows independent indices), so
    texture mapping and shading are preserved up to the sub-cell position shift.
    Duplicate faces (same cells, same material, same orientation) are removed.
    """
    vmin = mesh.v.min(0)
    key = np.floor((mesh.v - vmin) / cell).astype(np.int64)
    _, cid = np.unique(key, axis=0, return_inverse=True)
    cid = cid.reshape(-1)
    n_c = int(cid.max()) + 1
    cnt = np.bincount(cid, minlength=n_c).astype(float)
    rep = np.stack([np.bincount(cid, weights=mesh.v[:, k], minlength=n_c) for k in range(3)], 1)
    rep /= cnt[:, None]
    fv = cid[mesh.fv]
    ok = (fv[:, 0] != fv[:, 1]) & (fv[:, 1] != fv[:, 2]) & (fv[:, 0] != fv[:, 2])
    fv, fuv, fn, fmat = fv[ok], mesh.fuv[ok], mesh.fn[ok], mesh.fmat[ok]
    # remove duplicates: canonical rotation of the cell triple (keeps orientation)
    r = np.argmin(fv, axis=1)
    idx = (np.arange(3)[None, :] + r[:, None]) % 3
    canon = np.take_along_axis(fv, idx, axis=1)
    _, first = np.unique(np.c_[canon, fmat], axis=0, return_index=True)
    first = np.sort(first)
    out = Mesh(rep, mesh.uv, mesh.vn, fv[first], fuv[first], fn[first], fmat[first],
               mesh.materials)
    return compact(out)


def decimate_to(mesh: Mesh, max_tris: int, log=print) -> Mesh:
    """Smallest clustering cell (bisection) giving <= max_tris triangles."""
    if max_tris is None or mesh.n_tris <= max_tris:
        return mesh
    lo, hi = 1e-4, 0.2
    best = None
    for _ in range(18):
        mid = math.sqrt(lo * hi)
        d = cluster_decimate(mesh, mid)
        if d.n_tris <= max_tris:
            best, hi = d, mid
        else:
            lo = mid
    if best is None:
        best = cluster_decimate(mesh, hi)
    log(f"    decimated {mesh.n_tris} -> {best.n_tris} triangles (cell {hi * 100:.2f} cm)")
    return best


# =============================================================================
# Unity material resolution
# =============================================================================
_DOC_RE = re.compile(r"^--- !u!(\d+) &(-?\d+)", re.M)


def parse_unity_yaml_docs(text: str) -> list[tuple[int, int, str]]:
    """Split a Unity YAML asset into (class id, file id, body) documents."""
    out = []
    ms = list(_DOC_RE.finditer(text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        out.append((int(m.group(1)), int(m.group(2)), text[m.end():end]))
    return out


def parse_material(text: str) -> dict:
    """Relevant fields of a Unity .mat (Standard / URP / HDRP Lit)."""
    info = {"name": None, "textures": {}, "colors": {}, "floats": {}}
    m = re.search(r"\nMaterial:\s*\n(?:.*\n)*?\s+m_Name: ?(.*)", text)
    if m:
        info["name"] = m.group(1).strip()
    for m in re.finditer(
            r"- (_\w+):\s*\n\s*m_Texture: \{fileID: (-?\d+)(?:, guid: (\w+))?[^}]*\}\s*\n"
            r"\s*m_Scale: \{x: ([-\d.eE]+), y: ([-\d.eE]+)\}\s*\n"
            r"\s*m_Offset: \{x: ([-\d.eE]+), y: ([-\d.eE]+)\}", text):
        key, _fid, guid, sx, sy, ox, oy = m.groups()
        if guid:
            info["textures"][key] = (guid, (float(sx), float(sy)), (float(ox), float(oy)))
    for m in re.finditer(r"- (_\w+): \{r: ([-\d.eE]+), g: ([-\d.eE]+), b: ([-\d.eE]+), a: ([-\d.eE]+)\}",
                         text):
        info["colors"][m.group(1)] = tuple(float(x) for x in m.groups()[1:])
    for m in re.finditer(r"- (_\w+): ([-\d.eE]+)\s*$", text, re.M):
        info["floats"][m.group(1)] = float(m.group(2))
    return info


class UnityAssets:
    """GUID-level view of one extracted .unitypackage."""

    def __init__(self, package_name: str):
        from unitypackage import UnityPackage
        self.pkg = UnityPackage(package_name)
        self.guid_path: dict[str, str] = {}     # guid -> unity pathname
        self.guid_asset: dict[str, str] = {}    # guid -> local asset file
        for guid in os.listdir(self.pkg.dir):
            d = os.path.join(self.pkg.dir, guid)
            pn, asset = os.path.join(d, "pathname"), os.path.join(d, "asset")
            if os.path.isfile(pn) and os.path.isfile(asset):
                with open(pn, "r", encoding="utf-8", errors="replace") as f:
                    self.guid_path[guid] = f.readline().strip()
                self.guid_asset[guid] = asset
        self._mat_cache: dict[str, dict] = {}

    def guid_of(self, asset_file: str) -> str:
        return os.path.basename(os.path.dirname(asset_file))

    def find(self, suffix: str) -> str | None:
        return self.pkg.find(suffix)

    def material(self, guid: str) -> dict | None:
        if guid not in self.guid_asset or not self.guid_path[guid].lower().endswith(".mat"):
            return None
        if guid not in self._mat_cache:
            with open(self.guid_asset[guid], "r", encoding="utf-8", errors="replace") as f:
                info = parse_material(f.read())
            info["guid"] = guid
            info["path"] = self.guid_path[guid]
            self._mat_cache[guid] = info
        return self._mat_cache[guid]

    def materials_named(self, name: str) -> list[dict]:
        out = []
        for guid, p in sorted(self.guid_path.items(), key=lambda kv: kv[1]):
            if p.lower().endswith(".mat") and os.path.splitext(os.path.basename(p))[0].lower() == name.lower():
                out.append(self.material(guid))
        return [m for m in out if m]

    def asset_by_basename(self, filename: str) -> str | None:
        base = os.path.basename(filename.replace("\\", "/")).lower()
        for guid, p in sorted(self.guid_path.items(), key=lambda kv: kv[1]):
            if os.path.basename(p).lower() == base:
                return self.guid_asset[guid]
        return None

    def prefab_renderers(self, prefab_name: str | None) -> list[dict]:
        """MeshRenderers of a prefab: {'go_name', 'mesh_guid', 'materials': [guid,..]}."""
        if not prefab_name:
            return []
        path = self.find(prefab_name)
        if not path:
            return []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            docs = parse_unity_yaml_docs(f.read())
        names, meshes, renderers = {}, {}, []
        for cls, fid, body in docs:
            if cls == 1:                                        # GameObject
                m = re.search(r"m_Name: ?(.*)", body)
                names[fid] = m.group(1).strip() if m else ""
            elif cls == 33:                                     # MeshFilter
                go = re.search(r"m_GameObject: \{fileID: (-?\d+)", body)
                mg = re.search(r"m_Mesh: \{fileID: -?\d+, guid: (\w+)", body)
                if go and mg:
                    meshes[int(go.group(1))] = mg.group(1)
            elif cls in (23, 137):                              # (Skinned)MeshRenderer
                go = re.search(r"m_GameObject: \{fileID: (-?\d+)", body)
                mats = re.findall(r"- \{fileID: 2100000, guid: (\w+)", body)
                if go:
                    renderers.append({"go": int(go.group(1)), "materials": mats})
        for r in renderers:
            r["go_name"] = names.get(r["go"], "")
            r["mesh_guid"] = meshes.get(r["go"])
        return renderers


def _texture_from_mat(assets: UnityAssets, mat: dict):
    """(asset path, scale, offset, key) of the albedo texture of a .mat, or None."""
    for key in ALBEDO_KEYS:
        t = mat["textures"].get(key)
        if t and t[0] in assets.guid_asset:
            return assets.guid_asset[t[0]], t[1], t[2], f"{key}:{assets.guid_path[t[0]]}"
    return None


def _fbx_material_texture(scene, mat_id):
    """(filename, embedded bytes or None) of the FBX texture on the diffuse slot."""
    best = None
    for c, prop, _t in scene.children.get(mat_id, []):
        if scene.kind.get(c) != "Texture":
            continue
        if prop is None or any(k in prop for k in ("DiffuseColor", "base_color", "Base", "Albedo")):
            fname = scene.texture_filename(c)
            if not fname:
                continue
            content = None
            for vid in scene.child_ids(c, "Video"):
                node = scene.objects[vid].find("Content")
                if node is not None and node.props and isinstance(node.props[0], (bytes, bytearray)) \
                        and len(node.props[0]) > 0:
                    content = bytes(node.props[0])
            best = (fname, content)
            if prop and "DiffuseColor" in prop:
                break
    return best


def resolve_materials(assets: UnityAssets, scene, model_id: int, used_order: list[int],
                      prefab_slots: list[str] | None, max_tex: int, log=print) -> list[Material]:
    """One Material per FBX material index of `model_id` (list indexed by the
    FBX LayerElementMaterial index)."""
    from fbx_reader import _p70
    fbx_mats = scene.materials_of_model(model_id)
    out: list[Material] = []
    slot_of = {}
    if prefab_slots is not None and len(prefab_slots) == len(used_order):
        for k, mi in enumerate(used_order):
            slot_of[mi] = prefab_slots[k]
    for mi, mid in enumerate(fbx_mats):
        fbx_name = scene.name(mid)
        p70 = _p70(scene.objects[mid])
        mat_info, how = None, ""
        if mi in slot_of:
            mat_info = assets.material(slot_of[mi])
            how = "prefab slot"
        if mat_info is None:
            cands = assets.materials_named(fbx_name)
            if cands:
                mat_info, how = cands[0], "name match"
        tint = (0.7, 0.7, 0.7)
        if "DiffuseColor" in p70 and len(p70["DiffuseColor"]) >= 3:
            tint = tuple(float(x) for x in p70["DiffuseColor"][:3])
        alpha, double_sided, emissive, transparent = 1.0, False, None, False
        tex_img, tex_key, tex_label, source = None, None, None, f"FBX material '{fbx_name}'"
        uv_scale, uv_off = (1.0, 1.0), (0.0, 0.0)
        if mat_info is not None:
            col = mat_info["colors"].get("_BaseColor") or mat_info["colors"].get("_Color")
            if col:
                tint, alpha = col[:3], col[3]
            fl = mat_info["floats"]
            double_sided = fl.get("_DoubleSidedEnable", 0.0) > 0.5 or fl.get("_CullMode", 2.0) < 0.5
            # HDRP: _SurfaceType is authoritative (URP/legacy: _Surface)
            transparent = fl.get("_SurfaceType", fl.get("_Surface", 0.0)) > 0.5
            if not transparent:
                alpha = 1.0
            else:
                # Unity glass relies on reflections; without them a 5 % opaque
                # pane is invisible in ogre2 -> keep a visible glass tint.
                alpha = max(float(alpha), MIN_GLASS_ALPHA)
            em = mat_info["colors"].get("_EmissiveColor")
            if em and max(em[:3]) > 0.05:
                emissive = tuple(float(min(1.0, c)) for c in em[:3])
            source = f"{os.path.basename(mat_info['path'])} ({how})"
            tex = _texture_from_mat(assets, mat_info)
            if tex is not None:
                path, uv_scale, uv_off, tex_key = tex
                tex_img = imread_any(path)
                tex_label = os.path.splitext(os.path.basename(assets.guid_path[assets.guid_of(path)]))[0]
                source += f" -> {os.path.basename(assets.guid_path[assets.guid_of(path)])}"
        if tex_img is None:
            ft = _fbx_material_texture(scene, mid)
            if ft is not None:
                fname, content = ft
                path = assets.asset_by_basename(fname)
                if path is not None:
                    tex_img = imread_any(path)
                    tex_key = "file:" + assets.guid_path[assets.guid_of(path)]
                    source += f" -> {os.path.basename(fname)} (FBX reference)"
                elif content is not None:
                    tex_img = imread_any(content)
                    tex_key = "embedded:" + fname
                    source += f" -> {os.path.basename(fname.replace(chr(92), '/'))} (embedded in FBX)"
                if tex_img is not None:
                    tex_label = os.path.splitext(os.path.basename(fname.replace("\\", "/")))[0]
        if tex_img is not None:
            if not transparent and tex_img.ndim == 3 and tex_img.shape[2] == 4:
                tex_img = tex_img[..., :3]
            tex_img = downscale(tex_img, max_tex)
        mat = Material(name=fbx_name, tint=tuple(float(c) for c in tint), alpha=float(alpha),
                       texture=tex_img, texture_key=tex_key, texture_label=tex_label,
                       double_sided=bool(double_sided), emissive=emissive,
                       transparent=bool(transparent), source=source)
        mat.uv_transform = (uv_scale, uv_off)           # type: ignore[attr-defined]
        out.append(mat)
    return out


# =============================================================================
# FBX -> Mesh
# =============================================================================
def mesh_from_fbx(spec: dict, max_tex: int = 1024, log=print) -> Mesh:
    """Load the FBX of a MODEL_SPECS entry into one Mesh in the G frame."""
    from fbx_reader import FBXScene
    assets = UnityAssets(spec["package"])
    fbx_path = assets.find(spec["fbx"])
    if fbx_path is None:
        raise FileNotFoundError(f"{spec['fbx']} not found in {spec['package']}")
    scene = FBXScene(fbx_path)
    fbx_guid = assets.guid_of(fbx_path)
    A = scene.axis_matrix()
    renderers = [r for r in assets.prefab_renderers(spec.get("prefab")) if r.get("mesh_guid") == fbx_guid]
    model_ids = [m for m in scene.ids_of("Model") if scene.child_ids(m, "Geometry")]
    wanted = spec.get("models")
    if wanted:
        model_ids = [m for m in model_ids if scene.name(m) in wanted]
        missing = set(wanted) - {scene.name(m) for m in model_ids}
        if missing:
            raise ValueError(f"models {sorted(missing)} not in {spec['fbx']}")
    # renderer <-> FBX model assignment: by GameObject name, then leftovers in order
    all_models = [m for m in scene.ids_of("Model") if scene.child_ids(m, "Geometry")]
    assign, free_r = {}, list(range(len(renderers)))
    for m in all_models:
        for ri in free_r:
            if renderers[ri]["go_name"] == scene.name(m):
                assign[m] = renderers[ri]
                free_r.remove(ri)
                break
    left_models = [m for m in all_models if m not in assign]
    if len(left_models) == len(free_r):
        for m, ri in zip(left_models, free_r):
            assign[m] = renderers[ri]

    parts = []
    for mid in model_ids:
        for gid in scene.child_ids(mid, "Geometry"):
            geo = scene.geometry(gid)
            if geo is None:
                continue
            M = A @ scene.world_matrix(mid) @ scene.geometric_matrix(mid)
            R, t = M[:3, :3], M[:3, 3]
            pos = geo["vertices"] @ R.T + t
            starts, ends, pv = geo["starts"], geo["ends"], geo["poly_vertex"]
            # fan triangulation (corner indices into the polygon-vertex arrays)
            counts = ends - starts + 1
            ntri = np.maximum(counts - 2, 0)
            poly_of_tri = np.repeat(np.arange(len(starts)), ntri)
            k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri)
            c0 = starts[poly_of_tri]
            corners = np.stack([c0, c0 + 1 + k, c0 + 2 + k], 1)
            fv = pv[corners]
            n_pv = len(pv)
            if geo["normals"] is not None:
                Rn = np.linalg.inv(R).T
                vn = geo["normals"] @ Rn.T
                vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
                fn = corners
            else:
                vn, fn = np.zeros((0, 3)), np.full_like(corners, -1)
            uv = geo["uvs"] if geo["uvs"] is not None else np.zeros((0, 2))
            fuv = corners if geo["uvs"] is not None else np.full_like(corners, -1)
            fmat_poly = geo["materials"]
            fmat = fmat_poly[poly_of_tri]
            # material slots in first-use order (Unity sub-mesh order)
            _, first = np.unique(fmat_poly, return_index=True)
            used_order = [int(x) for x in np.unique(fmat_poly)[np.argsort(first)]]
            r = assign.get(mid)
            mats = resolve_materials(assets, scene, mid, used_order,
                                     r["materials"] if r else None, max_tex, log)
            if not mats:
                mats = [Material(name=scene.name(mid))]
            fmat = np.clip(fmat, 0, len(mats) - 1)
            # bake per-material UV tiling/offset into a private uv copy per face corner
            if len(uv):
                uv_c = uv[np.clip(fuv, 0, n_pv - 1)].copy()
                for mi, mat in enumerate(mats):
                    (sx, sy), (ox, oy) = getattr(mat, "uv_transform", ((1, 1), (0, 0)))
                    if (sx, sy, ox, oy) != (1, 1, 0, 0):
                        sel = fmat == mi
                        uv_c[sel] = uv_c[sel] * (sx, sy) + (ox, oy)
                uv = uv_c.reshape(-1, 2)
                fuv = np.arange(len(uv)).reshape(-1, 3)
            parts.append(compact(Mesh(pos, uv, vn, fv, fuv, fn, fmat, mats)))
            log(f"    {scene.name(mid)}: {len(fv)} tris, materials: "
                + ", ".join(f"{m.name}" for m in mats))
    if not parts:
        raise ValueError(f"no geometry for {spec}")
    return compact(concat_meshes(parts))


# =============================================================================
# OBJ / MTL IO
# =============================================================================
def _material_ids(mesh: Mesh, prefix: str) -> list[str]:
    """Unique slugged MTL names (prefixed with the model name)."""
    out, seen = [], set()
    for m in mesh.materials:
        base = f"{prefix}_{slug(m.name)}"
        nm, k = base, 2
        while nm in seen:
            nm, k = f"{base}_{k}", k + 1
        seen.add(nm)
        out.append(nm)
    return out


def write_textures(mesh: Mesh, out_dir: str, prefix: str, jpg_quality: int = 90) -> list[str]:
    """Write one image per distinct texture; sets Material.texture_file."""
    import cv2
    written, by_key = [], {}
    for m in mesh.materials:
        if m.texture is None:
            continue
        key = m.texture_key or id(m.texture)
        if key in by_key:
            m.texture_file = by_key[key]
            continue
        use_png = m.transparent and m.texture.ndim == 3 and m.texture.shape[2] == 4
        base = f"{prefix}_{slug(m.texture_label or m.name)}"
        name, k = base + (".png" if use_png else ".jpg"), 2
        while name in written:
            name, k = f"{base}_{k}" + (".png" if use_png else ".jpg"), k + 1
        path = os.path.join(out_dir, name)
        if use_png:
            cv2.imwrite(path, m.texture, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(path, m.texture[..., :3], [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        m.texture_file = name
        by_key[key] = name
        written.append(name)
    return written


def write_mtl(path: str, mesh: Mesh, names: list[str], header: str = "") -> None:
    lines = [f"# {header}" if header else "# generated by tools/convert_environment.py"]
    for nm, m in zip(names, mesh.materials):
        kd = tuple(float(np.clip(c, 0, 1)) for c in m.tint)
        lines += ["", f"newmtl {nm}",
                  f"Ka {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}",
                  f"Kd {kd[0]:.4f} {kd[1]:.4f} {kd[2]:.4f}",
                  "Ks 0.1500 0.1500 0.1500",
                  "Ns 24.0"]
        if m.emissive is not None:
            e = m.emissive
            lines.append(f"Ke {e[0]:.4f} {e[1]:.4f} {e[2]:.4f}")
        lines.append(f"d {m.alpha:.4f}" if m.transparent else "d 1.0000")
        lines.append("illum 2")
        if m.texture_file:
            lines.append(f"map_Kd {m.texture_file}")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def write_obj(mesh: Mesh, out_dir: str, base: str, header: str = "") -> dict:
    """Write <base>.obj + <base>.mtl (+ textures) into out_dir; returns stats."""
    os.makedirs(out_dir, exist_ok=True)
    names = _material_ids(mesh, base)
    tex_files = write_textures(mesh, out_dir, base)
    write_mtl(os.path.join(out_dir, base + ".mtl"), mesh, names, header)
    obj_path = os.path.join(out_dir, base + ".obj")
    with open(obj_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# {header}\n" if header else "")
        f.write(f"# {len(mesh.v)} vertices, {mesh.n_tris} triangles, {len(mesh.materials)} materials\n")
        f.write(f"mtllib {base}.mtl\n")
        f.write("".join(f"v {x:.5f} {y:.5f} {z:.5f}\n" for x, y, z in mesh.v))
        if len(mesh.uv):
            f.write("".join(f"vt {u:.5f} {w:.5f}\n" for u, w in mesh.uv))
        if len(mesh.vn):
            f.write("".join(f"vn {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in mesh.vn))
        order = np.argsort(mesh.fmat, kind="stable")
        cur = None
        buf = []
        has_uv, has_n = len(mesh.uv) > 0, len(mesh.vn) > 0
        for fi in order:
            mi = int(mesh.fmat[fi])
            if mi != cur:
                buf.append(f"g {names[mi]}\nusemtl {names[mi]}\n")
                cur = mi
            corners = []
            for k in range(3):
                vi = mesh.fv[fi, k] + 1
                ti = mesh.fuv[fi, k] + 1 if has_uv and mesh.fuv[fi, k] >= 0 else 0
                ni = mesh.fn[fi, k] + 1 if has_n and mesh.fn[fi, k] >= 0 else 0
                if ti and ni:
                    corners.append(f"{vi}/{ti}/{ni}")
                elif ni:
                    corners.append(f"{vi}//{ni}")
                elif ti:
                    corners.append(f"{vi}/{ti}")
                else:
                    corners.append(f"{vi}")
            buf.append("f " + " ".join(corners) + "\n")
        f.write("".join(buf))
    return {"obj": os.path.basename(obj_path), "mtl": base + ".mtl", "textures": tex_files,
            "materials": names}


def read_mtl(path: str) -> dict:
    """name -> {'Kd': (r,g,b), 'map_Kd': filename|None, 'd': alpha}."""
    mats, cur = {}, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if not parts or parts[0].startswith("#"):
                continue
            key, rest = parts[0], parts[1] if len(parts) > 1 else ""
            if key == "newmtl":
                cur = rest.strip()
                mats[cur] = {"Kd": (0.7, 0.7, 0.7), "map_Kd": None, "d": 1.0}
            elif cur is None:
                continue
            elif key == "Kd":
                mats[cur]["Kd"] = tuple(float(x) for x in rest.split()[:3])
            elif key == "map_Kd":
                mats[cur]["map_Kd"] = rest.split()[-1]
            elif key == "d":
                mats[cur]["d"] = float(rest.split()[0])
    return mats


def read_obj(path: str, load_textures: bool = False, max_tex: int = 512) -> Mesh:
    """Minimal OBJ reader (v/vt/vn/f with usemtl; polygons fan-triangulated).
    Materials carry tint and, optionally, the texture image."""
    vs, vts, vns = [], [], []
    fv, fuv, fn, fmat = [], [], [], []
    mat_index: dict[str, int] = {}
    mtl = {}
    cur = 0
    base_dir = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            tag, _, rest = line.partition(" ")
            if tag == "v":
                vs.append(rest.split()[:3])
            elif tag == "vt":
                vts.append(rest.split()[:2])
            elif tag == "vn":
                vns.append(rest.split()[:3])
            elif tag == "f":
                toks = rest.split()
                idx = []
                for tk in toks:
                    a = tk.split("/")
                    vi = int(a[0])
                    ti = int(a[1]) if len(a) > 1 and a[1] else 0
                    ni = int(a[2]) if len(a) > 2 and a[2] else 0
                    vi = vi - 1 if vi > 0 else len(vs) + vi
                    ti = ti - 1 if ti > 0 else (len(vts) + ti if ti < 0 else -1)
                    ni = ni - 1 if ni > 0 else (len(vns) + ni if ni < 0 else -1)
                    idx.append((vi, ti, ni))
                for k in range(1, len(idx) - 1):
                    tri = (idx[0], idx[k], idx[k + 1])
                    fv.append([c[0] for c in tri])
                    fuv.append([c[1] for c in tri])
                    fn.append([c[2] for c in tri])
                    fmat.append(cur)
            elif tag == "usemtl":
                nm = rest.strip()
                if nm not in mat_index:
                    mat_index[nm] = len(mat_index)
                cur = mat_index[nm]
            elif tag == "mtllib":
                p = os.path.join(base_dir, rest.strip())
                if os.path.isfile(p):
                    mtl.update(read_mtl(p))
    if not mat_index:
        mat_index["default"] = 0
    mats = []
    for nm, _ in sorted(mat_index.items(), key=lambda kv: kv[1]):
        info = mtl.get(nm, {"Kd": (0.7, 0.7, 0.7), "map_Kd": None, "d": 1.0})
        m = Material(name=nm, tint=info["Kd"], alpha=info["d"], texture_file=info["map_Kd"])
        if load_textures and info["map_Kd"]:
            img = imread_any(os.path.join(base_dir, info["map_Kd"]))
            if img is not None:
                m.texture = downscale(img[..., :3] if img.shape[2] == 4 else img, max_tex)
        mats.append(m)
    return Mesh(np.asarray(vs, float).reshape(-1, 3), np.asarray(vts, float).reshape(-1, 2),
                np.asarray(vns, float).reshape(-1, 3), np.asarray(fv, np.int64).reshape(-1, 3),
                np.asarray(fuv, np.int64).reshape(-1, 3), np.asarray(fn, np.int64).reshape(-1, 3),
                np.asarray(fmat, np.int64), mats)


# =============================================================================
# model.config / model.sdf
# =============================================================================
def model_config_xml(name: str, description: str) -> str:
    return f"""<?xml version="1.0"?>
<model>
  <name>{name}</name>
  <version>1.0</version>
  <sdf version="{SDF_VERSION}">model.sdf</sdf>
  <author>
    <name>DusanAI</name>
  </author>
  <description>{description}</description>
</model>
"""


def thermal_plugin_xml(temp_k: float, indent: str = "        ") -> str:
    return (f'{indent}<plugin filename="gz-sim-thermal-system" name="gz::sim::systems::Thermal">\n'
            f"{indent}  <temperature>{temp_k:.2f}</temperature>\n"
            f"{indent}</plugin>\n")


def mesh_model_sdf(name: str, collision_box: tuple | None = None, thermal_k: float | None = None) -> str:
    """Static model with the OBJ visual (+ optional box collision given as
    (cx, cy, cz, sx, sy, sz))."""
    col = ""
    if collision_box is not None:
        cx, cy, cz, sx, sy, sz = collision_box
        col = f"""      <collision name="collision">
        <pose>{fmt_vec((cx, cy, cz))} 0 0 0</pose>
        <geometry>
          <box>
            <size>{fmt_vec((sx, sy, sz))}</size>
          </box>
        </geometry>
      </collision>
"""
    th = thermal_plugin_xml(thermal_k) if thermal_k else ""
    return f"""<?xml version="1.0"?>
<sdf version="{SDF_VERSION}">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <mesh>
            <uri>model://{name}/meshes/{name}.obj</uri>
          </mesh>
        </geometry>
{th}      </visual>
{col}    </link>
  </model>
</sdf>
"""


# =============================================================================
# Software rasteriser (verification renders)
# =============================================================================
def view_basis(azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(right, up, toward_viewer) unit vectors of an orthographic camera looking
    at the scene from azimuth (deg, from +X towards +Y) and elevation (deg)."""
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    back = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    world_up = np.array([0.0, 0.0, 1.0]) if abs(el) < math.radians(89.0) else \
        np.array([-math.cos(az), -math.sin(az), 0.0])
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    return right, up, back


def render_mesh(mesh: Mesh, right, up, back, width_px: int = 900, face_mask=None,
                bounds=None, light=None, bg=(32, 32, 32), height_px: int | None = None
                ) -> tuple[np.ndarray, tuple]:
    """Orthographic z-buffer render with per-pixel texture sampling and
    Lambert shading; the view fits into width_px x height_px (height optional).
    Returns (BGR image, (u0, u1, v0, v1) view bounds)."""
    right, up, back = (np.asarray(a, float) for a in (right, up, back))
    sel = np.arange(mesh.n_tris) if face_mask is None else np.nonzero(face_mask)[0]
    tri = mesh.v[mesh.fv[sel]]
    P = np.stack([tri @ right, tri @ up, tri @ back], -1)          # (T,3,3)
    if bounds is None:
        u0, v0 = P[..., 0].min(), P[..., 1].min()
        u1, v1 = P[..., 0].max(), P[..., 1].max()
        pad = 0.03 * max(u1 - u0, v1 - v0, 1e-3)
        bounds = (u0 - pad, u1 + pad, v0 - pad, v1 + pad)
    u0, u1, v0, v1 = bounds
    res = (u1 - u0) / width_px
    if height_px is not None:
        res = max(res, (v1 - v0) / height_px)
    W = max(1, int(math.ceil((u1 - u0) / res)))
    H = max(1, int(math.ceil((v1 - v0) / res)))
    zbuf = np.full((H, W), -np.inf)
    img = np.empty((H, W, 3), np.float32)
    img[:] = np.asarray(bg, np.float32)
    L = np.asarray(light if light is not None else (0.35 * right + 0.55 * up + 0.75 * back), float)
    L /= np.linalg.norm(L)
    t3 = mesh.v[mesh.fv[sel]]
    fnrm = np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0])
    fnrm /= np.maximum(np.linalg.norm(fnrm, axis=1, keepdims=True), 1e-12)
    shade = 0.30 + 0.70 * np.abs(fnrm @ L)
    has_uv = len(mesh.uv) > 0
    texs = [m.texture for m in mesh.materials]
    flat = [np.asarray(m.mean_colour[::-1] * 255.0, np.float32) for m in mesh.materials]
    tints = [np.asarray(m.tint[::-1], np.float32) for m in mesh.materials]
    us = (P[..., 0] - u0) / res - 0.5
    vs = (v1 - P[..., 1]) / res - 0.5
    for k in range(len(sel)):
        fi = sel[k]
        tu, tv, tz = us[k], vs[k], P[k, :, 2]
        i0 = max(0, int(math.floor(tu.min()))); i1 = min(W - 1, int(math.ceil(tu.max())))
        j0 = max(0, int(math.floor(tv.min()))); j1 = min(H - 1, int(math.ceil(tv.max())))
        if i1 < i0 or j1 < j0:
            continue
        d = (tv[1] - tv[2]) * (tu[0] - tu[2]) + (tu[2] - tu[1]) * (tv[0] - tv[2])
        if abs(d) < 1e-12:
            continue
        I, J = np.meshgrid(np.arange(i0, i1 + 1, dtype=np.float64), np.arange(j0, j1 + 1, dtype=np.float64))
        l1 = ((tv[1] - tv[2]) * (I - tu[2]) + (tu[2] - tu[1]) * (J - tv[2])) / d
        l2 = ((tv[2] - tv[0]) * (I - tu[2]) + (tu[0] - tu[2]) * (J - tv[2])) / d
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        if not inside.any():
            continue
        z = l1 * tz[0] + l2 * tz[1] + l3 * tz[2]
        zb = zbuf[j0:j1 + 1, i0:i1 + 1]
        ok = inside & (z > zb)
        if not ok.any():
            continue
        zb[ok] = z[ok]
        mi = int(mesh.fmat[fi])
        tex = texs[mi] if mi < len(texs) else None
        if tex is not None and has_uv and mesh.fuv[fi, 0] >= 0:
            uvc = mesh.uv[mesh.fuv[fi]]
            uu = l1[ok] * uvc[0, 0] + l2[ok] * uvc[1, 0] + l3[ok] * uvc[2, 0]
            vv = l1[ok] * uvc[0, 1] + l2[ok] * uvc[1, 1] + l3[ok] * uvc[2, 1]
            th, tw = tex.shape[:2]
            px = np.mod(np.floor(uu * tw), tw).astype(np.int64)
            py = np.mod(np.floor((1.0 - vv) * th), th).astype(np.int64)   # OBJ v=0 = bottom row
            col = tex[py, px, :3].astype(np.float32) * tints[mi]
        else:
            col = flat[mi] if mi < len(flat) else np.float32([150, 150, 150])
        img[j0:j1 + 1, i0:i1 + 1][ok] = col * shade[k]
    return np.clip(img, 0, 255).astype(np.uint8), bounds


def _label(img, text, org=(8, 22), scale=0.55):
    import cv2
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _scale_bar(img, res_m_per_px: float):
    import cv2
    h, w = img.shape[:2]
    target = res_m_per_px * w * 0.2
    nice = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    L = max([n for n in nice if n <= target] or [nice[0]])
    px = int(L / res_m_per_px)
    y = h - 14
    cv2.line(img, (10, y), (10 + px, y), (255, 255, 255), 3)
    _label(img, f"{L:g} m", (14 + px, y + 5), 0.5)


# Preview views per model: (label, azimuth, elevation, region) where region =
# (x0, x1, y0, y1, z0, z1) in the MODEL frame or None.  Triangles whose bbox
# overlaps the region are drawn and the view is framed on the region, so the
# building shells can be cut open (floor levels, stairs, openings visible).
_INF = 1e9
DEFAULT_VIEWS = [
    ("top", -90.0, 90.0, None),
    ("front (looking +Y)", -90.0, 0.0, None),
    ("side (looking -X)", 0.0, 0.0, None),
    ("isometric", -55.0, 35.0, None),
]
PREVIEW_VIEWS = {
    "factory_hall": [
        ("top, roof removed (z < 3.2)", -90.0, 90.0, (-_INF, _INF, -_INF, _INF, -_INF, 3.2)),
        ("looking +Y, south wall removed", -90.0, 0.0, (-_INF, _INF, -8.6, _INF, -_INF, _INF)),
        ("looking -X, east wall removed", 0.0, 0.0, (-_INF, 14.0, -_INF, _INF, -_INF, _INF)),
        ("isometric from SE, z < 3.2, S/E walls removed", -50.0, 38.0,
         (-_INF, 14.5, -8.6, _INF, -_INF, 3.2)),
    ],
    "corridor": [
        ("top, ceiling removed (z < 2.9)", -90.0, 90.0, (-_INF, _INF, -_INF, _INF, -_INF, 2.9)),
        ("descent zone looking +Y, south wall removed", -90.0, 0.0, (39.5, 48.0, -1.75, _INF, -_INF, 2.0)),
        ("descent zone isometric (south wall removed)", -60.0, 35.0, (39.5, 48.0, -1.75, _INF, -_INF, 2.4)),
        ("west door end isometric (south wall removed)", -130.0, 30.0, (9.0, 15.5, -1.75, _INF, -_INF, 2.9)),
    ],
}


def preview_sheet(mesh: Mesh, title: str, out_png: str, views=None, width_px: int = 760) -> str:
    """2x2 sheet of orthographic renders (default: top / front / side / iso)."""
    import cv2
    views = views or DEFAULT_VIEWS
    tri = mesh.triangles()
    tmin, tmax = tri.min(1), tri.max(1)
    tiles = []
    for label, az, el, region in views:
        r, u, b = view_basis(az, el)
        mask, bounds = None, None
        if region is not None:
            lo = np.array(region[0::2], float)
            hi = np.array(region[1::2], float)
            mask = np.all((tmax >= lo) & (tmin <= hi), axis=1)
            glo, ghi = tmin[mask].min(0), tmax[mask].max(0)
            lo, hi = np.maximum(lo, glo), np.minimum(hi, ghi)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                                for z in (lo[2], hi[2])])
            pu, pv = corners @ r, corners @ u
            pad = 0.03 * max(np.ptp(pu), np.ptp(pv), 1e-3)
            bounds = (pu.min() - pad, pu.max() + pad, pv.min() - pad, pv.max() + pad)
        img, bnd = render_mesh(mesh, r, u, b, width_px=width_px, face_mask=mask, bounds=bounds,
                               height_px=int(width_px * 0.8))
        _label(img, f"{title} - {label}")
        _scale_bar(img, (bnd[1] - bnd[0]) / img.shape[1])
        tiles.append(img)
    while len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 32))
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1], cv2.BORDER_CONSTANT,
                                value=(32, 32, 32)) for t in tiles]
    rows = [np.hstack(tiles[k:k + 2]) for k in range(0, len(tiles), 2)]
    sheet = np.vstack(rows)
    if sheet.shape[1] > 1600:
        f = 1600 / sheet.shape[1]
        sheet = cv2.resize(sheet, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    cv2.imwrite(out_png, sheet)
    return out_png


# =============================================================================
# Conversion driver
# =============================================================================
def recentre_transform(mesh: Mesh) -> np.ndarray:
    """Translation putting the bbox centre at x = y = 0 and z_min at 0."""
    lo, hi = mesh.bbox()
    T = np.eye(4)
    T[:3, 3] = (-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2])
    return T


def _r4(x) -> float:
    """Round to 0.1 mm for the manifest (no '-0.0')."""
    return round(float(x), 4) + 0.0


def dir_size(path: str) -> int:
    total = 0
    for root, _d, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total


def convert_model(name: str, spec: dict, out_root: str = MODELS_DIR, max_tex: int = 1024,
                  max_tris: int | None = None, preview: bool = True, log=print) -> dict:
    """Convert one MODEL_SPECS entry; returns a manifest record."""
    t0 = time.time()
    log(f"== {name}  ({spec['package']}:{spec['fbx']})")
    mesh = mesh_from_fbx(spec, max_tex=max_tex, log=log)
    raw_tris = mesh.n_tris
    lo, hi = mesh.bbox()
    offset = np.zeros(3)
    if spec.get("recentre"):
        T = recentre_transform(mesh)
        offset = T[:3, 3].copy()
        mesh = mesh.transformed(T)
    budget = max_tris if max_tris is not None else spec.get("max_tris")
    if budget:
        mesh = decimate_to(mesh, int(budget), log=log)
    if spec.get("double_sided") == "all":
        mask = np.array([not m.transparent for m in mesh.materials], bool)
    else:
        mask = np.array([m.double_sided for m in mesh.materials], bool)
    single_tris = mesh.n_tris
    mesh = make_double_sided(mesh, mask)

    model_dir = os.path.join(out_root, name)
    mesh_dir = os.path.join(model_dir, "meshes")
    if os.path.isdir(mesh_dir):                       # stale textures from earlier runs
        for f in os.listdir(mesh_dir):
            if f.startswith(name) and os.path.splitext(f)[1].lower() in (".obj", ".mtl", ".jpg", ".png"):
                os.remove(os.path.join(mesh_dir, f))
    header = f"{name}: converted from {spec['package']}.unitypackage/{spec['fbx']} by tools/convert_environment.py"
    files = write_obj(mesh, mesh_dir, name, header)
    lo2, hi2 = mesh.bbox()
    collision_box = None
    if spec.get("collision") == "bbox":
        c, s = (lo2 + hi2) / 2.0, hi2 - lo2
        collision_box = (c[0], c[1], c[2], s[0], s[1], s[2])
    with open(os.path.join(model_dir, "model.config"), "w", encoding="utf-8", newline="\n") as f:
        f.write(model_config_xml(name, spec["description"]))
    with open(os.path.join(model_dir, "model.sdf"), "w", encoding="utf-8", newline="\n") as f:
        f.write(mesh_model_sdf(name, collision_box, spec.get("thermal_k")))
    record = {
        "source": f"{spec['package']}.unitypackage / {spec['fbx']}",
        "fbx_models": spec.get("models") or "all",
        "triangles_source": int(raw_tris),
        "triangles_single_sided": int(single_tris),
        "triangles_obj": int(mesh.n_tris),
        "vertices": int(len(mesh.v)),
        "bbox_native_min": [_r4(x) for x in lo],
        "bbox_native_max": [_r4(x) for x in hi],
        "recentre_offset": [_r4(x) for x in offset],
        "bbox_model_min": [_r4(x) for x in lo2],
        "bbox_model_max": [_r4(x) for x in hi2],
        "size_bytes": int(dir_size(model_dir)),
        "materials": [{"name": m.name, "mtl": nm, "texture": m.texture_file,
                       "tint": [_r4(c) for c in m.tint],
                       "double_sided": bool(mask[i]), "source": m.source}
                      for i, (m, nm) in enumerate(zip(mesh.materials, files["materials"]))],
    }
    if collision_box is not None:
        record["collision_box"] = [_r4(x) for x in collision_box]
    if preview:
        back = read_obj(os.path.join(mesh_dir, name + ".obj"), load_textures=True, max_tex=512)
        png = preview_sheet(back, name, os.path.join(IMAGES_DIR, f"world_model_{name}.png"),
                            views=PREVIEW_VIEWS.get(name))
        record["preview"] = os.path.relpath(png, REPO).replace("\\", "/")
    log(f"    -> {mesh.n_tris} tris ({single_tris} single-sided), {len(files['textures'])} textures, "
        f"{record['size_bytes'] / 1e6:.2f} MB, bbox {np.round(lo2, 3)} .. {np.round(hi2, 3)} "
        f"[{time.time() - t0:.1f} s]")
    return record


def write_manifest(records: dict, path: str) -> None:
    import yaml
    old = {}
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            old = yaml.safe_load(f) or {}
    models = dict(old.get("models", {}))
    models.update(records)
    data = {
        "generated_by": "tools/convert_environment.py",
        "note": "Converted Unity environment models (visual meshes). bbox_native = G frame of the "
                "FBX (metres, Z up); bbox_model = model frame after recentring.",
        "total_size_bytes": int(sum(m.get("size_bytes", 0) for m in models.values())),
        "models": models,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(data, f, sort_keys=False, width=120)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODEL_SPECS), help="comma separated subset")
    ap.add_argument("--out", default=MODELS_DIR, help="models output directory")
    ap.add_argument("--max-tex", type=int, default=1024, help="max texture size in px")
    ap.add_argument("--max-tris", type=int, default=None,
                    help="decimate every selected model above this many (single sided) triangles")
    ap.add_argument("--no-preview", action="store_true", help="skip verification renders")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    unknown = [n for n in names if n not in MODEL_SPECS]
    if unknown:
        ap.error(f"unknown models {unknown}; choose from {list(MODEL_SPECS)}")
    records = {}
    for n in names:
        records[n] = convert_model(n, MODEL_SPECS[n], args.out, args.max_tex, args.max_tris,
                                   preview=not args.no_preview)
    if os.path.abspath(args.out) == os.path.abspath(MODELS_DIR):
        write_manifest(records, os.path.join(GENERATED_DIR, "model_manifest.yaml"))
    total = sum(r["size_bytes"] for r in records.values())
    print(f"done: {len(records)} models, {total / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
