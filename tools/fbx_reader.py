#!/usr/bin/env python3
"""
Minimal, dependency-light binary FBX (7.x) reader.

Used to convert the Unity asset package meshes (Doosanenerbility/*.unitypackage)
into Gazebo friendly OBJ/MTL files without needing Blender or the Autodesk SDK.

Supported:
  * binary FBX 7.0 ~ 7.7 (32/64-bit node headers, zlib compressed arrays)
  * Model hierarchy with full FBX transform chain
    (T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1)
  * Geometry (polygons, normals, UVs, per-polygon material index)
  * Materials / Textures / embedded Video content
  * Skin deformers (clusters with Indexes / Weights / TransformLink)
  * GlobalSettings axis system and unit scale

Only numpy is required.
"""
import struct
import zlib
import math
import numpy as np


# --------------------------------------------------------------------------
# Low level binary parsing
# --------------------------------------------------------------------------
class FBXNode:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name = name
        self.props = props
        self.children = children

    def find(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def find_all(self, name):
        return [c for c in self.children if c.name == name]

    def __repr__(self):
        short = []
        for p in self.props:
            if isinstance(p, np.ndarray):
                short.append(f"array[{p.dtype},{p.size}]")
            elif isinstance(p, (bytes, str)) and len(p) > 40:
                short.append(repr(p[:40]) + "...")
            else:
                short.append(repr(p))
        return f"FBXNode({self.name}, [{', '.join(short)}], {len(self.children)} children)"


_ARRAY_TYPES = {
    ord("f"): ("<f4", 4),
    ord("d"): ("<f8", 8),
    ord("l"): ("<i8", 8),
    ord("i"): ("<i4", 4),
    ord("b"): ("<u1", 1),
}


def _read_prop(buf, off):
    t = buf[off]
    off += 1
    if t == ord("Y"):
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if t == ord("C"):
        return bool(buf[off]), off + 1
    if t == ord("I"):
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if t == ord("F"):
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if t == ord("D"):
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if t == ord("L"):
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if t in _ARRAY_TYPES:
        length, encoding, comp_len = struct.unpack_from("<III", buf, off)
        off += 12
        dtype, size = _ARRAY_TYPES[t]
        raw = buf[off:off + comp_len]
        off += comp_len
        if encoding == 1:
            raw = zlib.decompress(raw)
        arr = np.frombuffer(raw, dtype=dtype, count=length).copy()
        return arr, off
    if t == ord("S") or t == ord("R"):
        length = struct.unpack_from("<I", buf, off)[0]
        off += 4
        data = bytes(buf[off:off + length])
        off += length
        if t == ord("S"):
            return data, off
        return data, off
    raise ValueError(f"Unknown FBX property type {chr(t)!r} at {off - 1}")


def _read_node(buf, off, version):
    if version >= 7500:
        end, nprops, plen = struct.unpack_from("<QQQ", buf, off)
        off += 24
        null_len = 25
    else:
        end, nprops, plen = struct.unpack_from("<III", buf, off)
        off += 12
        null_len = 13
    name_len = buf[off]
    off += 1
    if end == 0:
        return None, off + name_len
    name = bytes(buf[off:off + name_len]).decode("ascii", "replace")
    off += name_len
    props = []
    for _ in range(nprops):
        p, off = _read_prop(buf, off)
        props.append(p)
    children = []
    while off < end:
        if end - off == null_len and not any(buf[off:end]):
            off = end
            break
        child, off = _read_node(buf, off, version)
        if child is None:
            break
        children.append(child)
    return FBXNode(name, props, children), end


def parse_fbx(path):
    with open(path, "rb") as f:
        buf = memoryview(f.read())
    magic = bytes(buf[:21])
    if not magic.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path}: not a binary FBX file")
    version = struct.unpack_from("<I", buf, 23)[0]
    off = 27
    nodes = []
    while off < len(buf):
        node, off = _read_node(buf, off, version)
        if node is None:
            break
        nodes.append(node)
    return version, FBXNode("ROOT", [], nodes)


# --------------------------------------------------------------------------
# Math helpers
# --------------------------------------------------------------------------
def _rot_axis(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    m = np.eye(4)
    if axis == 0:
        m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    elif axis == 1:
        m[0, 0], m[0, 2], m[2, 0], m[2, 2] = c, s, -s, c
    else:
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    return m


# FBX rotation order enum -> order the axes are applied in
_ROT_ORDERS = {
    0: (0, 1, 2),  # XYZ
    1: (0, 2, 1),  # XZY
    2: (1, 2, 0),  # YZX
    3: (1, 0, 2),  # YXZ
    4: (2, 0, 1),  # ZXY
    5: (2, 1, 0),  # ZYX
}


def euler_to_mat(rx, ry, rz, order=0):
    angles = (rx, ry, rz)
    m = np.eye(4)
    for ax in _ROT_ORDERS.get(order, (0, 1, 2)):
        m = _rot_axis(ax, angles[ax]) @ m
    return m


def translate(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


def scale(v):
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = v
    return m


# --------------------------------------------------------------------------
# Scene level helpers
# --------------------------------------------------------------------------
def _p70(node):
    """Return dict of Properties70 entries -> list of values."""
    out = {}
    if node is None:
        return out
    p = node.find("Properties70")
    if p is None:
        return out
    for e in p.children:
        if e.name != "P" or not e.props:
            continue
        key = e.props[0].decode("utf-8", "replace") if isinstance(e.props[0], bytes) else e.props[0]
        out[key] = e.props[4:]
    return out


def _clean_name(raw):
    if isinstance(raw, bytes):
        raw = raw.split(b"\x00\x01")[0]
        return raw.decode("utf-8", "replace")
    return str(raw)


class FBXScene:
    def __init__(self, path):
        self.path = path
        self.version, self.root = parse_fbx(path)
        self.objects = {}      # id -> node
        self.kind = {}         # id -> node.name (Model/Geometry/Material/...)
        self.parents = {}      # child id -> list of (parent id, prop)
        self.children = {}     # parent id -> list of (child id, prop)
        self._index()
        self._global_settings()
        self._world_cache = {}

    # ---------------------------------------------------------------
    def _index(self):
        objs = self.root.find("Objects")
        for n in objs.children if objs else []:
            if not n.props:
                continue
            oid = n.props[0]
            self.objects[oid] = n
            self.kind[oid] = n.name
        conns = self.root.find("Connections")
        for c in conns.children if conns else []:
            if c.name != "C":
                continue
            ctype = c.props[0]
            child, parent = c.props[1], c.props[2]
            prop = c.props[3].decode() if len(c.props) > 3 and isinstance(c.props[3], bytes) else None
            self.parents.setdefault(child, []).append((parent, prop, ctype))
            self.children.setdefault(parent, []).append((child, prop, ctype))

    def _global_settings(self):
        gs = _p70(self.root.find("GlobalSettings"))
        self.unit_scale = float(gs.get("UnitScaleFactor", [1.0])[0])
        self.up_axis = int(gs.get("UpAxis", [1])[0])
        self.up_sign = int(gs.get("UpAxisSign", [1])[0])
        self.front_axis = int(gs.get("FrontAxis", [2])[0])
        self.front_sign = int(gs.get("FrontAxisSign", [1])[0])
        self.coord_axis = int(gs.get("CoordAxis", [0])[0])
        self.coord_sign = int(gs.get("CoordAxisSign", [1])[0])
        self.original_up_axis = int(gs.get("OriginalUpAxis", [-1])[0])

    # ---------------------------------------------------------------
    def name(self, oid):
        n = self.objects.get(oid)
        return _clean_name(n.props[1]) if n is not None and len(n.props) > 1 else str(oid)

    def subtype(self, oid):
        n = self.objects.get(oid)
        if n is None or len(n.props) < 3:
            return ""
        return n.props[2].decode() if isinstance(n.props[2], bytes) else str(n.props[2])

    def ids_of(self, kind):
        return [i for i, k in self.kind.items() if k == kind]

    def parent_model(self, oid):
        for p, _prop, _t in self.parents.get(oid, []):
            if self.kind.get(p) == "Model":
                return p
        return None

    def child_ids(self, oid, kind=None):
        return [c for c, _p, _t in self.children.get(oid, []) if kind is None or self.kind.get(c) == kind]

    def parent_ids(self, oid, kind=None):
        return [p for p, _pr, _t in self.parents.get(oid, []) if kind is None or self.kind.get(p) == kind]

    # ---------------------------------------------------------------
    def local_matrix(self, mid):
        p = _p70(self.objects[mid])

        def v3(key, default):
            if key in p:
                vals = p[key]
                return np.array([float(vals[0]), float(vals[1]), float(vals[2])])
            return np.array(default, dtype=float)

        order = int(p.get("RotationOrder", [0])[0]) if "RotationOrder" in p else 0
        T = translate(v3("Lcl Translation", (0, 0, 0)))
        R = euler_to_mat(*v3("Lcl Rotation", (0, 0, 0)), order=order)
        S = scale(v3("Lcl Scaling", (1, 1, 1)))
        Roff = translate(v3("RotationOffset", (0, 0, 0)))
        Rp = translate(v3("RotationPivot", (0, 0, 0)))
        Soff = translate(v3("ScalingOffset", (0, 0, 0)))
        Sp = translate(v3("ScalingPivot", (0, 0, 0)))
        Rpre = euler_to_mat(*v3("PreRotation", (0, 0, 0)))
        Rpost = euler_to_mat(*v3("PostRotation", (0, 0, 0)))
        return (T @ Roff @ Rp @ Rpre @ R @ np.linalg.inv(Rpost) @ np.linalg.inv(Rp)
                @ Soff @ Sp @ S @ np.linalg.inv(Sp))

    def geometric_matrix(self, mid):
        p = _p70(self.objects[mid])

        def v3(key, default):
            if key in p:
                vals = p[key]
                return np.array([float(vals[0]), float(vals[1]), float(vals[2])])
            return np.array(default, dtype=float)

        return (translate(v3("GeometricTranslation", (0, 0, 0)))
                @ euler_to_mat(*v3("GeometricRotation", (0, 0, 0)))
                @ scale(v3("GeometricScaling", (1, 1, 1))))

    def world_matrix(self, mid):
        if mid in self._world_cache:
            return self._world_cache[mid]
        local = self.local_matrix(mid)
        parent = self.parent_model(mid)
        w = local if parent is None else self.world_matrix(parent) @ local
        self._world_cache[mid] = w
        return w

    def axis_matrix(self):
        """Matrix converting FBX file axes (in metres) to Gazebo Z-up / X-forward."""
        m = np.zeros((4, 4))
        m[3, 3] = 1.0
        # file coord axis -> +X, file up axis -> +Z, third by right-hand rule
        ex = np.zeros(3)
        ex[self.coord_axis] = self.coord_sign
        ez = np.zeros(3)
        ez[self.up_axis] = self.up_sign
        ey = np.cross(ez, ex)
        m[0, :3] = ex
        m[1, :3] = ey
        m[2, :3] = ez
        s = self.unit_scale / 100.0
        return scale((s, s, s)) @ m

    # ---------------------------------------------------------------
    def geometry(self, gid):
        """Return dict with vertices (N,3), polygons (list of index lists),
        normals per polygon-vertex (or None), uvs per polygon-vertex (or None),
        material per polygon (array)."""
        g = self.objects[gid]
        verts_node = g.find("Vertices")
        pvi_node = g.find("PolygonVertexIndex")
        if verts_node is None or pvi_node is None:
            return None
        verts = np.asarray(verts_node.props[0], dtype=np.float64).reshape(-1, 3)
        pvi = np.asarray(pvi_node.props[0], dtype=np.int64)
        ends = np.nonzero(pvi < 0)[0]
        poly_vertex = pvi.copy()
        poly_vertex[ends] = -poly_vertex[ends] - 1
        starts = np.concatenate([[0], ends[:-1] + 1])

        def layer(name, data_key, index_key):
            ln = g.find(name)
            if ln is None:
                return None
            mapping = ln.find("MappingInformationType")
            ref = ln.find("ReferenceInformationType")
            data = ln.find(data_key)
            if data is None:
                return None
            mapping = mapping.props[0].decode() if mapping is not None else "ByPolygonVertex"
            ref = ref.props[0].decode() if ref is not None else "Direct"
            arr = np.asarray(data.props[0], dtype=np.float64)
            idx = ln.find(index_key)
            idx = np.asarray(idx.props[0], dtype=np.int64) if idx is not None else None
            return mapping, ref, arr, idx

        def per_polyvertex(lay, width):
            if lay is None:
                return None
            mapping, ref, arr, idx = lay
            arr = arr.reshape(-1, width)
            if mapping in ("ByPolygonVertex",):
                sel = np.arange(len(pvi))
            elif mapping in ("ByVertice", "ByVertex"):
                sel = poly_vertex
            elif mapping == "ByPolygon":
                sel = np.repeat(np.arange(len(ends)), ends - starts + 1)
            elif mapping == "AllSame":
                sel = np.zeros(len(pvi), dtype=np.int64)
            else:
                return None
            if ref == "IndexToDirect" and idx is not None:
                sel = idx[sel] if mapping != "ByPolygonVertex" else idx
            sel = np.clip(sel, 0, len(arr) - 1)
            return arr[sel]

        normals = per_polyvertex(layer("LayerElementNormal", "Normals", "NormalsIndex"), 3)
        uvs = per_polyvertex(layer("LayerElementUV", "UV", "UVIndex"), 2)

        mats = np.zeros(len(ends), dtype=np.int64)
        lm = g.find("LayerElementMaterial")
        if lm is not None and lm.find("Materials") is not None:
            marr = np.asarray(lm.find("Materials").props[0], dtype=np.int64)
            mapping = lm.find("MappingInformationType")
            mapping = mapping.props[0].decode() if mapping is not None else "AllSame"
            if mapping == "ByPolygon" and len(marr) >= len(ends):
                mats = marr[:len(ends)]
            elif len(marr) > 0:
                mats[:] = marr[0]
        return {
            "vertices": verts,
            "poly_vertex": poly_vertex,
            "starts": starts,
            "ends": ends,
            "normals": normals,
            "uvs": uvs,
            "materials": mats,
        }

    # ---------------------------------------------------------------
    def materials_of_model(self, mid):
        """Material ids in connection order (index used by LayerElementMaterial)."""
        return [c for c, _p, _t in self.children.get(mid, []) if self.kind.get(c) == "Material"]

    def material_info(self, matid):
        p = _p70(self.objects[matid])
        info = {"name": self.name(matid), "diffuse": (0.7, 0.7, 0.7), "texture": None}
        for key in ("DiffuseColor", "Diffuse"):
            if key in p and len(p[key]) >= 3:
                info["diffuse"] = tuple(float(x) for x in p[key][:3])
                break
        # texture connected to diffuse colour
        for c, prop, _t in self.children.get(matid, []):
            if self.kind.get(c) != "Texture":
                continue
            if prop is None or "Diffuse" in prop or "Base" in prop or "Albedo" in prop:
                info["texture"] = self.texture_filename(c)
                if prop and "Diffuse" in prop:
                    break
        return info

    def texture_filename(self, tid):
        t = self.objects[tid]
        for key in ("RelativeFilename", "FileName"):
            n = t.find(key)
            if n is not None and n.props and n.props[0]:
                return n.props[0].decode("utf-8", "replace")
        return None

    def skin_clusters(self, gid):
        """Return list of (bone model id, indexes, weights, transform_link) for a geometry."""
        out = []
        for skin in self.child_ids(gid, "Deformer"):
            if self.subtype(skin) != "Skin":
                continue
            for cl in self.child_ids(skin, "Deformer"):
                node = self.objects[cl]
                idx = node.find("Indexes")
                w = node.find("Weights")
                tl = node.find("TransformLink")
                bone = None
                for b in self.child_ids(cl, "Model"):
                    bone = b
                if bone is None:
                    continue
                out.append((
                    bone,
                    np.asarray(idx.props[0], dtype=np.int64) if idx is not None else np.zeros(0, np.int64),
                    np.asarray(w.props[0], dtype=np.float64) if w is not None else np.zeros(0),
                    np.asarray(tl.props[0], dtype=np.float64).reshape(4, 4).T if tl is not None else None,
                ))
        return out
