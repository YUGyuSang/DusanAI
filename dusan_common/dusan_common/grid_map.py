"""
2-D occupancy grid utilities (ROS map_server compatible YAML + PGM).

Storage convention (same as nav_msgs/OccupancyGrid):
    occ[iy, ix]   iy = 0 is the row at y = origin_y (bottom of the PGM image)
    values: 0 = free, 100 = occupied, -1 = unknown
"""
import math
import os
import heapq
import numpy as np
import yaml

try:  # OpenCV is optional, only used to speed up distance transforms
    import cv2  # noqa: F401
    _HAVE_CV2 = True
except Exception:  # pragma: no cover
    _HAVE_CV2 = False


def read_pgm(path, return_maxval=False):
    """Read binary (P5) or ascii (P2) PGM -> uint8/uint16 2-D array (row 0 = top).

    With ``return_maxval=True`` returns ``(array, maxval)`` (the header's
    maximum grey value, needed to normalise the pixels).  A truncated or
    malformed file raises ValueError."""
    with open(path, "rb") as f:
        data = f.read()
    n = len(data)
    tokens = []
    idx = 0
    # parse header tokens skipping comments
    while len(tokens) < 4:
        while idx < n and data[idx:idx + 1].isspace():
            idx += 1
        if idx >= n:
            raise ValueError(f"truncated PGM header in {path}")
        if data[idx:idx + 1] == b"#":
            while idx < n and data[idx:idx + 1] not in (b"\n", b"\r"):
                idx += 1
            continue
        start = idx
        while idx < n and not data[idx:idx + 1].isspace():
            idx += 1
        tokens.append(data[start:idx])
    magic, w, h, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    if magic not in (b"P5", b"P2"):
        raise ValueError(f"Unsupported PGM type {magic!r}")
    if not 0 < maxval < 65536 or w <= 0 or h <= 0:
        raise ValueError(f"invalid PGM header in {path}: {w}x{h}, maxval {maxval}")
    idx += 1  # single whitespace after maxval
    if magic == b"P5":
        dtype = np.uint8 if maxval < 256 else np.dtype(">u2")
        arr = np.frombuffer(data, dtype=dtype, count=w * h, offset=idx).reshape(h, w)
        arr = arr.astype(np.uint16 if maxval >= 256 else np.uint8)
    else:
        vals = np.array(data[idx:].split(), dtype=np.int64)[: w * h]
        arr = vals.reshape(h, w).astype(np.uint8 if maxval < 256 else np.uint16)
    return (arr, maxval) if return_maxval else arr


def write_pgm(path, img):
    img = np.asarray(img, dtype=np.uint8)
    h, w = img.shape
    with open(path, "wb") as f:
        f.write(f"P5\n# dusan plant map\n{w} {h}\n255\n".encode("ascii"))
        f.write(img.tobytes())


class OccupancyGridMap:
    def __init__(self, occ, resolution, origin_x, origin_y, frame_id="map"):
        self.occ = np.asarray(occ, dtype=np.int8)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.frame_id = frame_id
        self._inflated_cache = {}
        self._dist = None

    # ------------------------------------------------------------------ io
    @property
    def height(self):
        return self.occ.shape[0]

    @property
    def width(self):
        return self.occ.shape[1]

    @classmethod
    def load(cls, yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            meta = yaml.safe_load(f)
        img_path = meta["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
        img, maxval = read_pgm(img_path, return_maxval=True)
        img = img.astype(np.float64)
        maxval = float(maxval)                 # header maxval (map_server normalises by it)
        negate = int(meta.get("negate", 0))
        p = img / maxval if negate else (maxval - img) / maxval   # p = occupancy probability
        occ_t = float(meta.get("occupied_thresh", 0.65))
        free_t = float(meta.get("free_thresh", 0.25))
        occ = np.full(img.shape, -1, dtype=np.int8)
        occ[p > occ_t] = 100
        occ[p < free_t] = 0
        occ = np.flipud(occ)  # PGM row 0 = top  ->  ROS row 0 = bottom
        origin = meta.get("origin", [0.0, 0.0, 0.0])
        return cls(occ, meta["resolution"], origin[0], origin[1])

    def save(self, yaml_path, image_name=None):
        image_name = image_name or os.path.splitext(os.path.basename(yaml_path))[0] + ".pgm"
        img = np.full(self.occ.shape, 205, dtype=np.uint8)
        img[self.occ == 0] = 254
        img[self.occ == 100] = 0
        write_pgm(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image_name), np.flipud(img))
        meta = {
            "image": image_name,
            "mode": "trinary",
            "resolution": self.resolution,
            "origin": [self.origin_x, self.origin_y, 0.0],
            "negate": 0,
            "occupied_thresh": 0.65,
            # map_saver convention: unknown pixels (205 -> p = 50/255 = 0.196)
            # must not fall below free_thresh, or they reload as free
            "free_thresh": 0.196,
        }
        # LF on every OS (the generated map is committed; Windows text mode would write CRLF)
        with open(yaml_path, "w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump(meta, f, sort_keys=False)

    # ------------------------------------------------------------- frames
    def world_to_grid(self, x, y):
        """World metres -> (ix, iy).  A non-finite coordinate (NaN/inf pose)
        maps to (-1, -1), i.e. outside the map (``in_bounds`` is False)."""
        f = self._cell_coords(x, y)
        if f is None:
            return -1, -1
        return int(math.floor(f[0])), int(math.floor(f[1]))

    def _cell_coords(self, x, y):
        """Continuous cell coordinates of (x, y), or None when not finite."""
        fx = (x - self.origin_x) / self.resolution
        fy = (y - self.origin_y) / self.resolution
        if not (math.isfinite(fx) and math.isfinite(fy)):
            return None
        return fx, fy

    def world_to_grid_array(self, xy):
        """Vectorised world_to_grid.  Cells inside the map are exact; anything
        outside is clamped to just outside (-1 or width/height) and non-finite
        coordinates give -1, so the result never overflows and still fails
        every bounds check."""
        xy = np.asarray(xy, dtype=float)
        with np.errstate(invalid="ignore", over="ignore"):
            fx = np.floor((xy[..., 0] - self.origin_x) / self.resolution)
            fy = np.floor((xy[..., 1] - self.origin_y) / self.resolution)
            ix = np.where(np.isfinite(fx), np.clip(fx, -1, self.width), -1).astype(int)
            iy = np.where(np.isfinite(fy), np.clip(fy, -1, self.height), -1).astype(int)
        return ix, iy

    def grid_to_world(self, ix, iy):
        return (self.origin_x + (ix + 0.5) * self.resolution,
                self.origin_y + (iy + 0.5) * self.resolution)

    def in_bounds(self, ix, iy):
        return 0 <= ix < self.width and 0 <= iy < self.height

    # ---------------------------------------------------------- queries
    def obstacle_mask(self, unknown_is_obstacle=True):
        m = self.occ >= 50
        if unknown_is_obstacle:
            m |= self.occ < 0
        return m

    def distance_field(self):
        """Euclidean distance (m) from every cell to the nearest obstacle/unknown cell."""
        if self._dist is not None:
            return self._dist
        obst = self.obstacle_mask()
        if _HAVE_CV2:
            free = (~obst).astype(np.uint8)
            d = cv2.distanceTransform(free, cv2.DIST_L2, 5).astype(np.float64)
        else:  # brushfire BFS fallback (8-connected, approximate euclidean)
            d = np.full(obst.shape, np.inf)
            h, w = obst.shape
            pq = [(0.0, int(iy), int(ix)) for iy, ix in zip(*np.nonzero(obst))]
            for _, iy, ix in pq:
                d[iy, ix] = 0.0
            heapq.heapify(pq)
            nbrs = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414), (0, -1, 1.0),
                    (0, 1, 1.0), (1, -1, 1.414), (1, 0, 1.0), (1, 1, 1.414)]
            while pq:
                dist, iy, ix = heapq.heappop(pq)
                if dist > d[iy, ix]:
                    continue
                for dy, dx, c in nbrs:
                    ny, nx = iy + dy, ix + dx
                    if 0 <= ny < h and 0 <= nx < w and dist + c < d[ny, nx]:
                        d[ny, nx] = dist + c
                        heapq.heappush(pq, (dist + c, ny, nx))
        self._dist = d * self.resolution
        return self._dist

    def inflated_obstacles(self, radius):
        """Boolean mask of cells closer than `radius` metres to an obstacle."""
        key = round(radius, 3)
        if key not in self._inflated_cache:
            self._inflated_cache[key] = self.distance_field() < radius
        return self._inflated_cache[key]

    def free_labels(self, radius):
        """Labels of the 8-connected components of the cells that are free at
        ``radius`` (int32, 0 = blocked, cached per radius; the cells of
        ``inflated_obstacles(radius)`` are the blocked ones).  Two free cells
        with different labels have no 8-connected A* path between them, so a
        planner can reject an impossible request in ~1 ms instead of
        exhausting the whole free space (OpenCV when available, else BFS)."""
        blocked = self.inflated_obstacles(radius)          # exactly the cells astar() treats as blocked
        # kept next to the inflation masks and tied to the mask object: whoever
        # resets the inflation cache (after editing occ) resets the labels too
        key = ("labels", round(radius, 3))
        hit = self._inflated_cache.get(key)
        if hit is not None and hit[0] is blocked:
            return hit[1]
        lab = _label_components(~blocked)
        self._inflated_cache[key] = (blocked, lab)
        return lab

    def is_free(self, x, y, radius=0.0):
        ix, iy = self.world_to_grid(x, y)
        if not self.in_bounds(ix, iy):
            return False
        if radius > 0.0:
            return not self.inflated_obstacles(radius)[iy, ix]
        return self.occ[iy, ix] == 0

    def clearance(self, x, y):
        ix, iy = self.world_to_grid(x, y)
        if not self.in_bounds(ix, iy):
            return 0.0
        return float(self.distance_field()[iy, ix])

    def raycast_crossings(self, x0, y0, x1, y1, step=None):
        """Walk the segment and return (number of free->occupied transitions,
        fraction of samples inside obstacles).  Used for line-of-sight and
        radio wall attenuation.

        Only cells >= 50 count as obstacles: samples outside the map and
        unknown cells (-1) count as free (module3 comm_model.crossings_batch
        relies on these exact semantics).  A non-finite endpoint returns
        (0, 0.0); a segment reaching far outside the map is clipped to the
        map rectangle first (only the part inside the map can cross walls)."""
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return 0, 0.0
        step = step or self.resolution * 0.5
        dist = math.hypot(x1 - x0, y1 - y0)
        if dist > 4.0 * (self.width + self.height) * self.resolution:
            # longer than twice the map's perimeter (never the case for an
            # in-map segment): a bogus pose far away, sample only the clipped
            # part so the sample count stays bounded
            seg = self._clip_segment(x0, y0, x1, y1)
            if seg is None or not all(math.isfinite(v) for v in seg):
                return 0, 0.0
            x0, y0, x1, y1 = seg
            dist = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(dist / step) + 1)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        ix, iy = self.world_to_grid_array(np.stack([xs, ys], -1))
        valid = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        occ = np.zeros(n, dtype=bool)
        occ[valid] = self.occ[iy[valid], ix[valid]] >= 50
        crossings = int(np.count_nonzero(occ[1:] & ~occ[:-1]))
        return crossings, float(occ.mean())

    def _clip_segment(self, x0, y0, x1, y1):
        """Liang-Barsky clip of a segment to the map rectangle, or None."""
        xmin, ymin = self.origin_x, self.origin_y
        xmax = xmin + self.width * self.resolution
        ymax = ymin + self.height * self.resolution
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
            if p == 0.0:
                if q < 0.0:
                    return None
                continue
            r = q / p
            if p < 0.0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None
        return x0 + t0 * dx, y0 + t0 * dy, x0 + t1 * dx, y0 + t1 * dy

    def line_of_sight(self, x0, y0, x1, y1, radius=0.0):
        """True when every sample of the segment lies inside the map on a free
        (radius-inflated) cell.  Non-finite or off-map endpoints -> False."""
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return False
        # the endpoints are samples too: checking them first is equivalent and
        # bounds the sample count by the map size
        if not (self.in_bounds(*self.world_to_grid(x0, y0)) and self.in_bounds(*self.world_to_grid(x1, y1))):
            return False
        step = self.resolution * 0.5
        dist = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(dist / step) + 1)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)
        ix, iy = self.world_to_grid_array(np.stack([xs, ys], -1))
        if np.any((ix < 0) | (ix >= self.width) | (iy < 0) | (iy >= self.height)):
            return False
        mask = self.inflated_obstacles(radius) if radius > 0 else self.obstacle_mask()
        return not bool(np.any(mask[iy, ix]))

    def nearest_free(self, x, y, radius=0.0, max_search=3.0):
        """Closest free (inflated) cell centre to (x, y), or None (also for a
        non-finite position)."""
        if self._cell_coords(x, y) is None:   # NaN/inf pose: no search around the (-1, -1) sentinel
            return None
        mask = self.inflated_obstacles(radius) if radius > 0 else self.obstacle_mask()
        ix0, iy0 = self.world_to_grid(x, y)
        r = int(max_search / self.resolution)
        best = None
        for rr in range(0, r + 1):
            ys, xs = np.mgrid[iy0 - rr:iy0 + rr + 1, ix0 - rr:ix0 + rr + 1]
            ring = (np.abs(ys - iy0) == rr) | (np.abs(xs - ix0) == rr)
            ys, xs = ys[ring], xs[ring]
            ok = (xs >= 0) & (xs < self.width) & (ys >= 0) & (ys < self.height)
            ys, xs = ys[ok], xs[ok]
            free = ~mask[ys, xs]
            if np.any(free):
                d = (xs[free] - ix0) ** 2 + (ys[free] - iy0) ** 2
                k = int(np.argmin(d))
                best = self.grid_to_world(int(xs[free][k]), int(ys[free][k]))
                break
        return best

    # -------------------------------------------------------------- ROS
    def to_msg(self, stamp=None):
        from nav_msgs.msg import OccupancyGrid
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        if stamp is not None:
            msg.header.stamp = stamp
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = self.occ.flatten().astype(int).tolist()
        return msg

    @classmethod
    def from_msg(cls, msg):
        occ = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        return cls(occ, msg.info.resolution, msg.info.origin.position.x,
                   msg.info.origin.position.y, msg.header.frame_id or "map")


def _label_components(free):
    """8-connected component labels of a boolean mask (int32, 0 = not free)."""
    free = np.asarray(free, dtype=bool)
    if _HAVE_CV2:
        _, lab = cv2.connectedComponents(free.astype(np.uint8), connectivity=8, ltype=cv2.CV_32S)
        return lab.astype(np.int32, copy=False)
    h, w = free.shape
    lab = np.zeros(h * w, dtype=np.int32)
    flat = free.ravel()
    nxt = 0
    from collections import deque
    for s in np.flatnonzero(flat):
        if lab[s]:
            continue
        nxt += 1
        lab[s] = nxt
        q = deque([int(s)])
        while q:
            c = q.popleft()
            cy, cx = divmod(c, w)
            for dy in (-1, 0, 1):
                ny = cy + dy
                if ny < 0 or ny >= h:
                    continue
                for dx in (-1, 0, 1):
                    nx = cx + dx
                    if (dx or dy) and 0 <= nx < w:
                        n = ny * w + nx
                        if flat[n] and not lab[n]:
                            lab[n] = nxt
                            q.append(n)
    return lab.reshape(h, w)


def submap(grid, x0, y0, x1, y1):
    """Crop of ``grid`` covering the world rectangle (x0, y0) .. (x1, y1),
    clamped to the grid -> (OccupancyGridMap, (ix0, iy0)) with (ix0, iy0) the
    index of the crop's cell [0, 0] in ``grid``; (None, (0, 0)) when the
    rectangle misses the grid."""
    vals = [float(v) for v in (x0, y0, x1, y1)]
    if not all(math.isfinite(v) for v in vals):
        return None, (0, 0)
    xa, xb = min(vals[0], vals[2]), max(vals[0], vals[2])
    ya, yb = min(vals[1], vals[3]), max(vals[1], vals[3])
    ix0 = int(math.floor((xa - grid.origin_x) / grid.resolution))
    iy0 = int(math.floor((ya - grid.origin_y) / grid.resolution))
    ix1 = int(math.floor((xb - grid.origin_x) / grid.resolution))
    iy1 = int(math.floor((yb - grid.origin_y) / grid.resolution))
    ix0, iy0 = max(ix0, 0), max(iy0, 0)
    ix1, iy1 = min(ix1, grid.width - 1), min(iy1, grid.height - 1)
    if ix1 < ix0 or iy1 < iy0:
        return None, (0, 0)
    sub = OccupancyGridMap(grid.occ[iy0:iy1 + 1, ix0:ix1 + 1].copy(), grid.resolution,
                           grid.origin_x + ix0 * grid.resolution, grid.origin_y + iy0 * grid.resolution,
                           grid.frame_id)
    return sub, (ix0, iy0)


def scalar_field_to_occupancy_msg(field, resolution, origin_x, origin_y, vmin, vmax,
                                  frame_id="map", stamp=None, nan_value=-1):
    """Encode any 2-D float field (row 0 = y origin) as an OccupancyGrid 0..100
    (handy for RViz / dashboard visualisation of gas, RSSI, cost maps)."""
    from nav_msgs.msg import OccupancyGrid
    f = np.asarray(field, dtype=np.float64)
    scaled = np.clip((f - vmin) / max(1e-9, (vmax - vmin)) * 100.0, 0, 100)
    data = np.where(np.isfinite(f), np.round(scaled), nan_value).astype(np.int8)
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.info.resolution = float(resolution)
    msg.info.width = int(f.shape[1])
    msg.info.height = int(f.shape[0])
    msg.info.origin.position.x = float(origin_x)
    msg.info.origin.position.y = float(origin_y)
    msg.info.origin.orientation.w = 1.0
    msg.data = data.flatten().astype(int).tolist()
    return msg
