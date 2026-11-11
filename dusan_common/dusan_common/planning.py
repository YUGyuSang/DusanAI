"""
Global path planning on an OccupancyGridMap (8-connected A*).

  path = astar(grid, (x0, y0), (x1, y1), robot_radius=0.30)
  path -> list of (x, y) world points (start .. goal) or None

A soft clearance cost keeps paths away from walls; `extra_cost` (same shape
as grid.occ, float >= 0) lets callers penalise rough terrain etc.
(``dusan_common.terrain_cost`` rasterises the crossable debris of the layout).
``smooth_path`` / ``plan`` respect the same ``extra_cost``: a line-of-sight
shortcut is only taken when it does not cross costlier terrain than the A*
cells it replaces, so smoothing never cuts back across a pipe or rubble field
that A* went around.
"""
import heapq
import math
import numpy as np

_NBRS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
         (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]


def astar(grid, start, goal, robot_radius=0.30, clearance_weight=2.0, clearance_range=0.8,
          extra_cost=None, snap_radius=2.0, max_expansions=2_000_000):
    """A* search.  start/goal in world metres.  Returns list of (x, y) or None."""
    blocked = grid.inflated_obstacles(robot_radius)
    s = grid.nearest_free(start[0], start[1], robot_radius, snap_radius)
    g = grid.nearest_free(goal[0], goal[1], robot_radius, snap_radius)
    if s is None or g is None:
        return None
    sx, sy = grid.world_to_grid(*s)
    gx, gy = grid.world_to_grid(*g)
    h, w = blocked.shape
    if (sx, sy) != (gx, gy) and hasattr(grid, "free_labels"):
        # different 8-connected free components: no path - answer at once instead of
        # expanding the whole reachable free space (a blocked corridor on a large map
        # took 0.3-0.6 s per failing search)
        lab = grid.free_labels(robot_radius)
        if lab[sy, sx] != lab[gy, gx]:
            return None

    dist = grid.distance_field()
    # soft cost: 0 at >= clearance_range from obstacles, up to clearance_weight near them
    soft = clearance_weight * np.clip((clearance_range - dist) / max(clearance_range, 1e-6), 0.0, 1.0)
    if extra_cost is not None:
        soft = soft + np.asarray(extra_cost, dtype=float)
    soft_flat = soft.ravel()
    blocked_flat = blocked.ravel()

    start_i = sy * w + sx
    goal_i = gy * w + gx
    g_cost = {start_i: 0.0}
    parent = {start_i: -1}
    closed = set()

    def heur(ix, iy):
        dx, dy = abs(ix - gx), abs(iy - gy)
        return (dx + dy) + (math.sqrt(2) - 2.0) * min(dx, dy)

    pq = [(heur(sx, sy), 0.0, start_i)]
    expansions = 0
    while pq:
        _, gc, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal_i:
            break
        closed.add(cur)
        expansions += 1
        if expansions > max_expansions:
            return None
        cy, cx = divmod(cur, w)
        for dx, dy, c in _NBRS:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or ny < 0 or nx >= w or ny >= h:
                continue
            ni = ny * w + nx
            if blocked_flat[ni] or ni in closed:
                continue
            ng = gc + c * (1.0 + soft_flat[ni])
            if ng < g_cost.get(ni, math.inf):
                g_cost[ni] = ng
                parent[ni] = cur
                heapq.heappush(pq, (ng + heur(nx, ny), ng, ni))
    if goal_i not in parent:
        return None
    cells = []
    cur = goal_i
    while cur != -1:
        cy, cx = divmod(cur, w)
        cells.append(grid.grid_to_world(cx, cy))
        cur = parent[cur]
    cells.reverse()
    return cells


def _segment_cost(grid, cost, a, b):
    """(integral [cost x m], max) of a cost layer along the segment a -> b
    (midpoint samples every half cell; samples outside the map cost 0)."""
    step = 0.5 * grid.resolution
    L = math.hypot(b[0] - a[0], b[1] - a[1])
    n = max(1, int(math.ceil(L / step)))
    t = (np.arange(n) + 0.5) / n
    ix, iy = grid.world_to_grid_array(np.stack([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])], -1))
    h, w = cost.shape
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    vals = np.zeros(n)
    vals[ok] = cost[iy[ok], ix[ok]]
    return float(vals.sum() * L / n), float(vals.max(initial=0.0))


def smooth_path(grid, path, robot_radius=0.30, max_segment=1.5, extra_cost=None, cost_slack=0.05):
    """Greedy line-of-sight shortcutting followed by densification.

    With ``extra_cost`` (the A* terrain cost layer) a shortcut path[i] -> path[j]
    must also (a) not enter cells costlier than the most expensive cell of the
    polyline path[i..j] it replaces (never cuts a debris corner the A* path
    went around) and (b) not accumulate more cost than that polyline
    (+ ``cost_slack`` relative); otherwise a shorter shortcut is tried."""
    if not path or len(path) < 3:
        return list(path) if path else path
    cost = None
    if extra_cost is not None:
        cost = np.asarray(extra_cost, dtype=float)
        if cost.shape != grid.occ.shape or not np.any(cost > 0.0):
            cost = None
    if cost is not None:
        seg = [_segment_cost(grid, cost, a, b) for a, b in zip(path[:-1], path[1:])]
        cum = np.concatenate([[0.0], np.cumsum([s[0] for s in seg])])
        seg_max = np.array([s[1] for s in seg])
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if grid.line_of_sight(path[i][0], path[i][1], path[j][0], path[j][1], robot_radius):
                if cost is None:
                    break
                c_int, c_max = _segment_cost(grid, cost, path[i], path[j])
                p_int, p_max = cum[j] - cum[i], float(seg_max[i:j].max())
                if c_max <= p_max + 1e-9 and c_int <= p_int * (1.0 + cost_slack) + 1e-6:
                    break
            j -= 1
        out.append(path[j])
        i = j
    return densify(out, max_segment)


def densify(path, max_segment=0.25):
    if not path:
        return path
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(math.ceil(d / max_segment)))
        for k in range(1, n + 1):
            t = k / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return out


def path_length(path):
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, dtype=float)
    return float(np.sum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))))


def plan(grid, start, goal, robot_radius=0.30, extra_cost=None, smooth=True, smooth_radius_scale=1.0):
    """Convenience: A* + smoothing.  Returns list of (x, y) or None.

    ``extra_cost`` (e.g. ``terrain_cost.terrain_cost_grid``) is used by both
    the A* search and the smoother.  Shortcuts are checked against ``robot_radius * smooth_radius_scale``; the
    default 1.0 keeps the clearance A* guarantees (every A* cell is at least
    ``robot_radius`` from an obstacle, so the smoother can always fall back to
    the A* polyline)."""
    p = astar(grid, start, goal, robot_radius=robot_radius, extra_cost=extra_cost)
    if p is None:
        return None
    return smooth_path(grid, p, robot_radius * float(smooth_radius_scale), extra_cost=extra_cost) if smooth else p


def path_to_msg(path, frame_id="map", stamp=None, z=0.0):
    """list of (x, y) or (x, y, yaw) -> nav_msgs/Path (yaw from the path direction
    when not given)."""
    from nav_msgs.msg import Path
    from geometry_msgs.msg import PoseStamped
    msg = Path()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    for i, pt in enumerate(path or []):
        ps = PoseStamped()
        ps.header = msg.header
        ps.pose.position.x = float(pt[0])
        ps.pose.position.y = float(pt[1])
        ps.pose.position.z = float(z)
        if len(pt) >= 3:
            yaw = float(pt[2])
        elif i + 1 < len(path):
            yaw = math.atan2(path[i + 1][1] - pt[1], path[i + 1][0] - pt[0])
        elif i > 0:
            yaw = math.atan2(pt[1] - path[i - 1][1], pt[0] - path[i - 1][0])
        else:
            yaw = 0.0
        ps.pose.orientation.z = math.sin(yaw / 2.0)
        ps.pose.orientation.w = math.cos(yaw / 2.0)
        msg.poses.append(ps)
    return msg
