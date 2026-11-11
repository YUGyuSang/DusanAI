"""
ROS-free image rendering for the dashboard (numpy + OpenCV).

* render_map_png      occupancy grid -> dark-theme PNG (north up)
* render_gas_png      gas grid 0..100 -> RGBA PNG, one-hue sequential orange
                      ramp; zero recedes to fully transparent, high values are
                      lighter AND more opaque (dark-surface anchor flip)
* render_rssi_png     RSSI grid 0..100 (= -100..-30 dBm) -> RGBA PNG, diverging
                      red <-> blue around the comm-shadow threshold with a
                      neutral gray midpoint
* encode_jpeg         BGR frame -> JPEG bytes (downscaled for the browser)

Grids use the nav_msgs/OccupancyGrid convention (row 0 = y origin = SOUTH),
images are flipped so that row 0 is NORTH (as drawn on screen).
"""
import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is a hard runtime dependency on the robot PC
    cv2 = None

# dark theme map colours (BGR)
MAP_FREE = (29, 31, 31)        # #1f1f1d
MAP_OCCUPIED = (85, 90, 91)    # #5b5a55
MAP_UNKNOWN = (13, 13, 13)     # #0d0d0d

# sequential orange (dark-surface ramp, low -> high), RGB
GAS_RAMP_RGB = np.array([[0x6b, 0x24, 0x0c], [0xa8, 0x3c, 0x14], [0xd9, 0x59, 0x26],
                         [0xf0, 0x86, 0x55], [0xff, 0xc4, 0xa3]], dtype=np.float64)
# diverging red <-> gray <-> blue, RGB
RSSI_LOW_RGB = np.array([0xe6, 0x67, 0x67], dtype=np.float64)
RSSI_MID_RGB = np.array([0x38, 0x38, 0x35], dtype=np.float64)
RSSI_HIGH_RGB = np.array([0x39, 0x87, 0xe5], dtype=np.float64)

RSSI_DBM_MIN, RSSI_DBM_MAX = -100.0, -30.0


def _require_cv2():
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV (python3-opencv) is required for dashboard rendering")


def png_bytes(img):
    _require_cv2()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def render_map_png(occ, scale=1):
    """int8 occupancy grid (row 0 south) -> PNG bytes, north up."""
    occ = np.asarray(occ)
    img = np.empty(occ.shape + (3,), np.uint8)
    img[:] = MAP_UNKNOWN
    img[occ == 0] = MAP_FREE
    img[(occ > 0) & (occ < 50)] = tuple(int(0.5 * (a + b)) for a, b in zip(MAP_FREE, MAP_OCCUPIED))
    img[occ >= 50] = MAP_OCCUPIED
    img = np.ascontiguousarray(np.flipud(img))
    if scale and int(scale) > 1:
        img = np.repeat(np.repeat(img, int(scale), axis=0), int(scale), axis=1)
    return png_bytes(img)


def _ramp(t, stops):
    """Piecewise-linear colour ramp; t in [0, 1] (array) -> (..., 3) RGB."""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    n = len(stops) - 1
    x = t * n
    i = np.minimum(np.floor(x).astype(int), n - 1)
    f = (x - i)[..., None]
    return stops[i] * (1.0 - f) + stops[i + 1] * f


def gas_rgba(values, min_visible=1.0, alpha_max=0.80):
    """0..100 grid -> RGBA float array (row order unchanged).

    Opacity grows with value**0.6 so faint plume edges stay visible without
    washing the whole map, and the core is not a saturated block; unknown
    (-1) and ~0 are transparent."""
    v = np.asarray(values, dtype=np.float64)
    t = np.clip(v / 100.0, 0.0, 1.0)
    rgb = _ramp(t, GAS_RAMP_RGB)
    a = np.where(v >= min_visible, alpha_max * t ** 0.6, 0.0)
    return np.concatenate([rgb, (a * 255.0)[..., None]], axis=-1)


def rssi_rgba(values, threshold_dbm=-80.0, alpha=0.42):
    """0..100 grid (= -100..-30 dBm) -> RGBA float array."""
    v = np.asarray(values, dtype=np.float64)
    dbm = RSSI_DBM_MIN + np.clip(v, 0.0, 100.0) / 100.0 * (RSSI_DBM_MAX - RSSI_DBM_MIN)
    below = np.clip((threshold_dbm - dbm) / max(1e-6, threshold_dbm - RSSI_DBM_MIN), 0.0, 1.0)
    above = np.clip((dbm - threshold_dbm) / max(1e-6, RSSI_DBM_MAX - threshold_dbm), 0.0, 1.0)
    rgb = np.where((dbm < threshold_dbm)[..., None],
                   RSSI_MID_RGB * (1 - below[..., None]) + RSSI_LOW_RGB * below[..., None],
                   RSSI_MID_RGB * (1 - above[..., None]) + RSSI_HIGH_RGB * above[..., None])
    a = np.where(v >= 0, alpha, 0.0)
    return np.concatenate([rgb, (a * 255.0)[..., None]], axis=-1)


def _rgba_png(rgba):
    rgba = np.clip(np.round(rgba), 0, 255).astype(np.uint8)
    bgra = rgba[..., [2, 1, 0, 3]]
    return png_bytes(np.ascontiguousarray(np.flipud(bgra)))


def render_gas_png(values, min_visible=1.0):
    return _rgba_png(gas_rgba(values, min_visible))


def render_rssi_png(values, threshold_dbm=-80.0):
    return _rgba_png(rssi_rgba(values, threshold_dbm))


def encode_jpeg(bgr, quality=80, max_width=640):
    """BGR (or gray) uint8 frame -> JPEG bytes, downscaled to max_width."""
    _require_cv2()
    img = np.asarray(bgr)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    if max_width and w > max_width:
        nh = max(1, int(round(h * max_width / float(w))))
        img = cv2.resize(img, (int(max_width), nh), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def placeholder_jpeg(text, width=320, height=240):
    """Dark placeholder frame with a centred caption ('no signal')."""
    _require_cv2()
    img = np.full((height, width, 3), 26, np.uint8)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (56, 56, 53), 1)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, text, ((width - tw) // 2, (height + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (137, 135, 129), 1, cv2.LINE_AA)
    return encode_jpeg(img, 70, width)
