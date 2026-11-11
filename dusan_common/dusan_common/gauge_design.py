"""
Analog pressure-gauge dial design shared by the world generator (Gazebo
textures) and Module 2 (synthetic benchmark + reader calibration).

Angle convention (dial plane, viewed from the front, y up):
    phi = angle counter-clockwise from +x (3 o'clock), in degrees
    min value  at phi = 225 deg (7:30 o'clock)
    max value  at phi = -45 deg (4:30 o'clock)
    the scale runs CLOCKWISE over SWEEP_DEG = 270 deg, the 90 deg gap is at
    the bottom (6 o'clock).  A red "danger" band covers the last 15 % of the
    scale (asymmetric marker -> identifies the max end / mirroring).
Image coordinates: u right, v down  ->  (cx + r cos phi, cy - r sin phi).
"""
import math
import numpy as np

SWEEP_DEG = 270.0
PHI_MIN_DEG = 225.0
PHI_MAX_DEG = PHI_MIN_DEG - SWEEP_DEG   # -45
DANGER_FRACTION = 0.15

# radii as a fraction of the texture size
R_HOUSING = 0.49
R_RIM_OUT = 0.47
R_FACE = 0.44
R_TICK_OUT = 0.40
R_MAJOR_IN = 0.335
R_MINOR_IN = 0.37
R_LABEL = 0.27
R_NEEDLE = 0.36
R_HUB = 0.035

COLORS = {  # BGR
    "housing": (70, 70, 70),
    "rim": (25, 25, 25),
    "face": (240, 243, 245),
    "ink": (20, 20, 20),
    "danger": (40, 40, 200),
    "needle": (15, 15, 15),
    "hub": (50, 50, 50),
}


def value_to_angle(value, vmin, vmax):
    f = (float(value) - vmin) / (vmax - vmin)
    return PHI_MIN_DEG - f * SWEEP_DEG


def angle_to_value(phi_deg, vmin, vmax):
    """Inverse of value_to_angle; phi in degrees (any wrapping)."""
    phi = float(phi_deg)
    # bring phi into (PHI_MAX - 45, PHI_MIN + 45] = (-90, 270]
    while phi <= PHI_MAX_DEG - 45.0:
        phi += 360.0
    while phi > PHI_MIN_DEG + 45.0:
        phi -= 360.0
    f = (PHI_MIN_DEG - phi) / SWEEP_DEG
    return vmin + f * (vmax - vmin)


def nice_major_step(vmin, vmax, target_divisions=8):
    span = float(vmax - vmin)
    raw = span / target_divisions
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag + 1e-12:
            return m * mag
    return 10 * mag


def format_label(v, step):
    if abs(step - round(step)) < 1e-9 and abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    decimals = 1 if abs(step * 10 - round(step * 10)) < 1e-9 else 2
    return f"{v:.{decimals}f}"


def render_gauge_face(value, vmin, vmax, unit="bar", tag="", size=1024, draw_needle=True):
    """Return a BGR uint8 image (size x size) of the dial face."""
    import cv2
    S = int(size)
    img = np.zeros((S, S, 3), np.uint8)
    img[:] = COLORS["housing"]
    c = (S // 2, S // 2)
    aa = cv2.LINE_AA

    def pt(r, phi_deg):
        a = math.radians(phi_deg)
        return (int(round(c[0] + r * S * math.cos(a))), int(round(c[1] - r * S * math.sin(a))))

    cv2.circle(img, c, int(R_RIM_OUT * S), COLORS["rim"], -1, aa)
    cv2.circle(img, c, int(R_FACE * S), COLORS["face"], -1, aa)

    # danger band (last 15 % of the scale), drawn as a thick arc
    phi0 = PHI_MIN_DEG - (1.0 - DANGER_FRACTION) * SWEEP_DEG
    pts = [pt(R_TICK_OUT + 0.012, a) for a in np.linspace(phi0, PHI_MAX_DEG, 40)]
    pts += [pt(R_TICK_OUT - 0.035, a) for a in np.linspace(PHI_MAX_DEG, phi0, 40)]
    cv2.fillPoly(img, [np.array(pts, np.int32)], COLORS["danger"], aa)

    major = nice_major_step(vmin, vmax)
    minor = major / 5.0
    n_minor = int(round((vmax - vmin) / minor))
    for k in range(n_minor + 1):
        v = vmin + k * minor
        phi = value_to_angle(v, vmin, vmax)
        is_major = abs((v - vmin) / major - round((v - vmin) / major)) < 1e-6
        r_in = R_MAJOR_IN if is_major else R_MINOR_IN
        th = max(1, int(0.012 * S)) if is_major else max(1, int(0.005 * S))
        cv2.line(img, pt(r_in, phi), pt(R_TICK_OUT, phi), COLORS["ink"], th, aa)
        if is_major:
            label = format_label(v, major)
            scale = S / 1024 * 1.25
            thick = max(1, int(S / 1024 * 3))
            (tw, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            p = pt(R_LABEL, phi)
            cv2.putText(img, label, (p[0] - tw // 2, p[1] + th_ // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, COLORS["ink"], thick, aa)

    # unit + tag in the bottom gap
    for text, dy, sc in ((unit, 0.17, 1.6), (tag, 0.30, 0.9)):
        if not text:
            continue
        scale = S / 1024 * sc
        thick = max(1, int(S / 1024 * (4 if sc > 1 else 2)))
        (tw, th_), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(img, text, (c[0] - tw // 2, int(c[1] + dy * S) + th_ // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, COLORS["ink"], thick, aa)

    if draw_needle:
        draw_needle_on(img, value, vmin, vmax)
    return img


def draw_needle_on(img, value, vmin, vmax):
    import cv2
    S = img.shape[0]
    c = np.array([S / 2.0, S / 2.0])
    a = math.radians(value_to_angle(value, vmin, vmax))
    d = np.array([math.cos(a), -math.sin(a)])
    n = np.array([-d[1], d[0]])
    tip = c + d * R_NEEDLE * S
    tail = c - d * 0.08 * S
    w0, w1 = 0.022 * S, 0.006 * S
    poly = np.array([tail + n * w0, tip + n * w1, tip - n * w1, tail - n * w0])
    cv2.fillPoly(img, [np.round(poly).astype(np.int32)], COLORS["needle"], cv2.LINE_AA)
    cv2.circle(img, (int(c[0]), int(c[1])), int(R_HUB * S), COLORS["hub"], -1, cv2.LINE_AA)
    cv2.circle(img, (int(c[0]), int(c[1])), max(1, int(0.01 * S)), (160, 160, 160), -1, cv2.LINE_AA)
    return img
