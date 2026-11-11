"""
ROS-free compositor of the demo HUD (plant_simulation demo_hud node).

The HUD frame imitates the reference video (.dev/sample_video_notes.md): the
robot camera image with an overlay of

* top bar     "현재 가스 농도: 123.4 ppm" (ASCII fallback "GAS: 123.4 ppm"),
              coloured normal / warning / alarm, a log-scale ppm bar with
              tick marks at the warning and alarm thresholds, and the battery
              gauge + percent (charging marker) on the right;
* bottom bar  mission state (+ target gauge and progress), locomotion mode /
              controller / gait, sim time;
* alarm       blinking red frame and a "가스 경보" badge while the gas alarm is
              active (mission gas_alarm, i.e. SEEK, or ppm >= gas_alarm_ppm);
* no camera   dark placeholder "카메라 신호 없음" / "NO CAMERA SIGNAL".

Korean text needs Pillow and a Korean font (Noto Sans CJK, Nanum, Malgun,
...).  Without either the HUD falls back to English ASCII text drawn with
OpenCV, so it works on every machine.  compose_hud() is pure numpy / OpenCV
and is unit-tested without ROS (simulation/test/test_plant_demo_hud.py).
"""
import glob
import math
import os

import cv2
import numpy as np

HUD_DEFAULTS = {
    "rate": 5.0,                     # Hz HUD frames
    "width": 640,                    # px HUD frame (the 800 x 600 inspection camera is scaled to fit)
    "height": 480,
    "hud_topic": "/demo/hud",        # bgr8 for ROS tools (rqt_image_view, ros2 bag)
    "gz_hud_topic": "/demo/hud_rgb",  # rgb8 copy for ros_gz_bridge -> Gazebo topic /demo/hud ('' = off)
    "camera_timeout": 2.0,           # s (wall) without a camera frame -> 'no camera signal'
    "status_timeout": 5.0,           # s (wall) without gas / battery / locomotion / mission data -> '--'
    "gas_warn_ppm": 10.0,            # ppm, amber
    "gas_alarm_ppm": 25.0,           # ppm, red (= mission_manager gas_alarm_ppm)
    "gas_bar_max_ppm": 1000.0,       # ppm at the right end of the log-scale bar (leak ~1200 ppm)
    "font_path": "",                 # '' = search a Korean font (Noto Sans CJK, Nanum, Malgun ...)
    "language": "ko",                # ko | en (ko falls back to en without Pillow or a Korean font)
    "title": "DUSAN AI",
}

# colours (BGR)
COLORS = {
    "normal": (120, 230, 92),
    "warning": (25, 178, 250),
    "alarm": (77, 77, 255),
    "unknown": (170, 170, 170),
}
WHITE = (245, 245, 245)
GREY = (200, 200, 200)
DARK = (28, 28, 28)

# Korean-capable fonts, most specific first (Ubuntu fonts-noto-cjk / fonts-nanum, Windows, macOS)
KOREAN_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf",
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)
FONT_GLOBS = ("/usr/share/fonts/**/*CJK*", "/usr/share/fonts/**/Nanum*", "/usr/share/fonts/**/*KR*",
              "/usr/local/share/fonts/**/*CJK*", "~/.local/share/fonts/**/*CJK*", "~/.fonts/**/*CJK*")

MISSION_LABELS = {
    "IDLE": ("대기", "IDLE"),
    "PATROL": ("순찰 이동", "PATROL"),
    "INSPECTING": ("계기 판독", "INSPECTING"),
    "SEEK": ("가스 누출원 탐색", "GAS SEARCH"),
    "RTH": ("도크로 복귀", "RETURN HOME"),
    "DOCKED": ("도킹 (충전)", "DOCKED"),
    "MANUAL": ("수동 조작", "MANUAL"),
    "PAUSED": ("일시 정지", "PAUSED"),
    "DONE": ("완료", "DONE"),
}

# HUD input: the node fills these keys (None = unknown / stale)
EMPTY_STATUS = {
    "gas_ppm": None,          # float ppm (/gas/sensor)
    "gas_alarm": False,       # bool (/mission/state gas_alarm)
    "battery_pct": None,      # float 0..100 (/battery/state)
    "charging": False,
    "loco_mode": "",          # controller state, e.g. WALK / STAND (/locomotion/state mode)
    "locomotion_mode": "",    # stable | mpc
    "gait": "",               # trot | crawl ...
    "mission_state": "",      # /mission/state state
    "mission_target": "",     # gauge id
    "mission_progress": None,  # [done, total]
    "sim_time": None,         # s
}


# ----------------------------------------------------------------------------
# small pure helpers
# ----------------------------------------------------------------------------
def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def gas_level(ppm, warn_ppm, alarm_ppm, alarm_flag=False):
    """'unknown' | 'normal' | 'warning' | 'alarm'."""
    if alarm_flag:
        return "alarm"
    c = _finite(ppm)
    if c is None:
        return "unknown"
    if c >= float(alarm_ppm):
        return "alarm"
    if c >= float(warn_ppm):
        return "warning"
    return "normal"


def battery_level(pct):
    p = _finite(pct)
    if p is None:
        return "unknown"
    return "normal" if p >= 50.0 else ("warning" if p >= 20.0 else "alarm")


def gas_bar_fraction(ppm, max_ppm):
    """Log-scale fill fraction 0..1: log10(1 + c) / log10(1 + max)."""
    c = _finite(ppm)
    if c is None or c <= 0.0:
        return 0.0
    return float(min(1.0, math.log10(1.0 + c) / math.log10(1.0 + max(1.0, float(max_ppm)))))


def format_ppm(ppm):
    c = _finite(ppm)
    if c is None:
        return "--"
    return f"{c:.1f}" if abs(c) < 10000.0 else f"{c:.0f}"


def gas_text(ppm, korean):
    return f"현재 가스 농도: {format_ppm(ppm)} ppm" if korean else f"GAS: {format_ppm(ppm)} ppm"


def mission_text(state, target="", progress=None, korean=True):
    st = str(state or "").upper()
    if not st:
        return "임무: --" if korean else "MISSION: --"
    ko, en = MISSION_LABELS.get(st, (st, st))
    out = f"임무: {ko}" if korean else f"MISSION: {en}"
    if target and st in ("PATROL", "INSPECTING"):
        out += (" → " if korean else " -> ") + str(target)
    if isinstance(progress, (list, tuple)) and len(progress) == 2:
        try:
            done, total = int(progress[0]), int(progress[1])
            if total > 0:
                out += f" ({done}/{total})"
        except (TypeError, ValueError):
            pass
    return out


def locomotion_text(mode="", locomotion_mode="", gait="", korean=True):
    parts = [str(p) for p in (mode, locomotion_mode, gait) if p]
    if not parts:
        return "보행: --" if korean else "LOCOMOTION: --"
    return ("보행: " + " · ".join(parts)) if korean else ("LOCOMOTION: " + " / ".join(parts))


def battery_text(pct):
    p = _finite(pct)
    return "--%" if p is None else f"{p:.0f}%"


def fit_image(img, width, height):
    """Scale an image into width x height keeping its aspect ratio (black bars)."""
    out = np.zeros((height, width, 3), np.uint8)
    if img is None or img.ndim < 2 or img.shape[0] == 0 or img.shape[1] == 0:
        return out
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(width / float(w), height / float(h))
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(img[..., :3], (nw, nh), interpolation=interp)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    out[y0:y0 + nh, x0:x0 + nw] = small
    return out


def _shade(img, x0, y0, x1, y1, alpha=0.55, color=(0, 0, 0)):
    """Blend a translucent rectangle into img (in place)."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(x0)), min(w, int(x1))
    y0, y1 = max(0, int(y0)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    over = np.empty_like(roi)
    over[:] = color
    cv2.addWeighted(over, float(alpha), roi, 1.0 - float(alpha), 0.0, dst=roi)


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
def find_korean_font(extra=()):
    """Path of the first readable Korean-capable font, or None."""
    for p in list(extra) + list(KOREAN_FONT_CANDIDATES):
        if p and os.path.isfile(p):
            return p
    for pattern in FONT_GLOBS:
        try:
            hits = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        except OSError:
            continue
        hits = [h for h in hits if h.lower().endswith((".ttf", ".ttc", ".otf"))]
        bold = [h for h in hits if "bold" in os.path.basename(h).lower()]
        if bold or hits:
            return (bold or hits)[0]
    return None


class TextRenderer:
    """Draws HUD text: Pillow + a Korean font when available, else OpenCV
    Hershey (ASCII only, English labels).  ``korean`` tells which one is used."""

    def __init__(self, font_path="", language="ko"):
        self.font_file = None
        self.reason = ""
        self._pil = None                                     # (Image, ImageDraw, ImageFont) modules
        self._fonts = {}
        want_korean = str(language or "ko").lower().startswith("ko")
        try:
            from PIL import Image, ImageDraw, ImageFont
        except Exception:                                    # Pillow missing -> OpenCV text
            self.reason = "Pillow (python3-pil) not installed"
        else:
            if font_path and not os.path.isfile(font_path):
                self.reason = f"font_path {font_path} not found"
            path = font_path if (font_path and os.path.isfile(font_path)) else find_korean_font()
            if path:
                self._pil = (Image, ImageDraw, ImageFont)
                self.font_file = path
                try:
                    self._font(20)
                except Exception as exc:                     # unreadable font file
                    self.reason = f"cannot load {path}: {exc}"
                    self._pil, self.font_file, self._fonts = None, None, {}
            elif not self.reason:
                self.reason = "no Korean font found (install fonts-noto-cjk or fonts-nanum)"
        self.pillow = self.font_file is not None
        self.korean = self.pillow and want_korean

    # -- Pillow font cache
    def _font(self, size):
        size = max(6, int(size))
        f = self._fonts.get(size)
        if f is None:
            path = self.font_file
            # Noto Sans CJK collections: face 1 = Korean (KR) glyph variants
            index = 1 if ("CJK" in os.path.basename(path) and path.lower().endswith(".ttc")) else 0
            try:
                f = self._pil[2].truetype(path, size, index=index)
            except OSError:
                f = self._pil[2].truetype(path, size, index=0)
            self._fonts[size] = f
        return f

    @staticmethod
    def _ascii(text):
        return "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in str(text))

    @staticmethod
    def _cv_scale(size):
        return max(0.3, float(size) / 30.0), max(1, int(round(size / 15.0)))

    def measure(self, text, size):
        """(width, height) in px of text drawn at the given pixel size."""
        if self.pillow:
            f = self._font(size)
            return int(math.ceil(f.getlength(str(text)))), int(size)
        scale, th = self._cv_scale(size)
        (w, h), _ = cv2.getTextSize(self._ascii(text), cv2.FONT_HERSHEY_SIMPLEX, scale, th)
        return int(w), int(size)

    def draw(self, img, items):
        """Draw [(text, x, y_top, size_px, bgr)] into img (BGR uint8, in place)."""
        items = [it for it in items if it and str(it[0])]
        if not items:
            return img
        if self.pillow:
            # the array is BGR; Pillow does not care about channel meaning, so
            # colours are passed in BGR order and no channel swap is needed
            pil = self._pil[0].fromarray(img)
            d = self._pil[1].Draw(pil)
            for text, x, y, size, color in items:
                f = self._font(size)
                d.text((int(x), int(y)), str(text), font=f, fill=tuple(int(c) for c in color),
                       anchor="la", stroke_width=max(1, int(size) // 14), stroke_fill=(0, 0, 0))
            img[:] = np.asarray(pil)
            return img
        for text, x, y, size, color in items:
            scale, th = self._cv_scale(size)
            t = self._ascii(text)
            (_, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
            org = (int(x), int(y + h + max(1, size * 0.12)))
            cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, scale, tuple(int(c) for c in color), th,
                        cv2.LINE_AA)
        return img


_DEFAULT_RENDERER = {}


def default_renderer(language="ko", font_path=""):
    key = (language, font_path)
    if key not in _DEFAULT_RENDERER:
        _DEFAULT_RENDERER[key] = TextRenderer(font_path, language)
    return _DEFAULT_RENDERER[key]


# ----------------------------------------------------------------------------
# compositor
# ----------------------------------------------------------------------------
def _battery_icon(img, x, y, w, h, pct, color):
    t = max(1, int(round(h / 10.0)))
    cv2.rectangle(img, (x, y), (x + w, y + h), WHITE, t, cv2.LINE_AA)
    nub_h = max(2, h // 2)
    cv2.rectangle(img, (x + w, y + (h - nub_h) // 2), (x + w + max(2, w // 12), y + (h + nub_h) // 2), WHITE, -1)
    p = _finite(pct)
    if p is not None:
        inner = max(0, int(round((w - 4 * t) * min(100.0, max(0.0, p)) / 100.0)))
        if inner > 0:
            cv2.rectangle(img, (x + 2 * t, y + 2 * t), (x + 2 * t + inner, y + h - 2 * t), color, -1)


def compose_hud(camera_bgr, status=None, cfg=None, renderer=None, t=None):
    """One HUD frame (BGR uint8, cfg height x width).

    camera_bgr  robot camera image (BGR / mono uint8) or None (no signal)
    status      dict with the EMPTY_STATUS keys (missing keys = unknown)
    cfg         HUD_DEFAULTS overrides (width, height, thresholds, title ...)
    renderer    TextRenderer (default: shared one for cfg language/font_path)
    t           wall time [s] for the alarm blink (None = no blinking, alarm drawn)
    """
    c = dict(HUD_DEFAULTS, **(cfg or {}))
    st = dict(EMPTY_STATUS, **(status or {}))
    W, H = max(160, int(c["width"])), max(120, int(c["height"]))
    r = renderer if renderer is not None else default_renderer(c["language"], c["font_path"])
    ko = bool(r.korean)
    s = min(W / 640.0, H / 480.0)

    def S(v):
        return max(1, int(round(v * s)))

    # ------------------------------------------------------------ background
    have_cam = camera_bgr is not None and getattr(camera_bgr, "size", 0) > 0
    if have_cam:
        img = fit_image(np.asarray(camera_bgr), W, H)
    else:
        img = np.full((H, W, 3), DARK, np.uint8)
        cv2.line(img, (0, 0), (W, H), (45, 45, 45), S(2), cv2.LINE_AA)
        cv2.line(img, (0, H), (W, 0), (45, 45, 45), S(2), cv2.LINE_AA)
    texts = []
    if not have_cam:
        msg = "카메라 신호 없음" if ko else "NO CAMERA SIGNAL"
        tw, th = r.measure(msg, S(26))
        _shade(img, (W - tw) // 2 - S(14), H // 2 - S(22), (W + tw) // 2 + S(14), H // 2 + S(24), 0.7)
        texts.append((msg, (W - tw) // 2, H // 2 - S(15), S(26), GREY))

    # ------------------------------------------------------------- top bar
    top_h = S(72)
    _shade(img, 0, 0, W, top_h, 0.55)
    level = gas_level(st["gas_ppm"], c["gas_warn_ppm"], c["gas_alarm_ppm"], bool(st["gas_alarm"]))
    gcol = COLORS[level]
    texts.append((gas_text(st["gas_ppm"], ko), S(12), S(8), S(28), gcol if level != "normal" else WHITE))
    # ppm bar (log scale) + threshold ticks
    bx0, by0 = S(12), S(48)
    bx1, by1 = int(W * 0.55), S(60)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (70, 70, 70), -1)
    frac = gas_bar_fraction(st["gas_ppm"], c["gas_bar_max_ppm"])
    if frac > 0.0:
        cv2.rectangle(img, (bx0, by0), (bx0 + max(1, int((bx1 - bx0) * frac)), by1), gcol, -1)
    for thr in (c["gas_warn_ppm"], c["gas_alarm_ppm"]):
        tx = bx0 + int((bx1 - bx0) * gas_bar_fraction(thr, c["gas_bar_max_ppm"]))
        cv2.line(img, (tx, by0 - S(3)), (tx, by1 + S(3)), WHITE, max(1, S(2)))
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (150, 150, 150), 1)
    # battery (right)
    bpct = st["battery_pct"]
    bcol = COLORS[battery_level(bpct)]
    btxt = battery_text(bpct)
    tw, _ = r.measure(btxt, S(22))
    iw, ih = S(44), S(22)
    tx = W - S(12) - tw
    ix = tx - S(10) - iw - S(4)
    _battery_icon(img, ix, S(12), iw, ih, bpct, bcol)
    texts.append((btxt, tx, S(10), S(22), WHITE))
    sub = ("충전 중" if ko else "CHARGING") if st["charging"] else ("배터리" if ko else "BATTERY")
    sw, _ = r.measure(sub, S(15))
    texts.append((sub, W - S(12) - sw, S(42), S(15), COLORS["normal"] if st["charging"] else GREY))

    # ----------------------------------------------------------- bottom bar
    bot_h = S(58)
    _shade(img, 0, H - bot_h, W, H, 0.55)
    texts.append((mission_text(st["mission_state"], st["mission_target"], st["mission_progress"], ko),
                  S(12), H - bot_h + S(6), S(20), WHITE))
    texts.append((locomotion_text(st["loco_mode"], st["locomotion_mode"], st["gait"], ko),
                  S(12), H - bot_h + S(32), S(17), GREY))
    cam_label = "로봇 카메라" if ko else "ROBOT CAM"
    title = f"{c['title']}{' · ' if ko else ' | '}{cam_label}" if c["title"] else cam_label
    tw, _ = r.measure(title, S(15))
    texts.append((title, W - S(12) - tw, H - bot_h + S(8), S(15), GREY))
    simt = _finite(st["sim_time"])
    if simt is not None:
        ttxt = f"t = {simt:.1f} s"
        tw, _ = r.measure(ttxt, S(15))
        texts.append((ttxt, W - S(12) - tw, H - bot_h + S(33), S(15), GREY))

    # ---------------------------------------------------------------- alarm
    if level == "alarm":
        blink_on = t is None or int(float(t) * 2.0) % 2 == 0
        if blink_on:
            th = S(6)
            cv2.rectangle(img, (th // 2, th // 2), (W - 1 - th // 2, H - 1 - th // 2), COLORS["alarm"], th)
        badge = "가스 경보" if ko else "GAS ALARM"
        bw, _ = r.measure(badge, S(24))
        x0, y0 = (W - bw) // 2 - S(14), top_h + S(10)
        cv2.rectangle(img, (x0, y0), (x0 + bw + S(28), y0 + S(36)),
                      COLORS["alarm"] if blink_on else (40, 40, 160), -1)
        texts.append((badge, x0 + S(14), y0 + S(5), S(24), WHITE))

    r.draw(img, texts)
    return np.ascontiguousarray(img)


class HudCompositor:
    """compose_hud() with a fixed configuration and text renderer."""

    def __init__(self, cfg=None, renderer=None):
        self.cfg = dict(HUD_DEFAULTS, **(cfg or {}))
        self.renderer = renderer if renderer is not None else TextRenderer(self.cfg["font_path"],
                                                                           self.cfg["language"])

    def compose(self, camera_bgr, status, t=None):
        return compose_hud(camera_bgr, status, self.cfg, self.renderer, t)
