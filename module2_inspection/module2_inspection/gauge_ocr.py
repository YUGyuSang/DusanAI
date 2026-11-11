"""
Scale-label OCR for analog gauges (ROS-free, numpy + OpenCV).

The dial labels are printed with ``cv2.FONT_HERSHEY_SIMPLEX`` by
``dusan_common.gauge_design``.  This module reads them with *template
matching* of glyph images:

* digit / '.' / '-' templates rendered with ``cv2.FONT_HERSHEY_SIMPLEX`` at the
  label parameters of several texture sizes (runtime OpenCV), plus
* the classic Hershey "simplex" stroke glyphs (OpenCV <= 4.x rendering of
  FONT_HERSHEY_SIMPLEX), drawn from their public-domain vector data at several
  stroke weights, plus
* an embedded bitmap set of the OpenCV >= 5 rendering of FONT_HERSHEY_SIMPLEX
  (OpenCV 5 replaced the stroke fonts by a TrueType face).

Including all three makes the OCR independent of which OpenCV version
rendered the Gazebo dial textures and which one runs the reader (the textures
are generated on the development PC, the reader runs on the robot PC).

Two recognition paths are provided:

1. ``DigitOCR.read`` - character level: binarise the label patch, segment
   connected components (with splitting of merged glyphs), classify each glyph
   by normalised cross-correlation against all templates; '.' and '-' are
   recognised by their geometry.  Works when the label is >= ~8 px tall.
2. ``DigitOCR.score_string`` - analysis by synthesis for low resolution: render
   candidate label strings and correlate them with the patch.  Used by
   ``infer_range`` to verify a small set of standard (EN 837-1) gauge ranges
   when single glyphs are too small to classify reliably.

``infer_range`` combines the label reads at every major tick (positions known
from the calibrated dial geometry) by a RANSAC fit  label_k = vmin + k * step
with a "nice step" plausibility check.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from dusan_common import gauge_design as gd

TEMPLATE_H = 20                       # normalised glyph height of all templates [px]
DIGITS = "0123456789"
LABEL_SCALE_PER_S = 1.25 / 1024.0     # gauge_design label fontScale per texture pixel
LABEL_HEIGHT_DIAL = 0.026             # digit height in dial units (fraction of texture side)
RUNTIME_TEXTURE_SIZES = (256, 512, 1024, 2048)   # label weights of these texture sizes
HERSHEY_STROKE_RATIOS = (0.08, 0.12, 0.16)       # stroke width / cap height of the stroke templates

# Standard pressure-gauge full-scale values (EN 837-1 / ASME B40.100 R10-like series)
STANDARD_SPANS = (0.6, 1.0, 1.6, 2.5, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0, 60.0, 100.0, 160.0,
                  250.0, 400.0, 600.0, 1000.0)

# ----------------------------------------------------------------------------------------
# Classic Hershey "simplex" glyphs (public-domain vector font by A. V. Hershey, the font
# behind OpenCV's FONT_HERSHEY_SIMPLEX up to 4.x).  (advance, [polyline, ...]), y up,
# baseline y = 0, cap height 21.
# ----------------------------------------------------------------------------------------
_HERSHEY_SIMPLEX = {
    "0": (20, [[(9, 21), (6, 20), (4, 17), (3, 12), (3, 9), (4, 4), (6, 1), (9, 0), (11, 0), (14, 1),
                (16, 4), (17, 9), (17, 12), (16, 17), (14, 20), (11, 21), (9, 21)]]),
    "1": (20, [[(6, 17), (8, 18), (11, 21), (11, 0)]]),
    "2": (20, [[(4, 16), (4, 17), (5, 19), (6, 20), (8, 21), (12, 21), (14, 20), (15, 19), (16, 17),
                (16, 15), (15, 13), (13, 10), (3, 0), (17, 0)]]),
    "3": (20, [[(5, 21), (16, 21), (10, 13), (13, 13), (15, 12), (16, 11), (17, 8), (17, 6), (16, 3),
                (14, 1), (11, 0), (8, 0), (5, 1), (4, 2), (3, 4)]]),
    "4": (20, [[(13, 21), (3, 7), (18, 7)], [(13, 21), (13, 0)]]),
    "5": (20, [[(15, 21), (5, 21), (4, 12), (5, 13), (8, 14), (11, 14), (14, 13), (16, 11), (17, 8),
                (17, 6), (16, 3), (14, 1), (11, 0), (8, 0), (5, 1), (4, 2), (3, 4)]]),
    "6": (20, [[(16, 18), (15, 20), (12, 21), (10, 21), (7, 20), (5, 17), (4, 12), (4, 7), (5, 3),
                (7, 1), (10, 0), (11, 0), (14, 1), (16, 3), (17, 6), (17, 7), (16, 10), (14, 12),
                (11, 13), (10, 13), (7, 12), (5, 10), (4, 7)]]),
    "7": (20, [[(17, 21), (7, 0)], [(3, 21), (17, 21)]]),
    "8": (20, [[(8, 21), (5, 20), (4, 18), (4, 16), (5, 14), (7, 13), (11, 12), (14, 11), (16, 9),
                (17, 7), (17, 4), (16, 2), (15, 1), (12, 0), (8, 0), (5, 1), (4, 2), (3, 4), (3, 7),
                (4, 9), (6, 11), (9, 12), (13, 13), (15, 14), (16, 16), (16, 18), (15, 20), (12, 21),
                (8, 21)]]),
    "9": (20, [[(16, 14), (15, 11), (13, 9), (10, 8), (9, 8), (6, 9), (4, 11), (3, 14), (3, 15),
                (4, 18), (6, 20), (9, 21), (10, 21), (13, 20), (15, 18), (16, 14), (16, 9), (15, 4),
                (13, 1), (10, 0), (8, 0), (5, 1), (4, 3)]]),
    ".": (10, [[(5, 2), (4, 1), (5, 0), (6, 1), (5, 2)]]),
    "-": (26, [[(4, 9), (22, 9)]]),
}

# ----------------------------------------------------------------------------------------
# OpenCV >= 5 rendering of FONT_HERSHEY_SIMPLEX digits with the gauge_design label
# parameters (texture 512 -> light weight, 1024 -> bold), ink 0..15 per pixel, row-major,
# cropped to the ink bounding box and resized to TEMPLATE_H rows.
# Generated once with cv2.putText (OpenCV 5.0.0).
# ----------------------------------------------------------------------------------------
_CV5_GLYPHS = (
    ("0", "cv5_512", 15, 20,
     "00058bdda830000026adededc9510006effdabeffd4003afe94239ffe9308dfa30028cffe60cfe61006b"
     "7aff80dfd30009d46ef90ffc2003ac13cf92ffa2017a813afa3ffa202bb403afa3ffa204e9003afa3ffa"
     "117c7004afa3ffb03b94004bfa2efc27d61005df91cfe6ad30007ff80adfcc810009ff705bffd72036df"
     "b4019fffd77befe710039dffcdffc7200002aeffffd71000"),
    ("1", "cv5_512", 14, 20,
     "0000028b930000000049cdc400000015bfffe50000027cffffe5000027efc8bfe500004afc409fe50000"
     "1574009fe500000010009fe500000000009fe500000000009fe500000000009fe500000000009fe50000"
     "0000009fe500000000009fe500000000009fe500000000009fe50000012212afe60122258878cfea7777"
     "38ddddeffedddd4affffffffffff"),
    ("2", "cv5_512", 14, 20,
     "00037addb950000149cddddda72004bffcaadffe7018ffa32238dfa45cfc4000029fd87ce71000006eea"
     "47830000005eea12200000028fc80000000006dfa4000000005cfe7000000016bfe9200000006cfe9200"
     "000016cff9200000016dfe9200000006cff9200000016cfe8200000016dffb401222217bffd987777777"
     "bdffeccdddddcccdffffffffffff"),
    ("3", "cv5_512", 14, 20,
     "47bbbbbbbbbb8559ddddddccdda748ccbbbbceffb7123333235cfe71000000028ee82000000028ff9100"
     "0000018ef92000000028efa3000000007efd6200000003afffdca72000038bbbdffd730000122248cfda"
     "00000000116efe11000000002bff54200000002aefcc920000003cffdff72111238feb7cfea6558befb7"
     "26beecbcdeda52015bfffffea410"),
    ("4", "cv5_512", 17, 20,
     "00000000049b840000000000039dda5100000000028dffd710000000005deefd71000000004bfa9fd610"
     "0000002afd45fd6100000017ee715fd610000005df9106fd61000003afb4006fd61000018ee60006fd61"
     "00015ee810006fd600004cfa200006fd610028ffa555559fea6435bffcbccccdffed975affffffffffff"
     "fc82466666666afea75400000000005fd600000000000006fd610000000000006fd610000000000006ed"
     "6100"),
    ("5", "cv5_512", 14, 20,
     "038abbbbbbbb7304bddddddddd8406ffdbbbbbcc7307fe822333332107fc500000000008fc5000000000"
     "19fa300000000029fb767876420039fedfffffb6204bffda99befd7238cb732247cfc702320000005efd"
     "00000000003bff22100000002aef88610000002aefbdc40000004dfd9dfa4111139fd94affb7557bffa4"
     "149decbbced9410039effffe9300"),
    ("6", "cv5_512", 15, 20,
     "00000016b8300000000005cd720000000003cfd400000000029fe720000000007ef910000000014bfc50"
     "000000002affa52100000017effcba96200005dfffffffe930028ffc8668cfd8207cfc610015bfe60bee"
     "61000006df91efc30000003afb4efc200000039fb4cfd20000003afa3aee50000005df906bfb511123af"
     "e6018ffc8558bfe820038ceccccdc83000017cefffc83000"),
    ("7", "cv5_512", 14, 20,
     "48bbbbbbbbbb9769edddddddddba59ccccccccefdb1233333235afc80000000003bfa30000000008ff70"
     "000000002bfc50000000016ef82000000003afd40000000005efa20000000018fd60000000005cfa3000"
     "0000008ff70000000002bfc40000000016de82000000003bfc40000000006ef910000000018fd7000000"
     "0015cf920000000017de60000000"),
    ("8", "cv5_512", 15, 20,
     "00059bddb940000027bdccccda620017efda88befd5104afb622135cf9307de60000008fd509de400000"
     "07ee607de60000008fd604afb622236cf93006efd978aefc50003bffffffffa10016beeccccdeb5107cf"
     "d731147dfd50ced61000018ef80ffa10000003bfa2fe9100000039fb4ffa10000004bfa2cee62111128f"
     "f706bfe95446aefc5015beecbbcdd9510015bfffffe94000"),
    ("9", "cv5_512", 14, 20,
     "00049bddb95000027adddddda72017efecaaceff716bfd832237dfb7bee81000017eecefd30000003cff"
     "ffb20000002affffb20000002bffdfe40000004dfe9dfb400004afda38efb6546affa404aeffeeffff70"
     "0137accdffe9300000246affb3000000004bfd61000000018ef81000000016efa3000000005cfc410000"
     "00029fd61000000016cf92000000"),
    ("0", "cv5_1024", 16, 20,
     "0000369bca7510000017dffffffea300019efffffffffc4006efffdbacffffa22bfffd4106ffffe53eff"
     "f9100bfffff95ffff5004ebbfffc7fffd4019e49fffe7fffd303db09fffe8fffc307f509ffff8fffc32c"
     "c209ffff7fffd36f7009fffe6fffe4bc301afffd4ffffae9102cfffb3dffffe4005efff819ffffd435cf"
     "ffd404dffffdceffff81006dffffffffea200004adffffec71000000134675320000"),
    ("1", "cv5_1024", 15, 20,
     "00000289972000000004cfffe500000007dffffe50000029ffffffe500003bfffffffe500008ffffbfff"
     "e500004dfe83fffe5000005b501fffe50000001001fffe50000000001fffe50000000001fffe50000000"
     "001fffe50000000001fffe50000000001fffe50000000001fffe50000000001fffe500003bbbbcffffcb"
     "bb86fffffffffffffd6fffffffffffffd5effffffffffffb"),
    ("2", "cv5_1024", 15, 20,
     "000259cdca62000004bfffffffb30004cfffffffffb302bffffddeffff915fffe9214cfffc28fffb2000"
     "6fffe4aeee700004ffff41222000009fffd3000000004efffa100000004cfffd400000006dfffe600000"
     "017efffe800000019efffe71000001affffe500000029ffffe700000019efffe710000008effffecbbbb"
     "bb7ffffffffffffffbffffffffffffffbefffffffffffff9"),
    ("3", "cv5_1024", 16, 20,
     "049999999999996119ffffffffffffc219ffffffffffffd207cccccccdffffa1012222224cfffb300000"
     "0003bfffc2000000005dfffa2000000005dfffa2000000004dffffc8410000006ffffffffb3000004cdd"
     "efffffb300000111249efff700000000001afffb133310000019fffe9eeeb200003cfffd6ffffa4235af"
     "fff92cffffdcdeffffd403bffffffffffb300028ceffffec82000000235775321000"),
    ("4", "cv5_1024", 18, 20,
     "000000001489994000000000005dffff900000000002cfffff90000000001affffff90000000007fffff"
     "ff9000000004dffccfff900000002bffe5afff900000019ffe81afff90000006fffb20afff9000003dff"
     "e300afff900002bffe6000afff900019fff91000afffa1104efffb9999dfffdbb86ffffffffffffffffd"
     "6ffffffffffffffffd4cccccccccefffecc90222222222cfffa2210000000000bfff90000000000000bf"
     "ff900000000000009fff7000"),
    ("5", "cv5_1024", 15, 20,
     "02899999999994006effffffffffa108fffffffffffa119fffcccccccc711affd42222222102bffc2000"
     "0000002cffb22332100003dffc9bccba50003efffffffffe9204fffffddeffff914eeed7334bfffe5012"
     "2100003cfff900000000019fffc88861000019fffceffe810003cfff8cfffd7334bffff47ffffeccdfff"
     "fb217dfffffffffa20005cefffffd9100000246887531000"),
    ("6", "cv5_1024", 16, 20,
     "00000004999500000000004dfff70000000002bfffb10000000009fffd30000000005effe60000000003"
     "cfffa1000000001affffa7530000007ffffffffc710003dfffffffffe91019ffffdbbdfffe713dfffb31"
     "14bfffc26fffe200003fffe48fffc100001cfff66fffc100001dfff53efff400004fffd32bfffd6336df"
     "ffa106efffeddefffe50005effffffffd4000005beffffea40000000246776410000"),
    ("7", "cv5_1024", 15, 20,
     "3899999999999977fffffffffffffe7ffffffffffffff6eeeeeeeeeffffe1444444447efffa000000000"
     "8fffe4000000003cfffa2000000007fffe5000000001cfffb2000000005fffe7000000001cfffc200000"
     "0004efff8000000001afffe2000000004dfff90000000018fffd4000000003dfffa1000000007fffe500"
     "0000001dfffa1000000006fffe60000000009fffb1000000"),
    ("8", "cv5_1024", 16, 20,
     "0001468bcb9620000018efffffffc40002affffffffffd5006effea658dfffb219fff910005dffe32bff"
     "e500001afff51afff700003cffe307effd7335bfffb2019fffeedefffd40005ffffffffffa2005dffffe"
     "effffea22cfffd52239efff85fffd400001afffc8fffb2000006efff7fffc3000008fffd4ffff920016d"
     "fff91affffcaabefffe502befffffffffc500027cefffffda3000000235776531000"),
    ("9", "cv5_1024", 16, 20,
     "0001369bca7410000017dffffffea300019efffffffffc4007efffeddeffffb23dfffd5113bffff65fff"
     "e600004dfffa7fffc300001afffc7fffc300001afffc5fffe500003cfff93dfffd3001affff606effffd"
     "cfffffa2018efffffffffe500016dffffffff8100000147bffffc1000000001afffd40000000007effe7"
     "1000000004dfffa1000000002bfffc20000000018fffe50000000002cffe81000000"),
)


def _decode_glyph(w: int, h: int, hexs: str) -> np.ndarray:
    a = np.frombuffer(hexs.encode("ascii"), np.uint8)
    v = np.where(a >= ord("a"), a - ord("a") + 10, a - ord("0")).astype(np.float32) / 15.0
    return v.reshape(h, w)


# ---------------------------------------------------------------------------------------- utils
def crop_ink(ink: np.ndarray, thr: float = 0.2) -> Optional[np.ndarray]:
    """Crop an ink map (1 = ink) to its bounding box; None if empty."""
    ys, xs = np.where(ink > thr)
    if len(ys) == 0:
        return None
    return ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def normalise_height(ink: np.ndarray, h: int = TEMPLATE_H) -> np.ndarray:
    """Resize an ink crop to height ``h`` keeping its aspect ratio."""
    hh, ww = ink.shape
    nw = max(1, int(round(ww * h / float(hh))))
    interp = cv2.INTER_AREA if hh > h else cv2.INTER_LINEAR
    return np.clip(cv2.resize(ink.astype(np.float32), (nw, h), interpolation=interp), 0, 1)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation of two equally sized arrays."""
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    return float(np.dot(a, b) / den) if den > 1e-12 else 0.0


def render_hershey_glyph(ch: str, cap_px: float = 56.0, stroke_ratio: float = 0.11) -> np.ndarray:
    """Draw a classic Hershey simplex glyph as an ink map (cropped)."""
    adv, strokes = _HERSHEY_SIMPLEX[ch]
    sc = cap_px / 21.0
    th = max(1, int(round(stroke_ratio * cap_px)))
    pad = th + 4
    W = int(adv * sc) + 2 * pad
    H = int(cap_px) + 2 * pad
    img = np.zeros((H, W), np.uint8)
    for line in strokes:
        pts = np.array([[pad + x * sc, H - pad - y * sc] for x, y in line]) * 16.0
        cv2.polylines(img, [np.round(pts).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=4)
    if ch == ".":   # the Hershey period is a tiny filled diamond
        c = np.array([pad + 5 * sc, H - pad - 1 * sc])
        cv2.circle(img, (int(c[0]), int(c[1])), max(1, int(1.2 * sc)), 255, -1, cv2.LINE_AA)
    return crop_ink(img.astype(np.float32) / 255.0)


def render_hershey_text(text: str, cap_px: float = 56.0, stroke_ratio: float = 0.11) -> Optional[np.ndarray]:
    """Draw a string with the classic Hershey simplex glyphs and their advance widths."""
    if any(ch not in _HERSHEY_SIMPLEX for ch in text):
        return None
    sc = cap_px / 21.0
    th = max(1, int(round(stroke_ratio * cap_px)))
    pad = th + 4
    W = int(sum(_HERSHEY_SIMPLEX[ch][0] for ch in text) * sc) + 2 * pad
    H = int(cap_px) + 2 * pad
    img = np.zeros((H, W), np.uint8)
    xo = float(pad)
    for ch in text:
        adv, strokes = _HERSHEY_SIMPLEX[ch]
        for line in strokes:
            pts = np.array([[xo + x * sc, H - pad - y * sc] for x, y in line]) * 16.0
            cv2.polylines(img, [np.round(pts).astype(np.int32)], False, 255, th, cv2.LINE_AA, shift=4)
        if ch == ".":
            cv2.circle(img, (int(xo + 5 * sc), int(H - pad - 1 * sc)), max(1, int(1.2 * sc)), 255, -1, cv2.LINE_AA)
        xo += adv * sc
    return crop_ink(img.astype(np.float32) / 255.0)


def render_cv_text(text: str, font_scale: float, thickness: int) -> Optional[np.ndarray]:
    """Render text with cv2.FONT_HERSHEY_SIMPLEX (runtime OpenCV) as a cropped ink map."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = int(10 + 2 * thickness)
    img = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
    cv2.putText(img, text, (pad, pad + th), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, thickness, cv2.LINE_AA)
    return crop_ink(img.astype(np.float32) / 255.0)


# ---------------------------------------------------------------------------------------- results
@dataclass
class GlyphResult:
    char: str
    score: float          # best NCC (minus aspect penalty)
    margin: float         # best - best score of any other character
    box: Tuple[int, int, int, int] = (0, 0, 0, 0)   # x, y, w, h in the patch


@dataclass
class LabelRead:
    text: str
    value: Optional[float]
    confidence: float
    glyphs: List[GlyphResult] = field(default_factory=list)
    reason: str = ""


@dataclass
class RangeEstimate:
    vmin: float
    vmax: float
    confidence: float
    method: str                       # 'chars' | 'hypothesis'
    n_div: int
    labels: List[Tuple[int, str, float]] = field(default_factory=list)   # (k, text, conf)


# ---------------------------------------------------------------------------------------- OCR
class DigitOCR:
    """Template-matching OCR for gauge scale labels (digits, '.', '-').

    Segmentation works on ink projections rather than connected components so that it
    survives the blur of up-sampled low-resolution labels: digit columns are found in the
    *upper* 62 % of the text line (the decimal point lives at the baseline and therefore never
    merges two digits there), decimal points are looked for in the baseline zone of the gaps
    between digits, minus signs are short mid-height runs.  Templates are blurred to the
    estimated blur of the observation before the normalised cross-correlation."""

    def __init__(self, template_height: int = TEMPLATE_H, aspect_weight: float = 0.6):
        self.h = int(template_height)
        self.aspect_weight = float(aspect_weight)
        self.templates: Dict[str, List[Tuple[str, np.ndarray]]] = {c: [] for c in DIGITS}
        self._blurred: Dict[int, Dict[str, List[Tuple[str, np.ndarray]]]] = {}
        self._strings: Dict[Tuple[str, str, int], Optional[np.ndarray]] = {}
        self._string_blur: Dict[Tuple[str, str, int], np.ndarray] = {}
        self._build_templates()

    # ------------------------------------------------------------ templates
    def _add(self, ch: str, name: str, ink: Optional[np.ndarray]):
        if ink is None or ink.size == 0 or ch not in self.templates:
            return
        self.templates[ch].append((name, normalise_height(ink, self.h)))

    def _build_templates(self):
        # (1) runtime OpenCV FONT_HERSHEY_SIMPLEX with the label parameters of several texture sizes
        for S in RUNTIME_TEXTURE_SIZES:
            scale = S * LABEL_SCALE_PER_S
            thick = max(1, int(S / 1024 * 3))
            for ch in DIGITS:
                self._add(ch, f"cv_{S}", render_cv_text(ch, scale, thick))
        # (2) classic Hershey strokes at several stroke weights
        for ratio in HERSHEY_STROKE_RATIOS:
            for ch in DIGITS:
                self._add(ch, f"hershey_{ratio}", render_hershey_glyph(ch, 56.0, ratio))
        # (3) embedded OpenCV-5 bitmaps
        for ch, name, w, h, hexs in _CV5_GLYPHS:
            self._add(ch, name, _decode_glyph(w, h, hexs))

    def font_variants(self) -> List[str]:
        return sorted({name for lst in self.templates.values() for name, _ in lst})

    @staticmethod
    def _blur_crop(ink: np.ndarray, sigma: float, h: int) -> np.ndarray:
        """Blur an ink crop (padding first), re-crop at 15 % of its peak and re-normalise."""
        if sigma < 0.3:
            return ink
        p = int(math.ceil(3 * sigma)) + 1
        b = cv2.GaussianBlur(np.pad(ink, p), (0, 0), sigma)
        c = crop_ink(b, 0.15 * float(b.max()))
        return normalise_height(c / max(float(c.max()), 1e-6), h) if c is not None else ink

    def bank(self, sigma: float) -> Dict[str, List[Tuple[str, np.ndarray]]]:
        """Templates blurred by ``sigma`` (template-canvas pixels), cached in 0.25 px steps."""
        key = int(round(max(sigma, 0.0) * 4))
        if key == 0:
            return self.templates
        if key not in self._blurred:
            s = key / 4.0
            self._blurred[key] = {ch: [(n, self._blur_crop(t, s, self.h)) for n, t in lst]
                                  for ch, lst in self.templates.items()}
        return self._blurred[key]

    # ------------------------------------------------------------ glyph classification
    def classify(self, ink: np.ndarray, sigma: float = 0.0) -> GlyphResult:
        """Classify one digit glyph (ink crop, any size) -> best character."""
        crop = crop_ink(ink, 0.15 * float(ink.max()) if ink.size else 0.2)
        if crop is None or crop.shape[0] < 3:
            return GlyphResult("?", 0.0, 0.0)
        g = normalise_height(crop / max(float(crop.max()), 1e-6), self.h)
        a_g = g.shape[1] / float(self.h)
        best: Dict[str, float] = {}
        for ch, lst in self.bank(sigma).items():
            s_best = -1.0
            for _name, t in lst:
                tw = t.shape[1]
                a_t = tw / float(self.h)
                gr = cv2.resize(g, (tw, self.h), interpolation=cv2.INTER_LINEAR) if g.shape[1] != tw else g
                s = ncc(gr, t) - self.aspect_weight * abs(math.log(max(a_g, 1e-3) / max(a_t, 1e-3)))
                s_best = max(s_best, s)
            best[ch] = s_best
        order = sorted(best.items(), key=lambda kv: -kv[1])
        (c1, s1), (_c2, s2) = order[0], order[1]
        return GlyphResult(c1, float(s1), float(s1 - s2))

    # ------------------------------------------------------------ patch -> ink
    @staticmethod
    def ink_map(patch: np.ndarray, min_contrast: float = 12.0) -> Optional[np.ndarray]:
        """Gray patch (dark text on light face) -> ink map 0..1, None if no contrast."""
        p = patch.astype(np.float32)
        if p.ndim == 3:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        bg = float(np.percentile(p, 85))
        # the ink may cover only ~1-3 % of the patch: dark level from a lightly smoothed
        # low percentile (robust to isolated noise pixels)
        dk = float(np.percentile(cv2.GaussianBlur(p, (3, 3), 0), 0.4))
        if bg - dk < min_contrast:
            return None
        return np.clip((bg - p) / (bg - dk), 0.0, 1.0)

    def segment(self, ink: np.ndarray, expected_h: Optional[float] = None) -> Optional[dict]:
        """Projection-based segmentation of the label centred in the patch.

        Returns {'top', 'bottom', 'h', 'items': [(kind, x0, x1)], 'flags'} where kind is
        'digit' | '.' | '-', or None when no text line is found."""
        H, W = ink.shape
        flags: List[str] = []
        exp_h = float(expected_h) if expected_h else 0.5 * H
        cm = np.zeros(W, bool)
        cm[int(0.1 * W):int(0.9 * W)] = True
        rows = (ink[:, cm] > 0.35).sum(axis=1).astype(float)
        if rows.max() <= 0:
            return None
        # text line: the band of significant rows around the patch centre
        c0 = int(np.argmax(rows[int(0.25 * H):int(0.75 * H)])) + int(0.25 * H)
        thr = 0.15 * rows[c0]
        top = c0
        while top > 0 and rows[top - 1] > thr:
            top -= 1
        bot = c0
        while bot < H - 1 and rows[bot + 1] > thr:
            bot += 1
        h = bot - top + 1
        if top == 0 or bot == H - 1:
            flags.append("line_clipped")         # needle / tick crossing the label vertically
        if h < 0.45 * exp_h or h > 1.8 * exp_h:
            flags.append("bad_height")
            if h > 1.8 * exp_h:
                # keep a band of the expected height around the strongest row
                top = max(0, int(round(c0 - 0.5 * exp_h)))
                bot = min(H - 1, int(round(c0 + 0.5 * exp_h)))
                h = bot - top + 1
        upper = ink[top:top + max(2, int(round(0.62 * h))), :]
        cp = (upper > 0.3).sum(axis=0).astype(float)
        if cp.max() <= 0:
            return None
        on = cp > 0.12 * cp.max()
        runs: List[List[int]] = []
        x = 0
        while x < W:
            if on[x]:
                x0 = x
                while x < W and on[x]:
                    x += 1
                runs.append([x0, x - 1])
            else:
                x += 1
        # merge runs separated by tiny gaps (broken strokes)
        merged: List[List[int]] = []
        for r in runs:
            if merged and r[0] - merged[-1][1] - 1 < max(1, int(0.06 * h)):
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        items: List[Tuple[str, int, int]] = []
        for x0, x1 in merged:
            w = x1 - x0 + 1
            if w < max(2, int(0.1 * h)):
                continue
            if (x0 == 0 or x1 == W - 1) and abs((x0 + x1) / 2.0 - W / 2.0) > 0.28 * W:
                continue                       # neighbouring label / tick at the patch border
            col = ink[top:bot + 1, x0:x1 + 1]
            rows_on = np.where((col > 0.35).any(axis=1))[0]
            if len(rows_on) and (rows_on[-1] - rows_on[0] + 1) < 0.3 * h and w > 0.25 * h:
                mid = top + 0.5 * (rows_on[0] + rows_on[-1])
                if top + 0.3 * h < mid < top + 0.75 * h:
                    items.append(("-", x0, x1))
                    continue
            n_split = int(round(w / (0.72 * h))) if w > 1.02 * h else 1
            if n_split <= 1:
                items.append(("digit", x0, x1))
                continue
            # split at the weakest columns near the expected glyph boundaries
            bounds = [x0]
            for k in range(1, n_split):
                xe = x0 + k * w / n_split
                lo_, hi_ = int(xe - 0.18 * h), int(xe + 0.18 * h) + 1
                lo_, hi_ = max(bounds[-1] + 2, lo_), min(x1 - 1, hi_)
                xs = lo_ + int(np.argmin(cp[lo_:hi_])) if hi_ > lo_ else int(xe)
                bounds.append(xs)
            bounds.append(x1 + 1)
            for k in range(n_split):
                items.append(("digit", bounds[k], bounds[k + 1] - 1))
        items.sort(key=lambda it: it[1])
        # decimal points: a separate blob in the baseline zone of a gap between two digits.  The
        # baseline ink profile must have an interior maximum (a serif of '1' or the foot of a
        # digit decays monotonically from the neighbouring glyph instead).
        out: List[Tuple[str, int, int]] = []
        base = ink[top + int(0.66 * h):bot + 1, :]
        bprof = base.max(axis=0) if base.shape[0] > 0 else np.zeros(W)
        for k, it in enumerate(items):
            out.append(it)
            if not (k + 1 < len(items) and it[0] == "digit" and items[k + 1][0] == "digit"):
                continue
            g0, g1 = it[2] + 1, items[k + 1][1] - 1
            if g1 - g0 + 1 < max(1, int(0.08 * h)):
                continue
            # blobs of the baseline profile inside the gap that touch neither neighbour
            on_b = bprof[g0:g1 + 1] > 0.45
            x = 0
            while x < on_b.size:
                if not on_b[x]:
                    x += 1
                    continue
                b0 = x
                while x < on_b.size and on_b[x]:
                    x += 1
                b1 = x - 1
                touches = ((b0 == 0 and bprof[max(0, g0 - 1)] > 0.45) or
                           (b1 == on_b.size - 1 and bprof[min(W - 1, g1 + 1)] > 0.45))
                area = float(base[:, g0 + b0:g0 + b1 + 1].sum())
                if not touches and area > 0.012 * h * h and (b1 - b0 + 1) < 0.5 * h:
                    out.append((".", g0 + b0, g0 + b1))
                    break
        # digit crops: extend the upper-zone runs over full-height ink (feet, serifs) up to the
        # neighbouring item / gap middle
        line = ink[top:bot + 1, :]
        colmax = line.max(axis=0)
        final: List[Tuple[str, int, int]] = []
        for k, (kind, x0, x1) in enumerate(out):
            if kind == "digit":
                lim_l = (out[k - 1][2] + 1 if out[k - 1][0] == "." else (out[k - 1][2] + x0) // 2 + 1) if k > 0 else 0
                lim_r = (out[k + 1][1] - 1 if out[k + 1][0] == "." else (x1 + out[k + 1][1]) // 2) if k + 1 < len(out) else W - 1
                while x0 - 1 >= lim_l and colmax[x0 - 1] > 0.35:
                    x0 -= 1
                while x1 + 1 <= lim_r and colmax[x1 + 1] > 0.35:
                    x1 += 1
            final.append((kind, x0, x1))
        return {"top": top, "bottom": bot, "h": h, "items": final, "flags": flags}

    def read(self, patch: np.ndarray, expected_h: Optional[float] = None, blur_sigma: float = 0.0) -> LabelRead:
        """Read the label centred in ``patch`` (gray, text upright, dark on light).

        blur_sigma: estimated blur of the patch [patch px] (templates are blurred to match)."""
        ink = self.ink_map(patch)
        if ink is None:
            return LabelRead("", None, 0.0, reason="no_contrast")
        seg = self.segment(ink, expected_h)
        if seg is None or not seg["items"]:
            return LabelRead("", None, 0.0, reason="no_glyphs")
        h = seg["h"]
        if h < 5:
            return LabelRead("", None, 0.0, reason="too_small")
        sig_c = blur_sigma * self.h / float(h)
        glyphs: List[GlyphResult] = []
        for kind, x0, x1 in seg["items"]:
            box = (x0, seg["top"], x1 - x0 + 1, h)
            if kind == "digit":
                g = self.classify(ink[seg["top"]:seg["bottom"] + 1, x0:x1 + 1], sig_c)
                g.box = box
            else:
                g = GlyphResult(kind, 0.9, 0.9, box)
            glyphs.append(g)
        text = "".join(g.char for g in glyphs)
        value = parse_label(text)
        if value is None:
            return LabelRead(text, None, 0.0, glyphs, reason="unparsable")
        digit_g = [g for g in glyphs if g.char in DIGITS]
        # confidence: weakest glyph (score and margin), damped for tiny / disturbed text
        s = min(g.score for g in digit_g)
        m = min(g.margin for g in digit_g)
        conf = float(np.clip((s - 0.3) / 0.45, 0, 1) * np.clip(m / 0.1, 0, 1)
                     * np.clip((h - 4.0) / 6.0, 0.2, 1.0))
        if seg["flags"]:
            conf *= 0.6
        return LabelRead(text, value, conf, glyphs, reason=",".join(seg["flags"]))

    # ------------------------------------------------------------ analysis by synthesis
    def render_string(self, text: str, variant: str) -> Optional[np.ndarray]:
        """Ink image of ``text`` in one font variant, cropped and normalised to height h."""
        key = (text, variant, self.h)
        if key in self._strings:
            return self._strings[key]
        out = None
        if variant.startswith("cv_"):
            S = int(variant[3:])
            out = render_cv_text(text, S * LABEL_SCALE_PER_S, max(1, int(S / 1024 * 3)))
        elif variant.startswith("hershey_"):
            out = render_hershey_text(text, 56.0, float(variant[8:]))
        else:
            # embedded monospace bitmaps: digits on a fixed advance, small square period
            parts = []
            for ch in text:
                if ch in DIGITS:
                    t = next((tt for name, tt in self.templates[ch] if name == variant), None)
                    if t is None:
                        return None
                    cell = np.zeros((self.h, int(round(0.78 * self.h))), np.float32)
                    x0 = (cell.shape[1] - t.shape[1]) // 2
                    cell[:, max(0, x0):max(0, x0) + min(t.shape[1], cell.shape[1])] = t[:, :cell.shape[1]]
                    parts.append(cell)
                elif ch == ".":
                    cell = np.zeros((self.h, int(round(0.78 * self.h))), np.float32)
                    d = max(2, int(round(0.2 * self.h)))
                    c = cell.shape[1] // 2
                    cell[self.h - d:, c - d // 2:c - d // 2 + d] = 1.0
                    parts.append(cell)
                elif ch == "-":
                    cell = np.zeros((self.h, int(round(0.78 * self.h))), np.float32)
                    cell[int(0.45 * self.h):int(0.58 * self.h), 2:-2] = 1.0
                    parts.append(cell)
                else:
                    return None
            out = crop_ink(np.concatenate(parts, axis=1)) if parts else None
        if out is not None:
            out = normalise_height(out, self.h)
        self._strings[key] = out
        return out

    def score_string(self, patch: np.ndarray, text: str, expected_h: Optional[float] = None,
                     blur_sigma: float = 0.0) -> float:
        """Best NCC between the label ink in ``patch`` and ``text`` rendered in any font variant
        (both blurred to a common level; width mismatch penalised)."""
        ink = self.ink_map(patch, min_contrast=6.0)
        if ink is None:
            return 0.0
        seg = self.segment(ink, expected_h)
        if seg is None or not seg["items"]:
            return 0.0
        x0 = min(it[1] for it in seg["items"])
        x1 = max(it[2] for it in seg["items"])
        obs = ink[seg["top"]:seg["bottom"] + 1, x0:x1 + 1]
        if obs.shape[0] < 3 or obs.shape[1] < 2:
            return 0.0
        obs_n = normalise_height(obs, self.h)
        sig_c = max(1.0, blur_sigma * self.h / float(seg["h"]))
        obs_b = cv2.GaussianBlur(obs_n, (0, 0), 0.5)
        best = -1.0
        skey = int(round(sig_c * 4))
        for var in self.font_variants():
            ck = (text, var, skey)
            rb = self._string_blur.get(ck)
            if rb is None:
                r = self.render_string(text, var)
                if r is None:
                    continue
                rb = self._blur_crop(r, skey / 4.0, self.h)
                if len(self._string_blur) > 4000:
                    self._string_blur.clear()
                self._string_blur[ck] = rb
            o = cv2.resize(obs_b, (rb.shape[1], self.h), interpolation=cv2.INTER_LINEAR)
            pen = abs(math.log(obs_n.shape[1] / float(rb.shape[1])))
            best = max(best, ncc(o, rb) - 0.8 * pen)
        return float(best)


# ---------------------------------------------------------------------------------------- labels
_LABEL_RE = re.compile(r"^-?\d+(\.\d+)?$")


def parse_label(text: str) -> Optional[float]:
    """'1.6' -> 1.6 ; rejects malformed strings ('1..6', '.5', '05')."""
    if not text or not _LABEL_RE.match(text):
        return None
    body = text.lstrip("-")
    if len(body) > 1 and body[0] == "0" and body[1] != ".":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def n_major_divisions(vmin: float, vmax: float) -> int:
    """Number of major divisions gauge_design prints for a range."""
    step = gd.nice_major_step(vmin, vmax)
    return int(round((vmax - vmin) / step))


def major_label_angles(n_div: int) -> List[float]:
    """Dial angles (deg, gauge_design convention) of the n_div + 1 major ticks / labels."""
    return [gd.PHI_MIN_DEG - k * gd.SWEEP_DEG / n_div for k in range(n_div + 1)]


def expected_labels(vmin: float, vmax: float) -> List[str]:
    step = gd.nice_major_step(vmin, vmax)
    n = int(round((vmax - vmin) / step))
    return [gd.format_label(vmin + k * step, step) for k in range(n + 1)]


def candidate_ranges(n_div: int, spans: Sequence[float] = STANDARD_SPANS,
                     vmins: Sequence[float] = (0.0, -1.0)) -> List[Tuple[float, float]]:
    """Standard ranges whose printed scale has exactly ``n_div`` major divisions."""
    out = []
    for v0 in vmins:
        for sp in spans:
            v1 = v0 + sp if v0 == 0.0 else sp            # compound gauges: -1 .. span
            if v1 <= v0:
                continue
            step = gd.nice_major_step(v0, v1)
            n = (v1 - v0) / step
            if abs(n - round(n)) < 1e-6 and int(round(n)) == n_div:
                out.append((v0, v1))
    return out


def infer_range(reads: Sequence[Tuple[int, LabelRead]], n_div: int,
                min_conf: float = 0.25) -> Optional[RangeEstimate]:
    """Fit label_k = vmin + k * step to character-level reads (RANSAC over pairs).

    The step must equal gauge_design.nice_major_step(vmin, vmin + n_div * step); the
    estimate needs >= 2 consistent labels, one of them at an end of the scale or >= 3.
    """
    good = [(k, r) for k, r in reads if r.value is not None and r.confidence >= min_conf]
    best = None
    for i in range(len(good)):
        for j in range(i + 1, len(good)):
            (k1, r1), (k2, r2) = good[i], good[j]
            if k1 == k2:
                continue
            step = (r2.value - r1.value) / float(k2 - k1)
            if step <= 0:
                continue
            v0 = r1.value - k1 * step
            v1 = v0 + n_div * step
            nice = gd.nice_major_step(v0, v1)
            if abs(nice - step) > 1e-6 * max(1.0, abs(step)):
                continue
            inl = [(k, r) for k, r in good
                   if abs(r.value - (v0 + k * step)) <= 1e-6 * max(1.0, abs(step)) + 1e-9]
            score = sum(r.confidence for _, r in inl)
            if best is None or score > best[0]:
                best = (score, v0, v1, inl)
    if best is None:
        return None
    score, v0, v1, inl = best
    ks = {k for k, _ in inl}
    if len(inl) < 2 or (len(inl) == 2 and not ({0, n_div} & ks)):
        return None
    conf = float(np.clip(score / 2.0, 0, 1)) * min(1.0, len(inl) / 3.0 + 0.34)
    return RangeEstimate(round(v0, 6), round(v1, 6), conf, "chars", n_div,
                         [(k, r.text, r.confidence) for k, r in inl])


def score_ranges(ocr: DigitOCR, patches: Dict[int, np.ndarray], n_div: int,
                 expected_h: Optional[float] = None,
                 candidates: Optional[Sequence[Tuple[float, float]]] = None,
                 blur_sigma: float = 0.0) -> List[Tuple[float, float, float]]:
    """Analysis-by-synthesis scores [(score, vmin, vmax)] of candidate ranges.

    For every candidate the expected label strings are rendered (all font variants) and
    correlated with the label patches at the major ticks; the score is the mean NCC after
    dropping the worst label (the needle may cover one)."""
    cands = list(candidates) if candidates is not None else candidate_ranges(n_div)
    out = []
    for v0, v1 in cands:
        labels = expected_labels(v0, v1)
        if len(labels) != n_div + 1:
            continue
        sc = [ocr.score_string(p, labels[k], expected_h, blur_sigma) for k, p in patches.items() if k < len(labels)]
        sc = sorted(x for x in sc if x > -0.5)
        if not sc:
            continue
        if len(sc) >= 3:
            sc = sc[1:]
        out.append((float(np.mean(sc)), float(v0), float(v1)))
    return out


def hypothesis_confidence(best: float, runner_up: float) -> float:
    """Confidence of the best range hypothesis from its score and its margin."""
    return float(np.clip((best - 0.45) / 0.3, 0, 1) * np.clip((best - runner_up) / 0.12, 0, 1))


def verify_ranges(ocr: DigitOCR, patches: Dict[int, np.ndarray], n_div: int,
                  expected_h: Optional[float] = None,
                  candidates: Optional[Sequence[Tuple[float, float]]] = None,
                  blur_sigma: float = 0.0) -> Optional[RangeEstimate]:
    """Best standard range for one division count (see ``score_ranges``)."""
    scored = score_ranges(ocr, patches, n_div, expected_h, candidates, blur_sigma)
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    s1, v0, v1 = scored[0]
    s2 = scored[1][0] if len(scored) > 1 else -1.0
    return RangeEstimate(v0, v1, hypothesis_confidence(s1, s2), "hypothesis", n_div, [])
