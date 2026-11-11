"""
sensor_msgs/Image <-> numpy conversion without cv_bridge.

Supported encodings: rgb8, bgr8, rgba8, bgra8, mono8, mono16, 16UC1, 32FC1,
L8 / L16 (Gazebo thermal names).  Colour images are returned as BGR uint8
(OpenCV convention) by image_msg_to_bgr().
"""
import array

import numpy as np

_ENC = {
    "rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "rgba8": (np.uint8, 4), "bgra8": (np.uint8, 4),
    "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1), "l8": (np.uint8, 1),
    "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1), "l16": (np.uint16, 1),
    "32fc1": (np.float32, 1), "64fc1": (np.float64, 1),
}


def image_msg_to_numpy(msg):
    """Return the raw array (H, W[, C]) in the message's own encoding."""
    enc = msg.encoding.lower()
    if enc not in _ENC:
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    dtype, ch = _ENC[enc]
    dt = np.dtype(dtype)
    if msg.is_bigendian and dt.itemsize > 1:
        dt = dt.newbyteorder(">")
    buf = np.frombuffer(bytes(msg.data), dtype=dt)
    row_elems = msg.step // dt.itemsize if msg.step else msg.width * ch
    arr = buf[: msg.height * row_elems].reshape(msg.height, row_elems)[:, : msg.width * ch]
    arr = arr.reshape(msg.height, msg.width, ch) if ch > 1 else arr.reshape(msg.height, msg.width)
    return arr.astype(dt.newbyteorder("=")) if dt.byteorder == ">" else arr


def image_msg_to_bgr(msg):
    """Any colour/mono 8-bit image -> BGR uint8 (H, W, 3)."""
    arr = image_msg_to_numpy(msg)
    enc = msg.encoding.lower()
    if enc == "rgb8":
        return arr[..., ::-1].copy()
    if enc == "rgba8":
        return arr[..., 2::-1].copy()
    if enc == "bgra8":
        return arr[..., :3].copy()
    if enc == "bgr8":
        return arr.copy()
    if arr.ndim == 2:
        a = arr
        if a.dtype != np.uint8:
            a = a.astype(np.float64)
            a = (255 * (a - np.nanmin(a)) / max(1e-9, np.nanmax(a) - np.nanmin(a))).astype(np.uint8)
        return np.repeat(a[..., None], 3, axis=2)
    raise ValueError(f"cannot convert {msg.encoding} to bgr")


def numpy_to_image_msg(arr, encoding, frame_id="", stamp=None):
    """numpy array -> sensor_msgs/Image (encoding must match the array layout)."""
    from sensor_msgs.msg import Image
    arr = np.ascontiguousarray(arr)
    msg = Image()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    ch = arr.shape[2] if arr.ndim == 3 else 1
    msg.step = int(msg.width * ch * arr.dtype.itemsize)
    _assign_uint8_data(msg, arr.tobytes())
    return msg


def _assign_uint8_data(msg, buf):
    """Assign a ``uint8[]`` message field without per-element validation.

    The rosidl-generated setter accepts an ``array.array('B')`` as is; any
    other sequence (``bytes`` included) is checked element by element in
    Python while ``__debug__`` is on (~0.13 s for an 800 x 600 bgr8 image).
    The ``bytes`` fallback covers rosidl versions that reject array.array."""
    try:
        msg.data = array.array("B", buf)
    except (AssertionError, TypeError, ValueError):  # pragma: no cover - other rosidl versions
        msg.data = buf


def bgr_to_image_msg(bgr, frame_id="", stamp=None):
    return numpy_to_image_msg(bgr.astype(np.uint8), "bgr8", frame_id, stamp)
