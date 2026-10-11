"""2D code encoding (QR / DataMatrix) and decoding helpers.

Corner convention used everywhere in this project:
  corners = [TL, TR, BR, BL] of the symbol in the symbol's OWN orientation
  (TL = matrix row 0 / col 0), clockwise, quiet zone excluded,
  continuous pixel coordinates where (0, 0) is the top-left EDGE of the top-left
  pixel (COCO / YOLO convention). OpenCV pixel centres are at +0.5.
"""
import string

import cv2
import numpy as np
import segno
import zxingcpp

try:  # pylibdmtx imports distutils; setuptools provides the shim on Python 3.12
    import setuptools  # noqa: F401
except ImportError:
    pass
from pylibdmtx.pylibdmtx import encode as dmtx_encode

ZX_FORMATS = {"qr": zxingcpp.BarcodeFormat.QRCode, "datamatrix": zxingcpp.BarcodeFormat.DataMatrix}


def modules_for(cfg, symbology):
    return int(cfg["code"]["CODE_MODULES"][symbology])


def encode_matrix(symbology, payload, modules, qr_error):
    """Return a bool matrix (True = dark) with exactly `modules` x `modules` cells."""
    if symbology == "qr":
        if (modules - 17) % 4 != 0:
            raise ValueError(f"QR modules must be 21, 25, 29, ...; got {modules}")
        version = (modules - 17) // 4
        qr = segno.make(payload, version=version, error=qr_error, boost_error=False, micro=False)
        m = np.array(qr.matrix, dtype=bool)
    elif symbology == "datamatrix":
        enc = dmtx_encode(payload.encode("ascii"), size=f"{modules}x{modules}")
        px = np.frombuffer(enc.pixels, dtype=np.uint8).reshape(enc.height, enc.width, 3)[..., 0]
        dark = px < 128
        ys, xs = np.nonzero(dark)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        step = (x1 - x0) / modules
        centres = ((np.arange(modules) + 0.5) * step).astype(int)
        m = dark[y0:y1, x0:x1][np.ix_(centres, centres)]
    else:
        raise ValueError(symbology)
    if m.shape != (modules, modules):
        raise ValueError(f"{symbology} encoded to {m.shape}, expected {modules}x{modules}")
    return m


def random_sample_id(rng):
    yy = rng.integers(60, 70)
    num = rng.integers(0, 100000)
    block = string.ascii_uppercase[rng.integers(0, 8)]
    slide = f"{rng.integers(1, 10)}{string.ascii_uppercase[rng.integers(0, 6)]}"
    return f"S{yy}-{num:05d} {block} {slide}"


def make_payload(cfg, rng, sample_id):
    kind = cfg["code"]["payload"]
    if kind == "sample_id":
        return sample_id
    if kind == "random_alnum":
        chars = string.ascii_uppercase + string.digits
        return "".join(chars[i] for i in rng.integers(0, len(chars), size=12))
    raise ValueError(kind)


def _corners_to_cv(corners):
    return np.asarray(corners, dtype=np.float32) - 0.5


def rectify(img, corners, modules, px_per_module, quiet_modules):
    """Warp the quad to an axis-aligned symbol with a white quiet zone."""
    side = modules * px_per_module
    q = quiet_modules * px_per_module
    dst = np.array([[q, q], [q + side, q], [q + side, q + side], [q, q + side]], np.float32) - 0.5
    H = cv2.getPerspectiveTransform(_corners_to_cv(corners), dst)
    size = side + 2 * q
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.warpPerspective(gray, H, (size, size), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # force the quiet zone white so neighbouring label text cannot interfere
    mask = np.ones_like(out, dtype=bool)
    mask[q:q + side, q:q + side] = False
    out[mask] = 255
    return out


def decode_rectified(img, corners, symbology, modules, px_per_module=8, quiet_modules=4):
    patch = rectify(img, corners, modules, px_per_module, quiet_modules)
    res = zxingcpp.read_barcodes(patch, formats=ZX_FORMATS[symbology])
    return res[0].text if res else None


def decode_direct(img, formats=None):
    """Plain zxing read on the image (no geometry hint). Returns list of (text, format)."""
    res = zxingcpp.read_barcodes(img, formats=formats) if formats else zxingcpp.read_barcodes(img)
    return [(r.text, str(r.format)) for r in res]


def sample_modules(img, corners, modules, px_per_module=8):
    """Sample each module centre after rectification; returns bool matrix (True = dark)."""
    patch = rectify(img, corners, modules, px_per_module, 0).astype(np.float32)
    c = (np.arange(modules) + 0.5) * px_per_module - 0.5
    ci = np.round(c).astype(int)
    vals = patch[np.ix_(ci, ci)]
    thr = 0.5 * (np.percentile(vals, 5) + np.percentile(vals, 95))
    return vals < thr
