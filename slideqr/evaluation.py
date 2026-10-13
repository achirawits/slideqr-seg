"""Model-independent evaluation: quads from masks, matching, corner error,
three ways of feeding the decoder, and per-ppm-bin aggregation.

All coordinates are full-resolution, edge-based pixel coordinates (see codes.py).
Every number produced here comes from SYNTHETIC images.
"""
from collections import defaultdict

import cv2
import numpy as np
import zxingcpp
from pycocotools import mask as mask_utils

from . import codes as C

ZX_ALL = zxingcpp.BarcodeFormat.DataMatrix | zxingcpp.BarcodeFormat.QRCode
METHODS = ["full_image", "bbox_crop", "quad_warp", "warp_then_crop"]


# ---------------------------------------------------------------- geometry

def mask_to_poly(mask):
    """Largest external contour of a bool mask, edge-based coordinates (N x 2)."""
    m = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    return c + 0.5  # pixel centres -> edge-based


def poly_to_quad(poly):
    """Four corners, clockwise in image coordinates (y down), starting top-left-most.

    approxPolyDP with a growing tolerance; falls back to the minimum-area rectangle.
    The symbol's own orientation is unknown from a mask, so corner error is
    measured up to a cyclic shift (see corner_error)."""
    if poly is None or len(poly) < 4:
        return None
    pts = poly.astype(np.float32).reshape(-1, 1, 2)
    peri = cv2.arcLength(pts, True)
    quad = None
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(pts, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float64)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(pts)).astype(np.float64)
    # clockwise in image coords == positive shoelace area with y pointing down
    x, y = quad[:, 0], quad[:, 1]
    if np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)) < 0:
        quad = quad[::-1]
    start = int(np.argmin(quad.sum(1)))
    return np.roll(quad, -start, axis=0)


def corner_error(quad, gt):
    """Mean corner distance (px), minimised over the 4 cyclic orderings."""
    gt = np.asarray(gt, float).reshape(4, 2)
    return float(min(np.linalg.norm(np.roll(quad, k, axis=0) - gt, axis=1).mean() for k in range(4)))


# ---------------------------------------------------------------- matching

def gt_instances(coco, image_id, cat_id):
    anns = [a for a in coco.imgToAnns[image_id] if a["category_id"] == cat_id]
    return anns


def owners(preds, gts, h, w, inside_thr=0.5):
    """For each prediction, the code physically under it: the ground truth that covers at
    least `inside_thr` of the predicted mask area (None otherwise). Decoded text is judged
    against this code, so a poor but well-placed mask that reads the code beneath it is
    not counted as a wrong read."""
    if not preds or not gts:
        return [None] * len(preds)
    g_rle = [mask_utils.merge(mask_utils.frPyObjects(g["segmentation"], h, w)) for g in gts]
    out = []
    for p in preds:
        area = max(float(mask_utils.area(p["rle"])), 1.0)
        inside = [float(mask_utils.area(mask_utils.merge([p["rle"], g], intersect=True))) / area for g in g_rle]
        j = int(np.argmax(inside))
        out.append(j if inside[j] >= inside_thr else None)
    return out


def match(preds, gts, h, w, iou_thr):
    """Greedy one-to-one matching by mask IoU (preds sorted by score).
    Returns dict pred_index -> gt_index."""
    if not preds or not gts:
        return {}
    g_rle = [mask_utils.merge(mask_utils.frPyObjects(g["segmentation"], h, w)) for g in gts]
    p_rle = [p["rle"] for p in preds]
    ious = mask_utils.iou(p_rle, g_rle, [0] * len(g_rle))
    order = np.argsort([-p["score"] for p in preds])
    used, out = set(), {}
    for i in order:
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and j not in used:
            used.add(j)
            out[int(i)] = j
    return out


# ---------------------------------------------------------------- decoding

# zxing's default LocalAverage binarizer uses a fixed local window: when modules are
# larger than it (high ppm, tight crops) they come out hollow and the read fails.
# GlobalHistogram is tried as a fallback. All three decoder inputs use the same reader.
BINARIZERS = (zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram)


def zx_read(img, first_only):
    texts = []
    for b in BINARIZERS:
        for r in zxingcpp.read_barcodes(img, formats=ZX_ALL, binarizer=b):
            if r.text not in texts:
                texts.append(r.text)
        if texts and first_only:
            break
    return texts


def decode_full(img):
    return zx_read(img, first_only=False)


def decode_bbox_crop(img, poly, margin_frac):
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    mx, my = (x1 - x0) * margin_frac, (y1 - y0) * margin_frac
    H, W = img.shape[:2]
    xa, ya = int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my)))
    xb, yb = int(min(W, np.ceil(x1 + mx))), int(min(H, np.ceil(y1 + my)))
    if xb - xa < 4 or yb - ya < 4:
        return None
    res = zx_read(np.ascontiguousarray(img[ya:yb, xa:xb]), first_only=True)
    return res[0] if res else None


def decode_quad_warp(img, quad, px_per_module, nominal_modules=20):
    """Rectify the predicted quad to a square (modules unknown -> nominal size)."""
    patch = C.rectify(img, quad, nominal_modules, px_per_module, 3)
    res = zx_read(patch, first_only=True)
    return res[0] if res else None


def decode_all(img, poly, quad, cfg_eval):
    """Per-instance decoder inputs. warp_then_crop = 4-corner warp first, the
    axis-aligned crop only when the warp gives no read."""
    crop = decode_bbox_crop(img, poly, cfg_eval["crop_margin_frac"]) if poly is not None else None
    warp = decode_quad_warp(img, quad, cfg_eval["rectify_px_per_module"]) if quad is not None else None
    return {"bbox_crop": crop, "quad_warp": warp, "warp_then_crop": warp if warp is not None else crop}


# ---------------------------------------------------------------- aggregation

def ppm_bin(ppm, edges):
    k = int(np.searchsorted(edges, ppm, side="right")) - 1
    if k < 0 or k >= len(edges) - 1:
        return None
    return f"[{edges[k]:g},{edges[k + 1]:g})"


class Tally:
    """Per-bin counters for detection, corner error and the three decode methods."""

    def __init__(self, edges, focus):
        self.edges = edges
        self.focus = focus
        self.rows = defaultdict(lambda: defaultdict(float))
        self.corner = defaultdict(list)

    def _keys(self, ppm):
        keys = ["all"]
        b = ppm_bin(ppm, self.edges)
        if b:
            keys.append(b)
        for name, (lo, hi) in self.focus.items():
            if lo <= ppm < hi:
                keys.append(f"{name} [{lo:g},{hi:g})")
        return keys

    def add(self, ppm, field, value=1.0):
        for k in self._keys(ppm):
            self.rows[k][field] += value

    def add_corner(self, ppm, err_px):
        for k in self._keys(ppm):
            self.corner[k].append(err_px)

    def table(self):
        out = []
        order = ["all"] + [ppm_bin(e + 1e-9, self.edges) for e in self.edges[:-1]] + \
            [f"{n} [{lo:g},{hi:g})" for n, (lo, hi) in self.focus.items()]
        for k in order:
            if k not in self.rows:
                continue
            r = self.rows[k]
            n = r["n_codes"]
            row = dict(bin=k, n_codes=int(n), detected=r["detected"] / n if n else np.nan)
            ce = self.corner.get(k, [])
            row["corner_err_px_mean"] = float(np.mean(ce)) if ce else np.nan
            row["corner_err_px_median"] = float(np.median(ce)) if ce else np.nan
            for m in METHODS:
                row[f"{m}_read_rate"] = r[f"{m}_ok"] / n if n else np.nan
                row[f"{m}_wrong_reads"] = int(r[f"{m}_wrong"])
            out.append(row)
        return out


def evaluate_image(img, gts_code, preds, cfg_eval, tally, full_texts=None):
    """Update `tally` for one image. gts_code: COCO anns of class code (with payload, ppm,
    corners, iscrowd). preds: code-class predictions with score, rle, poly."""
    H, W = img.shape[:2]
    thr = cfg_eval["score_thr"]
    preds = [p for p in preds if p["score"] >= thr]
    scored = [g for g in gts_code if not g["iscrowd"]]
    m = match(preds, scored, H, W, cfg_eval["iou_match"])
    payloads = {g["payload"] for g in gts_code}

    # method 1: whole image straight into the decoder (model independent)
    texts = decode_full(img) if full_texts is None else full_texts
    for g in scored:
        tally.add(g["ppm"], "n_codes")
        if g["payload"] in texts:
            tally.add(g["ppm"], "full_image_ok")
    for t in texts:
        if t not in payloads:
            tally.add(_nearest_ppm(scored), "full_image_wrong")

    # per predicted instance. Detection and corner error use the IoU match; a decoded
    # text is judged against the code physically under the prediction (owners()).
    own = owners(preds, gts_code, H, W)
    hit = {m: set() for m in METHODS if m != "full_image"}
    for i, p in enumerate(preds):
        j = m.get(i)
        g = scored[j] if j is not None else None
        quad = poly_to_quad(p["poly"])
        if g is not None:
            tally.add(g["ppm"], "detected")
            if quad is not None:
                tally.add_corner(g["ppm"], corner_error(quad, g["corners"]))
        reads = decode_all(img, p["poly"], quad, cfg_eval)
        o = gts_code[own[i]] if own[i] is not None else None
        ppm = o["ppm"] if o is not None else _nearest_ppm(scored)
        for meth, text in reads.items():
            if text is None:
                continue
            if o is not None and text == o["payload"]:
                if not o["iscrowd"] and o["id"] not in hit[meth]:   # ignored codes are not scored
                    hit[meth].add(o["id"])
                    tally.add(ppm, f"{meth}_ok")
            else:
                tally.add(ppm, f"{meth}_wrong")  # not the code under the prediction, or no code under it
    return texts


def _nearest_ppm(gts):
    """ppm used to bin reads that cannot be tied to one code (image median)."""
    return float(np.median([g["ppm"] for g in gts])) if gts else -1.0


def det_to_record(score, cls_name, mask_full):
    """Compact prediction record: RLE for AP/IoU, contour for quads and crops."""
    rle = mask_utils.encode(np.asfortranarray(mask_full.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    return dict(score=float(score), cls=cls_name, rle=rle, poly=mask_to_poly(mask_full),
                area=float(mask_full.sum()))
