"""Figures for the theory slide, built from a random DataMatrix (no real label data).

  python theory_figs.py <deck/assets>

Needs opencv-python-headless, numpy and zxing-cpp. Prints what zxing-cpp reads from the warped code.
"""
import sys

import cv2
import numpy as np
import zxingcpp

out = sys.argv[1]
rng = np.random.default_rng(7)
TEXT = "s%02d-%05d-" % (rng.integers(60, 70), rng.integers(0, 100000)) + "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz"), 13))

# one pixel per module, with the white quiet zone
bc = zxingcpp.create_barcode(TEXT, zxingcpp.BarcodeFormat.DataMatrix)
dm = np.array(bc.to_image(scale=1, add_quiet_zones=False), dtype=np.uint8)
print("modules", dm.shape, TEXT)
QZ = 2
dm = cv2.copyMakeBorder(dm, QZ, QZ, QZ, QZ, cv2.BORDER_CONSTANT, value=255)


def camera(img):
    """Soften and add sensor noise, as in the synthetic images."""
    img = cv2.GaussianBlur(img, (0, 0), 0.7).astype(np.float32)
    img = 60 + img / 255 * 150 + rng.normal(0, 7, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def nearest(img, s):
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)


# 4 px/module, shown enlarged so each pixel is visible
ppm = 4
sharp = cv2.resize(dm, None, fx=ppm, fy=ppm, interpolation=cv2.INTER_NEAREST)
cv2.imwrite(f"{out}/theory_dm.png", nearest(camera(sharp), 6))

# tilted view of the same code at about 4 px/module, on a pale label
hi = cv2.resize(dm, None, fx=20, fy=20, interpolation=cv2.INTER_NEAREST)
n = hi.shape[0]
pad = 70
W = n + 2 * pad
quad = np.float32([[pad + 90, pad + 10], [pad + n - 10, pad + 130], [pad + n - 80, pad + n - 20], [pad - 30, pad + n - 120]])
H = cv2.getPerspectiveTransform(np.float32([[0, 0], [n, 0], [n, n], [0, n]]), quad)
tilted = cv2.warpPerspective(hi, H, (W, W), flags=cv2.INTER_AREA, borderValue=255)
scale = ppm / 20
small = cv2.resize(tilted, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
small = camera(small)
big = 5
shown = cv2.cvtColor(nearest(small, big), cv2.COLOR_GRAY2BGR)
pts = (quad * scale * big).astype(int)
cv2.polylines(shown, [pts], True, (31, 54, 168), 3, cv2.LINE_AA)
for p in pts:
    cv2.circle(shown, tuple(int(v) for v in p), 9, (31, 54, 168), -1, cv2.LINE_AA)
cv2.imwrite(f"{out}/theory_tilt.png", shown)

# homography from the four corners back to a square
S = 40 * dm.shape[0] // 4 * 2
Hw = cv2.getPerspectiveTransform(quad * scale, np.float32([[0, 0], [S, 0], [S, S], [0, S]]))
warped = cv2.warpPerspective(small, Hw, (S, S), flags=cv2.INTER_CUBIC, borderValue=255)
cv2.imwrite(f"{out}/theory_warp.png", warped)
print("zxing on warped:", [r.text for r in zxingcpp.read_barcodes(warped)])
print("zxing on tilted:", [r.text for r in zxingcpp.read_barcodes(small)])

# instance masks: each code gets its own colour on a crop of the synthetic strip
raw = cv2.imread(f"{out}/strip_raw.png")
mask = cv2.imread(f"{out}/strip_mask.png")
g = (mask[:, :, 1] > 100).astype(np.uint8)
_, lab, st, _ = cv2.connectedComponentsWithStats(g)
ids = sorted((i for i in range(1, len(st)) if st[i][4] > 200), key=lambda i: st[i][0])
BGR = [(31, 54, 168), (92, 138, 42), (160, 90, 30)]
comp = raw.astype(np.float32).copy()
rows = []
for c, i in zip(BGR, ids[2:5]):
    m = lab == i
    comp[m] = comp[m] * 0.35 + np.float32(c) * 0.65
    x, y, w, h = st[i][:4]
    rows.append(comp[y - 44:y + 44, x - 110:x + 46].astype(np.uint8))
gap = np.full((4, rows[0].shape[1], 3), 251, np.uint8)
stack = np.vstack([rows[0], gap, rows[1], gap, rows[2]])
cv2.imwrite(f"{out}/theory_seg.png", cv2.resize(stack, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
