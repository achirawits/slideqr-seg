"""Procedural textures drawn in millimetre space.

A Canvas is a texture whose pixel (i, j) covers the mm square
origin + [i, i+1) / res. Colours are float32 BGR in [0, 1].
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import geometry as G

SHIFT = 4


class Canvas:
    def __init__(self, x0, y0, w_mm, h_mm, res, fill=(0, 0, 0), alpha=0.0):
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.w = max(1, int(np.ceil(w_mm * res)))
        self.h = max(1, int(np.ceil(h_mm * res)))
        self.img = np.empty((self.h, self.w, 3), np.float32)
        self.img[:] = fill
        self.alpha = np.full((self.h, self.w), alpha, np.float32)

    def tex_to_world(self):
        """Edge-based texture pixel coords -> world mm."""
        return G.translate(self.x0, self.y0) @ G.scale(1.0 / self.res)

    def px(self, pts_mm):
        """mm -> fixed-point pixel-centre coords for cv2 drawing."""
        p = (np.asarray(pts_mm, float) - (self.x0, self.y0)) * self.res - 0.5
        return np.round(p * (1 << SHIFT)).astype(np.int32)

    def _roi(self, pts_fixed, pad):
        p = pts_fixed >> SHIFT
        x0 = max(0, int(p[:, 0].min()) - pad)
        y0 = max(0, int(p[:, 1].min()) - pad)
        x1 = min(self.w, int(p[:, 0].max()) + pad + 2)
        y1 = min(self.h, int(p[:, 1].max()) + pad + 2)
        return x0, y0, x1, y1

    def fill_poly(self, pts_mm, color, alpha=1.0):
        """Fill a polygon; returns the full-size coverage map."""
        pts = self.px(pts_mm)
        full = np.zeros((self.h, self.w), np.float32)
        x0, y0, x1, y1 = self._roi(pts, 2)
        if x1 <= x0 or y1 <= y0:
            return full
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0, y0], np.int32) << SHIFT
        cv2.fillPoly(mask, [pts - off], 255, cv2.LINE_AA, SHIFT)
        a = (mask.astype(np.float32) / 255.0) * alpha
        self.blend(a, color, (x0, y0))
        full[y0:y1, x0:x1] = a
        return full

    def blend(self, a, color, origin=(0, 0)):
        x0, y0 = origin
        h, w = a.shape
        img = self.img[y0:y0 + h, x0:x0 + w]
        a3 = a[..., None]
        img *= 1 - a3
        img += a3 * np.asarray(color, np.float32)
        al = self.alpha[y0:y0 + h, x0:x0 + w]
        np.maximum(al, a, out=al)

    def line(self, p0, p1, width_mm, color, alpha=1.0):
        th = max(1, int(round(width_mm * self.res)))
        pts = self.px([p0, p1])
        x0, y0, x1, y1 = self._roi(pts, th + 2)
        if x1 <= x0 or y1 <= y0:
            return
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        off = np.array([x0, y0], np.int32) << SHIFT
        q = pts - off
        cv2.line(mask, tuple(map(int, q[0])), tuple(map(int, q[1])), 255, th, cv2.LINE_AA, SHIFT)
        self.blend(mask.astype(np.float32) / 255.0 * alpha, color, (x0, y0))

    def mm_grid(self):
        xs = self.x0 + (np.arange(self.w) + 0.5) / self.res
        ys = self.y0 + (np.arange(self.h) + 0.5) / self.res
        return np.meshgrid(xs, ys)


def smooth_noise(shape, cell_px, rng, octaves=1):
    """Low-frequency noise in [-1, 1] with feature size ~cell_px."""
    h, w = shape
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    for o in range(octaves):
        c = max(1.0, cell_px / (2 ** o))
        gh, gw = int(np.ceil(h / c)) + 2, int(np.ceil(w / c)) + 2
        g = rng.standard_normal((gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(gw * c), int(gh * c)), interpolation=cv2.INTER_CUBIC)[:h, :w]
        if up.shape != (h, w):
            up = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= 0.5
    out /= total
    m = np.abs(out).max() + 1e-6
    return out / m


def rounded_rect(x, y, w, h, r, n=6):
    pts = []
    for cx, cy, a0 in [(x + w - r, y + r, -90), (x + w - r, y + h - r, 0),
                       (x + r, y + h - r, 90), (x + r, y + r, 180)]:
        for a in np.linspace(a0, a0 + 90, n):
            t = np.deg2rad(a)
            pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
    return np.array(pts)


# ---------------------------------------------------------------- background

def draw_background(cv, layout, rng):
    """Papers and folders inside a white wire basket, floor outside."""
    floor = rng.uniform(0.28, 0.45)
    cv.img[:] = (floor * 1.08, floor, floor * 0.95)
    n = smooth_noise((cv.h, cv.w), 30 * cv.res, rng, 2)
    cv.img *= (1 + 0.06 * n)[..., None]
    tx0, ty0, tw, th = layout["tray_rect"]
    cxm, cym = tx0 + tw / 2, ty0 + th / 2
    basket = (tx0 - rng.uniform(20, 60), ty0 - rng.uniform(20, 60),
              tw + rng.uniform(40, 120), th + rng.uniform(60, 160))
    # basket floor (white wire mesh) under the papers
    bx, by, bw, bh = basket
    cv.fill_poly([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], (0.55, 0.56, 0.56))
    sheets = []
    yellow = (0.42, 0.80, 0.93)
    for i in range(rng.integers(3, 7)):
        if i == 0 and rng.random() < layout["folder_prob"]:
            col, size = yellow, (320, 240)
        else:
            w = rng.uniform(0.86, 0.96)
            tint = rng.choice([0, 1, 2], p=[0.8, 0.1, 0.1])
            col = [(w, w, w * 0.99), (w * 0.85, w * 0.95, w), (w * 0.9, w * 0.88, w)][tint]
            size = (297, 210) if rng.random() < 0.5 else (210, 297)
        ang = rng.uniform(-8, 8)
        cx = cxm + rng.uniform(-0.4, 0.4) * tw
        cy = cym + rng.uniform(-0.3, 0.5) * th
        sheets.append((col, size, ang, cx, cy))
    for col, (sw, sh), ang, cx, cy in sheets:
        R = G.rotate(ang)
        rect = np.array([[-sw / 2, -sh / 2], [sw / 2, -sh / 2], [sw / 2, sh / 2], [-sw / 2, sh / 2]])
        pts = rect @ R[:2, :2].T + (cx, cy)
        cv.fill_poly(pts + (1.5, 2.5), (0.15, 0.15, 0.15), alpha=0.35)
        cv.fill_poly(pts, col)
        if rng.random() < 0.5:  # faint printed lines
            for k in range(rng.integers(5, 25)):
                yy = -sh / 2 + 20 + k * 7
                if yy > sh / 2 - 15:
                    break
                ln = rng.uniform(40, sw - 40)
                p0 = np.array([-sw / 2 + 20, yy]) @ R[:2, :2].T + (cx, cy)
                p1 = np.array([-sw / 2 + 20 + ln, yy]) @ R[:2, :2].T + (cx, cy)
                cv.line(p0, p1, 1.2, (0.45, 0.45, 0.45), alpha=0.35)
    if rng.random() < layout["wire_basket_prob"]:
        _draw_basket_wires(cv, basket, rng)
    cv.alpha[:] = 1.0


def _draw_basket_wires(cv, basket, rng):
    bx, by, bw, bh = basket
    pitch = rng.uniform(22, 38)
    wall = rng.uniform(30, 70)
    rod = rng.uniform(2.5, 4.0)
    col = np.array((0.86, 0.88, 0.88)) * rng.uniform(0.9, 1.05)
    # front and right walls seen from above: parallel rods outside the basket edge
    for k in range(int(wall // 12) + 1):
        y = by + bh + 4 + k * 12
        cv.line((bx - wall, y), (bx + bw + wall, y), rod, col * 0.6, 0.8)
        cv.line((bx - wall, y - 0.4), (bx + bw + wall, y - 0.4), rod * 0.5, col)
        x = bx + bw + 4 + k * 12
        cv.line((x, by - wall), (x, by + bh + wall), rod, col * 0.6, 0.8)
        cv.line((x - 0.4, by - wall), (x - 0.4, by + bh + wall), rod * 0.5, col)
    for x in np.arange(bx - wall, bx + bw + wall, pitch):
        cv.line((x, by + bh), (x, by + bh + wall), rod, col * 0.9)
    for y in np.arange(by - wall, by + bh + wall, pitch):
        cv.line((bx + bw, y), (bx + bw + wall, y), rod, col * 0.9)


# ---------------------------------------------------------------- tray

def draw_tray(cv, layout, rng):
    tx0, ty0, tw, th = layout["tray_rect"]
    outline = rounded_rect(tx0, ty0, tw, th, 4.0)
    base = rng.uniform(0.62, 0.74)
    tint = rng.uniform(0.0, 0.03)
    tray_a = cv.fill_poly(outline, (base * (1 + tint), base, base * (1 - tint)))
    # brushed finish: noise smeared along X
    k = max(3, int(8 * cv.res) | 1)
    n = rng.standard_normal((cv.h, cv.w)).astype(np.float32)
    n = cv2.blur(n, (k, 1))
    n /= n.std() + 1e-6
    big = smooth_noise((cv.h, cv.w), 40 * cv.res, rng, 2)
    # broad soft reflection bands across the steel
    X, Y = cv.mm_grid()
    ang = np.deg2rad(rng.uniform(-60, 60))
    proj = X * np.cos(ang) + Y * np.sin(ang)
    band = np.zeros_like(proj, dtype=np.float32)
    for _ in range(rng.integers(1, 3)):
        c = rng.uniform(proj.min(), proj.max())
        band += np.exp(-((proj - c) / rng.uniform(20, 70)) ** 2).astype(np.float32)
    gain = 1 + 0.025 * n + 0.05 * big + 0.12 * band
    cv.img *= np.where(tray_a[..., None] > 0, gain[..., None], 1.0)
    # bevelled edge
    cv.line((tx0 + 3, ty0 + 0.8), (tx0 + tw - 3, ty0 + 0.8), 1.0, (0.92, 0.92, 0.92), 0.6)
    cv.line((tx0 + 3, ty0 + th - 0.8), (tx0 + tw - 3, ty0 + th - 0.8), 1.2, (0.3, 0.3, 0.3), 0.6)
    cv.line((tx0 + 0.8, ty0 + 3), (tx0 + 0.8, ty0 + th - 3), 1.0, (0.85, 0.85, 0.85), 0.5)
    cv.line((tx0 + tw - 0.8, ty0 + 3), (tx0 + tw - 0.8, ty0 + th - 3), 1.0, (0.35, 0.35, 0.35), 0.5)
    # slot markers, holes and ruler ticks
    white = (0.95, 0.96, 0.96)
    for s, sx in enumerate(layout["slot_x"]):
        cx = sx + layout["slide_w"] / 2
        my = layout["slide_y0"] + layout["slide_h"] + 3.0
        cv.fill_poly([(cx - 2, my), (cx + 2, my), (cx + 2, my + 3), (cx - 2, my + 3)], white)
        hy = my + 10
        hole = np.array([(cx + 1.2 * np.cos(t), hy + 1.2 * np.sin(t)) for t in np.linspace(0, 2 * np.pi, 12)])
        cv.fill_poly(hole, (0.22, 0.22, 0.22), 0.9)
    ry = ty0 + th - 14
    for i, x in enumerate(np.arange(tx0 + 8, tx0 + tw - 8, 1.0)):
        L = 4.0 if i % 10 == 0 else (2.5 if i % 5 == 0 else 1.5)
        cv.line((x, ry), (x, ry + L), 0.18, (0.2, 0.2, 0.2), 0.8)
    for s, sx in enumerate(layout["slot_x"]):
        _text_mm(cv, str(s + 1), sx + layout["slide_w"] / 2 - 1, ry + 5.0, 3.0, (0.25, 0.25, 0.25))
    for _ in range(rng.integers(5, 25)):  # light scratches
        x, y = rng.uniform(tx0, tx0 + tw), rng.uniform(ty0, ty0 + th)
        a, L = rng.uniform(0, np.pi), rng.uniform(3, 30)
        cv.line((x, y), (x + L * np.cos(a), y + L * np.sin(a)), 0.15, (0.9, 0.9, 0.9), 0.25)
    cv.alpha[:] = tray_a


# ---------------------------------------------------------------- other surfaces

SURFACE_MATERIAL = {"steel_tray": "metal", "plastic_tray": "plastic", "desk": "wood",
                    "paper": "background", "slide_box": "plastic", "pla_base": "pla"}


def draw_surface(cv, layout, rng):
    kind = layout["surface"]
    fn = {"steel_tray": draw_tray, "plastic_tray": _plastic_tray, "desk": _desk, "paper": _paper_sheet,
          "slide_box": _slide_box, "pla_base": _pla_base}[kind]
    fn(cv, layout, rng)


def _shade_rect_edges(cv, x0, y0, x1, y1, w, light, dark, alpha):
    """Raised edge: highlight on top/left, shadow on bottom/right."""
    cv.line((x0, y0), (x1, y0), w, light, alpha)
    cv.line((x0, y0), (x0, y1), w, light, alpha)
    cv.line((x0, y1), (x1, y1), w, dark, alpha)
    cv.line((x1, y0), (x1, y1), w, dark, alpha)


def _apply_gain(cv, mask, gain):
    cv.img *= np.where(mask[..., None] > 0, gain[..., None], 1.0).astype(np.float32)


def _slot_dividers(cv, layout, y0, y1, light, dark, width, alpha):
    """Ribs between neighbouring slots (moulded trays, slide boxes)."""
    sw = layout["slide_w"]
    xs = layout["slot_x"]
    for a, b in zip(xs[:-1], xs[1:]):
        x = (a + sw + b) / 2
        cv.line((x - width * 0.35, y0), (x - width * 0.35, y1), width * 0.5, light, alpha)
        cv.line((x + width * 0.35, y0), (x + width * 0.35, y1), width * 0.5, dark, alpha)


def _plastic_tray(cv, layout, rng):
    x, y, w, h = layout["surface_rect"]
    rim = layout["surface_extra"]["rim_mm"]
    v = rng.uniform(0.86, 0.94) if layout["surface_variant"] == "white" else rng.uniform(0.5, 0.68)
    col = (v * rng.uniform(0.98, 1.02), v, v * rng.uniform(0.98, 1.01))
    a = cv.fill_poly(rounded_rect(x, y, w, h, 6.0), col)
    gain = 1 + 0.03 * smooth_noise((cv.h, cv.w), 30 * cv.res, rng, 2) \
        + 0.012 * rng.standard_normal((cv.h, cv.w)).astype(np.float32)
    _apply_gain(cv, a, gain)
    _shade_rect_edges(cv, x + 1, y + 1, x + w - 1, y + h - 1, 1.2, (1, 1, 1), (0.2, 0.2, 0.2), 0.35)
    _shade_rect_edges(cv, x + rim, y + rim, x + w - rim, y + h - rim, 0.8, (0.25, 0.25, 0.25), (1, 1, 1), 0.3)
    _slot_dividers(cv, layout, y + rim, y + h - rim, (1, 1, 1), (0.25, 0.25, 0.25), 1.6, 0.3)
    cv.alpha[:] = a


def _desk(cv, layout, rng):
    var = layout["surface_variant"]
    X, Y = cv.mm_grid()
    if var.startswith("laminate"):
        base = {"laminate_white": (0.88, 0.89, 0.89), "laminate_grey": (0.62, 0.63, 0.63),
                "laminate_beige": (0.70, 0.80, 0.86)}[var]
        cv.img[:] = np.asarray(base, np.float32) * rng.uniform(0.93, 1.05)
        speck = rng.standard_normal((cv.h, cv.w)).astype(np.float32)
        gain = 1 + 0.02 * smooth_noise((cv.h, cv.w), 60 * cv.res, rng, 2) + 0.015 * speck
    else:
        base = (0.36, 0.55, 0.75) if var == "wood_light" else (0.16, 0.27, 0.42)
        cv.img[:] = np.asarray(base, np.float32) * rng.uniform(0.9, 1.08)
        ang = np.deg2rad(rng.uniform(-10, 10))
        u = X * np.cos(ang) + Y * np.sin(ang)
        v = -X * np.sin(ang) + Y * np.cos(ang)
        warp = 6 * smooth_noise((cv.h, cv.w), 80 * cv.res, rng, 2)
        rings = np.sin((v + warp) * 2 * np.pi / rng.uniform(3, 8))
        k = max(3, int(25 * cv.res) | 1)
        streak = cv2.blur(rng.standard_normal((cv.h, cv.w)).astype(np.float32), (k, 1))
        streak /= streak.std() + 1e-6
        gain = 1 + 0.07 * rings + 0.05 * streak
        del u
    cv.img *= gain[..., None]
    cv.alpha[:] = 1.0


def _paper_sheet(cv, layout, rng):
    e = layout["surface_extra"]
    var = layout["surface_variant"]
    col = {"white_paper": (0.93, 0.94, 0.94), "yellow_folder": (0.42, 0.80, 0.93),
           "manila_folder": (0.58, 0.78, 0.86)}[var]
    col = np.asarray(col) * rng.uniform(0.93, 1.03)
    R = G.rotate(e["angle"])[:2, :2]
    rect = np.array([[-e["w"] / 2, -e["h"] / 2], [e["w"] / 2, -e["h"] / 2],
                     [e["w"] / 2, e["h"] / 2], [-e["w"] / 2, e["h"] / 2]]) @ R.T + (e["cx"], e["cy"])
    shadow = cv.fill_poly(rect + (1.5, 2.0), (0.1, 0.1, 0.1), alpha=0.3)
    a = cv.fill_poly(rect, col)
    gain = 1 + 0.015 * smooth_noise((cv.h, cv.w), 50 * cv.res, rng, 2)
    _apply_gain(cv, a, gain)
    if var == "white_paper" and rng.random() < 0.6:
        for k in range(int(rng.integers(6, 20))):
            yy = -e["h"] / 2 + 15 + k * 8
            if yy > e["h"] / 2 - 10:
                break
            ln = rng.uniform(60, e["w"] - 40)
            p0 = np.array([-e["w"] / 2 + 15, yy]) @ R.T + (e["cx"], e["cy"])
            p1 = np.array([-e["w"] / 2 + 15 + ln, yy]) @ R.T + (e["cx"], e["cy"])
            cv.line(p0, p1, 1.0, (0.5, 0.5, 0.5), alpha=0.25)
    cv.alpha[:] = np.maximum(a, shadow)


def _slide_box(cv, layout, rng):
    x, y, w, h = layout["surface_rect"]
    rim = layout["surface_extra"]["rim_mm"]
    v = rng.uniform(0.05, 0.11) if layout["surface_variant"] == "black" else rng.uniform(0.22, 0.36)
    a = cv.fill_poly(rounded_rect(x, y, w, h, 3.0), (v, v, v * 1.02))
    floor = [(x + rim, y + rim), (x + w - rim, y + rim), (x + w - rim, y + h - rim), (x + rim, y + h - rim)]
    cv.fill_poly(floor, (v * 0.85, v * 0.85, v * 0.87))
    gain = 1 + 0.05 * smooth_noise((cv.h, cv.w), 25 * cv.res, rng, 2) \
        + 0.02 * rng.standard_normal((cv.h, cv.w)).astype(np.float32)
    _apply_gain(cv, a, gain)
    hi = min(1.0, v + 0.25)
    _shade_rect_edges(cv, x + 0.8, y + 0.8, x + w - 0.8, y + h - 0.8, 1.0, (hi, hi, hi), (0, 0, 0), 0.5)
    _slot_dividers(cv, layout, y + rim, y + h - rim, (hi, hi, hi), (0, 0, 0), 2.0, 0.5)
    cv.alpha[:] = a


def _pla_base(cv, layout, rng):
    """Flat light-grey 3D-printed PLA plate: faint layer lines, a seam across the middle
    (printed in two halves), an engraved frame and short ticks on the frame's top edge.
    No grooves and no stops."""
    x, y, w, h = layout["surface_rect"]
    e = layout["surface_extra"]
    v = rng.uniform(0.70, 0.80)
    col = (v * rng.uniform(0.99, 1.03), v, v * rng.uniform(0.97, 1.0))
    a = cv.fill_poly(rounded_rect(x, y, w, h, 3.0), col)
    X, Y = cv.mm_grid()
    layer = rng.uniform(0.15, 0.25)  # layer height in mm; lines run along X
    lines = np.sin(Y * 2 * np.pi / layer + 0.4 * smooth_noise((cv.h, cv.w), 20 * cv.res, rng, 1))
    amp = 0.02 * float(np.clip((cv.res * layer - 2) / 2, 0, 1))  # fade out when under-sampled (no moire)
    seam_x = x + w / 2
    half = np.where(X < seam_x, 1.0, 1.0 + rng.uniform(-0.025, 0.025)).astype(np.float32)
    gain = half * (1 + amp * lines + 0.015 * smooth_noise((cv.h, cv.w), 15 * cv.res, rng, 2)
                   + 0.01 * rng.standard_normal((cv.h, cv.w)).astype(np.float32))
    _apply_gain(cv, a, gain)
    # seam between the two printed halves
    cv.line((seam_x, y + 0.5), (seam_x, y + h - 0.5), 0.35, (v * 0.55,) * 3, 0.8)
    cv.line((seam_x + 0.35, y + 0.5), (seam_x + 0.35, y + h - 0.5), 0.25, (min(1, v * 1.15),) * 3, 0.4)
    # engraved frame (thin groove: dark line + light lip) and ticks
    fx, fy, fw, fh = e["frame"]
    dark, lip = (v * 0.5,) * 3, (min(1, v * 1.15),) * 3
    for p0, p1 in [((fx, fy), (fx + fw, fy)), ((fx + fw, fy), (fx + fw, fy + fh)),
                   ((fx + fw, fy + fh), (fx, fy + fh)), ((fx, fy + fh), (fx, fy))]:
        cv.line(p0, p1, 0.4, dark, 0.85)
        cv.line(np.add(p0, 0.3), np.add(p1, 0.3), 0.25, lip, 0.4)
    n, pitch, L = e["n_ticks"], e["tick_pitch"], e["tick_len"]
    for i in range(n):  # ticks straddle the top frame line: half inside, half outside
        tx = fx + fw / 2 + (i - (n - 1) / 2) * pitch
        cv.line((tx, fy - L / 2), (tx, fy + L / 2), 0.4, dark, 0.85)
        cv.line((tx + 0.3, fy - L / 2), (tx + 0.3, fy + L / 2), 0.25, lip, 0.4)
    _shade_rect_edges(cv, x + 0.6, y + 0.6, x + w - 0.6, y + h - 0.6, 0.8, (1, 1, 1), (0.25, 0.25, 0.25), 0.3)
    cv.alpha[:] = a


def _text_mm(cv, text, x, y, h_mm, color):
    size_px = h_mm * cv.res
    if size_px < 4:
        return
    font_scale = size_px / 22.0
    org = cv.px([(x, y + h_mm)])[0]
    th = max(1, int(round(size_px / 10)))
    mask = np.zeros((cv.h, cv.w), np.uint8)
    cv2.putText(mask, text, (int(org[0] >> SHIFT), int(org[1] >> SHIFT)), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, 255, th, cv2.LINE_AA)
    cv.blend(mask.astype(np.float32) / 255 * 0.8, color)


# ---------------------------------------------------------------- slide glass + tissue

def draw_slide(slide, res, rng):
    """Returns (transmission BGR, glass alpha, highlight colour, highlight alpha, tissue alpha)
    on a canvas in slide-local mm (0..25 x 0..75, label end at y=0)."""
    w, h = slide["size"]
    cv = Canvas(0, 0, w, h, res, fill=(1, 1, 1))
    rect = [(0, 0), (w, 0), (w, h), (0, h)]
    glass_a = cv.fill_poly(rect, (1, 1, 1))
    g = rng.uniform(0.0, 0.04)
    trans = np.empty_like(cv.img)
    trans[:] = (0.95 - g * 0.3, 0.97, 0.93 - g)  # faint green-blue glass tint
    # tissue under the label (Beer-Lambert absorption)
    X, Y = cv.mm_grid()
    od = np.zeros((cv.h, cv.w, 3), np.float32)
    tissue_a = np.zeros((cv.h, cv.w), np.float32)
    eosin = np.array([0.15, 0.62, 0.08], np.float32)       # B, G, R optical density
    hema = np.array([0.30, 0.70, 0.65], np.float32)
    faint = slide["tissue_faint"]
    for p in slide["tissue"]:
        cx, cy, rx, ry, ang = p
        a = np.deg2rad(ang)
        u = ((X - cx) * np.cos(a) + (Y - cy) * np.sin(a)) / rx
        v = (-(X - cx) * np.sin(a) + (Y - cy) * np.cos(a)) / ry
        env = 1 - np.sqrt(u ** 2 + v ** 2)
        n = smooth_noise((cv.h, cv.w), 2.5 * res, rng, 3)
        m = np.clip((env + 0.35 * n) * 6, 0, 1).astype(np.float32)
        dens = rng.uniform(0.25, 0.9) * (0.25 if faint else 1.0)
        tex = 0.6 + 0.4 * smooth_noise((cv.h, cv.w), 0.6 * res, rng, 2)
        nuclei = np.clip(smooth_noise((cv.h, cv.w), 0.12 * res + 1, rng, 1) * 3 - 1.2, 0, 1)
        hfrac = rng.uniform(0.2, 0.8)
        od += (m * dens * tex)[..., None] * ((1 - hfrac) * eosin + hfrac * hema)
        od += (m * dens * nuclei * 0.8)[..., None] * hema
        tissue_a = np.maximum(tissue_a, m * min(1.0, dens * 2))
    trans *= np.exp(-od * 2.2)
    # glass edges: dark outer line, bright inner line
    edge = max(0.25, 1.5 / res)
    hi = np.zeros((cv.h, cv.w), np.float32)
    inner = [(edge, edge), (w - edge, edge), (w - edge, h - edge), (edge, h - edge)]
    ring = np.zeros((cv.h, cv.w), np.uint8)
    cv2.polylines(ring, [cv.px(rect)], True, 255, max(1, int(round(edge * res))), cv2.LINE_AA, SHIFT)
    trans *= (1 - 0.35 * ring.astype(np.float32) / 255)[..., None]
    ring[:] = 0
    cv2.polylines(ring, [cv.px(inner)], True, 255, max(1, int(round(edge * res * 0.6))), cv2.LINE_AA, SHIFT)
    hi = np.maximum(hi, ring.astype(np.float32) / 255 * 0.35)
    return trans, glass_a, hi, tissue_a


# ---------------------------------------------------------------- label

def load_font(path, size_px):
    # no silent fallback: a missing font would change every label without notice
    if not os.path.isfile(path):
        raise FileNotFoundError(f"label font {path} not found (sudo apt install fonts-dejavu, "
                                "or point scene.fonts.dir to a folder with the DejaVu .ttf files)")
    return ImageFont.truetype(path, size_px)


def draw_label(label, matrix, rng):
    """High-resolution label texture with the code aligned to the pixel grid.

    Returns (bgr float image, alpha, res px/mm, code corners in label mm [TL,TR,BR,BL]).
    """
    lw, lh = label["size"]
    mods = matrix.shape[0]
    mpx = label["px_per_module"]
    res = mpx * mods / label["code_mm"]
    W, H = int(round(lw * res)), int(round(lh * res))
    paper = label["paper"]
    img = np.empty((H, W, 3), np.float32)
    img[:] = paper
    img *= (1 + 0.012 * rng.standard_normal((H, W))).astype(np.float32)[..., None]
    # text
    pil = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(pil)
    code_x_px = label["code_x_px"]
    text_w_px = code_x_px - int(0.8 * res) - int(label["text_x_mm"] * res)
    y = label["text_y_mm"] * res
    font_px = int(label["font_mm"] * res)
    for i, line in enumerate(label["text"]):
        path = label["font_bold"] if i == 1 else label["font"]   # sample id in bold
        f = load_font(path, font_px)
        while f.getlength(line) > text_w_px and font_px > 6:
            font_px -= 1
            f = load_font(path, font_px)
        d.text((label["text_x_mm"] * res, y), line, fill=255, font=f)
        y += font_px * label["line_spacing"]
    ink = np.asarray(pil, np.float32)[..., None] / 255.0
    img = img * (1 - ink) + ink * label["ink"]
    # code: one integer block of mpx pixels per module
    cx, cy = label["code_x_px"], label["code_y_px"]
    block = np.kron(matrix.astype(np.float32), np.ones((mpx, mpx), np.float32))
    region = img[cy:cy + mods * mpx, cx:cx + mods * mpx]
    region[:] = region * (1 - block[..., None]) + block[..., None] * np.asarray(label["code_ink"], np.float32)
    alpha_cv = Canvas(0, 0, lw, lh, res)
    alpha_cv.w, alpha_cv.h = W, H
    alpha_cv.alpha = np.zeros((H, W), np.float32)
    alpha_cv.img = np.zeros((H, W, 3), np.float32)
    a = alpha_cv.fill_poly(rounded_rect(0, 0, lw, lh, 0.8), (1, 1, 1))
    x0, y0 = cx / res, cy / res
    s = mods * mpx / res
    corners = np.array([[x0, y0], [x0 + s, y0], [x0 + s, y0 + s], [x0, y0 + s]])
    return img, a, res, corners


def label_stain(shape, res, rng, strength):
    """Multiplicative stain / dirt map for a label texture."""
    h, w = shape
    out = np.ones((h, w, 3), np.float32)
    for _ in range(rng.integers(1, 4)):
        n = smooth_noise((h, w), rng.uniform(1.5, 5) * res, rng, 3)
        m = np.clip((n - rng.uniform(0.2, 0.6)) * 4, 0, 1)
        col = np.array(rng.choice([(0.55, 0.8, 0.95), (0.8, 0.7, 0.6), (0.7, 0.7, 0.7), (0.85, 0.6, 0.85)]),
                       np.float32)
        s = strength * rng.uniform(0.5, 1.0)
        out *= 1 - (s * m)[..., None] * (1 - col)
    return out


def label_wrinkle(shape, res, rng, strength):
    """Shading of creases / wrinkles (no geometric displacement)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = np.zeros((h, w), np.float32)
    for _ in range(rng.integers(1, 4)):
        a = rng.uniform(0, np.pi)
        c = rng.uniform(0, max(w, h))
        d = (xx * np.cos(a) + yy * np.sin(a) - c) / (rng.uniform(0.4, 1.5) * res)
        shade += np.tanh(d) * np.exp(-d ** 2 / 8)
    shade += 0.5 * smooth_noise((h, w), 3 * res, rng, 2)
    return np.repeat((1 + strength * shade)[..., None], 3, axis=2).astype(np.float32)
