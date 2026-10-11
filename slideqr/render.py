"""Compose a scene into an image and compute exact annotations.

Every object is drawn as a texture in its own mm frame and warped to the image
ONCE with the full homography (camera x placement x texture scale), so the code
corners we report are the exact image of the printed symbol corners.
"""
import cv2
import numpy as np

from . import draw as D
from . import geometry as G
from . import scene as S

MAT = {"background": 0, "metal": 1, "glass": 2, "label": 3, "tissue": 4, "hand": 5,
       "plastic": 6, "wood": 7, "pla": 8}
MAX_TEX_PIXELS = 24e6
SKIN = [(0.78, 0.86, 0.96), (0.62, 0.74, 0.90), (0.50, 0.63, 0.82), (0.38, 0.50, 0.70),
        (0.27, 0.37, 0.55), (0.18, 0.25, 0.40)]   # BGR, light to dark


class Frame:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.img = np.zeros((H, W, 3), np.float32)
        self.mat = np.zeros((H, W), np.uint8)
        self.occ = np.zeros((H, W), np.float32)    # things above the labels (hand, paper)

    def warp(self, tex, H_img_from_tex, interp=cv2.INTER_LINEAR):
        """Warp texture into the bounding box it covers. Returns (out, (x0,y0,x1,y1)) or None."""
        h, w = tex.shape[:2]
        corners = G.apply_h(H_img_from_tex, [(0, 0), (w, 0), (w, h), (0, h)])
        x0, y0 = np.floor(corners.min(0)).astype(int) - 2
        x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, self.W), min(y1, self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        Hc = G.to_cv(G.translate(-x0, -y0) @ H_img_from_tex)
        out = cv2.warpPerspective(tex, Hc, (x1 - x0, y1 - y0), flags=interp,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out, (x0, y0, x1, y1)

    def over(self, color, alpha, H, material=None):
        """Alpha-composite a texture (premultiplied on the fly)."""
        tex = np.concatenate([color * alpha[..., None], alpha[..., None]], axis=2)
        r = self.warp(tex, H)
        if r is None:
            return None
        out, (x0, y0, x1, y1) = r
        a = np.clip(out[..., 3:4], 0, 1)
        roi = self.img[y0:y1, x0:x1]
        roi *= 1 - a
        roi += out[..., :3]
        if material is not None:
            self.mat[y0:y1, x0:x1][a[..., 0] > 0.5] = material
        return a[..., 0], (x0, y0, x1, y1)

    def multiply(self, trans, alpha, H):
        tex = np.concatenate([trans, alpha[..., None]], axis=2)
        r = self.warp(tex, H)
        if r is None:
            return
        out, (x0, y0, x1, y1) = r
        a = np.clip(out[..., 3:4], 0, 1)
        t = out[..., :3] / np.maximum(a, 1e-4)
        roi = self.img[y0:y1, x0:x1]
        roi *= 1 - a + a * t


def _visible_bbox(cam, z, margin=10.0):
    Hinv = np.linalg.inv(cam.plane_h(z))
    pts = G.apply_h(Hinv, [(0, 0), (cam.W, 0), (cam.W, cam.H), (0, cam.H)])
    return pts.min(0) - margin, pts.max(0) + margin


def _world_canvas(cam, z, rect, oversample):
    """Canvas over (visible region of plane z) âˆ© rect with a resolution matching the image."""
    lo, hi = _visible_bbox(cam, z)
    if rect is not None:
        lo = np.maximum(lo, rect[:2])
        hi = np.minimum(hi, (rect[0] + rect[2], rect[1] + rect[3]))
    if np.any(hi <= lo):
        return None
    Hp = cam.plane_h(z)
    centre = (lo + hi) / 2
    res = max(G.local_scale(Hp, p)[1] for p in [lo, hi, (lo[0], hi[1]), (hi[0], lo[1]), centre])
    res = res * oversample
    area = (hi - lo).prod()
    res = min(res, np.sqrt(MAX_TEX_PIXELS / area))
    return D.Canvas(lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1], res)


def _label_px_per_module(cam, slide, oversample):
    A = S.slide_to_world(slide)
    centre = G.apply_h(A, [(12.5, 10.0)])[0]
    _, smax = G.local_scale(cam.plane_h(S.Z_LABEL), centre)
    ppm = smax * slide["label"]["code_mm"] / slide["modules"]
    return int(min(48, max(6, np.ceil(ppm * oversample * 1.5))))


def render_scene(cfg, layout, cam, rng, label_overlays=True, stain_params=None):
    """Render a clean scene. Returns dict with image, material map, occluder map,
    per-code annotations and the label overlay (stain/wrinkle multiplier in image space)."""
    W, H = cam.W, cam.H
    over = cfg["render"]["texture_oversample"]
    fr = Frame(W, H)

    # 1. background papers / basket
    cvb = _world_canvas(cam, S.Z_BACKGROUND, None, over)
    D.draw_background(cvb, layout, rng)
    fr.over(cvb.img, cvb.alpha, cam.plane_h(S.Z_BACKGROUND) @ cvb.tex_to_world(), MAT["background"])

    # 2. surface the slides lie on (tray, desk, paper, slide box, PLA base)
    cvt = _world_canvas(cam, S.Z_TRAY, layout["surface_rect"], over)
    if cvt is not None:
        D.draw_surface(cvt, layout, rng)
        fr.over(cvt.img, cvt.alpha, cam.plane_h(S.Z_TRAY) @ cvt.tex_to_world(),
                MAT[D.SURFACE_MATERIAL[layout["surface"]]])

    # 3. slides + labels
    overlay = np.ones((H, W, 3), np.float32)
    codes = []
    for si, s in enumerate(layout["slides"]):
        A = S.slide_to_world(s)
        Hs = cam.plane_h(S.Z_SLIDE) @ A
        Hl = cam.plane_h(S.Z_LABEL) @ A @ S.label_to_slide(s)
        outline = [(0, 0), (25, 0), (25, 75), (0, 75)]
        if not G.projects_into(Hs, outline, W, H, margin=2):
            # entirely outside the frame (or behind the camera): annotate, skip drawing
            cl = S.code_corners_label(s)
            codes.append(dict(slide_index=si, slot=s["slot"], symbology=s["symbology"], modules=s["modules"],
                              payload=s["payload"],
                              corners=_project_or_far(Hl, cl), slide_corners=_project_or_far(Hs, outline),
                              matrix=s["matrix"]))
            continue
        smax = max(G.local_scale(Hs, p)[1] for p in [(0, 0), (25, 75)])
        res = min(smax * over, 20.0)
        trans, ga, hi, ta = D.draw_slide(s, res, rng)
        Ht = Hs @ G.scale(1.0 / res)
        fr.multiply(trans, ga, Ht)
        r = fr.over(np.ones_like(trans), hi, Ht)
        rg = fr.warp(np.stack([ga, ta], -1), Ht)
        if rg is not None:
            out, (x0, y0, x1, y1) = rg
            m = fr.mat[y0:y1, x0:x1]
            m[out[..., 0] > 0.5] = MAT["glass"]
            m[out[..., 1] > 0.5] = MAT["tissue"]

        lab = s["label"]
        lab["px_per_module"] = _label_px_per_module(cam, s, over)
        res_hi = lab["px_per_module"] * s["modules"] / lab["code_mm"]
        lab["code_x_px"] = int(round(lab["code_x_mm"] * res_hi))
        lab["code_y_px"] = int(round(lab["code_y_mm"] * res_hi))
        limg, la, lres, corners_label = D.draw_label(lab, s["matrix"], rng)
        Hl = cam.plane_h(S.Z_LABEL) @ A @ S.label_to_slide(s)
        lh, lw = limg.shape[:2]
        Htex = Hl @ G.scale(lab["size"][0] / lw, lab["size"][1] / lh)
        # soft contact shadow of the sticker edge
        fr.over(np.zeros_like(limg), la * 0.25, Hl @ G.translate(0.25, 0.35) @ G.scale(lab["size"][0] / lw, lab["size"][1] / lh))
        fr.over(limg, la, Htex, MAT["label"])
        if label_overlays and stain_params is not None:
            mult = np.ones_like(limg)
            sp = stain_params.get(si, {})
            if "stain" in sp:
                mult *= D.label_stain((lh, lw), lres, rng, sp["stain"])
            if "wrinkle" in sp:
                mult *= D.label_wrinkle((lh, lw), lres, rng, sp["wrinkle"])
            if "stain" in sp or "wrinkle" in sp:
                rw = fr.warp(np.concatenate([mult, la[..., None]], 2), Htex)
                if rw is not None:
                    out, (x0, y0, x1, y1) = rw
                    a = np.clip(out[..., 3:4], 0, 1)
                    t = out[..., :3] / np.maximum(a, 1e-4)
                    overlay[y0:y1, x0:x1] *= 1 - a + a * t
        corners_img = G.apply_h(Hl, corners_label)
        slide_img = G.apply_h(cam.plane_h(S.Z_SLIDE) @ A, [(0, 0), (25, 0), (25, 75), (0, 75)])
        codes.append(dict(slide_index=si, slot=s["slot"], symbology=s["symbology"], modules=s["modules"],
                          payload=s["payload"],
                          corners=corners_img, slide_corners=slide_img, matrix=s["matrix"]))

    # 4. foreground occluders (image space, above the labels)
    occ_params = {}
    if rng.random() < cfg["scene"]["paper_edge_prob"]:
        occ_params["paper_edge"] = _paper_edge(fr, rng)
    if rng.random() < cfg["scene"]["hand"]["prob"]:
        s_img = G.local_scale(cam.plane_h(0), cam.target)[0]
        occ_params["hand"] = _hand(fr, rng, s_img, cfg["scene"]["hand"]["max_fingers"])

    # 5. visibility of each code and slide
    for c in codes:
        c.update(_visibility(c["corners"], fr.occ, W, H))
        sv = _visibility(c["slide_corners"], fr.occ, W, H)
        c["slide_visible_frac"] = sv["visible_frac"]
        c["slide_poly_clipped"] = sv["poly_clipped"]
    return dict(image=np.clip(fr.img, 0, 1), material=fr.mat, occluder=fr.occ, codes=codes,
                overlay=overlay, occluder_params=occ_params)


def _project_or_far(H, pts):
    p = np.column_stack([np.asarray(pts, float), np.ones(len(pts))]) @ H.T
    if np.any(p[:, 2] <= 1e-9):
        return np.full((len(pts), 2), -1e5)  # behind the camera: zero area, never in frame
    return p[:, :2] / p[:, 2:3]


def _visibility(corners, occ, W, H):
    area = G.polygon_area(corners)
    clipped = G.clip_polygon_rect(corners, 0, 0, W, H)
    in_frame = G.polygon_area(clipped) / max(area, 1e-9)
    occluded = 0.0
    if len(clipped) >= 3:
        x0, y0 = np.floor(clipped.min(0)).astype(int)
        x1, y1 = np.ceil(clipped.max(0)).astype(int)
        x1, y1 = min(x1, W), min(y1, H)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.round((clipped - (x0, y0) - 0.5) * 16).astype(np.int32)
        cv2.fillPoly(m, [pts], 255, cv2.LINE_AA, 4)
        w = m.astype(np.float32) / 255
        if w.sum() > 0:
            occluded = float((w * (occ[y0:y1, x0:x1] > 0.5)).sum() / w.sum()) * in_frame
    return dict(in_frame_frac=float(in_frame), occluded_frac=float(occluded),
                visible_frac=float(max(0.0, in_frame - occluded)), poly_clipped=clipped)


def _paper_edge(fr, rng):
    W, H = fr.W, fr.H
    side = rng.choice(["left", "right", "bottom", "top"])
    depth = rng.uniform(0.04, 0.15)
    tilt = rng.uniform(-0.08, 0.08)
    if side in ("left", "right"):
        d0, d1 = depth * W, (depth + tilt) * W
        xs = (d0, d1) if side == "left" else (W - d0, W - d1)
        edge = (0, xs[0]) if side == "left" else (W, xs[0])
        pts = np.array([(edge[0], -5), (xs[0], -5), (xs[1], H + 5), (edge[0], H + 5)])
    else:
        d0, d1 = depth * H, (depth + tilt) * H
        ys = (H - d0, H - d1) if side == "bottom" else (d0, d1)
        e = H + 5 if side == "bottom" else -5
        pts = np.array([(-5, e), (-5, ys[0]), (W + 5, ys[1]), (W + 5, e)])
    shade = np.zeros((H, W), np.uint8)
    cv2.fillPoly(shade, [np.round((pts - 0.5) * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    a = shade.astype(np.float32) / 255
    sh = cv2.GaussianBlur(a, (0, 0), 8)
    fr.img *= (1 - 0.35 * np.clip(sh - a, 0, 1))[..., None]
    w = rng.uniform(0.85, 0.97)
    col = np.array((w * 0.98, w, w), np.float32)
    fr.img[:] = fr.img * (1 - a[..., None]) + a[..., None] * col * (1 + 0.02 * D.smooth_noise((H, W), 200, rng))[..., None]
    fr.mat[a > 0.5] = MAT["background"]
    fr.occ = np.maximum(fr.occ, a)
    return dict(side=str(side), depth=float(depth))


def _hand(fr, rng, px_per_mm, max_fingers):
    """Fingers entering from an image edge, skin tone, cylindrical shading, defocused."""
    W, H = fr.W, fr.H
    s = px_per_mm * rng.uniform(1.05, 1.3)  # hand is above the tray -> slightly larger
    side = rng.choice(["left", "right", "bottom", "top"])
    nrm = {"left": (1, 0), "right": (-1, 0), "bottom": (0, -1), "top": (0, 1)}[side]
    nrm = np.array(nrm, float)
    tang = np.array([-nrm[1], nrm[0]])
    along = rng.uniform(0.15, 0.85) * (H if side in ("left", "right") else W)
    entry = {"left": (0, along), "right": (W, along), "bottom": (along, H), "top": (along, 0)}[side]
    entry = np.array(entry, float)
    ang = np.deg2rad(rng.uniform(-30, 30))
    d = nrm * np.cos(ang) + tang * np.sin(ang)
    perp = np.array([-d[1], d[0]])
    mask = np.zeros((H, W), np.uint8)
    nf = int(rng.integers(1, max_fingers + 1))
    fw = rng.uniform(15, 19) * s
    reach = rng.uniform(25, 70) * s
    tips = []
    palm_c = entry - d * (45 * s)
    cv2.ellipse(mask, tuple(np.round(palm_c).astype(int)), (int(55 * s), int(45 * s)),
                float(np.rad2deg(np.arctan2(d[1], d[0]))), 0, 360, 255, -1, cv2.LINE_AA)
    for k in range(nf):
        off = (k - (nf - 1) / 2) * fw * 1.05
        L = reach * rng.uniform(0.8, 1.1) * (1 - 0.12 * abs(k - (nf - 1) / 2))
        base = entry - d * (20 * s) + perp * off
        tip = base + d * (L + 20 * s)
        cv2.line(mask, tuple(np.round(base).astype(int)), tuple(np.round(tip).astype(int)), 255,
                 int(fw), cv2.LINE_AA)
        tips.append(tip)
    a = mask.astype(np.float32) / 255
    if a.sum() == 0:
        return None
    # cast shadow
    shift = (rng.uniform(-1, 1, 2) * 12 * s).astype(np.float32)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    sh = cv2.warpAffine(a, M, (W, H))
    sh = cv2.GaussianBlur(sh, (0, 0), max(2.0, 6 * s))
    fr.img *= (1 - rng.uniform(0.25, 0.5) * sh * (1 - a))[..., None]
    tone = np.array(SKIN[int(rng.integers(0, len(SKIN)))], np.float32) * rng.uniform(0.92, 1.08)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    shade = 0.62 + 0.38 * np.sqrt(np.clip(dist / (fw / 2), 0, 1))
    tex = 1 + 0.04 * D.smooth_noise((H, W), 6 * s, rng, 2)
    col = tone[None, None, :] * (shade * tex)[..., None]
    for tip in tips:  # nails
        nail = np.zeros((H, W), np.uint8)
        c = tip - d * (8 * s)
        cv2.ellipse(nail, tuple(np.round(c).astype(int)), (int(6 * s), int(5 * s)),
                    float(np.rad2deg(np.arctan2(d[1], d[0]))), 0, 360, 255, -1, cv2.LINE_AA)
        na = (nail.astype(np.float32) / 255 * a)[..., None]
        col = col * (1 - na * 0.6) + na * 0.6 * np.minimum(1, tone * 1.25 + 0.1)
    blur = rng.uniform(1.0, 4.0) * max(1.0, s / 8)
    colp = cv2.GaussianBlur(col * a[..., None], (0, 0), blur)   # premultiplied blur, no dark halo
    a = cv2.GaussianBlur(a, (0, 0), blur)
    fr.img[:] = fr.img * (1 - a[..., None]) + colp
    fr.mat[a > 0.5] = MAT["hand"]
    fr.occ = np.maximum(fr.occ, a)
    return dict(side=str(side), fingers=nf, skin_index=int(np.argmin(np.abs(np.array(SKIN)[:, 1] - tone[1]))))
