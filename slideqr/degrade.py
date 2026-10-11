"""Image degradations. Each one is toggled and scaled independently from config.

sample_params() draws every parameter up front so they can be logged to
metadata; apply() is deterministic given (params, rng).
"""
import cv2
import numpy as np

from . import draw as D
from .render import MAT


def _on(cfg, name, rng, p=None):
    c = cfg["degrade"][name]
    return bool(c["enabled"]) and rng.random() < (c["p"] if p is None else p)


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def sample_params(cfg, mode, rng, n_labels):
    dc = cfg["degrade"]
    P = {}
    stains = {}
    for i in range(n_labels):
        sp = {}
        if _on(cfg, "label_stain", rng):
            sp["stain"] = _u(rng, dc["label_stain"]["strength"])
        if _on(cfg, "label_wrinkle", rng):
            sp["wrinkle"] = _u(rng, dc["label_wrinkle"]["strength"])
        if sp:
            stains[i] = sp
    P["label_stain_count"] = sum("stain" in v for v in stains.values())
    P["label_wrinkle_count"] = sum("wrinkle" in v for v in stains.values())
    P["color_temp_k"] = _u(rng, dc["color_temp"]["kelvin"]) if _on(cfg, "color_temp", rng) else 0.0
    P["uneven_light"] = _u(rng, dc["uneven_light"]["strength"]) if _on(cfg, "uneven_light", rng) else 0.0
    P["shadow"] = _u(rng, dc["shadow"]["strength"]) if _on(cfg, "shadow", rng) else 0.0
    P["glare"] = _u(rng, dc["glare"]["strength"]) if _on(cfg, "glare", rng) else 0.0
    P["defocus_sigma"] = _u(rng, dc["defocus"]["sigma_px"]) if _on(cfg, "defocus", rng) else 0.0
    mp = dc["motion_blur"]["p_handheld"] if mode == "handheld" else dc["motion_blur"]["p"]
    P["motion_len"] = _u(rng, dc["motion_blur"]["length_px"]) if _on(cfg, "motion_blur", rng, mp) else 0.0
    P["motion_angle"] = float(rng.uniform(0, 180))
    P["vignetting"] = _u(rng, dc["vignetting"]["strength"]) if _on(cfg, "vignetting", rng) else 0.0
    if _on(cfg, "noise", rng):
        P["noise_sigma"] = _u(rng, dc["noise"]["sigma"])
        P["noise_shot"] = _u(rng, dc["noise"]["shot"])
    else:
        P["noise_sigma"] = P["noise_shot"] = 0.0
    P["jpeg_quality"] = int(rng.integers(dc["jpeg"]["quality"][0], dc["jpeg"]["quality"][1] + 1)) \
        if _on(cfg, "jpeg", rng) else 0
    return P, stains


def kelvin_gain(k):
    """Approximate BGR gain of a blackbody illuminant relative to 6500 K."""
    def rgb(t):
        t = t / 100.0
        r = 255.0 if t <= 66 else 329.7 * (t - 60) ** -0.1332
        g = 99.47 * np.log(t) - 161.1 if t <= 66 else 288.1 * (t - 60) ** -0.0755
        b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5 * np.log(t - 10) - 305.0)
        return np.array([b, g, r], np.float32)
    g = rgb(k) / rgb(6500)
    return g / g.max()


def apply(img, material, P, cfg, rng):
    H, W = img.shape[:2]
    out = img.astype(np.float32).copy()
    if P["color_temp_k"]:
        # camera auto white balance removes most of the cast; keep ~half of it
        out *= kelvin_gain(P["color_temp_k"]) ** 0.5
    if P["uneven_light"]:
        n = D.smooth_noise((H // 8, W // 8), max(H, W) / 8 / 2, rng, 1)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        out *= (1 + P["uneven_light"] * n)[..., None]
    if P["shadow"]:
        n = D.smooth_noise((H // 8, W // 8), max(H, W) / 8 / 3, rng, 1)
        n = cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)
        m = 1 / (1 + np.exp(-(n - rng.uniform(-0.2, 0.4)) * 12))
        out *= (1 - P["shadow"] * m)[..., None]
    if P["glare"]:
        out = _glare(out, material, P["glare"], cfg, rng)
    if P["defocus_sigma"]:
        out = cv2.GaussianBlur(out, (0, 0), P["defocus_sigma"])
    if P["motion_len"]:
        out = _motion_blur(out, P["motion_len"], P["motion_angle"])
    if P["vignetting"]:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / ((W / 2) ** 2 + (H / 2) ** 2)
        out *= (1 - P["vignetting"] * r2)[..., None]
    out = np.clip(out, 0, 1) * 255
    if P["noise_sigma"]:
        sig = np.sqrt(P["noise_sigma"] ** 2 + P["noise_shot"] * 255 * out)
        out = out + rng.standard_normal(out.shape).astype(np.float32) * sig
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _glare(img, material, strength, cfg, rng):
    H, W = img.shape[:2]
    refl_cfg = cfg["degrade"]["glare"]["reflectivity"]
    refl = np.zeros(len(MAT), np.float32)
    for k, v in MAT.items():
        refl[v] = refl_cfg[k]
    rmap = refl[material]
    yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4
    g = np.zeros((H // 4, W // 4), np.float32)
    # broad bands from overhead tube lights
    for _ in range(rng.integers(1, 3)):
        a = rng.uniform(0, np.pi)
        proj = xx * np.cos(a) + yy * np.sin(a)
        c = rng.uniform(proj.min(), proj.max())
        g += np.exp(-((proj - c) / (rng.uniform(0.04, 0.15) * max(H, W))) ** 2)
    # a few compact hot spots (point lights)
    for _ in range(rng.integers(0, 3)):
        cx, cy = rng.uniform(0, W), rng.uniform(0, H)
        s = rng.uniform(0.02, 0.08) * max(H, W)
        g += 1.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    glare = np.clip(strength * g * rmap, 0, 1.5)
    # screen blend; >1 saturates (blown highlights)
    return 1 - (1 - img) * np.clip(1 - glare, 0, 1)[..., None]


def _motion_blur(img, length, angle):
    L = max(1.0, length)
    k = int(np.ceil(L)) | 1
    k = max(k, 3)
    kern = np.zeros((k, k), np.float32)
    c = (k - 1) / 2
    a = np.deg2rad(angle)
    for t in np.linspace(-L / 2, L / 2, int(L * 4) + 1):
        x, y = c + t * np.cos(a), c + t * np.sin(a)
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for dx, dy, w in [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]:
            if 0 <= x0 + dx < k and 0 <= y0 + dy < k:
                kern[y0 + dy, x0 + dx] += w
    kern /= kern.sum()
    return cv2.filter2D(img, -1, kern, borderType=cv2.BORDER_REFLECT)
