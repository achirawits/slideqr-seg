"""Random scene layout and camera sampling (geometry only, no pixels)."""
import os

import numpy as np

from . import codes as C
from . import geometry as G

STAINS = ["H&E", "H&E", "H&E", "PAS", "GMS", "AFB", "IHC CK7", "IHC CD20", "Ki-67", "Masson"]
HEADERS = ["Pathology", "Surg Path", "Histology", "Path Lab", "Cytology"]

Z_BACKGROUND = 12.0   # papers lie below the tray top
Z_TRAY = 0.0
Z_SLIDE = -1.0        # top surface of the 1 mm glass
Z_LABEL = -1.1


def _u(rng, r):
    return float(rng.uniform(r[0], r[1]))


def _choose_symbology(cfg, rng):
    s = cfg["code"]["symbology"]
    if s == "mixed":
        return "qr" if rng.random() < cfg["code"]["mixed_qr_fraction"] else "datamatrix"
    return s


def _random_date(rng):
    return f"{rng.integers(1, 29):02d}/{rng.integers(1, 13):02d}/{rng.integers(2023, 2027)}"


SURFACE_VARIANTS = {
    "pla_base": ["light_grey"],
    "steel_tray": ["brushed"],
    "plastic_tray": ["white", "grey"],
    "desk": ["laminate_white", "laminate_grey", "laminate_beige", "wood_light", "wood_dark"],
    "paper": ["white_paper", "yellow_folder", "manila_folder"],
    "slide_box": ["black", "grey"],
}


def _sample_surface_kind(sc, rng):
    types = list(sc["surface"]["types"])
    w = np.asarray(sc["surface"]["weights"], float)
    if len(w) != len(types) or w.sum() <= 0:
        raise ValueError("scene.surface.weights must match scene.surface.types")
    kind = str(types[int(rng.choice(len(types), p=w / w.sum()))])
    if kind not in SURFACE_VARIANTS:
        raise ValueError(f"unknown surface type {kind}")
    return kind, str(rng.choice(SURFACE_VARIANTS[kind]))


def sample_layout(cfg, rng, n_slides=None):
    sc = cfg["scene"]
    sw, sh = sc["slide_mm"]
    n = int(n_slides) if n_slides is not None else int(rng.integers(sc["min_slides"], sc["max_slides"] + 1))
    surface, variant = _sample_surface_kind(sc, rng)
    if surface == "pla_base":
        geom = _pla_geometry(sc, rng, n, sw, sh)
    else:
        geom = _slot_geometry(sc, rng, n, sw, sh, surface)
    slides = [dict(pose, size=(sw, sh), **_slide_content(cfg, rng, sw, sh)) for pose in geom.pop("poses")]
    return dict(slides=slides, slide_w=sw, slide_h=sh, surface=surface, surface_variant=variant,
                folder_prob=sc["background"]["folder_prob"],
                wire_basket_prob=sc["background"]["wire_basket_prob"], **geom)


def _slide_content(cfg, rng, sw, sh):
    """Code, label text and tissue of one slide (independent of where it lies)."""
    sc = cfg["scene"]
    code_mm = float(cfg["code"]["CODE_SIZE_MM"])
    lw, lh = sc["label_mm"]
    sym = _choose_symbology(cfg, rng)
    mods = C.modules_for(cfg, sym)
    sample_id = C.random_sample_id(rng)
    payload = C.make_payload(cfg, rng, sample_id)
    matrix = C.encode_matrix(sym, payload, mods, cfg["code"]["qr_error"])
    quiet = cfg["code"]["quiet_zone_modules"] * code_mm / mods
    n_lines = int(rng.integers(sc["label_text_lines"][0], sc["label_text_lines"][1] + 1))
    lines = [rng.choice(HEADERS), sample_id, rng.choice(STAINS), _random_date(rng)]
    if n_lines == 3:
        lines.pop(int(rng.choice([0, 3])))
    fc = sc["fonts"]
    regular, bold = fc["families"][int(rng.integers(0, len(fc["families"])))]
    right = max(quiet, 1.0) + rng.uniform(0, 0.6)
    code_x = lw - right - code_mm
    free_y = lh - code_mm
    code_y = np.clip(free_y / 2 + rng.uniform(-1.5, 1.5), max(quiet, 0.8), max(quiet, free_y - max(quiet, 0.8)))
    pieces = []
    for _ in range(rng.integers(sc["tissue"]["pieces"][0], sc["tissue"]["pieces"][1] + 1)):
        rx, ry = rng.uniform(2.5, 9), rng.uniform(3, 16)
        pieces.append((rng.uniform(5, sw - 5), rng.uniform(lh + 6 + ry * 0.6, sh - 4 - ry * 0.6),
                       rx, ry, rng.uniform(-40, 40)))
    p = rng.uniform(0.86, 0.97)
    return dict(
        tissue=pieces,
        tissue_faint=bool(rng.random() < sc["tissue"]["faint_prob"]),
        symbology=sym, modules=mods, payload=payload, sample_id=sample_id, matrix=matrix,
        label=dict(size=(lw, lh), code_mm=code_mm, code_x_mm=code_x, code_y_mm=code_y,
                   text=lines, font=os.path.join(fc["dir"], regular), font_bold=os.path.join(fc["dir"], bold),
                   font_mm=rng.uniform(1.6, 2.1), line_spacing=rng.uniform(1.15, 1.35),
                   text_x_mm=rng.uniform(0.8, 1.4), text_y_mm=rng.uniform(1.0, 2.5),
                   paper=(p * rng.uniform(0.96, 1.0), p, p * rng.uniform(0.99, 1.02)),
                   ink=rng.uniform(0.05, 0.2), code_ink=rng.uniform(0.04, 0.15)))


# ---------------------------------------------------------------- slot surfaces (fixed pitch)

def _slot_geometry(sc, rng, n, sw, sh, surface):
    """Tray-like surfaces: slides sit in slots at a fixed pitch, small tilt."""
    pitch = float(sc["slide_pitch_mm"])
    gap = pitch - sw
    if gap <= 0:
        raise ValueError(f"slide_pitch_mm {pitch} must exceed the slide width {sw}")
    n_slots = int(sc["tray_slots"])
    if n_slots < sc["max_slides"]:
        raise ValueError("tray_slots must be >= max_slides")
    margin = 10.0
    slot_x = [margin + i * pitch for i in range(n_slots)]
    slide_y0 = 8.0
    tray_rect = (0.0, 0.0, 2 * margin + n_slots * pitch - gap, slide_y0 + sh + 42.0)
    if rng.random() < 0.6:  # contiguous run, the usual way slides are laid out
        start = int(rng.integers(0, n_slots - n + 1))
        occupied = list(range(start, start + n))
    else:
        occupied = sorted(rng.choice(n_slots, size=n, replace=False).tolist())
    max_tilt = float(sc["slide_rot_deg"])
    placed = []
    for slot in occupied:
        pose = _place_in_slot(sc, rng, slot_x[slot], slide_y0, (sw, sh), placed, max_tilt)
        pose["slot"] = slot
        placed.append(dict(pose, size=(sw, sh)))
    rect, extra = _slot_surface_rect(surface, rng, tray_rect, slot_x, sw, sh, slide_y0)
    centres = np.array([slide_outline_world(p).mean(0) for p in placed])
    return dict(poses=[{k: p[k] for k in ("x", "y", "rot", "slot")} for p in placed],
                tray_rect=tray_rect, slot_x=slot_x, slide_y0=slide_y0, gap=gap,
                surface_rect=rect, surface_extra=extra, max_tilt_deg=max_tilt,
                min_gap_mm=float(sc["min_slide_clearance_mm"]), max_gap_mm=None, frame_rect=None,
                row_center=tuple(centres.mean(0)))


def _place_in_slot(sc, rng, x0, y0, size, placed, max_tilt):
    """Sample a small jitter/tilt that keeps clearance to already placed slides."""
    jit, clear = sc["slide_jitter_mm"], sc["min_slide_clearance_mm"]
    for attempt in range(100):
        scale_down = 1.0 if attempt < 50 else 0.5
        pose = dict(x=x0 + rng.uniform(-1, 1) * jit * scale_down,
                    y=y0 + rng.uniform(-1, 1) * jit * scale_down,
                    rot=rng.uniform(-1, 1) * max_tilt * scale_down)
        cand = slide_outline_world(dict(pose, size=size))
        if all(G.convex_gap(cand, slide_outline_world(p)) >= clear for p in placed):
            return pose
    return dict(x=x0, y=y0, rot=0.0)


def _slot_surface_rect(kind, rng, tray_rect, slot_x, sw, sh, slide_y0):
    x_lo, x_hi = slot_x[0], slot_x[-1] + sw
    extra = {}
    if kind == "steel_tray":
        rect = tray_rect
    elif kind in ("plastic_tray", "slide_box"):
        rim = rng.uniform(6, 12)
        rect = (x_lo - rim, slide_y0 - rim, (x_hi - x_lo) + 2 * rim, sh + 2 * rim + rng.uniform(0, 25))
        extra["rim_mm"] = rim
    elif kind == "paper":
        pw, ph = rng.uniform(300, 460), rng.uniform(220, 330)
        cx = (x_lo + x_hi) / 2 + rng.uniform(-40, 40)
        cy = slide_y0 + sh / 2 + rng.uniform(-30, 30)
        extra.update(cx=cx, cy=cy, w=pw, h=ph, angle=rng.uniform(-6, 6))
        r = np.hypot(pw, ph) / 2
        rect = (cx - r, cy - r, 2 * r, 2 * r)
    elif kind == "desk":
        rect = None  # covers the whole view
    else:
        raise ValueError(kind)
    return rect, extra


# ---------------------------------------------------------------- flat PLA base (the real rig)

def _pla_geometry(sc, rng, n, sw, sh):
    """Flat PLA plate with an engraved frame; slides lie freely in one row inside it."""
    pc = sc["pla_base"]
    PW, PH = pc["plate_mm"]
    FW, FH = pc["frame_mm"]
    gmin, gmax = pc["gap_mm"]
    max_tilt = float(pc["max_tilt_deg"])
    fx0, fy0 = (PW - FW) / 2, (PH - FH) / 2
    inset = 0.5
    for attempt in range(200):
        shrink = max(0.0, 1.0 - attempt / 150)
        g_hi = gmin + (gmax - gmin) * shrink
        poses, prev = [], None
        for i in range(n):
            rot = rng.uniform(-1, 1) * max_tilt * max(shrink, 0.2)
            out0 = slide_outline_world(dict(x=0.0, y=0.0, rot=rot, size=(sw, sh)))
            slack_y = (FH - 2 * inset) - (out0[:, 1].max() - out0[:, 1].min())
            if slack_y < 0:
                raise ValueError("pla_base.frame_mm too short for a tilted slide")
            y = fy0 + inset - out0[:, 1].min() + rng.uniform(0, slack_y)
            pose = dict(x=0.0, y=y, rot=rot, slot=i)
            if prev is not None:
                target = rng.uniform(gmin, g_hi)
                pose["x"] = prev["x"] + sw + target
                for _ in range(6):  # SAT gap is ~linear in x: a few correction steps
                    g = G.convex_gap(slide_outline_world(dict(prev, size=(sw, sh))),
                                     slide_outline_world(dict(pose, size=(sw, sh))))
                    pose["x"] += target - g
            poses.append(pose)
            prev = pose
        outs = [slide_outline_world(dict(p, size=(sw, sh))) for p in poses]
        xs = np.concatenate([o[:, 0] for o in outs])
        width = xs.max() - xs.min()
        if width > FW - 2 * inset:
            continue
        shift = fx0 + inset - xs.min() + rng.uniform(0, FW - 2 * inset - width)
        for p in poses:
            p["x"] += shift
        frame = (fx0, fy0, FW, FH)
        return dict(poses=poses, tray_rect=(0.0, 0.0, PW, PH), slot_x=[], slide_y0=fy0, gap=None,
                    surface_rect=(0.0, 0.0, PW, PH), frame_rect=frame,
                    surface_extra=dict(frame=frame, tick_pitch=float(pc["tick_pitch_mm"]),
                                       n_ticks=int(pc["n_ticks"]), tick_len=float(pc["tick_len_mm"])),
                    max_tilt_deg=max_tilt, min_gap_mm=float(gmin), max_gap_mm=float(gmax),
                    # the rig camera hangs over the middle of the base
                    row_center=(fx0 + FW / 2, fy0 + FH / 2))
    raise RuntimeError(f"could not fit {n} slides in the {FW} mm frame")


# ---------------------------------------------------------------- geometry helpers / checks

def slide_to_world(slide):
    """Affine (3x3) from slide-local mm (label end at y=0) to world mm."""
    w, h = slide["size"]
    # every slide has the same orientation (label at the top end); only a small tilt
    A = G.translate(slide["x"] + w / 2, slide["y"] + h / 2) @ G.rotate(slide["rot"])
    return A @ G.translate(-w / 2, -h / 2)


def slide_outline_world(slide):
    w, h = slide["size"]
    return G.apply_h(slide_to_world(slide), [(0, 0), (w, 0), (w, h), (0, h)])


def label_outline_world(slide):
    lw, lh = slide["label"]["size"]
    return G.apply_h(slide_to_world(slide) @ label_to_slide(slide), [(0, 0), (lw, 0), (lw, lh), (0, lh)])


def check_layout(layout, cfg):
    """Return a list of physical-consistency problems (empty list = OK)."""
    problems = []
    slides = layout["slides"]
    tol = 0.05
    for i, s in enumerate(slides):
        if abs(s["rot"]) > layout["max_tilt_deg"] + 1e-9:
            problems.append(f"slide {i}: tilt {s['rot']:.2f} deg > {layout['max_tilt_deg']}")
        out = slide_outline_world(s)
        lab = label_outline_world(s)
        # label must lie on its own glass ...
        local = G.apply_h(np.linalg.inv(slide_to_world(s)), lab)
        w, h = s["size"]
        if local.min() < -1e-6 or local[:, 0].max() > w + 1e-6 or local[:, 1].max() > h + 1e-6:
            problems.append(f"slide {i}: label not inside its slide")
        # ... at the top end (smaller tray Y than the slide centre)
        if lab[:, 1].mean() >= out[:, 1].mean():
            problems.append(f"slide {i}: label not at the top end")
        if layout["frame_rect"] is not None:
            fx, fy, fw, fh = layout["frame_rect"]
            if out[:, 0].min() < fx or out[:, 0].max() > fx + fw or out[:, 1].min() < fy or out[:, 1].max() > fy + fh:
                problems.append(f"slide {i}: outside the PLA frame")
        for j, t in enumerate(slides):
            if j <= i:
                continue
            if G.convex_gap(out, slide_outline_world(t)) < layout["min_gap_mm"] - tol:
                problems.append(f"slides {i},{j}: closer than {layout['min_gap_mm']} mm (overlap/touch)")
            if G.convex_gap(lab, slide_outline_world(t)) < 0:
                problems.append(f"label {i} overlaps slide {j}")
            if G.convex_gap(label_outline_world(t), out) < 0:
                problems.append(f"label {j} overlaps slide {i}")
    if layout["max_gap_mm"] is not None:  # free placement: neighbour gaps inside the allowed range
        order = sorted(range(len(slides)), key=lambda k: slides[k]["x"])
        for a, b in zip(order[:-1], order[1:]):
            g = G.convex_gap(slide_outline_world(slides[a]), slide_outline_world(slides[b]))
            if g > layout["max_gap_mm"] + tol:
                problems.append(f"slides {a},{b}: gap {g:.2f} mm > {layout['max_gap_mm']}")
    return problems


def label_to_slide(slide):
    return np.eye(3)  # label sits flush at the top end of the slide


def code_corners_world(slide, label_corners_mm):
    A = slide_to_world(slide) @ label_to_slide(slide)
    return G.apply_h(A, label_corners_mm)


def code_corners_label(slide):
    lab = slide["label"]
    x, y, s = lab["code_x_mm"], lab["code_y_mm"], lab["code_mm"]
    return np.array([[x, y], [x + s, y], [x + s, y + s], [x, y + s]])


def sample_camera(cfg, mode, layout, rng, ppm):
    cc = cfg["camera"][mode]
    W, H = cfg["camera"]["image_size"]
    hfov = _u(rng, cc["hfov_deg"])
    tilt = _u(rng, cc["tilt_deg"])
    roll = _u(rng, cc["roll_deg"])
    az = rng.uniform(0, 360)
    slides = layout["slides"]
    centres = np.array([code_corners_world(s, code_corners_label(s)).mean(0) for s in slides])
    code_mm = float(cfg["code"]["CODE_SIZE_MM"])
    mods_ref = slides[0]["modules"]
    px_per_mm = ppm * mods_ref / code_mm
    view_w = W / px_per_mm
    if mode == "rig":
        # bridge rig: camera hangs straight down over the middle of the row
        tx, ty = np.asarray(layout["row_center"]) + rng.uniform(-1, 1, 2) * cc["center_jitter_mm"]
        return _solve_distance(cam_from(W, H, hfov, tilt, az, roll, tx, ty, px_per_mm), slides, centres, tx, ppm)
    lo, hi = centres[:, 0].min(), centres[:, 0].max()
    span = hi - lo + layout["slide_w"] + 6
    if span < view_w * 0.92:
        slack = (view_w * 0.92 - span) / 2
        tx = (lo + hi) / 2 + rng.uniform(-slack, slack) * 0.8
    else:
        # not everything fits: aim near one slide so the frame never lands on an empty gap
        anchor = centres[int(rng.integers(0, len(centres))), 0]
        tx = anchor + rng.uniform(-1, 1) * max(0.0, view_w / 2 - 20) * 0.7
    view_h = H / px_per_mm
    code_y = centres[:, 1].mean()
    ty = code_y + rng.uniform(-0.1, 0.35) * view_h
    ty = min(ty, code_y + view_h * 0.4)
    return _solve_distance(cam_from(W, H, hfov, tilt, az, roll, tx, ty, px_per_mm), slides, centres, tx, ppm)


def cam_from(W, H, hfov, tilt, az, roll, tx, ty, px_per_mm):
    f = 0.5 * W / np.tan(np.deg2rad(hfov) / 2)
    return G.Camera(W, H, hfov, tilt, az, roll, (tx, ty), f / px_per_mm)


def _solve_distance(cam, slides, centres, tx, ppm):
    """Set the camera distance so the code nearest the aim point hits the target ppm."""
    ref = int(np.argmin(np.abs(centres[:, 0] - tx)))
    ref_s = slides[ref]
    world = code_corners_world(ref_s, code_corners_label(ref_s))
    for _ in range(4):  # solve distance so the reference code hits the target ppm
        img = G.apply_h(cam.plane_h(Z_LABEL), world)
        got = G.quad_ppm(img, ref_s["modules"])["ppm"]
        cam.set_distance(cam.distance * got / ppm)
    return cam
