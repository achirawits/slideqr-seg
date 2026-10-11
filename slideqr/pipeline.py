"""One synthetic sample = layout -> camera -> clean render -> degradations."""
import numpy as np

from . import degrade as DG
from . import geometry as G
from . import render as R
from . import scene as S

MODE_ID = {"rig": 0, "handheld": 1}


def rng_for(cfg, mode, index):
    return np.random.default_rng([int(cfg["seed"]), MODE_ID[mode], int(index)])


def stratified_ppm(cfg, n, rng):
    """n target ppm values, one per equal-width stratum (shuffled)."""
    lo, hi = cfg["scale"]["ppm_range"]
    u = (rng.permutation(n) + rng.random(n)) / n
    return lo + u * (hi - lo)


def generate(cfg, mode, index, ppm, n_slides=None, degrade=True):
    rng = rng_for(cfg, mode, index)
    layout = S.sample_layout(cfg, rng, n_slides)
    problems = S.check_layout(layout, cfg)
    if problems:
        raise RuntimeError(f"invalid layout {mode}#{index}: {problems}")
    cam = S.sample_camera(cfg, mode, layout, rng, ppm)
    P, stains = DG.sample_params(cfg, mode, rng, len(layout["slides"])) if degrade else ({}, {})
    out = R.render_scene(cfg, layout, cam, rng, stain_params=stains)
    clean = np.clip(out["image"] * 255 + 0.5, 0, 255).astype(np.uint8)
    if degrade:
        lit = out["image"] * out["overlay"]
        img = DG.apply(lit, out["material"], P, cfg, rng)
    else:
        img = clean
    min_vis = cfg["labels"]["min_visible_frac"]
    for i, c in enumerate(out["codes"]):
        c.update(G.quad_ppm(c["corners"], c["modules"]))
        c["slide_tilt_deg"] = float(layout["slides"][c["slide_index"]]["rot"])
        c["label_stain"] = stains.get(c["slide_index"], {}).get("stain", 0.0)
        c["label_wrinkle"] = stains.get(c["slide_index"], {}).get("wrinkle", 0.0)
        c["flag_out_of_frame"] = c["in_frame_frac"] < 0.999
        c["flag_occluded"] = c["occluded_frac"] > 0.5
        c["ignore"] = c["visible_frac"] < min_vis
        c["slide_ignore"] = c["slide_visible_frac"] < min_vis
    return dict(mode=mode, index=index, ppm_target=float(ppm), image=img, clean=clean,
                codes=out["codes"], camera=cam.params(), degrade=P, n_slides=len(layout["slides"]),
                occluders=out["occluder_params"], size=(cam.W, cam.H),
                slide_tilts=[sl["rot"] for sl in layout["slides"]],
                surface=layout["surface"], surface_variant=layout["surface_variant"])
