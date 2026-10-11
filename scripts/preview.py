"""Step-1 review pack: 12 sample scenes (1..10 slides, both camera modes),
polygon overlays, zoomed corner crops and a clean-image decode check.

  python scripts/preview.py --out runs/preview_step1
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import codes as C  # noqa: E402
from slideqr import geometry as G  # noqa: E402
from slideqr import pipeline as P  # noqa: E402
from slideqr import scene as S  # noqa: E402
from slideqr.config import load_config  # noqa: E402

# slide counts cover 1..10 in both modes; the PLA base (the real rig) appears 4 times,
# every other default surface at least twice
PLAN = [("rig", 1, "pla_base"), ("rig", 2, "steel_tray"), ("rig", 4, "plastic_tray"),
        ("rig", 6, "desk"), ("rig", 8, "paper"), ("rig", 10, "pla_base"),
        ("handheld", 1, "paper"), ("handheld", 3, "desk"), ("handheld", 5, "pla_base"),
        ("handheld", 7, "plastic_tray"), ("handheld", 9, "steel_tray"), ("handheld", 10, "pla_base")]
CORNER_COLORS = [(0, 0, 255), (0, 200, 255), (0, 255, 0), (255, 0, 0)]  # TL red, TR orange, BR green, BL blue


def find_scene(cfg, mode, n, start):
    """Search indices for a layout where all n codes are fully in frame."""
    code_mm = cfg["code"]["CODE_SIZE_MM"]
    lo, hi = cfg["scale"]["ppm_range"]
    W = cfg["camera"]["image_size"][0]
    for idx in range(start, start + 4000):
        rng = P.rng_for(cfg, mode, idx)
        span = n * cfg["scene"]["slide_pitch_mm"] + 15
        mods = C.modules_for(cfg, "qr" if cfg["code"]["symbology"] != "datamatrix" else "datamatrix")
        ppm_max = min(hi, W / span * code_mm / mods * (0.75 if mode == "handheld" else 0.95))
        ppm = float(np.random.default_rng(idx).uniform(lo, max(lo, ppm_max)))
        layout = S.sample_layout(cfg, rng, n)
        cam = S.sample_camera(cfg, mode, layout, rng, ppm)
        Hl = cam.plane_h(S.Z_LABEL)
        ok = all(_inside(G.apply_h(Hl, S.code_corners_world(s, S.code_corners_label(s))), cam.W, cam.H, 8)
                 for s in layout["slides"])
        if ok:
            return idx, ppm
    raise RuntimeError(f"no scene found for {mode} n={n}")


def _inside(pts, W, H, m):
    return bool(np.all(pts[:, 0] > m) and np.all(pts[:, 0] < W - m) and np.all(pts[:, 1] > m) and np.all(pts[:, 1] < H - m))


def fp(pts, shift=4):
    """Edge coords -> fixed-point pixel-centre coords for cv2 drawing."""
    return np.round((np.asarray(pts) - 0.5) * (1 << shift)).astype(np.int32)


def draw_overlay(img, s, decoded):
    out = img.copy()
    t = max(2, img.shape[1] // 900)
    for c, dec in zip(s["codes"], decoded):
        cv2.polylines(out, [fp(c["slide_corners"])], True, (255, 160, 0), t, cv2.LINE_AA, 4)
    for c, dec in zip(s["codes"], decoded):
        col = (0, 255, 0) if not c["ignore"] else (0, 0, 255)
        cv2.polylines(out, [fp(c["corners"])], True, col, t, cv2.LINE_AA, 4)
        for k, p in enumerate(c["corners"]):
            cv2.circle(out, tuple(fp([p])[0]), 4 * t, CORNER_COLORS[k], -1, cv2.LINE_AA, 4)
        org = c["corners"].min(0).astype(int) + (0, -12)
        txt = f"{c['ppm']:.1f}ppm {'OK' if dec else 'FAIL'}"
        cv2.putText(out, txt, tuple(org), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 0, 0), 3 * t, cv2.LINE_AA)
        cv2.putText(out, txt, tuple(org), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 255, 255), t, cv2.LINE_AA)
    return out


def crop_code(img, c, size=260):
    """Zoomed crop around a code with sub-pixel exact polygon and corner dots."""
    ctr = c["corners"].mean(0)
    half = np.linalg.norm(c["corners"][0] - c["corners"][2]) * 0.75
    zoom = size / (2 * half)
    M = np.float32([[zoom, 0, size / 2 - zoom * ctr[0]], [0, zoom, size / 2 - zoom * ctr[1]]])
    # edge-based affine -> pixel-centre convention
    M[:, 2] += 0.5 * (M[0, 0] - 1)
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=(40, 40, 40))
    pts = c["corners"] * zoom + (M[0, 2] - 0.5 * (M[0, 0] - 1), M[1, 2] - 0.5 * (M[0, 0] - 1))
    cv2.polylines(crop, [fp(pts)], True, (0, 255, 0), 1, cv2.LINE_AA, 4)
    for k, p in enumerate(pts):
        cv2.circle(crop, tuple(fp([p])[0]), 4 * 16, CORNER_COLORS[k], 1, cv2.LINE_AA, 4)
    cv2.putText(crop, f"{c['ppm']:.2f} ppm", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return crop


def grid(images, cols, cell_w):
    cells = []
    for im in images:
        h = int(im.shape[0] * cell_w / im.shape[1])
        cells.append(cv2.resize(im, (cell_w, h), interpolation=cv2.INTER_AREA))
    ch = max(c.shape[0] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=0) for c in cells]
    while len(cells) % cols:
        cells.append(np.zeros_like(cells[0]))
    rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
    return np.vstack(rows)


def bulk_layout_check(cfg, n):
    """Run the physical-consistency checks on many random layouts."""
    bad, n_slides = [], 0
    per = {}
    for i in range(n):
        layout = S.sample_layout(cfg, np.random.default_rng([int(cfg["seed"]), 7, i]))
        n_slides += len(layout["slides"])
        st = per.setdefault(layout["surface"], dict(layouts=0, max_tilt_deg=0.0, min_gap_mm=np.inf, max_gap_mm=0.0))
        st["layouts"] += 1
        st["max_tilt_deg"] = max(st["max_tilt_deg"], max(abs(s["rot"]) for s in layout["slides"]))
        sl = sorted(layout["slides"], key=lambda s: s["x"])
        for a, b in zip(sl[:-1], sl[1:]):
            g = G.convex_gap(S.slide_outline_world(a), S.slide_outline_world(b))
            st["min_gap_mm"], st["max_gap_mm"] = min(st["min_gap_mm"], g), max(st["max_gap_mm"], g)
        p = S.check_layout(layout, cfg)
        if p:
            bad.append((i, p))
    per = {k: {kk: (round(float(vv), 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}
           for k, v in per.items()}
    return dict(layouts=n, slides=n_slides, bad_layouts=len(bad), per_surface=per, examples=bad[:5])


def image_check(s):
    """In the rendered image every code must lie inside its own slide polygon only."""
    problems = []
    for i, c in enumerate(s["codes"]):
        for j, d in enumerate(s["codes"]):
            inside = [cv2.pointPolygonTest(d["slide_corners"].astype(np.float32), tuple(map(float, p)), False)
                      for p in c["corners"]]
            if i == j and min(inside) < 0:
                problems.append(f"code {i} not inside its slide")
            if i != j and max(inside) >= 0:
                problems.append(f"code {i} inside slide {j}")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--out", default="runs/preview_step1")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    bulk = bulk_layout_check(cfg, 5000)
    print("layout check:", bulk)
    report, overlays, raws, crops = [], [], [], []
    for k, (mode, n, surface) in enumerate(PLAN):
        scfg = copy.deepcopy(cfg)
        scfg["scene"]["surface"].update(types=[surface], weights=[1])
        idx, ppm = find_scene(scfg, mode, n, 100000 + 1000 * k)
        s = P.generate(scfg, mode, idx, ppm, n_slides=n)
        name = f"{k:02d}_{mode}_n{n:02d}_{surface}"
        cv2.imwrite(str(out / f"{name}.jpg"), s["image"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(out / f"{name}_clean.png"), s["clean"])
        dec_clean, dec_deg, bits, ov_status = [], [], [], []
        for c in s["codes"]:
            # decode test only on codes that are fully in frame and not covered (e.g. paper edge)
            full = c["in_frame_frac"] > 0.999 and c["occluded_frac"] == 0
            dc = C.decode_rectified(s["clean"], c["corners"], c["symbology"], c["modules"]) if full else None
            dd = C.decode_rectified(s["image"], c["corners"], c["symbology"], c["modules"]) if full else None
            be = int((C.sample_modules(s["clean"], c["corners"], c["modules"]) != c["matrix"]).sum()) if full else -1
            ov_status.append(dd == c["payload"])
            if full:
                dec_clean.append(dc == c["payload"])
                dec_deg.append(dd == c["payload"])
                bits.append(be)
        direct = C.decode_direct(s["clean"])
        direct_ok = len({t for t, _ in direct} & {c["payload"] for c in s["codes"]})
        ov = draw_overlay(s["image"], s, ov_status)
        cv2.imwrite(str(out / f"{name}_overlay.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 92])
        overlays.append(ov)
        raws.append(s["image"])
        for c in s["codes"][:2]:
            if c["in_frame_frac"] > 0.999:
                crops.append(crop_code(s["image"], c))
        rec = dict(name=name, mode=mode, n_slides=n, scene_index=idx, ppm_target=round(ppm, 3),
                   ppm_measured=[round(c["ppm"], 2) for c in s["codes"]],
                   visible=[round(c["visible_frac"], 2) for c in s["codes"]],
                   clean_rectified_decode=f"{sum(dec_clean)}/{len(dec_clean)}",
                   clean_bit_errors=bits,
                   clean_direct_zxing=f"{direct_ok}/{len(s['codes'])}",
                   degraded_rectified_decode=f"{sum(dec_deg)}/{len(dec_deg)}",
                   image_check=image_check(s), surface=f"{s['surface']}/{s['surface_variant']}",
                   tilts_deg=[round(float(x), 2) for x in s["slide_tilts"]],
                   hand=bool(s["occluders"].get("hand")), degrade={k2: (round(v, 3) if isinstance(v, float) else v)
                                                                  for k2, v in s["degrade"].items()})
        report.append(rec)
        print(name, rec["surface"], "image_check", rec["image_check"] or "OK",
              "max_tilt", max(abs(t) for t in rec["tilts_deg"]), "hand", rec["hand"])
        print("   ", rec["clean_rectified_decode"], "bits", bits, "direct", rec["clean_direct_zxing"],
              "degraded", rec["degraded_rectified_decode"], "ppm", rec["ppm_measured"])
    cv2.imwrite(str(out / "contact_images.jpg"), grid(raws, 4, 800), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / "contact_overlays.jpg"), grid(overlays, 4, 800), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / "contact_crops.jpg"), grid(crops, 6, 260), [cv2.IMWRITE_JPEG_QUALITY, 95])
    (out / "report.json").write_text(json.dumps(dict(layout_check=bulk, images=report), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
