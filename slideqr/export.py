"""Dataset writers: YOLO segmentation, COCO instances, metadata.csv."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from . import geometry as G

CLASSES = ["code", "slide"]
COCO_CAT = {"code": 1, "slide": 2}


def assign_splits(cfg, ppm_targets, rng):
    """Stratify by ppm: walk the images in ppm order and give each one to the split
    that is furthest behind its quota. Every split gets the same ppm distribution and
    the totals match the ratios exactly, even for tiny datasets."""
    ratios = cfg["split"]["ratios"]
    names = list(ratios)
    r = np.array([ratios[k] for k in names], float)
    r /= r.sum()
    ppm = np.asarray(ppm_targets, float)
    order = np.lexsort((rng.random(len(ppm)), ppm))   # random tie-break
    assigned = np.zeros(len(names))
    out = np.empty(len(ppm), dtype=object)
    for j, i in enumerate(order, start=1):
        k = int(np.argmax(j * r - assigned))
        assigned[k] += 1
        out[i] = names[k]
    return out


def save_image(path, img, jpeg_quality):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if jpeg_quality:
        path = path.with_suffix(".jpg")
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
    else:
        path = path.with_suffix(".png")
        cv2.imwrite(str(path), img)
    return path


def instances(sample, include_slide):
    """Yield (class name, clipped polygon, amodal corners, ignore, code dict)."""
    for c in sample["codes"]:
        if c["in_frame_frac"] > 0 and len(c["poly_clipped"]) >= 3:
            yield "code", c["poly_clipped"], c["corners"], c["ignore"], c
        if include_slide and c["slide_visible_frac"] > 0 and len(c["slide_poly_clipped"]) >= 3:
            yield "slide", c["slide_poly_clipped"], c["slide_corners"], c["slide_ignore"], c


def yolo_lines(sample, include_slide):
    W, H = sample["size"]
    lines = []
    for cls, poly, _, ignore, _ in instances(sample, include_slide):
        if ignore:
            continue  # YOLO has no ignore regions; heavily hidden objects are left out
        p = np.clip(poly / (W, H), 0, 1)
        lines.append(f"{CLASSES.index(cls)} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)))
    return lines


def coco_anns(sample, image_id, next_id, include_slide):
    anns = []
    for cls, poly, corners, ignore, c in instances(sample, include_slide):
        x0, y0 = poly.min(0)
        x1, y1 = poly.max(0)
        ann = dict(id=next_id + len(anns), image_id=image_id, category_id=COCO_CAT[cls],
                   segmentation=[[round(float(v), 3) for v in poly.reshape(-1)]],
                   area=round(G.polygon_area(poly), 3),
                   bbox=[round(float(x0), 3), round(float(y0), 3), round(float(x1 - x0), 3), round(float(y1 - y0), 3)],
                   iscrowd=int(bool(ignore)),
                   corners=[round(float(v), 3) for v in np.asarray(corners).reshape(-1)])
        if cls == "code":
            ann.update(payload=c["payload"], symbology=c["symbology"], ppm=round(c["ppm"], 4),
                       visible_frac=round(c["visible_frac"], 4))
        anns.append(ann)
    return anns


def metadata_rows(sample, split, file_name):
    rows = []
    for i, c in enumerate(sample["codes"]):
        r = dict(image=file_name, split=split, mode=sample["mode"], scene_index=sample["index"],
                 surface=sample["surface"], surface_variant=sample["surface_variant"],
                 code_index=i, slot=c["slot"], n_slides_in_scene=sample["n_slides"],
                 symbology=c["symbology"], modules=c["modules"], payload=c["payload"],
                 slide_tilt_deg=round(c["slide_tilt_deg"], 3), ppm_target=round(sample["ppm_target"], 4),
                 ppm=round(c["ppm"], 4), ppm_min_side=round(c["ppm_min_side"], 4),
                 ppm_max_side=round(c["ppm_max_side"], 4),
                 in_frame_frac=round(c["in_frame_frac"], 4), occluded_frac=round(c["occluded_frac"], 4),
                 visible_frac=round(c["visible_frac"], 4),
                 flag_out_of_frame=int(c["flag_out_of_frame"]), flag_occluded=int(c["flag_occluded"]),
                 ignore=int(c["ignore"]), label_stain=round(c["label_stain"], 4),
                 label_wrinkle=round(c["label_wrinkle"], 4))
        for k, (x, y) in enumerate(c["corners"]):
            r[f"c{k}x"], r[f"c{k}y"] = round(float(x), 3), round(float(y), 3)
        r.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in sample["camera"].items()})
        r.update({f"deg_{k}": (round(v, 4) if isinstance(v, float) else v) for k, v in sample["degrade"].items()})
        occ = sample["occluders"]
        r["hand"] = int(bool(occ.get("hand")))
        r["paper_edge"] = int(bool(occ.get("paper_edge")))
        rows.append(r)
    return rows


def write_csv(path, rows):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_coco(path, images, anns):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = dict(info=dict(description="slideqr-seg SYNTHETIC dataset", note="synthetic images only"),
                images=images, annotations=anns,
                categories=[dict(id=v, name=k, supercategory="object") for k, v in COCO_CAT.items()])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def write_data_yaml(root, splits):
    lines = [f"path: {Path(root).resolve().as_posix()}"]
    for s in splits:
        lines.append(f"{s}: images/{s}")
    lines.append("names:")
    for i, n in enumerate(CLASSES):
        lines.append(f"  {i}: {n}")
    (Path(root) / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
