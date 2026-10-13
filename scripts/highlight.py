"""Step 4: side-by-side highlight images for every model.

Three panels per image: original | predicted masks | overlay with one colour per code,
its confidence and the text decoded from the 4-corner warp. Uses the predictions
saved by evaluate.py (no GPU needed). Images are cropped to the slide row so the
small codes stay visible.

  python scripts/highlight.py --split test --n 6
"""
import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slideqr import evaluation as EV  # noqa: E402
from slideqr.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["maskrcnn", "yolo11n", "yolo26n", "rfdetr", "rfdetr_long"]  # rfdetr_long = supplementary run
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178),
           (213, 94, 0), (204, 121, 167), (255, 255, 255), (120, 200, 255), (200, 120, 255)]  # BGR-ish, distinct


def pick_images(coco, n):
    """Images spread over the ppm range (median ppm of their codes)."""
    meds = []
    for iid in sorted(coco.getImgIds()):
        p = [a["ppm"] for a in coco.imgToAnns[iid] if a["category_id"] == 1 and not a["iscrowd"]]
        if p:
            meds.append((float(np.median(p)), iid))
    meds.sort()
    idx = np.linspace(0, len(meds) - 1, n).round().astype(int)
    return [meds[i][1] for i in idx]


def crop_box(coco, iid, W, H, margin=0.08):
    anns = [a for a in coco.imgToAnns[iid] if a["category_id"] == 2] or coco.imgToAnns[iid]
    pts = np.concatenate([np.asarray(a["segmentation"][0]).reshape(-1, 2) for a in anns])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    mx, my = (x1 - x0) * margin + 20, (y1 - y0) * margin + 20
    return int(max(0, x0 - mx)), int(max(0, y0 - my)), int(min(W, x1 + mx)), int(min(H, y1 + my))


def label(img, text, org, scale, outline=True):
    if outline:  # dark halo for text drawn over the photo (a thicker stroke is also wider)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), max(2, int(4 * scale)), cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(1.5 * scale)), cv2.LINE_AA)


def panels(img, preds, thr, ppm_px):
    H, W = img.shape[:2]
    mask_panel = np.zeros_like(img)
    over = img.copy()
    preds = sorted([p for p in preds if p["score"] >= thr], key=lambda p: -p["score"])
    scale = max(0.6, W / 2500)
    texts = []
    for k, p in enumerate(preds):
        col = np.array(PALETTE[k % len(PALETTE)], np.uint8)
        m = mask_utils.decode(p["rle"]).astype(bool)
        mask_panel[m] = col
        over[m] = (0.45 * over[m] + 0.55 * col).astype(np.uint8)
        quad = EV.poly_to_quad(p["poly"])
        text = EV.decode_quad_warp(img, quad, ppm_px) if quad is not None else None
        if quad is not None:
            cv2.polylines(over, [np.round((quad - 0.5) * 16).astype(np.int32)], True, tuple(int(c) for c in col),
                          max(2, int(2 * scale)), cv2.LINE_AA, 4)
        x0, y0 = p["poly"].min(0).astype(int) if p["poly"] is not None else (0, 0)
        label(over, f"{p['score']:.2f}", (x0, max(15, y0 - 8)), scale * 0.8)
        texts.append((col, f"{p['score']:.2f}  {text or '(no read)'}"))
    return mask_panel, over, texts


def compose(img, mask_panel, over, texts, title, box):
    x0, y0, x1, y1 = box
    crops = [c[y0:y1, x0:x1] for c in (img, mask_panel, over)]
    h = crops[0].shape[0]
    gap = np.full((h, 12, 3), 40, np.uint8)
    row = np.hstack([crops[0], gap, crops[1], gap, crops[2]])
    target_w = 2400
    row = cv2.resize(row, (target_w, int(row.shape[0] * target_w / row.shape[1])), interpolation=cv2.INTER_AREA)
    lines = max(1, (len(texts) + 2) // 3)
    footer = np.full((80 + 50 * lines, target_w, 3), 25, np.uint8)
    label(footer, title, (12, 50), 1.3, outline=False)
    for i, (col, t) in enumerate(texts):
        x = 12 + (i % 3) * (target_w // 3)
        y = 110 + 50 * (i // 3)
        cv2.rectangle(footer, (x, y - 28), (x + 32, y + 4), tuple(int(c) for c in col), -1)
        label(footer, t, (x + 42, y), 1.05, outline=False)
    return np.vstack([row, footer])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "highlight"))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    ec = cfg["eval"]
    root = Path(cfg["data_root"])
    coco = COCO(str(root / "annotations" / f"instances_{args.split}.json"))
    preds = {}
    for m in ORDER:
        p = Path(args.eval_dir) / m / f"preds_{args.split}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                preds[m] = pickle.load(f)
    if not preds:
        raise SystemExit("no saved predictions; run evaluate.py first")
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    for iid in pick_images(coco, args.n):
        info = coco.imgs[iid]
        img = cv2.imread(str(root / "images" / args.split / info["file_name"]))
        H, W = img.shape[:2]
        box = crop_box(coco, iid, W, H)
        gts = [a for a in coco.imgToAnns[iid] if a["category_id"] == 1 and not a["iscrowd"]]
        med = float(np.median([a["ppm"] for a in gts])) if gts else float("nan")
        rows = []
        for m, pm in preds.items():
            mask_panel, over, texts = panels(img, pm.get(iid, []), ec["score_thr"], ec["rectify_px_per_module"])
            n_ok = sum(1 for _, t in texts if any(t.endswith(a["payload"]) for a in gts))
            title = f"{m} | {info['file_name']} | {len(gts)} codes, median {med:.2f} ppm | read {n_ok}/{len(gts)} (SYNTHETIC)"
            panel = compose(img, mask_panel, over, texts, title, box)
            cv2.imwrite(str(out / f"{Path(info['file_name']).stem}_{m}.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])
            rows.append(panel)
        w = max(r.shape[1] for r in rows)
        grid = np.vstack([cv2.copyMakeBorder(r, 0, 16, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in rows])
        cv2.imwrite(str(out / f"{Path(info['file_name']).stem}_all_models.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(out / f"{Path(info['file_name']).stem}_all_models.jpg")


if __name__ == "__main__":
    main()
