"""Per-code read outcomes for paired statistics (no GPU, uses saved predictions).

For every scored code: detected, and read / not read with each decoder input
(full_image, bbox_crop, quad_warp, warp_then_crop). The same scoring as evaluate.py.

  python scripts/per_code.py --models maskrcnn yolo11n yolo26n rfdetr rfdetr_long oracle

Writes runs/eval/<model>/per_code_<split>.csv. All numbers are from SYNTHETIC images.
"""
import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import SPLITS, full_image_cache  # noqa: E402
from slideqr import evaluation as EV  # noqa: E402
from slideqr.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def oracle_preds(img, gts):
    """Ground-truth quads as predictions, exactly as evaluate.py's oracle."""
    H, W = img.shape[:2]
    preds = []
    for g in gts:
        if g["iscrowd"]:
            continue
        m = np.zeros((H, W), np.uint8)
        q = np.asarray(g["corners"], float).reshape(4, 2)
        cv2.fillPoly(m, [np.round((q - 0.5) * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        preds.append(EV.det_to_record(1.0, "code", m.astype(bool)))
    return preds


def run(model, cfg, root, eval_dir):
    from pycocotools.coco import COCO
    ec = cfg["eval"]
    edges = np.asarray(ec["ppm_bins"], float)
    focus = {k: tuple(v) for k, v in ec["focus_bins"].items()}
    out = Path(eval_dir) / model
    for split in SPLITS:
        ann = root / "annotations" / f"instances_{split}.json"
        pkl = out / f"preds_{split}.pkl"
        if not ann.exists() or (model != "oracle" and not pkl.exists()):
            print("skip", model, split)
            continue
        coco = COCO(str(ann))
        preds = None
        if model != "oracle":
            with open(pkl, "rb") as f:
                preds = pickle.load(f)
        full = full_image_cache(root, split, coco, eval_dir)
        tally = EV.Tally(edges, focus)   # not written; keeps evaluate_image unchanged
        records = []
        for iid in sorted(coco.getImgIds()):
            img = cv2.imread(str(root / "images" / split / coco.imgs[iid]["file_name"]))
            gts = EV.gt_instances(coco, iid, 1)
            p = oracle_preds(img, gts) if model == "oracle" else preds.get(iid, [])
            EV.evaluate_image(img, gts, p, ec, tally, full_texts=full[iid], records=records, image_id=iid)
        path = out / f"per_code_{split}.csv"
        out.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            w.writeheader()
            w.writerows(records)
        # cross-check against the published per-bin table
        rows = {r["bin"]: r for r in tally.table()}
        print(model, split, len(records), "codes; all-bin warp_then_crop",
              f"{rows['all']['warp_then_crop_read_rate']:.4f}", "->", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--config", default=str(ROOT / "configs" / "train.yaml"))
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--eval-dir", default=str(ROOT / "runs" / "eval"))
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    root = Path(cfg["data_root"])
    for m in args.models:
        run(m, cfg, root, args.eval_dir)


if __name__ == "__main__":
    main()
